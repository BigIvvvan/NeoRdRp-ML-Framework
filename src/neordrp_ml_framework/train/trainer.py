"""Generic task trainer.

The trainer owns the optimisation loop, checkpointing, and dev metrics.  It
does not own model architecture or data parsing; those come from
``build_model`` and ``read_records`` respectively, so swapping a component
never requires touching this file.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import AminoAcidTokenizer, ProteinDataset, collate_records, read_records
from ..eval.metrics import binary_metrics, boundary_metrics, multiclass_metrics
from ..nn.models import build_model
from .config import TrainConfig
from .losses import detection_loss, family_loss, localization_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class Trainer:
    """Train one task model from a :class:`TrainConfig`."""

    def __init__(self, config: TrainConfig, model: nn.Module | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = AminoAcidTokenizer(max_len=config.data.max_len)

        train_records = read_records(
            config.data.train,
            id_column=config.data.id_column,
            sequence_column=config.data.sequence_column,
        )
        if not train_records:
            raise ValueError(f"no records read from {config.data.train}")
        self.family_to_index: dict[str, int] = {}
        if config.task == "family":
            names = sorted({record.family for record in train_records if record.family is not None})
            if not names:
                raise ValueError("family task requires a 'family' column with non-empty values")
            self.family_to_index = {name: index for index, name in enumerate(names)}

        self.train_dataset = ProteinDataset(train_records, self.tokenizer, config.task, self.family_to_index)
        self.dev_dataset = None
        if config.data.dev:
            dev_records = read_records(
                config.data.dev,
                id_column=config.data.id_column,
                sequence_column=config.data.sequence_column,
            )
            self.dev_dataset = ProteinDataset(dev_records, self.tokenizer, config.task, self.family_to_index)

        self.model = (model or build_model(config.model)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = self._make_scheduler()
        self.class_weights = self._class_weights() if config.task == "family" else None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_scheduler(self) -> Any:
        if self.config.scheduler == "none":
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.config.epochs)
        )

    def _class_weights(self) -> torch.Tensor | None:
        if not self.config.class_weights:
            return None
        counts = np.zeros(len(self.family_to_index), dtype=np.float64)
        for record in self.train_dataset.records:
            if record.family in self.family_to_index:
                counts[self.family_to_index[record.family]] += 1
        weights = np.where(counts > 0, len(counts) / np.maximum(counts, 1), 1.0)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _loader(self, dataset: ProteinDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=lambda batch: collate_records(batch, self.config.task),
            pin_memory=self.device.type == "cuda",
        )

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["mask"].to(self.device)
        if self.config.task == "detection":
            logits = self.model(input_ids, mask)
            loss = detection_loss(logits, batch["label"].to(self.device), self.config.pos_weight)
            return loss, logits
        if self.config.task == "family":
            logits = self.model(input_ids, mask)
            loss = family_loss(
                logits,
                batch["label"].to(self.device),
                self.class_weights,
                self.config.label_smoothing,
            )
            return loss, logits
        logits = self.model(input_ids, mask)
        loss = localization_loss(
            logits,
            batch["start"].to(self.device),
            batch["end"].to(self.device),
            mask,
            self.config.pos_weight,
        )
        return loss, logits

    def _dev_metrics(self) -> dict[str, float]:
        if self.dev_dataset is None:
            return {}
        self.model.eval()
        loader = self._loader(self.dev_dataset, shuffle=False)
        all_labels: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        all_starts: list[torch.Tensor] = []
        all_ends: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["mask"].to(self.device)
                logits = self.model(input_ids, mask)
                all_logits.append(logits.detach().cpu())
                if self.config.task == "localization":
                    all_starts.append(batch["start"])
                    all_ends.append(batch["end"])
                else:
                    all_labels.append(batch["label"])
        logits = torch.cat(all_logits)
        if self.config.task == "detection":
            labels = torch.cat(all_labels).numpy()
            scores = torch.sigmoid(logits).numpy()
            return binary_metrics(labels, scores)
        if self.config.task == "family":
            labels = torch.cat(all_labels).numpy()
            return multiclass_metrics(labels, logits.numpy())
        starts = torch.cat(all_starts).numpy()
        ends = torch.cat(all_ends).numpy()
        predicted_start = logits[:, :, 0].argmax(dim=1).numpy()
        predicted_end = logits[:, :, 1].argmax(dim=1).numpy()
        return boundary_metrics(starts, ends, predicted_start, predicted_end)

    def _score(self, metrics: dict[str, float]) -> float:
        if self.config.task == "detection":
            return float(metrics.get("auroc", -math.inf))
        if self.config.task == "family":
            return float(metrics.get("macro_f1", -math.inf))
        return float(metrics.get("both_e10", -math.inf))

    def fit(self) -> dict[str, Any]:
        """Run training and return the final metrics dictionary."""
        seed_everything(self.config.seed)
        train_loader = self._loader(self.train_dataset, shuffle=True)
        best_score = -math.inf
        best_epoch = -1
        patience = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}/{self.config.epochs}", leave=False):
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self._forward_loss(batch)
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                running += float(loss.detach())
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = self._dev_metrics()
            score = self._score(metrics)
            record = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), **metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience = 0
                self._save_checkpoint(epoch, metrics)
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                    break

        result = {
            "task": self.config.task,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
            "config": asdict(self.config),
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return result

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "model_config": self.config.model,
            "task": self.config.task,
            "family_to_index": self.family_to_index,
            "max_len": self.tokenizer.max_len,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(payload, self.output_dir / "best.pt")

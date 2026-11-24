"""Checkpoint loading and batched inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import AminoAcidTokenizer, ProteinDataset, ProteinRecord, collate_records
from .nn.decode import decode_pair
from .nn.models import build_model


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a model and metadata saved by :class:`Trainer`.

    ``torch.load`` uses ``weights_only=False`` because the checkpoint contains
    plain dictionaries as well as tensors.  Only load checkpoints you trust.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_records(
    model: torch.nn.Module,
    payload: dict[str, Any],
    records: list[ProteinRecord],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Run inference on records and return a prediction table."""
    task = payload.get("task", "detection")
    tokenizer = AminoAcidTokenizer(max_len=int(payload.get("max_len", 1024)))
    dataset = ProteinDataset(records, tokenizer, task, payload.get("family_to_index", {}))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_records(batch, task),
    )
    index_to_family = {index: name for name, index in payload.get("family_to_index", {}).items()}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            logits = model(input_ids, mask).detach().cpu()
            if task == "detection":
                scores = torch.sigmoid(logits).numpy()
                for protein_id, score in zip(batch["protein_id"], scores):
                    rows.append({"protein_id": protein_id, "score": float(score), "label": int(score >= 0.5)})
            elif task == "family":
                probabilities = torch.softmax(logits, dim=-1).numpy()
                for protein_id, row in zip(batch["protein_id"], probabilities):
                    index = int(np.argmax(row))
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "prediction": index_to_family.get(index, str(index)),
                            "family_id": index,
                            "confidence": float(row[index]),
                        }
                    )
            else:
                probabilities = torch.sigmoid(logits).numpy()
                for protein_id, row, valid in zip(batch["protein_id"], probabilities, batch["mask"].numpy()):
                    length = int(valid.sum())
                    start, end = decode_pair(row[:length, 0], row[:length, 1], valid[:length])
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "start_pred": start + 1,
                            "end_pred": end + 1,
                            "length": length,
                            "start_prob": float(row[start, 0]),
                            "end_prob": float(row[end, 1]),
                        }
                    )
    return pd.DataFrame(rows)

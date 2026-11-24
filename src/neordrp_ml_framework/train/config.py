"""Training configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset paths and column aliases."""

    train: str
    dev: str | None = None
    max_len: int = 1024
    id_column: str | None = None
    sequence_column: str | None = None


@dataclass
class TrainConfig:
    """Everything the trainer needs, loaded from one YAML file."""

    task: str
    data: DataConfig
    model: dict[str, Any]
    output_dir: str = "runs/experiment"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    dropout: float | None = None
    pos_weight: float = 1.0
    label_smoothing: float = 0.0
    class_weights: bool = True
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    early_stopping_patience: int = 5
    scheduler: str = "cosine"
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config file.

    Unknown top-level keys are kept in ``extra`` so custom trainers can use
    them without changing the package.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    known = {
        "task",
        "data",
        "model",
        "output_dir",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "pos_weight",
        "label_smoothing",
        "class_weights",
        "grad_clip",
        "num_workers",
        "seed",
        "device",
        "early_stopping_patience",
        "scheduler",
    }
    extra = {key: value for key, value in raw.items() if key not in known}
    data_raw = raw.get("data", {})
    data = DataConfig(
        train=data_raw["train"],
        dev=data_raw.get("dev"),
        max_len=int(data_raw.get("max_len", 1024)),
        id_column=data_raw.get("id_column"),
        sequence_column=data_raw.get("sequence_column"),
    )
    return TrainConfig(
        task=raw["task"],
        data=data,
        model=raw["model"],
        output_dir=raw.get("output_dir", "runs/experiment"),
        epochs=int(raw.get("epochs", 20)),
        batch_size=int(raw.get("batch_size", 32)),
        lr=float(raw.get("lr", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        dropout=raw.get("dropout"),
        pos_weight=float(raw.get("pos_weight", 1.0)),
        label_smoothing=float(raw.get("label_smoothing", 0.0)),
        class_weights=bool(raw.get("class_weights", True)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        num_workers=int(raw.get("num_workers", 0)),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        early_stopping_patience=int(raw.get("early_stopping_patience", 5)),
        scheduler=str(raw.get("scheduler", "cosine")),
        extra=extra,
    )

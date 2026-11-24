"""Composed task models and the YAML-driven builder.

A model is deliberately thin: encoder -> pooling -> head.  The builder is the
only place that knows about config keys, which keeps training and inference
code independent of model choices.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..registry import build
from . import encoders as _encoders  # noqa: F401  (populate the registry)
from . import heads as _heads  # noqa: F401


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens; returns a zero vector for fully padded rows."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class Detector(nn.Module):
    """Sequence-level RdRp detector."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, mask)
        return self.head(masked_mean(hidden, mask))


class FamilyClassifier(nn.Module):
    """Sequence-level family classifier.

    Accepts either token ids ``(batch, length)`` or pooled features
    ``(batch, in_dim)``.  The branch is chosen by tensor rank, so the same
    wrapper works with a Transformer or a frozen-embedding encoder.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pooled-feature encoders receive float tensors; token encoders receive
        # integer ids.  Rank alone cannot distinguish the two branches because
        # both are (batch, length) or (batch, features).
        if x.is_floating_point():
            return self.head(self.encoder(x))
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        hidden = self.encoder(x, mask)
        return self.head(masked_mean(hidden, mask))


class Localizer(nn.Module):
    """Token-level start/end localizer."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, mask)
        return self.head(hidden)


_MODELS: dict[str, type[nn.Module]] = {
    "detector": Detector,
    "family": FamilyClassifier,
    "localizer": Localizer,
}


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from a plain dictionary (usually one YAML section).

    Expected shape::

        model: detector | family | localizer
        encoder:
          name: transformer
          d_model: 384
        head:
          name: detection
          hidden: 384

    ``head.d_model`` defaults to the encoder width, so custom encoders do not
    need to repeat it in every config.
    """
    if "encoder" not in config or "head" not in config:
        raise KeyError("model config requires 'encoder' and 'head' sections")

    encoder_cfg = dict(config["encoder"])
    encoder_name = encoder_cfg.pop("name")
    encoder = build("encoder", encoder_name, **encoder_cfg)

    head_cfg = dict(config["head"])
    head_name = head_cfg.pop("name")
    head_cfg.setdefault("d_model", getattr(encoder, "d_model", 384))
    head = build("head", head_name, **head_cfg)

    model_name = config.get("model", "detector")
    try:
        cls = _MODELS[model_name]
    except KeyError as exc:
        raise KeyError(f"unknown model '{model_name}', available: {sorted(_MODELS)}") from exc
    return cls(encoder, head)

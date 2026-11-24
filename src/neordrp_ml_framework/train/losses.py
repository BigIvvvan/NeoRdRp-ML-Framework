"""Task losses."""

from __future__ import annotations

import torch
from torch import nn


def detection_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy for sequence-level detection."""
    weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weight)


def family_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy for 65-way family classification."""
    return nn.functional.cross_entropy(
        logits,
        labels.long(),
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


def localization_loss(
    logits: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 20.0,
) -> torch.Tensor:
    """Masked binary cross-entropy for start and end channels.

    ``logits`` has shape ``(batch, length, 2)``.  Targets are zero-based
    indices; padding positions never contribute to the loss.
    """
    batch, length, channels = logits.shape
    if channels != 2:
        raise ValueError(f"expected 2 boundary channels, got {channels}")

    target = torch.zeros((batch, length, 2), device=logits.device, dtype=logits.dtype)
    rows = torch.arange(batch, device=logits.device)
    target[rows, start.long(), 0] = 1.0
    target[rows, end.long(), 1] = 1.0

    valid = mask.bool().unsqueeze(-1).expand_as(target)
    weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[valid], target[valid], pos_weight=weight
    )
    return loss

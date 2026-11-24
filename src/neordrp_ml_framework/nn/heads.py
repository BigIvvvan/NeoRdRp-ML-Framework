"""Task heads.

A head consumes encoder output and returns task logits.  Heads do not own
pooling, masking, or length handling; the composed model does that so the
same head can be reused with token or pooled encoders.
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import register


@register("head", "detection")
class DetectionHead(nn.Module):
    """Binary sequence-level head returning one logit per protein."""

    def __init__(self, d_model: int = 384, hidden: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


@register("head", "family")
class FamilyHead(nn.Module):
    """Multi-class family head returning ``(batch, n_classes)`` logits."""

    def __init__(
        self,
        d_model: int = 384,
        n_classes: int = 65,
        hidden: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


@register("head", "boundary")
class BoundaryHead(nn.Module):
    """Token-level start/end head returning ``(batch, length, 2)``.

    Channel 0 is the start logit and channel 1 is the end logit.  Keeping
    both channels in one tensor makes the localizer output easy to decode
    and easy to evaluate with a single boundary metric function.
    """

    def __init__(self, d_model: int = 384, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.start = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.end = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(hidden)
        start = self.start(hidden)
        end = self.end(hidden)
        return torch.cat([start, end], dim=-1)

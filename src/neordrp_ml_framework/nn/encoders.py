"""Interchangeable token and feature encoders.

All token encoders consume ``(input_ids, mask)`` and return
``(batch, length, d_model)``.  Pooled encoders consume ``(batch, in_dim)``
and return ``(batch, d_model)``.  That is the only contract heads rely on.
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import register


class _PositionalEmbedding(nn.Module):
    """Learned absolute position embedding with a fixed maximum length."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(self.max_len, d_model)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        positions = torch.arange(length, device=device)
        return self.embedding(positions).unsqueeze(0)


class _TokenEncoderBase(nn.Module):
    """Shared token embedding and padding-mask handling."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.position_embedding = _PositionalEmbedding(max_len, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(input_ids.shape[1], input_ids.device).to(x.dtype)
        return self.dropout(x)


@register("encoder", "transformer")
class TransformerEncoder(_TokenEncoderBase):
    """Bidirectional Transformer encoder.

    Args:
        vocab_size: Number of amino-acid tokens, including padding and unknown.
        d_model: Hidden width.
        n_layers: Number of Transformer blocks.
        n_heads: Attention heads.
        dropout: Dropout applied to embeddings and inside each block.
        max_len: Maximum supported sequence length.
        dim_feedforward: Width of the position-wise feed-forward layer.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 4096,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(dim_feedforward or 4 * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~mask.bool()
        x = self._embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


@register("encoder", "conv")
class ConvEncoder(_TokenEncoderBase):
    """Lightweight 1-D convolutional encoder.

    Useful as a fast baseline and as a test fixture; it has no attention and
    scales linearly with sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 384,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096,
        kernel_size: int = 5,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._embed(input_ids).transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        return self.norm(x)


@register("encoder", "mamba")
class MambaEncoder(_TokenEncoderBase):
    """Bidirectional Mamba-2 encoder.

    This is an optional component: install the ``mamba`` extra before use::

        pip install "neordrp-ml-framework[mamba]"

    The forward and reverse passes are independent Mamba-2 blocks; their
    outputs are summed.  The class raises a clear error at construction time
    when ``mamba-ssm`` is missing, so a config typo never silently degrades.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 384,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 4096,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "MambaEncoder requires mamba-ssm. Install it with: "
                'pip install "neordrp-ml-framework[mamba]"'
            ) from exc

        self.forward_blocks = nn.ModuleList(
            [Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.backward_blocks = nn.ModuleList(
            [Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._embed(input_ids)
        forward = x
        for block in self.forward_blocks:
            forward = block(forward)
        backward = torch.flip(x, dims=[1])
        for block in self.backward_blocks:
            backward = block(backward)
        backward = torch.flip(backward, dims=[1])
        return self.norm(forward + backward)


@register("encoder", "pooled")
class PooledEncoder(nn.Module):
    """Projection for frozen pooled embeddings (ESM-C, ProTrek, ...).

    The encoder does not know which upstream model produced the vector; it
    only needs the input width.  This keeps precomputed-feature pipelines
    interchangeable with token encoders.
    """

    def __init__(
        self,
        in_dim: int = 1152,
        d_model: int = 384,
        hidden: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(features)

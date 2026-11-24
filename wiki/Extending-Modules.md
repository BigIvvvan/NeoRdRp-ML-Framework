# Extending modules

## Encoder

An encoder receives token ids and a mask, and returns hidden states:

```python
import torch
from torch import nn
from neordrp_ml_framework.registry import register

@register("encoder", "my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(24, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        return self.proj(hidden) * mask.unsqueeze(-1)
```

The class must expose `d_model`; `build_model` uses it when the head omits
`d_model`.

## Head

A head receives pooled or token-level hidden states and returns the task shape:

```python
@register("head", "my_head")
class MyHead(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x).squeeze(-1)
```

Shapes:

- detection: `(batch,)`;
- family: `(batch, n_classes)`;
- boundary: `(batch, length, 2)`.

## Wrapper

If your task does not fit `detector` / `family` / `localizer`, add a small
wrapper in `nn/models.py` and register it in `_MODELS`.  Keep it thin: encoder,
pooling, head.

## Data format

Extend `read_records` rather than adding a second loader.  The rest of the
framework depends on `ProteinRecord`, not on the file format.

## Loss

Add a function in `train/losses.py` and branch in
`Trainer._forward_loss`.  Keep the loss signature explicit; do not read global
state.

## Report section

Add a builder in `eval/quick.py` that returns a `Section` with a Markdown
table.  The report renderer is generic.

## Rules of thumb

- English docstrings.
- No network calls inside model or dataset code.
- No hidden file writes; return values.
- Keep public signatures stable; add optional keyword arguments.

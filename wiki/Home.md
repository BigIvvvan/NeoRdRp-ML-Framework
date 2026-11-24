# NeoRdRp-ML-Framework Wiki

Composable models for **RdRp detection**, **family classification**, and
**domain boundary localization**.

Start here:

- [Installation](Installation)
- [Training](Training)
- [Inference](Inference)
- [Quick evaluation](Quick-Evaluation)
- [Extending modules](Extending-Modules)
- [FAQ](FAQ)

The full narrative guide is in the repository:
[GUIDE.md](https://github.com/BigIvvvan/NeoRdRp-ML-Framework/blob/main/GUIDE.md).

## What is composable here?

The model is a sequence encoder plus one task head:

```yaml
model:
  model: detector
  encoder: {name: transformer, d_model: 384, n_layers: 6, n_heads: 8}
  head: {name: detection, hidden: 384}
```

Swap `encoder` for `conv`, `mamba`, or `pooled`, or register your own.  Swap
`head` for `family` or `boundary`.  The training loop and the data pipeline do
not change.

## Three tasks, three wrappers

| Task | `model` | `head` | Output |
|---|---|---|---|
| Detection | `detector` | `detection` | one logit per protein |
| Family | `family` | `family` | `n_classes` logits |
| Localization | `localizer` | `boundary` | `(length, 2)` start/end |

## Quick taste

```bash
pip install -e ".[dev]"
neordrp-train --config examples/detection.yaml
neordrp-eval --task detection --predictions predictions.csv --output report.md
```

The evaluator is not a thin AUC wrapper.  It reports length-binned recall,
metadata slices, false positives, per-family recall, and internal-versus-terminal
boundary accuracy.

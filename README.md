# NeoRdRp-ML-Framework

A small, composable deep-learning framework for **RNA-dependent RNA polymerase
(RdRp) detection**, **family classification**, and **domain boundary
localization**.  Encoders and task heads are independent registry entries, so a
new protein language model, a convolutional encoder, or a task-specific head can
be swapped through YAML without rewriting the training loop.

```
sequence -> encoder (transformer | conv | mamba | pooled)
         -> task head (detection | family | boundary)
         -> predictions + a biological performance report
```

## Why this exists

Most RdRp tools are single-purpose scripts: the detector cannot be reused for
family classification, the classifier cannot report localization, and swapping
an encoder means forking the training code.  This project separates the three
concerns:

- **Standard interfaces.** Every model is built by `build_model()` from a plain
  dictionary with `encoder`, `head`, and `model` keys.
- **Task-specific models.** `detector`, `family`, and `localizer` are thin
  wrappers; they share the registry but not an entangled multi-task loss.
- **Fast biological reports.** `neordrp-eval` turns a prediction table into a
  Markdown report with length bins, metadata slices, false positives by family,
  and boundary metrics for internal versus terminal domains.

## Install

```bash
git clone https://github.com/BigIvvvan/NeoRdRp-ML-Framework
cd NeoRdRp-ML-Framework
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional Mamba-2 support:

```bash
pip install -e ".[mamba]"
```

Python 3.10+ and PyTorch 2.0+ are required.

## Quick start

```python
from neordrp_ml_framework.nn.models import build_model

model = build_model(
    {
        "model": "detector",
        "encoder": {"name": "transformer", "d_model": 384, "n_layers": 6, "n_heads": 8},
        "head": {"name": "detection", "hidden": 384},
    }
)
```

Train from YAML:

```bash
neordrp-train --config examples/detection.yaml
```

Predict:

```bash
neordrp-predict --checkpoint runs/detection_demo/best.pt --input proteins.fasta --output predictions.csv
```

Evaluate biologically:

```bash
neordrp-eval --task detection --predictions predictions.csv --output report.md
```

## Biological quick report

The evaluator deliberately reports more than a single AUC:

- detection: AUROC, AUPRC, recall at a negative-only FPR budget, length-binned
  recall, per-metadata recall, and the first false positives/negatives;
- family: macro-F1, top-1/top-2, per-family recall, and a confusion summary;
- localization: exact-at-tolerance start/end/both, MAE, IoU, split into
  internal domains and domains that touch a protein terminus.

```python
from neordrp_ml_framework.eval import quick_evaluate, write_report

report = quick_evaluate(frame, "detection", group_cols=["host", "length_bin"])
print(report.to_markdown())
write_report(report, "report.md")
```

## Repository layout

```
src/neordrp_ml_framework/
  nn/          registry-backed encoders, heads, task models, decoding
  data/        schema-tolerant readers, tokenizer, datasets, collators
  train/       config, losses, generic trainer
  eval/        metrics and the quick biological report
  inference.py checkpoint loading and batched prediction
  cli.py       neordrp-train / neordrp-predict / neordrp-eval
examples/      YAML configs and a runnable quick-report example
tests/         unit tests for shapes, metrics, decoding, and reports
docs/          architecture, API, data schema
wiki/          ready-to-publish GitHub wiki pages
```

## Documentation

- [GUIDE.md](GUIDE.md) — end-to-end training, inference, and evaluation guide.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — component boundaries and
  extension points.
- [docs/API.md](docs/API.md) — public Python API.
- [docs/DATA.md](docs/DATA.md) — input schema and label conventions.
- [wiki/](wiki/) — GitHub wiki source pages.

## Status

Version `0.1.0`.  The registry, three task models, trainer, inference, CLI,
quick report, and tests are in place.  The package is intentionally small: it
provides the interfaces and reference implementations, not a bundled model
zoo.

## License

MIT.  See [LICENSE](LICENSE).

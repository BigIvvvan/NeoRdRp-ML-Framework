# NeoRdRp-ML-Framework guide

This guide walks through data preparation, training, inference, and the fast
biological report.  It is written for a new user who has a protein table and
wants a working RdRp model without reading the source first.

## 1. Install

```bash
git clone https://github.com/BigIvvvan/NeoRdRp-ML-Framework
cd NeoRdRp-ML-Framework
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Check the install:

```bash
neordrp --version
neordrp-train --help
pytest
```

## 2. Input data

All tabular inputs are CSV, TSV, or Parquet.  FASTA is accepted for inference
(no labels).  Column names are matched case-insensitively; the aliases below
are built in.

| Meaning | Preferred | Aliases |
|---|---|---|
| protein id | `protein_id` | `id`, `protein`, `accession`, `uniprot` |
| sequence | `sequence` | `seq`, `aa`, `protein_sequence` |
| binary label | `label` | `y`, `target`, `is_rdrp` |
| family name | `family` | `family_name`, `class`, `taxonomy` |
| domain start | `start` | `domain_start`, `qstart` |
| domain end | `end` | `domain_end`, `qend` |
| length | `length` | `seq_len`, `protein_length` |

**Boundary convention:** `start` and `end` are **one-based, inclusive**.
`ProteinDataset` converts them to zero-based indices internally.  If `start` or
`end` is missing for a localization record, the dataset falls back to the full
sequence, which is useful for negative or unlabeled examples.

Minimal detection table:

```csv
protein_id,sequence,label
p1,MKLV...,1
p2,MSAI...,0
```

Minimal family table:

```csv
protein_id,sequence,family
p1,MKLV...,Alphavirus
p2,MSAI...,Flavivirus
```

Minimal localization table:

```csv
protein_id,sequence,start,end
p1,MKLV...,120,480
```

## 3. Choose a task

| Task | Model wrapper | Head | Output |
|---|---|---|---|
| Detection | `detector` | `detection` | one logit per protein |
| Family classification | `family` | `family` | `n_classes` logits |
| Boundary localization | `localizer` | `boundary` | `(length, 2)` start/end logits |

The task name in the config must match the wrapper and head.  The trainer does
not infer it.

## 4. Train

```bash
neordrp-train --config examples/detection.yaml
```

The YAML has four required parts:

```yaml
task: detection
data:
  train: data/train.parquet
  dev: data/dev.parquet
  max_len: 1024
model:
  model: detector
  encoder:
    name: transformer
    d_model: 384
    n_layers: 6
    n_heads: 8
  head:
    name: detection
    hidden: 384
output_dir: runs/detection_demo
epochs: 10
batch_size: 32
lr: 1.0e-4
pos_weight: 10.0
```

Common options:

| Key | Default | Notes |
|---|---|---|
| `pos_weight` | `1.0` | positive-class weight for detection/localization BCE |
| `class_weights` | `true` | inverse-frequency weights for family classification |
| `label_smoothing` | `0.0` | family cross-entropy smoothing |
| `scheduler` | `cosine` | `cosine` or `none` |
| `early_stopping_patience` | `5` | epochs without dev improvement |
| `num_workers` | `0` | DataLoader workers |
| `device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |

Artifacts:

- `runs/<name>/best.pt` — checkpoint with weights and inference metadata;
- `runs/<name>/metrics.json` — per-epoch history and the best score.

### Selection metric

- detection: dev AUROC;
- family: dev macro-F1;
- localization: dev both-end exact-at-10.

For a release model, select on an **independent dev split**, not on the test
split and not on a split that shares proteins with training.

## 5. Predict

```bash
neordrp-predict \
  --checkpoint runs/detection_demo/best.pt \
  --input proteins.fasta \
  --output predictions.csv
```

The checkpoint carries `model_config`, `task`, `family_to_index`, and
`max_len`, so inference rebuilds the exact architecture without a YAML file.

Output columns:

- detection: `protein_id`, `score`, `label` (0/1 at 0.5);
- family: `protein_id`, `prediction`, `family_id`, `confidence`;
- localization: `protein_id`, `start_pred`, `end_pred`, `length`,
  `start_prob`, `end_prob`.  Coordinates are one-based.

Batch size and device:

```bash
neordrp-predict --checkpoint best.pt --input proteins.csv --output pred.csv --batch-size 64 --device cuda
```

## 6. Fast biological report

A single AUC hides the failure modes that matter in biology.  `neordrp-eval`
accepts the prediction table and produces a Markdown + JSON report:

```bash
neordrp-eval \
  --task detection \
  --predictions predictions.csv \
  --output reports/detection.md
```

### Detection

Required columns: `label`, `score`.  Optional: `length`, any metadata column.

The report includes:

- AUROC, AUPRC, and recall at a negative-only FPR budget (default `1e-3`);
- recall by protein-length bin (default 0–300, 300–600, 600–1000, 1000+);
- recall by each metadata column passed with `--group-cols`;
- the first false positives and false negatives, with ids and scores.

The threshold is always chosen from negatives, never from positives.  This is
the only defensible way to report “recall at 0.1% FPR” when positives are rare.

### Family classification

Required columns: `label`, `prediction` (names or ids).  The report includes
macro-F1, top-1, top-2, per-family recall, and a confusion summary.  Family
names are matched as strings; keep train and report mappings identical.

### Localization

Required columns: `start_true`, `end_true`, `start_pred`, `end_pred`, optional
`length`.

The report splits records into:

- **terminal** — the domain touches the first or last 10 residues;
- **internal** — the domain is inside the protein.

It then reports exact-at-tolerance (default 10 aa) for start, end, both, plus
MAE and IoU for each group.  Terminal domains are easy; internal domains are
the real test.  Always report both numbers.

## 7. Swap an encoder or head

Everything is registry-backed.  A custom encoder only needs `d_model`:

```python
import torch
from torch import nn
from neordrp_ml_framework.registry import register

@register("encoder", "my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(1280, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, length, 1280) embeddings
        return self.proj(x) * mask.unsqueeze(-1)
```

Use it in YAML:

```yaml
model:
  model: detector
  encoder: {name: my_encoder, d_model: 256}
  head: {name: detection, hidden: 256}
```

A custom head receives the encoder width and must return the right shape:

- detection: `(batch,)`;
- family: `(batch, n_classes)`;
- boundary: `(batch, length, 2)`.

## 8. Programmatic training

```python
from neordrp_ml_framework.train import Trainer, load_config

config = load_config("examples/detection.yaml")
result = Trainer(config).fit()
print(result["best_score"])
```

To inject a model (for tests or a custom wrapper):

```python
Trainer(config, model=my_model).fit()
```

## 9. Checkpoints and reproducibility

A checkpoint contains:

```python
{
  "state_dict": ...,
  "model_config": {...},
  "task": "detection",
  "family_to_index": {...},
  "max_len": 1024,
  "epoch": 7,
  "metrics": {...},
}
```

`torch.load(..., weights_only=False)` is used because the payload contains
plain dictionaries.  Only load checkpoints you trust.

For reproducible runs, fix `seed`, keep the data snapshot, and record the git
commit.  The trainer seeds Python, NumPy, and PyTorch, but full determinism also
depends on CUDA kernels and DataLoader workers.

## 10. Common problems

**`ModuleNotFoundError: sklearn`** — install the package with dependencies:
`pip install -e ".[dev]"`.

**`no sequence column found`** — pass `data.sequence_column` explicitly or
rename the column to `sequence`.

**Family labels missing in dev** — `ProteinDataset` raises when a dev family is
not in `family_to_index`.  Build the mapping from the training split and keep
it fixed; do not silently drop classes.

**Everything predicts the same class** — check class imbalance, `class_weights`,
and whether the split leaks near-duplicates.  The quick report's per-family
recall makes this visible.

**Boundary predictions look terminal** — that is a data property, not only a
model bug.  Inspect the internal/terminal split; many databases annotate
domains that run to the protein end.

## 11. Repository conventions

- English comments and docstrings.
- No hidden global state; every component is built from a config dictionary.
- Public functions keep the same signature across encoders and heads.
- Tests cover shapes, metrics, decoding, and report rendering.

## 12. Citation

If this framework helps your work, cite the repository URL and the version you
used.  Model and dataset citations belong to the corresponding papers; the
framework does not claim those benchmarks as its own.

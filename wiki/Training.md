# Training

## Config shape

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

Run:

```bash
neordrp-train --config examples/detection.yaml
```

## Task-specific notes

### Detection

- `pos_weight` compensates for rare positives.
- Selection uses dev AUROC.
- Keep negative sampling outside the model; supply the training table you want.

### Family classification

- `head.n_classes` must match the number of training families.
- `class_weights: true` uses inverse frequency.
- `label_smoothing` helps when some families are tiny.
- Selection uses dev macro-F1.

### Localization

- `start` and `end` are one-based in the table and become zero-based in the
  dataset.
- `pos_weight` is applied to both start and end channels.
- Selection uses both-end exact-at-10.
- Train on internal domains if you care about internal boundaries; terminal
  domains dominate many tables and are easy to fit.

## Artifacts

```
runs/<name>/
  best.pt
  metrics.json
```

`best.pt` contains weights plus `model_config`, `task`, `family_to_index`, and
`max_len`; inference needs only this file.

## Reproducibility

- Fix `seed`.
- Keep a frozen data snapshot.
- Do not select on the locked test.
- Record the git commit and the config file with every reported number.

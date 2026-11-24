# Public API

## Model construction

### `neordrp_ml_framework.nn.models.build_model(config: dict) -> nn.Module`

Build a `detector`, `family`, or `localizer` from a plain dictionary.

```python
{
  "model": "detector",
  "encoder": {"name": "transformer", "d_model": 384, "n_layers": 6, "n_heads": 8},
  "head": {"name": "detection", "hidden": 384},
}
```

`head.d_model` defaults to the encoder's `d_model`.

### `neordrp_ml_framework.registry`

- `register(kind, name)` — decorator;
- `build(kind, name, **kwargs)` — instantiate;
- `available(kind)` — sorted names.

Kinds: `encoder`, `head`.

## Data

### `read_records(path, *, id_column=None, sequence_column=None) -> list[ProteinRecord]`

Read CSV, TSV, Parquet, or FASTA.

### `AminoAcidTokenizer(max_len=1024)`

- `encode(sequence) -> (ids, mask)`;
- `decode(ids) -> str`;
- `vocab`: residue -> index.

Special indices: `0` padding, `1` end-of-sequence, `2` unknown.

### `ProteinDataset(records, tokenizer, task, family_to_index=None)`

Returns a task-specific item:

- detection: `input_ids`, `mask`, `label`;
- family: `input_ids`, `mask`, `label` (integer);
- localization: `input_ids`, `mask`, `start`, `end` (zero-based).

### `collate_records(batch, task) -> dict`

Pads ids and masks, stacks labels or boundaries.

## Training

### `load_config(path) -> TrainConfig`

Loads YAML.  Unknown top-level keys are stored in `config.extra`.

### `Trainer(config, model=None)`

- `fit() -> dict` — train, write `best.pt` and `metrics.json`;
- `device` — resolved from `config.device`;
- `family_to_index` — built from the training split.

Selection: detection AUROC, family macro-F1, localization both-end exact-at-10.

## Inference

### `load_model(checkpoint, device) -> (model, payload)`

Rebuilds the architecture from `payload["model_config"]`.

### `predict_records(model, payload, records, device, batch_size=32) -> DataFrame`

Returns task-specific columns; localization coordinates are one-based.

## Metrics

### `binary_metrics(labels, scores, fpr=1e-3) -> dict`

`n`, `n_pos`, `n_neg`, `auroc`, `auprc`, `recall_at_fpr`, `fpr_actual`,
`threshold`.

### `recall_at_fpr(labels, scores, fpr=1e-3) -> (recall, actual_fpr, threshold)`

Threshold is the negative-score quantile; positives never enter the choice.

### `multiclass_metrics(labels, logits) -> dict`

`n`, `macro_f1`, `top1`, `top2`.

### `boundary_metrics(start_true, end_true, start_pred, end_pred, tolerance=10) -> dict`

`n`, `start_e10`, `end_e10`, `both_e10`, `mae_start`, `mae_end`,
`iou_median`, `iou_mean`.

## Quick report

### `quick_evaluate(frame, task, **kwargs) -> Report`

`task` is `detection`, `family`, or `localization`.  Common kwargs:

- `label_col`, `score_col`, `pred_col`;
- `start_true`, `end_true`, `start_pred`, `end_pred`;
- `length_col`;
- `group_cols` (detection);
- `fpr` (detection);
- `tolerance` (localization).

### `Report`

- `summary: dict`;
- `sections: list[Section]`;
- `to_markdown() -> str`;
- `to_dict() -> dict`.

### `write_report(report, path) -> Path`

Writes Markdown at `path` and JSON next to it.

## CLI

```
neordrp --version
neordrp train   --config examples/detection.yaml
neordrp predict --checkpoint best.pt --input proteins.fasta --output pred.csv
neordrp eval    --task detection --predictions pred.csv --output report.md
```

`neordrp-eval` options: `--fpr`, `--tolerance`, `--group-cols`, and column
overrides (`--label-col`, `--score-col`, `--pred-col`, `--start-true`,
`--end-true`, `--start-pred`, `--end-pred`, `--length-col`).

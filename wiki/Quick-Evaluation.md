# Quick evaluation

The quick report answers “where does the model fail biologically?” rather than
only “what is the AUC?”.

```bash
neordrp-eval \
  --task detection \
  --predictions predictions.csv \
  --output reports/detection.md
```

It writes Markdown at `--output` and JSON next to it.

## Detection

Required: `label`, `score`.  Optional: `length`, metadata columns.

Report sections:

- **Summary**: AUROC, AUPRC, recall at a negative-only FPR budget.
- **By protein length**: recall per length bin.
- **By metadata**: recall per group when `--group-cols` is passed.
- **False positives / false negatives**: first examples with ids and scores.

The threshold is always chosen from negatives.  This matters when positives are
rare: a threshold that looks good on positives inflates recall at “0.1% FPR”.

## Family

Required: `label`, `prediction`.

Report sections:

- **Summary**: macro-F1, top-1, top-2.
- **Per-family recall**: which families are systematically missed.
- **Confusion**: the largest off-diagonal pairs.

## Localization

Required: `start_true`, `end_true`, `start_pred`, `end_pred`.  Optional:
`length`.

The report splits records into **internal** and **terminal**:

- terminal: start within 10 aa of the N-terminus or end within 10 aa of the
  C-terminus;
- internal: everything else.

Both groups report start/end/both exact-at-tolerance, MAE, and IoU.  Always read
the internal row; terminal domains can be predicted by a constant and still
look good.

## Programmatic use

```python
from neordrp_ml_framework.eval import quick_evaluate, write_report

report = quick_evaluate(frame, "detection", group_cols=["host"], fpr=1e-3)
print(report.to_markdown())
write_report(report, "report.md")
```

## Why a report, not a single number

A model can win AUROC and lose the biology: it may fail on short proteins, on a
host group, on a family, or on internal domain starts.  The report makes those
slices visible in one run.

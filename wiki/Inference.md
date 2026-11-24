# Inference

## CLI

```bash
neordrp-predict \
  --checkpoint runs/detection_demo/best.pt \
  --input proteins.fasta \
  --output predictions.csv
```

`--input` accepts FASTA, CSV, TSV, or Parquet.  FASTA has no labels and is the
normal path for new proteins.

Options:

- `--batch-size` (default 32);
- `--device` (`auto`, `cpu`, `cuda`, `mps`).

## Python

```python
from neordrp_ml_framework.inference import load_model, predict_records
from neordrp_ml_framework.data import read_records
from neordrp_ml_framework.train.trainer import resolve_device

device = resolve_device("auto")
model, payload = load_model("runs/detection_demo/best.pt", device)
records = read_records("proteins.fasta")
predictions = predict_records(model, payload, records, device)
```

## Output columns

Detection:

```
protein_id, score, label
```

Family:

```
protein_id, prediction, family_id, confidence
```

Localization:

```
protein_id, start_pred, end_pred, length, start_prob, end_prob
```

Coordinates are one-based.

## Long proteins

The tokenizer truncates at `max_len`.  For full-length boundary prediction,
run a sliding window and take the maximum start/end probability per residue:

1. split the sequence into overlapping windows (e.g. 768 with stride 512);
2. run the localizer on each window;
3. map local positions back to protein coordinates;
4. keep the maximum probability per residue;
5. decode one start/end pair.

This is the same logic used by the reference localization evaluation; it keeps
memory bounded and avoids truncating the C-terminus.

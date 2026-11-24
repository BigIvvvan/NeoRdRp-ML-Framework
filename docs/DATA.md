# Data schema

## Records

Every input is normalized to `ProteinRecord`:

```python
ProteinRecord(
    protein_id: str,
    sequence: str,
    label: float | None = None,
    family: str | None = None,
    start: int | None = None,
    end: int | None = None,
    length: int | None = None,
    metadata: dict[str, Any] = {},
)
```

`read_records` accepts CSV, TSV, Parquet, and FASTA.  FASTA records have no
labels; they are intended for inference.

## Column aliases

Matching is case-insensitive.

| Meaning | Preferred | Aliases |
|---|---|---|
| id | `protein_id` | `id`, `protein`, `accession`, `uniprot`, `uniprot_acc` |
| sequence | `sequence` | `seq`, `aa`, `protein_sequence` |
| binary label | `label` | `y`, `target`, `is_rdrp` |
| family | `family` | `family_name`, `class`, `taxonomy` |
| start | `start` | `domain_start`, `qstart` |
| end | `end` | `domain_end`, `qend` |
| length | `length` | `seq_len`, `protein_length` |

If no id column exists, synthetic ids `protein_0`, `protein_1`, ... are
generated.  If no sequence column exists, reading fails loudly.

## Label conventions

- **Detection**: `label` is 0/1.  Missing labels are treated as 0 by
  `ProteinDataset`; do not rely on that for training data — clean missing
  labels first.
- **Family**: `family` is a string.  The mapping is built from the training
  split in sorted order and stored in the checkpoint.  A dev/test family not
  present in training raises an error.
- **Localization**: `start` and `end` are **one-based, inclusive**.  If they
  are missing, the dataset uses the full sequence (0 .. length-1), which is a
  valid negative/terminal example but not an internal-domain example.

## Sequence handling

- Case is normalized to uppercase.
- Whitespace inside a sequence is removed.
- Non-standard residues map to `<unk>`; they are not dropped.
- Sequences longer than `max_len` are truncated in `AminoAcidTokenizer`.
  For long proteins, use a sliding-window inference pipeline (see
  `GUIDE.md`) rather than assuming the model sees the whole sequence.

## Splits

The framework does not create splits.  Supply explicit `train` and `dev` files.
For honest detection numbers:

1. split by protein identity, not by domain or fragment;
2. keep near-duplicates in one split (cluster at 30–50% identity);
3. select the threshold on negatives from the dev split;
4. report the locked test only once.

The quick report exposes length bins and metadata slices precisely because
random splits often hide length and host shifts.

## Internal versus terminal domains

The localization report classifies a domain as **terminal** when it starts
within the first 10 residues or ends within the last 10.  Everything else is
**internal**.

This split matters: many public domain tables annotate a domain that runs to
the protein terminus, and a model can score well on those by predicting the
terminus.  Internal exact-at-10 is the harder and more informative number.

## Prediction table for `neordrp-eval`

Detection:

```csv
protein_id,length,label,score,host
p1,420,1,0.97,virus
p2,180,0,0.04,host
```

Family:

```csv
protein_id,label,prediction
p1,Alphavirus,Alphavirus
p2,Flavivirus,Alphavirus
```

Localization:

```csv
protein_id,length,start_true,end_true,start_pred,end_pred
p1,900,120,480,118,475
p2,300,1,300,2,299
```

`length` is optional for detection/family but recommended; it enables length
bins and the internal/terminal split.

## Metadata

Any extra columns become `ProteinRecord.metadata`.  They are not used by the
model, but they are preserved for the quick report.  Pass them to
`--group-cols` to get per-slice recall.  Typical useful slices: host, taxonomy,
length bin, database source, year.

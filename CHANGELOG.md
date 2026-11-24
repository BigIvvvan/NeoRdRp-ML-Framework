# Changelog

## 0.1.0

Initial public scaffold.

### Added

- Registry-backed encoders: `transformer`, `conv`, `pooled`, optional `mamba`.
- Task heads: `detection`, `family`, `boundary`.
- Task wrappers: `detector`, `family`, `localizer`.
- YAML training config and generic `Trainer` with early stopping and
  self-describing checkpoints.
- Schema-tolerant readers for CSV, TSV, Parquet, and FASTA.
- `AminoAcidTokenizer`, `ProteinDataset`, and collators for three tasks.
- Inference: checkpoint loading and batched prediction.
- CLI: `neordrp-train`, `neordrp-predict`, `neordrp-eval`.
- Fast biological report:
  - detection length bins, metadata slices, false-positive/negative examples;
  - family per-class recall and confusion summary;
  - localization internal versus terminal split with E10/MAE/IoU.
- Unit tests for model shapes, tokenizer, metrics, decoding, and report.
- Example configs and a runnable quick-report example.
- Architecture, API, and data-schema documentation.
- Ready-to-publish GitHub wiki pages.

### Known limitations

- No bundled pretrained weights or model zoo.
- `mamba` requires the optional `mamba-ssm` package and a compatible CUDA
  build.
- Very long proteins are truncated at `max_len`; use sliding-window inference
  for full-length boundary prediction.
- The quick report assumes the prediction table already contains true labels;
  it does not compute labels from raw sequences.

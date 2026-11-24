# Architecture

## Design goal

The framework is a **composition layer**, not a model zoo.  Its job is to make
the three RdRp tasks share data plumbing, metrics, and a training loop while
keeping encoders and heads swappable.

```
                    +-------------------+
  YAML config ----> |  build_model()    |
                    |  registry lookup  |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
        encoder         task wrapper         head
   transformer/conv/     detector/         detection/
   mamba/pooled          family/localizer  family/boundary
              |               |               |
              +-------+-------+-------+-------+
                      |
                 Trainer / predict
                      |
             metrics -> quick report
```

## Modules

### `registry.py`

A single `REGISTRY` maps `(kind, name)` to a class.  Kinds are `encoder` and
`head`.  `register` is a decorator; `build` instantiates by name.  The registry
is intentionally tiny: it prevents the trainer from importing every model, and
it gives third-party code one stable extension point.

### `nn/encoders.py`

All encoders accept `(input_ids, mask)` and return `(batch, length, d_model)`.

| Name | Input | Notes |
|---|---|---|
| `transformer` | token ids | learned embeddings, sinusoidal positions, 4D attention mask |
| `conv` | token ids | dilated Conv1d stack; fast on CPU |
| `mamba` | token ids | optional `mamba-ssm`; falls back to an error if missing |
| `pooled` | `(batch, in_dim)` | frozen protein-language-model vectors |

`pooled` is the bridge for ESM/ProTrek-style embeddings.  It is used with the
`family` wrapper, not with the token-level localizer.

### `nn/heads.py`

- `DetectionHead`: masked mean -> LayerNorm -> MLP -> one logit.
- `FamilyHead`: masked mean -> LayerNorm -> MLP -> `n_classes` logits.
- `BoundaryHead` (imported from the localizer module): token-wise MLP -> two
  logits per residue.

### `nn/models.py`

Three wrappers, each a few lines:

- `Detector`: encoder -> masked mean -> detection head;
- `FamilyClassifier`: accepts token ids or pooled vectors; branches on
  `x.is_floating_point()`;
- `Localizer`: encoder -> token-wise boundary head, output `(B, L, 2)`.

`build_model` is the only function that reads config keys.  It fills
`head.d_model` from the encoder when omitted, so configs stay short.

### `data/`

`read_records` normalizes CSV, TSV, Parquet, and FASTA into `ProteinRecord`.
`ProteinDataset` tokenizes and converts boundary labels.  `collate_records`
pads to the batch maximum and returns a task-specific dictionary.

Boundary labels are stored **one-based** in records and converted to zero-based
indices in `ProteinDataset`.  This matches common domain tables and avoids
off-by-one confusion at the API boundary.

### `train/`

`TrainConfig` is a dataclass loaded from YAML.  `Trainer` owns the loop,
optimizer, scheduler, class weights, early stopping, and checkpointing.  Losses
live in `losses.py`:

- detection: BCE with `pos_weight`;
- family: cross-entropy with optional class weights and label smoothing;
- localization: per-residue BCE over start and end channels with `pos_weight`.

The trainer selects the best epoch by task:

| Task | Dev score |
|---|---|
| detection | AUROC |
| family | macro-F1 |
| localization | both-end exact-at-10 |

### `eval/`

`metrics.py` contains pure functions and no I/O.  `quick.py` builds a `Report`
with sections; each section is a table plus optional notes.  The report is
rendered to Markdown and JSON by `write_report`.

### `inference.py`

Loads a checkpoint, rebuilds the model from its stored config, and runs a
`DataLoader`.  Detection returns probabilities; family returns argmax and
confidence; localization decodes a start/end pair from the two probability
tracks.

## Boundaries and invariants

1. **No task logic in encoders.**  An encoder must not know whether it feeds a
   detector or a localizer.
2. **No model logic in the trainer.**  Swapping a model never edits
   `trainer.py`.
3. **No positive labels in threshold selection.**  `recall_at_fpr` uses
   negative scores only.
4. **Checkpoints are self-describing.**  `model_config`, `task`,
   `family_to_index`, and `max_len` travel with the weights.
5. **Evaluation is separate from training.**  The quick report can run on any
   prediction table, including baselines and external tools.

## Extension points

| Want to add | Do this |
|---|---|
| new encoder | subclass `nn.Module`, set `d_model`, `@register("encoder", "name")` |
| new head | return the task shape, `@register("head", "name")` |
| new wrapper | add to `_MODELS` in `nn/models.py` |
| new loss | add to `train/losses.py` and branch in `Trainer._forward_loss` |
| new metric section | add to `eval/quick.py` |
| new data format | extend `read_records`; keep `ProteinRecord` unchanged |

## Non-goals

- Bundling pretrained weights or a model zoo.
- Hiding the data split logic inside the trainer.
- Providing a single "SOTA" number.  Results depend on the dataset, split, and
  label definition; the framework reports the conditions alongside the metric.

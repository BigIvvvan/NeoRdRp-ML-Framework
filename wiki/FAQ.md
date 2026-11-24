# FAQ

**Does the framework include trained weights?**
No.  It provides the interfaces, task models, trainer, inference, and
evaluation.  Bring your own data and checkpoints.

**Can I use ESM or ProTrek embeddings?**
Yes.  Use the `pooled` encoder with precomputed vectors, or register an encoder
that calls a protein language model.  The `family` wrapper accepts pooled
vectors directly.

**Why is my localization model predicting the protein end?**
Many domain tables contain domains that run to the terminus.  Check the
internal/terminal split in the quick report.  If internal domains are rare,
train on internal-only examples or add an explicit terminal class.

**Why not one multi-task model?**
The tasks have different label definitions, resolutions, and selection metrics.
Separate wrappers keep the loss and the evaluation honest.  Sharing an encoder
is still possible; sharing a loss is not required.

**How do I choose the detection threshold?**
Use negatives from a dev split.  `recall_at_fpr` picks the negative quantile at
the requested FPR.  Do not tune on the locked test.

**Does `neordrp-eval` need labels?**
Yes.  It evaluates a prediction table; it does not assign labels from
sequences.  For unlabeled FASTA, use `neordrp-predict` only.

**Can I run on CPU?**
Yes, but it is slow.  `conv` is the fastest CPU encoder; `mamba` requires CUDA.

**What is the checkpoint format?**
A dictionary with `state_dict`, `model_config`, `task`, `family_to_index`,
`max_len`, `epoch`, and `metrics`.  See `GUIDE.md`.

**How do I report a result?**
State the data snapshot, split strategy, threshold rule, and the quick report’s
internal/terminal split.  A bare AUC is not enough to compare tools.

**Why does the report show NaN AUROC for a group?**
That group contains only one class.  AUROC is undefined; the report returns
NaN instead of a misleading number.

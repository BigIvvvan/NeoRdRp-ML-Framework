# Installation

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- NumPy, pandas, scikit-learn, PyYAML, tqdm

## From source

```bash
git clone https://github.com/BigIvvvan/NeoRdRp-ML-Framework
cd NeoRdRp-ML-Framework
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify:

```bash
neordrp --version
neordrp-train --help
pytest
```

## Optional Mamba-2

```bash
pip install -e ".[mamba]"
```

`mamba-ssm` needs a CUDA toolchain and a compatible PyTorch build.  On CPU or
unsupported GPUs, use `transformer` or `conv` instead.

## Editable install without dependencies

For development against an existing environment:

```bash
pip install -e . --no-deps
```

You are then responsible for NumPy, pandas, scikit-learn, PyTorch, PyYAML, and
tqdm being present.

## GPU notes

`device: auto` chooses CUDA, then MPS, then CPU.  Set `device: cuda:1` to pin a
specific GPU.  The trainer does not shard across GPUs; use one process per GPU
for now.

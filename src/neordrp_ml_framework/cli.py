"""Command-line entry points for training, prediction, and quick evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .data import read_records
from .eval.quick import quick_evaluate, write_report
from .inference import load_model, predict_records
from .train.config import load_config
from .train.trainer import Trainer, resolve_device


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a NeoRdRp-ML task model.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = Trainer(config).fit()
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2, ensure_ascii=False))


def predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference on FASTA or tabular inputs.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--input", required=True, help="FASTA, CSV, TSV, or Parquet with sequences.")
    parser.add_argument("--output", required=True, help="Destination CSV.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, payload = load_model(args.checkpoint, device)
    records = read_records(args.input)
    predictions = predict_records(model, payload, records, device, batch_size=args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"wrote {len(predictions)} predictions to {output}")


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fast biological performance report from a prediction table.")
    parser.add_argument("--task", required=True, choices=["detection", "family", "localization"])
    parser.add_argument("--predictions", required=True, help="CSV or Parquet with predictions and labels.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--pred-col", default="prediction")
    parser.add_argument("--start-true", default="start_true")
    parser.add_argument("--end-true", default="end_true")
    parser.add_argument("--start-pred", default="start_pred")
    parser.add_argument("--end-pred", default="end_pred")
    parser.add_argument("--length-col", default="length")
    parser.add_argument("--fpr", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--group-cols", nargs="*", default=None)
    args = parser.parse_args(argv)

    path = Path(args.predictions)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    report = quick_evaluate(
        frame,
        args.task,
        label_col=args.label_col,
        score_col=args.score_col,
        pred_col=args.pred_col,
        start_true=args.start_true,
        end_true=args.end_true,
        start_pred=args.start_pred,
        end_pred=args.end_pred,
        length_col=args.length_col,
        group_cols=args.group_cols,
        fpr=args.fpr,
        tolerance=args.tolerance,
    )
    output = write_report(report, args.output)
    print(report.to_markdown())
    print(f"\nwrote {output}")


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else __import__("sys").argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: neordrp {train,predict,eval} [...]")
        print("       neordrp --version")
        return
    if argv[0] == "--version":
        print(__version__)
        return
    commands = {"train": train_main, "predict": predict_main, "eval": eval_main}
    command, command_args = argv[0], argv[1:]
    if command not in commands:
        raise SystemExit(f"unknown command '{command}'; choose from {sorted(commands)}")
    commands[command](command_args)


if __name__ == "__main__":
    main()

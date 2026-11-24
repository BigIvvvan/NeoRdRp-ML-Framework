"""Fast biological performance reports.

The goal is a one-command answer to "is this model biologically useful?".
Beyond AUROC, the report slices detection performance by protein length and
optional metadata groups, surfaces the false positives/negatives a biologist
would inspect first, and separates internal from terminal boundary cases.

Everything here operates on a prediction table and never loads a model, so a
report takes seconds even for hundreds of thousands of proteins.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import binary_metrics, boundary_metrics, multiclass_metrics

DETECTION_LENGTH_BINS = [(0, 300), (301, 600), (601, 1000), (1001, 10**9)]
BIN_LABELS = ["<=300 aa", "301-600 aa", "601-1000 aa", ">1000 aa"]


@dataclass
class Section:
    """A named table inside a report."""

    title: str
    rows: list[dict[str, Any]]
    note: str = ""


@dataclass
class Report:
    """Structured result plus a rendered Markdown body."""

    task: str
    title: str
    summary: dict[str, Any]
    sections: list[Section] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append(_markdown_table([self.summary]))
        for section in self.sections:
            lines.extend(["", f"## {section.title}", ""])
            if section.note:
                lines.extend([section.note, ""])
            lines.append(_markdown_table(section.rows))
        if self.examples:
            lines.extend(["", "## First cases to inspect", ""])
            lines.append(_markdown_table(self.examples))
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend([f"- {note}" for note in self.notes])
        return "\n".join(lines).rstrip() + "\n"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _length_bin_table(frame: pd.DataFrame, labels: pd.Series, scores: pd.Series, fpr: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "length" in frame:
        lengths = frame["length"].astype(int)
    elif "sequence" in frame:
        lengths = frame["sequence"].str.len()
    else:
        # Prediction tables may contain only ids, labels, and scores.  Length
        # bins are useful but optional; skip them instead of failing.
        return rows
    for (low, high), label in zip(DETECTION_LENGTH_BINS, BIN_LABELS):
        selected = (lengths >= low) & (lengths <= high)
        subset_labels = labels[selected].to_numpy()
        subset_scores = scores[selected].to_numpy()
        if len(subset_labels) == 0 or subset_labels.sum() == 0:
            continue
        metrics = binary_metrics(subset_labels, subset_scores, fpr=fpr)
        rows.append(
            {
                "length bin": label,
                "n": int(metrics["n"]),
                "pos": int(metrics["n_pos"]),
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "recall@FPR": metrics["recall_at_fpr"],
            }
        )
    return rows


def _group_table(
    frame: pd.DataFrame,
    labels: pd.Series,
    scores: pd.Series,
    group_cols: list[str],
    fpr: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_col in group_cols:
        if group_col not in frame:
            continue
        for value, index in frame.groupby(group_col, dropna=False).groups.items():
            subset_labels = labels.loc[index].to_numpy()
            subset_scores = scores.loc[index].to_numpy()
            if len(subset_labels) < 5:
                continue
            metrics = binary_metrics(subset_labels, subset_scores, fpr=fpr)
            rows.append(
                {
                    "group": f"{group_col}={value}",
                    "n": int(metrics["n"]),
                    "pos": int(metrics["n_pos"]),
                    "AUROC": metrics["auroc"],
                    "recall@FPR": metrics["recall_at_fpr"],
                }
            )
    return sorted(rows, key=lambda row: row["n"], reverse=True)


def _detection_report(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    fpr: float,
    group_cols: list[str] | None,
) -> Report:
    labels = frame[label_col].astype(float)
    scores = frame[score_col].astype(float)
    summary = binary_metrics(labels.to_numpy(), scores.to_numpy(), fpr=fpr)
    sections: list[Section] = []
    length_rows = _length_bin_table(frame, labels, scores, fpr)
    if length_rows:
        sections.append(Section(title="By protein length", rows=length_rows))
    if group_cols:
        sections.append(Section(title="By metadata group", rows=_group_table(frame, labels, scores, group_cols, fpr)))

    examples: list[dict[str, Any]] = []
    if "protein_id" in frame:
        false_positive = frame[(labels == 0)].assign(score=scores).sort_values("score", ascending=False).head(5)
        false_negative = frame[(labels == 1)].assign(score=scores).sort_values("score").head(5)
        for _, row in false_positive.iterrows():
            examples.append({"case": "false positive", "protein_id": row["protein_id"], "length": int(row.get("length", 0)), "score": float(row["score"])})
        for _, row in false_negative.iterrows():
            examples.append({"case": "false negative", "protein_id": row["protein_id"], "length": int(row.get("length", 0)), "score": float(row["score"])})
    notes = [
        "Thresholds are chosen on negatives only, so recall@FPR does not use test positives.",
        "Length bins expose the common failure where a model only works on short training proteins.",
    ]
    return Report(
        task="detection",
        title="RdRp detection — biological performance",
        summary={
            "n": int(summary["n"]),
            "positives": int(summary["n_pos"]),
            "negatives": int(summary["n_neg"]),
            "AUROC": summary["auroc"],
            "AUPRC": summary["auprc"],
            "recall@FPR": summary["recall_at_fpr"],
            "actual FPR": summary["fpr_actual"],
            "threshold": summary["threshold"],
        },
        sections=sections,
        examples=examples,
        notes=notes,
    )


def _family_report(frame: pd.DataFrame, label_col: str, pred_col: str) -> Report:
    labels = frame[label_col].astype(int)
    predictions = frame[pred_col].astype(int)
    logits = frame.filter(regex=r"^(score|prob)_\d+$")
    if logits.shape[1] == 0:
        # Build a one-hot score matrix from predicted labels so the report still works.
        n_classes = int(max(labels.max(), predictions.max())) + 1
        scores = np.zeros((len(frame), n_classes), dtype=np.float64)
        scores[np.arange(len(frame)), predictions.to_numpy()] = 1.0
    else:
        scores = logits.to_numpy()
    summary = multiclass_metrics(labels.to_numpy(), scores)

    per_class: list[dict[str, Any]] = []
    for class_id in sorted(labels.unique()):
        mask = labels == class_id
        per_class.append(
            {
                "family id": int(class_id),
                "n": int(mask.sum()),
                "recall": float((predictions[mask] == class_id).mean()),
            }
        )
    wrong = frame[labels != predictions]
    pairs = (
        wrong.groupby([label_col, pred_col]).size().sort_values(ascending=False).head(10).reset_index(name="count")
        if len(wrong)
        else pd.DataFrame(columns=[label_col, pred_col, "count"])
    )
    sections = [
        Section(title="Per-family recall", rows=per_class),
        Section(title="Most common confusions", rows=pairs.to_dict(orient="records")),
    ]
    return Report(
        task="family",
        title="Family classification — biological performance",
        summary={
            "n": int(summary["n"]),
            "macro F1": summary["macro_f1"],
            "top-1": summary["top1"],
            "top-2": summary["top2"],
            "errors": int((labels != predictions).sum()),
        },
        sections=sections,
        notes=["Use per-family recall to spot rare families that a high macro-F1 can hide."],
    )


def _localization_report(
    frame: pd.DataFrame,
    start_true: str,
    end_true: str,
    start_pred: str,
    end_pred: str,
    tolerance: int,
) -> Report:
    summary = boundary_metrics(
        frame[start_true].to_numpy(),
        frame[end_true].to_numpy(),
        frame[start_pred].to_numpy(),
        frame[end_pred].to_numpy(),
        tolerance=tolerance,
    )
    sections: list[dict[str, Any]] = []
    internal = None
    if "length" in frame and {start_true, end_true}.issubset(frame.columns):
        lengths = frame["length"].astype(int).to_numpy()
        internal = (frame[start_true].to_numpy() > 10) & (frame[end_true].to_numpy() < lengths - 10)
    if internal is not None and internal.any():
        internal_metrics = boundary_metrics(
            frame.loc[internal, start_true].to_numpy(),
            frame.loc[internal, end_true].to_numpy(),
            frame.loc[internal, start_pred].to_numpy(),
            frame.loc[internal, end_pred].to_numpy(),
            tolerance=tolerance,
        )
        sections.append(
            Section(
                title="Internal domains only",
                rows=[{"n": int(internal_metrics["n"]), "start E10": internal_metrics["start_e10"], "end E10": internal_metrics["end_e10"], "both E10": internal_metrics["both_e10"], "IoU median": internal_metrics["iou_median"]}],
                note="Terminal domains are easy; a model that only snaps to the protein end will look good here without localizing internal domains.",
            )
        )
    return Report(
        task="localization",
        title="RdRp boundary localization — biological performance",
        summary={
            "n": int(summary["n"]),
            f"start ±{tolerance} aa": summary["start_e10"],
            f"end ±{tolerance} aa": summary["end_e10"],
            f"both ±{tolerance} aa": summary["both_e10"],
            "MAE start": summary["mae_start"],
            "MAE end": summary["mae_end"],
            "IoU median": summary["iou_median"],
        },
        sections=sections,
        notes=[
            "Report internal and terminal domains separately; mixing them hides the hard cases.",
            "A high end accuracy with zero internal start accuracy usually means the model learned a terminal prior, not domain biology.",
        ],
    )


def quick_evaluate(
    predictions: pd.DataFrame,
    task: str,
    *,
    label_col: str = "label",
    score_col: str = "score",
    pred_col: str = "prediction",
    start_true: str = "start_true",
    end_true: str = "end_true",
    start_pred: str = "start_pred",
    end_pred: str = "end_pred",
    length_col: str = "length",
    group_cols: list[str] | None = None,
    fpr: float = 1e-3,
    tolerance: int = 10,
) -> Report:
    """Build a biological performance report from a prediction table.

    Args:
        predictions: One row per protein; must contain the columns for the task.
        task: ``detection``, ``family``, or ``localization``.
        label_col: Binary label for detection or integer family id for family.
        score_col: Detection score (higher means more likely RdRp).
        pred_col: Predicted family id for family classification.
        start_true/end_true/start_pred/end_pred: One-based boundary columns.
        length_col: Protein length; inferred from ``sequence`` when absent.
        group_cols: Optional metadata columns (for example ``taxon`` or ``host``).
        fpr: False-positive budget for detection recall.
        tolerance: Boundary tolerance in residues (default 10).
    """
    frame = predictions.copy()
    if length_col not in frame and "sequence" in frame:
        frame[length_col] = frame["sequence"].str.len()
    if task == "detection":
        return _detection_report(frame, label_col, score_col, fpr, group_cols)
    if task == "family":
        return _family_report(frame, label_col, pred_col)
    if task == "localization":
        return _localization_report(frame, start_true, end_true, start_pred, end_pred, tolerance)
    raise ValueError(f"unknown task: {task}")


def write_report(report: Report, output: str | Path) -> Path:
    """Write Markdown and a sibling JSON file; returns the Markdown path."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_markdown(), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output

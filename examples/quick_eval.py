"""Minimal programmatic quick report.

Run from the repository root:

    python examples/quick_eval.py
"""

from __future__ import annotations

import pandas as pd

from neordrp_ml_framework.eval import quick_evaluate, write_report

frame = pd.DataFrame(
    {
        "protein_id": ["p1", "p2", "p3", "p4", "p5"],
        "length": [180, 450, 900, 1500, 320],
        "label": [1, 0, 1, 0, 1],
        "score": [0.95, 0.2, 0.8, 0.05, 0.6],
        "taxon": ["virus", "host", "virus", "host", "virus"],
    }
)
report = quick_evaluate(frame, "detection", group_cols=["taxon"])
print(report.to_markdown())
write_report(report, "runs/quick_eval_report.md")

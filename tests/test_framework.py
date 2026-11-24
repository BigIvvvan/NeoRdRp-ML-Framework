import numpy as np
import pandas as pd
import torch

from neordrp_ml_framework.data import AminoAcidTokenizer
from neordrp_ml_framework.eval.metrics import binary_metrics, boundary_metrics, recall_at_fpr
from neordrp_ml_framework.eval.quick import quick_evaluate
from neordrp_ml_framework.nn.decode import decode_pair
from neordrp_ml_framework.nn.models import build_model
from neordrp_ml_framework.registry import available


def _config(task: str = "detector", head: str = "detection") -> dict:
    return {
        "model": task,
        "encoder": {
            "name": "transformer",
            "vocab_size": 24,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "max_len": 64,
        },
        "head": {"name": head, "hidden": 32, **({"n_classes": 5} if head == "family" else {})},
    }


def test_registry_has_core_components() -> None:
    assert "transformer" in available("encoder")
    assert "detection" in available("head")


def test_detector_forward_shape() -> None:
    model = build_model(_config())
    ids = torch.randint(2, 24, (3, 20))
    mask = torch.ones(3, 20, dtype=torch.bool)
    assert model(ids, mask).shape == (3,)


def test_family_forward_shape() -> None:
    model = build_model(_config("family", "family"))
    ids = torch.randint(2, 24, (4, 15))
    mask = torch.ones(4, 15, dtype=torch.bool)
    assert model(ids, mask).shape == (4, 5)


def test_localizer_forward_shape() -> None:
    model = build_model(_config("localizer", "boundary"))
    ids = torch.randint(2, 24, (2, 25))
    mask = torch.ones(2, 25, dtype=torch.bool)
    assert model(ids, mask).shape == (2, 25, 2)


def test_tokenizer_round_trip() -> None:
    tokenizer = AminoAcidTokenizer(max_len=10)
    ids, mask = tokenizer.encode("ACDX")
    assert ids[0] == tokenizer.vocab["A"]
    assert ids[-1] == 1
    assert tokenizer.decode(ids) == "ACDX"
    assert mask == [1, 1, 1, 1]


def test_recall_at_fpr_is_negative_only() -> None:
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.2, 0.1])
    recall, actual_fpr, threshold = recall_at_fpr(labels, scores, fpr=0.5)
    assert threshold == 0.2
    assert actual_fpr == 2 / 3
    assert recall == 1.0


def test_boundary_metrics() -> None:
    metrics = boundary_metrics(
        np.array([10, 100]), np.array([50, 200]), np.array([12, 120]), np.array([48, 190])
    )
    assert metrics["start_e10"] == 0.5
    assert metrics["end_e10"] == 1.0
    assert metrics["both_e10"] == 0.5


def test_decode_pair_respects_length_window() -> None:
    start = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    end = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
    mask = np.ones(5, dtype=bool)
    assert decode_pair(start, end, mask, min_len=3, max_len=5) == (0, 4)


def test_quick_detection_report() -> None:
    frame = pd.DataFrame(
        {
            "protein_id": ["a", "b", "c", "d"],
            "length": [100, 400, 700, 1200],
            "label": [1, 0, 1, 0],
            "score": [0.9, 0.8, 0.7, 0.1],
        }
    )
    report = quick_evaluate(frame, "detection")
    assert report.summary["AUROC"] == 0.75
    assert "By protein length" in report.to_markdown()

"""Decode token-level boundary logits into one start/end pair."""

from __future__ import annotations

import numpy as np


def decode_pair(
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    mask: np.ndarray,
    min_len: int = 30,
    max_len: int = 700,
    top_k: int = 16,
) -> tuple[int, int]:
    """Pick the highest-scoring start/end pair under a length window.

    Args:
        start_prob: 1-D start probabilities, one value per residue.
        end_prob: 1-D end probabilities, one value per residue.
        mask: Boolean valid-residue mask.
        min_len: Minimum accepted domain length in residues.
        max_len: Maximum accepted domain length in residues.
        top_k: Number of candidate starts and ends to consider.  Keeping this
            small makes decoding O(k^2) even for 4,096-residue proteins.

    Returns:
        ``(start, end)`` as zero-based inclusive indices.  If no candidate
        satisfies the length window, the independent argmax pair is returned,
        which mirrors what a model without joint decoding would predict.
    """
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0, 0

    start_prob = np.asarray(start_prob, dtype=np.float64)
    end_prob = np.asarray(end_prob, dtype=np.float64)
    starts = valid[np.argsort(-start_prob[valid])[:top_k]]
    ends = valid[np.argsort(-end_prob[valid])[:top_k]]

    best_score = -np.inf
    best_pair = (int(valid[0]), int(valid[-1]))
    for start in starts:
        for end in ends:
            length = int(end) - int(start) + 1
            if not (min_len <= length <= max_len):
                continue
            score = float(start_prob[start] + end_prob[end])
            if score > best_score:
                best_score = score
                best_pair = (int(start), int(end))
    if np.isfinite(best_score):
        return best_pair
    return int(valid[int(np.argmax(start_prob[valid]))]), int(valid[int(np.argmax(end_prob[valid]))])

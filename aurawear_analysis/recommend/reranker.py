from __future__ import annotations

from typing import List, Optional
import numpy as np



def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assume normalized
    return float(np.dot(a, b))


def update_preference(pref: Optional[np.ndarray], item_emb: np.ndarray, step: float) -> np.ndarray:
    """
    u <- normalize(u + step * e)
    """
    if pref is None:
        return normalize(item_emb)
    return normalize(pref + step * item_emb)


def _threshold_penalty(
    max_sim: float,
    hard_threshold: float,
    hard_penalty: float,
    soft_start: float,
    soft_weight: float,
    power: float,
) -> float:
    """Shared penalty curve: hard suppress above hard_threshold, smooth ramp above soft_start."""
    if max_sim > hard_threshold:
        return float(hard_penalty)
    if max_sim <= soft_start:
        return 0.0
    denom = max(1e-12, hard_threshold - soft_start)
    x = min(max((max_sim - soft_start) / denom, 0.0), 1.0)
    return float(soft_weight * (x ** power))


def neg_suppression_penalty(
    item_emb: np.ndarray,
    neg_vecs: List[np.ndarray],
    *,
    hard_threshold: float = 0.98,
    hard_penalty: float = 1e6,
    soft_start: float = 0.70,
    soft_weight: float = 0.50,
    power: float = 2.0,
) -> float:
    """Negative suppression: hard suppress when max cosine > hard_threshold,
    smooth soft penalty above soft_start.  Assumes L2-normalized embeddings."""
    if not neg_vecs:
        return 0.0
    max_sim = max(float(np.dot(item_emb, nv)) for nv in neg_vecs)
    return _threshold_penalty(max_sim, hard_threshold, hard_penalty, soft_start, soft_weight, power)


def dup_penalty(
    item_emb: np.ndarray,
    picked_embs: List[np.ndarray],
    threshold: float = 0.95,
    *,
    hard_penalty: float = 1e6,
    soft_start: float = 0.70,
    soft_weight: float = 0.30,
    power: float = 2.0,
) -> float:
    """Diversity penalty (MMR-like): hard suppress above threshold,
    smooth soft penalty above soft_start."""
    if not picked_embs:
        return 0.0
    max_sim = max(float(np.dot(item_emb, pe)) for pe in picked_embs)
    return _threshold_penalty(max_sim, threshold, hard_penalty, soft_start, soft_weight, power)



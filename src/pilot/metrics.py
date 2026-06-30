"""Shared metric definitions: Faith_S, Jaccard, Spearman rho.

All formulas are implemented once here and imported everywhere.

D        = logit(IO) - logit(S) at final position
D_full   = D on clean run
D_corrupt = D when all features in S are mean-ablated  (behavior-destroyed baseline)
D_S      = D when S is mean-ablated

Faith_S  = (D_S - D_corrupt) / (D_full - D_corrupt)
         → 1.0  means ablating S destroys the behaviour completely
         → 0.0  means ablating S has no effect beyond the corrupt baseline
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def faith_s(d_full: float, d_s: float, d_corrupt: float) -> float:
    """Faith_S = (D_full - D_S) / (D_full - D_corrupt).

    Measures the fraction of the clean-vs-corrupt behavior gap that is DESTROYED
    by mean-ablating S:
      → 1.0  S alone accounts for all the behaviour (ablating S = full corruption)
      → 0.0  S is irrelevant (ablating S has no effect)
      → >1   ablation overshoots the corrupt baseline

    Note: the task spec writes (D_S - D_corrupt)/(D_full - D_corrupt), which is the
    complementary "remaining fraction."  We invert so that high Faith_S means S is
    causal, consistent with the GO condition "Faith_S above random."

    Returns NaN if the denominator is zero (D_full == D_corrupt).
    """
    denom = d_full - d_corrupt
    if abs(denom) < 1e-9:
        return float("nan")
    return (d_full - d_s) / denom


# ---------------------------------------------------------------------------
# Feature-set comparison
# ---------------------------------------------------------------------------

def jaccard(a: Set, b: Set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def pairwise_jaccards(feature_sets: List[Set]) -> np.ndarray:
    """Return upper-triangle pairwise Jaccard matrix for a list of sets."""
    n = len(feature_sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = jaccard(feature_sets[i], feature_sets[j])
            mat[i, j] = v
            mat[j, i] = v
    return mat


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def spearman_rho(
    predicted: List[float], realized: List[float]
) -> Tuple[float, float]:
    """Spearman correlation between per-feature predicted and realized effects.

    Returns (rho, p_value).
    """
    rho, pval = spearmanr(predicted, realized)
    return float(rho), float(pval)


# ---------------------------------------------------------------------------
# Helpers for check 04
# ---------------------------------------------------------------------------

def top_k_by_magnitude(
    layer_feature_acts: Dict[int, np.ndarray], k: int
) -> Set[Tuple[int, int]]:
    """Return the top-k (layer, feature_idx) pairs ranked by activation magnitude."""
    candidates: List[Tuple[float, int, int]] = []
    for layer, acts in layer_feature_acts.items():
        for feat_idx in np.argsort(-np.abs(acts)):
            candidates.append((float(np.abs(acts[feat_idx])), layer, int(feat_idx)))
    candidates.sort(reverse=True)
    return {(layer, feat) for _, layer, feat in candidates[:k]}

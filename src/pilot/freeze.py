"""Compute the consensus feature set S across multiple source prompts.

S = features that appear in the top-k circuit of at least `min_hits` prompts.
Also reports pairwise Jaccard diagnostics.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

from .metrics import pairwise_jaccards


def build_consensus_set(
    circuits: List[List[Tuple[int, int, float]]],
    min_hits: int = 2,
) -> Set[Tuple[int, int]]:
    """Return (layer, feature_idx) pairs present in >= min_hits circuits.

    circuits: list of outputs from build_feature_circuit().
    """
    counter: Counter = Counter()
    for circuit in circuits:
        for layer, feat_idx, _ in circuit:
            counter[(layer, feat_idx)] += 1
    return {feat for feat, count in counter.items() if count >= min_hits}


def jaccard_diagnostics(
    circuits: List[List[Tuple[int, int, float]]],
) -> Dict[str, float]:
    """Return summary statistics of pairwise Jaccard among the circuit sets."""
    sets = [{(layer, feat) for layer, feat, _ in c} for c in circuits]
    mat = pairwise_jaccards(sets)
    n = len(sets)
    # upper triangle (excluding diagonal)
    upper = [mat[i, j] for i in range(n) for j in range(i + 1, n)]
    # top-feature overlap: just the single best feature per circuit
    top_feats = [
        {(c[0][0], c[0][1])} if c else set()
        for c in circuits
    ]
    top_mat = pairwise_jaccards(top_feats)
    top_upper = [top_mat[i, j] for i in range(n) for j in range(i + 1, n)]

    return {
        "median_topk_jaccard": float(np.median(upper)),
        "mean_topk_jaccard": float(np.mean(upper)),
        "min_topk_jaccard": float(np.min(upper)),
        "max_topk_jaccard": float(np.max(upper)),
        "median_top1_jaccard": float(np.median(top_upper)),
    }

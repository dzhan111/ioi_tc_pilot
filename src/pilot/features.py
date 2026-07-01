"""Build a feature circuit for a single prompt.

Returns the top-k (layer, feature_idx, c_n) triples ranked by IxG attribution
toward the logit-difference direction.

c_n = (W_dec[n] · range_normal) * feature_acts[n]

Results are disk-cached keyed by (prompt, model_name, k).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Dict, List, Tuple

import torch
import numpy as np

# Vendor path so transcoder_circuits imports resolve
_VENDOR = os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from sae_training.sparse_autoencoder import SparseAutoencoder
from transcoder_circuits.circuit_analysis import get_transcoder_ixg

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'features')
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(prompt: str, model_name: str, k: int) -> str:
    raw = f"{prompt}|{model_name}|{k}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_feature_circuit(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    range_normal: torch.Tensor,
    k: int = 20,
    token_pos: int = -1,
    use_cache: bool = True,
    tc_active: bool = True,
    tc_layers: List[int] = None,
) -> List[Tuple[int, int, float]]:
    """Return top-k (layer, feature_idx, c_n) triples for a prompt.

    range_normal: (d_model,) direction in residual stream.
    token_pos: position to attribute (default -1 = last token).
    tc_active: if True, replace tc_layers MLPs with transcoders when building
        the cache so attribution and ablation share the same activation space.
    tc_layers: which layers to attribute and replace (default: [11]).
        Limiting to one layer avoids compounding approximation error.
    """
    from transcoder_circuits.replacement_ctx import TranscoderReplacementContext

    if tc_layers is None:
        tc_layers = [11]

    model_name = "gpt2-small"
    layer_tag = "L" + "_".join(str(l) for l in sorted(tc_layers))
    suffix = f"_tc{layer_tag}" if tc_active else ""
    cache_path = os.path.join(_CACHE_DIR, f"{_cache_key(prompt, model_name, k)}{suffix}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        if tc_active:
            tc_subset = [transcoders[l] for l in tc_layers]
            with TranscoderReplacementContext(model, tc_subset):
                _, cache = model.run_with_cache(tokens)
        else:
            _, cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    pos = token_pos if token_pos >= 0 else seq_len + token_pos

    all_contribs: List[Tuple[int, int, float]] = []
    for layer in tc_layers:
        tc = transcoders[layer]
        is_post_ln = ('ln2' in tc.cfg.hook_point and 'normalized' in tc.cfg.hook_point)
        ixg, _ = get_transcoder_ixg(
            tc, cache, range_normal,
            input_layer=layer,
            input_token_idx=pos,
            return_numpy=False,
            is_transcoder_post_ln=is_post_ln,
            return_feature_activs=True,
        )
        # ixg shape: (d_sae,) — c_n for each feature
        top_vals, top_idxs = torch.topk(ixg, k=k)
        for feat_idx, c_n in zip(top_idxs.tolist(), top_vals.tolist()):
            all_contribs.append((layer, feat_idx, c_n))

    # Sort by c_n descending and keep top k overall
    all_contribs.sort(key=lambda x: x[2], reverse=True)
    result = all_contribs[:k]

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result


def feature_set_from_circuit(
    circuit: List[Tuple[int, int, float]]
) -> set:
    """Convert circuit triples to a set of (layer, feature_idx) identity pairs."""
    return {(layer, feat_idx) for layer, feat_idx, _ in circuit}


def get_feature_activations(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    token_pos: int = -1,
    layers: List[int] = None,
) -> Dict[int, np.ndarray]:
    """Return raw feature activations per layer at token_pos (for lazy baseline).

    layers: which layers to compute activations for (default: [11]).
    """
    if layers is None:
        layers = [11]

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    pos = token_pos if token_pos >= 0 else seq_len + token_pos

    result: Dict[int, np.ndarray] = {}
    for layer in layers:
        tc = transcoders[layer]
        act_name = get_act_name('normalized', layer, 'ln2')
        feat_acts = tc(cache[act_name])[1][0, pos].detach().cpu().numpy()
        result[layer] = feat_acts
    return result

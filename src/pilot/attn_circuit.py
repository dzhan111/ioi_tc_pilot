"""Build an attention-head circuit for a single prompt.

Returns top-k (layer, head, c_h) triples ranked by IxG attribution toward
the logit-difference direction.

c_h = sum over src positions of:
        attn_pattern[h, dst, src] * (v[src, h] @ W_O[h]) · range_normal

This is the exact linear attribution used by get_attn_head_contribs in
the vendored library, summed over source tokens at the target destination.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Dict, List, Set, Tuple

import torch
import numpy as np

_VENDOR = os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transcoder_circuits.circuit_analysis import get_attn_head_contribs

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'attn_circuits')
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(prompt: str, k: int) -> str:
    raw = f"{prompt}|gpt2-small-attn|{k}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_attn_circuit(
    model: HookedTransformer,
    prompt: str,
    range_normal: torch.Tensor,
    k: int = 20,
    token_pos: int = -1,
    use_cache: bool = True,
) -> List[Tuple[int, int, float]]:
    """Return top-k (layer, head, c_h) triples for a prompt.

    range_normal: (d_model,) direction — e.g. ∂D/∂resid_post.
    token_pos: destination position to attribute (default -1 = last token).
    c_h = sum_src contribs[0, head, dst, src], skipping BOS (src=0).
    """
    cache_path = os.path.join(_CACHE_DIR, f"{_cache_key(prompt, k)}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, act_cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    pos = token_pos if token_pos >= 0 else seq_len + token_pos

    all_contribs: List[Tuple[int, int, float]] = []
    for layer in range(model.cfg.n_layers):
        # contribs: (batch=1, n_heads, dst, src)
        contribs = get_attn_head_contribs(model, act_cache, layer, range_normal)
        # Sum over source tokens at the target destination, skip BOS (src=0)
        head_scores = contribs[0, :, pos, 1:].sum(dim=-1)  # (n_heads,)
        for head in range(model.cfg.n_heads):
            all_contribs.append((layer, head, head_scores[head].item()))

    all_contribs.sort(key=lambda x: x[2], reverse=True)
    result = all_contribs[:k]

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result


def head_set_from_circuit(circuit: List[Tuple[int, int, float]]) -> Set[Tuple[int, int]]:
    """Convert circuit triples to a set of (layer, head) pairs."""
    return {(layer, head) for layer, head, _ in circuit}


def get_head_output_magnitudes(
    model: HookedTransformer,
    prompt: str,
    token_pos: int = -1,
) -> Dict[Tuple[int, int], float]:
    """Return L2 norm of each head's output at token_pos (for magnitude baseline)."""
    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, act_cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    pos = token_pos if token_pos >= 0 else seq_len + token_pos

    result: Dict[Tuple[int, int], float] = {}
    for layer in range(model.cfg.n_layers):
        hook_name = get_act_name('result', layer)
        head_outs = act_cache[hook_name][0, pos, :, :]  # (n_heads, d_model)
        for head in range(model.cfg.n_heads):
            result[(layer, head)] = head_outs[head].norm().item()
    return result

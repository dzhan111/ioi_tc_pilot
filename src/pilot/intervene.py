"""Mean-ablate a feature set S on a prompt and return logit-diff D.

Mean activations are precomputed from a small neutral corpus and cached.
The ablation modifies each transcoder's output by replacing the selected
features' activations with their dataset mean before decoding.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import numpy as np

_VENDOR = os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from sae_training.sparse_autoencoder import SparseAutoencoder

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'mean_acts')
os.makedirs(_CACHE_DIR, exist_ok=True)

# Small neutral corpus for computing mean feature activations
_NEUTRAL_SENTENCES = [
    "The sun rose over the mountains in the early morning light.",
    "Scientists discovered a new species of fish in the Pacific Ocean.",
    "The library opened its doors to students every weekday morning.",
    "Rain began to fall softly on the quiet city streets.",
    "A large bridge was built across the river last summer.",
    "The chef prepared a delicious meal for the dinner guests.",
    "Children played in the park while their parents watched.",
    "The professor explained the concept of gravity to the class.",
    "Flowers bloomed across the meadow in the spring season.",
    "The train arrived at the station exactly on schedule.",
    "An engineer designed a new type of energy-efficient engine.",
    "The museum displayed ancient artifacts from many civilizations.",
    "Clouds gathered above the hills as the afternoon progressed.",
    "The athlete trained for months before the big competition.",
    "A group of researchers published their findings last week.",
    "The old clock on the tower chimed at noon every day.",
    "Students gathered in the auditorium for the annual lecture.",
    "The river flowed slowly through the valley toward the sea.",
    "Workers finished building the new hospital ahead of schedule.",
    "The telescope revealed distant galaxies never seen before.",
    "Books lined every wall of the scholar's small study.",
    "The concert drew a crowd of thousands to the park.",
    "Engineers tested the new software for several hours.",
    "The garden was filled with roses of every color.",
    "A storm swept through the coastal town overnight.",
    "The committee voted on several important proposals.",
    "The pilot guided the plane safely through turbulence.",
    "Historians debated the causes of the ancient conflict.",
    "The factory produced thousands of units each day.",
    "A doctor examined the patient carefully in the clinic.",
    "The photograph captured a moment of pure joy.",
    "Researchers analyzed data collected over ten years.",
    "The mountain trail led to a beautiful waterfall.",
    "Students submitted their essays before the deadline.",
    "The bakery opened each morning at six o'clock.",
    "A team of divers explored the sunken ship.",
    "The newspaper reported on events from around the world.",
    "The farmer harvested wheat from the golden fields.",
    "An architect designed a stunning new concert hall.",
    "The experiment produced unexpected but interesting results.",
    "The bus arrived late due to heavy traffic.",
    "A musician composed a symphony over three years.",
    "The satellite transmitted data back to mission control.",
    "Volunteers helped clean up the local park on Saturday.",
    "The judge reviewed the evidence presented by both sides.",
    "A young artist painted murals on the city walls.",
    "The wind turbine generated power for the entire village.",
    "Firefighters responded quickly to the emergency call.",
    "The novel won several prestigious literary awards.",
    "The telescope pointed toward a newly discovered comet.",
]


@torch.no_grad()
def compute_mean_activations(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    force_recompute: bool = False,
) -> Dict[int, torch.Tensor]:
    """Return per-layer mean feature activations over a neutral corpus.

    Returns {layer: tensor of shape (d_sae,)}.
    Cached to disk after first computation.
    """
    cache_path = os.path.join(_CACHE_DIR, "neutral_means.pkl")
    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n_layers = len(transcoders)
    sums = [torch.zeros(tc.cfg.d_sae) for tc in transcoders]
    counts = [0] * n_layers

    for sent in _NEUTRAL_SENTENCES:
        tokens = model.to_tokens(sent)
        _, cache = model.run_with_cache(tokens)
        # Use only the FINAL token position so the mean reflects
        # "expected activation when predicting the next word" — the same
        # context as the IOI task's final position.  Pooling all positions
        # gives inflated means for common features and flips ablation signs.
        for layer, tc in enumerate(transcoders):
            act_name = get_act_name('normalized', layer, 'ln2')
            feat_acts = tc(cache[act_name])[1][0, -1]  # (d_sae,) — final pos only
            sums[layer] += feat_acts.cpu()
            counts[layer] += 1

    means = {layer: sums[layer] / counts[layer] for layer in range(n_layers)}

    with open(cache_path, "wb") as f:
        pickle.dump(means, f)

    return means


class _AblationWrapper(nn.Module):
    """Drop-in MLP replacement that mean-ablates selected transcoder features."""

    def __init__(
        self,
        transcoder: SparseAutoencoder,
        feature_indices: List[int],
        mean_acts: torch.Tensor,
    ):
        super().__init__()
        self.transcoder = transcoder
        self.feature_indices = feature_indices
        self.mean_acts = mean_acts  # (d_sae,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sae_out, feature_acts = self.transcoder(x)[:2]
        if not self.feature_indices:
            return sae_out
        # Compute correction: swap selected features to their means
        diff = self.mean_acts[self.feature_indices].to(x.device) - feature_acts[..., self.feature_indices]
        # diff shape: (..., len(feature_indices))
        # W_dec[feature_indices] shape: (len(feature_indices), d_out)
        correction = torch.einsum(
            '...f,fd->...d',
            diff,
            self.transcoder.W_dec[self.feature_indices],
        )
        return sae_out + correction


class _PassthroughWrapper(nn.Module):
    """Replaces an MLP with the transcoder's full output (no feature ablation)."""

    def __init__(self, transcoder: SparseAutoencoder):
        super().__init__()
        self.transcoder = transcoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transcoder(x)[0]


@torch.no_grad()
def run_with_transcoders(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    io_name: str,
    s_name: str,
) -> float:
    """Run the model with ALL MLPs replaced by transcoders (no ablation).

    Use this as the baseline (D_tc_full) when computing Faith_S so that
    ablation effects aren't confounded with transcoder approximation error.
    """
    original_mlps = {l: model.blocks[l].mlp for l in range(len(transcoders))}
    for l, tc in enumerate(transcoders):
        model.blocks[l].mlp = _PassthroughWrapper(tc)
    try:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)[0, -1]
        io_tok = model.to_single_token(" " + io_name)
        s_tok = model.to_single_token(" " + s_name)
        return (logits[io_tok] - logits[s_tok]).item()
    finally:
        for l, orig in original_mlps.items():
            model.blocks[l].mlp = orig


@torch.no_grad()
def ablate_and_run(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    feature_set: Set[Tuple[int, int]],
    mean_acts: Dict[int, torch.Tensor],
    io_name: str,
    s_name: str,
) -> float:
    """Replace feature_set with mean activations; return logit-diff D.

    All layers use their transcoder as the MLP replacement so the ablation
    is measured against the transcoder baseline (D_tc_full), not the original
    MLP.  Layers NOT in feature_set use a passthrough (unmodified) transcoder.
    """
    by_layer: Dict[int, List[int]] = {}
    for layer, feat_idx in feature_set:
        by_layer.setdefault(layer, []).append(feat_idx)

    original_mlps = {}
    for l, tc in enumerate(transcoders):
        original_mlps[l] = model.blocks[l].mlp
        if l in by_layer:
            model.blocks[l].mlp = _AblationWrapper(tc, by_layer[l], mean_acts[l])
        else:
            model.blocks[l].mlp = _PassthroughWrapper(tc)

    try:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)[0, -1]
        io_tok = model.to_single_token(" " + io_name)
        s_tok = model.to_single_token(" " + s_name)
        return (logits[io_tok] - logits[s_tok]).item()
    finally:
        for l, orig in original_mlps.items():
            model.blocks[l].mlp = orig


@torch.no_grad()
def ablate_gt_and_run(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    feature_set: Set[Tuple[int, int]],
    mean_acts: Dict[int, torch.Tensor],
    high_tok: str,
    low_tok: str,
) -> float:
    """Ablate feature_set on a greater-than prompt; return logit(high) - logit(low).

    All layers use transcoder wrappers so the result is in the same framework
    as the attribution (consistent with ablate_and_run).
    """
    by_layer: Dict[int, List[int]] = {}
    for layer, feat_idx in feature_set:
        by_layer.setdefault(layer, []).append(feat_idx)

    original_mlps = {}
    for l, tc in enumerate(transcoders):
        original_mlps[l] = model.blocks[l].mlp
        if l in by_layer:
            model.blocks[l].mlp = _AblationWrapper(tc, by_layer[l], mean_acts[l])
        else:
            model.blocks[l].mlp = _PassthroughWrapper(tc)

    try:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)[0, -1]
        high_id = model.to_single_token(high_tok)
        low_id = model.to_single_token(low_tok)
        d = (logits[high_id] - logits[low_id]).item()
    finally:
        for l, orig in original_mlps.items():
            model.blocks[l].mlp = orig

    return d


@torch.no_grad()
def run_gt_with_transcoders(
    model: HookedTransformer,
    transcoders: List[SparseAutoencoder],
    prompt: str,
    high_tok: str,
    low_tok: str,
) -> float:
    """TC-baseline D for a greater-than prompt."""
    original_mlps = {l: model.blocks[l].mlp for l in range(len(transcoders))}
    for l, tc in enumerate(transcoders):
        model.blocks[l].mlp = _PassthroughWrapper(tc)
    try:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)[0, -1]
        high_id = model.to_single_token(high_tok)
        low_id = model.to_single_token(low_tok)
        return (logits[high_id] - logits[low_id]).item()
    finally:
        for l, orig in original_mlps.items():
            model.blocks[l].mlp = orig

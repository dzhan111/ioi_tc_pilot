"""Mean-ablate a set of attention heads and return logit-diff D.

Mean head outputs are computed over a neutral corpus at the final token
position and used to replace the selected heads' contributions at ALL
sequence positions during ablation — fully removing their causal effect.

No transcoder replacement. Ablation is exact (hook-based), so D_full
equals the original model's logit difference with no approximation error.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np

_VENDOR = os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'attn_means')
os.makedirs(_CACHE_DIR, exist_ok=True)

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
def compute_mean_head_outputs(
    model: HookedTransformer,
    force_recompute: bool = False,
) -> Dict[int, torch.Tensor]:
    """Return per-layer mean head outputs over a neutral corpus.

    Returns {layer: tensor of shape (n_heads, d_model)}.
    Mean is computed at the FINAL token position of each neutral sentence
    so it reflects the expected head output when predicting the next word —
    matching the IOI task context.
    Cached to disk after first computation.
    """
    cache_path = os.path.join(_CACHE_DIR, "neutral_head_means.pkl")
    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    sums = [torch.zeros(n_heads, d_model) for _ in range(n_layers)]
    counts = [0] * n_layers

    for sent in _NEUTRAL_SENTENCES:
        tokens = model.to_tokens(sent)
        _, cache = model.run_with_cache(tokens)
        for layer in range(n_layers):
            hook_name = get_act_name('result', layer)
            # cache[hook_name]: (batch=1, seq, n_heads, d_model)
            head_out = cache[hook_name][0, -1, :, :].cpu()  # (n_heads, d_model)
            sums[layer] += head_out
            counts[layer] += 1

    means = {layer: sums[layer] / counts[layer] for layer in range(n_layers)}

    with open(cache_path, "wb") as f:
        pickle.dump(means, f)

    return means


def _make_ablation_hook(heads: List[int], mean: torch.Tensor):
    """Return a hook that replaces selected heads with their mean at the final position only.

    We ablate only position -1 because we measure causal effect on the final-token
    prediction. Ablating all positions propagates errors through intermediate residuals
    and confounds the single-head causal estimate.
    """
    mean_cpu = mean.cpu()

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # value: (batch, seq, n_heads, d_model)
        for h in heads:
            value[:, -1, h, :] = mean_cpu[h].to(value.device)
        return value

    return hook_fn


@torch.no_grad()
def ablate_heads_and_run(
    model: HookedTransformer,
    prompt: str,
    head_set: Set[Tuple[int, int]],
    mean_head_outputs: Dict[int, torch.Tensor],
    io_name: str,
    s_name: str,
) -> float:
    """Replace head_set outputs with means at all positions; return logit-diff D.

    No transcoder replacement — this is exact hook-based ablation on the
    original model. D_full is the unmodified model's logit difference.
    """
    by_layer: Dict[int, List[int]] = {}
    for layer, head in head_set:
        by_layer.setdefault(layer, []).append(head)

    fwd_hooks = []
    for layer, heads in by_layer.items():
        hook_name = get_act_name('result', layer)
        fwd_hooks.append((hook_name, _make_ablation_hook(heads, mean_head_outputs[layer])))

    tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0, -1]
    io_tok = model.to_single_token(" " + io_name)
    s_tok = model.to_single_token(" " + s_name)
    return (logits[io_tok] - logits[s_tok]).item()


@torch.no_grad()
def ablate_heads_gt_and_run(
    model: HookedTransformer,
    prompt: str,
    head_set: Set[Tuple[int, int]],
    mean_head_outputs: Dict[int, torch.Tensor],
    high_tok: str,
    low_tok: str,
) -> float:
    """Ablate head_set on a Greater-Than prompt; return logit(high) - logit(low)."""
    by_layer: Dict[int, List[int]] = {}
    for layer, head in head_set:
        by_layer.setdefault(layer, []).append(head)

    fwd_hooks = []
    for layer, heads in by_layer.items():
        hook_name = get_act_name('result', layer)
        fwd_hooks.append((hook_name, _make_ablation_hook(heads, mean_head_outputs[layer])))

    tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0, -1]
    high_id = model.to_single_token(high_tok)
    low_id = model.to_single_token(low_tok)
    return (logits[high_id] - logits[low_id]).item()

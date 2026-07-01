"""Load GPT-2 small and expose the logit-difference readout D = logit(IO) - logit(S)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits'))

from typing import Optional

import torch
from transformer_lens import HookedTransformer

_MODEL: Optional[HookedTransformer] = None


def get_model() -> HookedTransformer:
    """Return a cached, eval-mode GPT-2 small instance pinned to CPU."""
    global _MODEL
    if _MODEL is None:
        _MODEL = HookedTransformer.from_pretrained(
            "gpt2",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        _MODEL.cfg.use_attn_result = True
        _MODEL.setup()
        _MODEL = _MODEL.cpu().eval()
    return _MODEL


@torch.no_grad()
def logit_diff(model: HookedTransformer, prompt: str, io_name: str, s_name: str) -> float:
    """D = logit(IO token) - logit(S token) at the final position."""
    tokens = model.to_tokens(prompt)
    logits = model(tokens)[0, -1]  # (d_vocab,)
    io_tok = model.to_single_token(" " + io_name)
    s_tok = model.to_single_token(" " + s_name)
    return (logits[io_tok] - logits[s_tok]).item()


@torch.no_grad()
def logit_diff_direction(model: HookedTransformer, io_name: str, s_name: str) -> torch.Tensor:
    """Approximate direction in residual-stream space for logit(IO) - logit(S).

    Shape: (d_model,).  Used as range_normal for IxG attribution.
    This is W_U[:,IO] - W_U[:,S], BEFORE the final LayerNorm correction.
    """
    io_tok = model.to_single_token(" " + io_name)
    s_tok = model.to_single_token(" " + s_name)
    return (model.W_U[:, io_tok] - model.W_U[:, s_tok]).detach().cpu()


def logit_diff_direction_grad(
    model: HookedTransformer,
    prompt: str,
    io_name: str,
    s_name: str,
) -> torch.Tensor:
    """True gradient ∂D/∂resid_post at the operating point (accounts for final LN).

    Shape: (d_model,).  More accurate than logit_diff_direction for attribution
    because it accounts for the non-linear final LayerNorm.
    """
    io_tok = model.to_single_token(" " + io_name)
    s_tok = model.to_single_token(" " + s_name)

    grad_vec: list = []

    def hook_resid(value: torch.Tensor, hook) -> torch.Tensor:
        value = value.detach().requires_grad_(True)
        grad_vec.append(value)
        return value

    with torch.enable_grad():
        model.run_with_hooks(
            model.to_tokens(prompt),
            fwd_hooks=[("blocks.11.hook_resid_post", hook_resid)],
        )
        resid = grad_vec[0]  # (1, seq, d_model)
        # Recompute logits from the hooked residual (forward through ln_final + W_U)
        normed = model.ln_final(resid)
        logits = model.unembed(normed)  # (1, seq, d_vocab)
        d = logits[0, -1, io_tok] - logits[0, -1, s_tok]
        d.backward()

    return resid.grad[0, -1].detach().cpu()  # (d_model,)

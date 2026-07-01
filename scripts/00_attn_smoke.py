"""Check 00 (attn) — Smoke test.

Attribute the top-K attention heads for one IOI prompt, ablate them jointly,
and verify D drops substantially. Individual-head ablations show nonlinear
compensation effects (early layers are corrected by later ones), so we check
the joint effect of the top-K set which is the quantity used in all downstream
faith_S measurements.

VERDICT: GO   if joint ablation of top-K heads drops D by > 1.0 logit
         NO-GO otherwise
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch
torch.manual_seed(42)

from pilot.model import get_model, logit_diff, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.attn_circuit import build_attn_circuit
from pilot.attn_ablate import compute_mean_head_outputs, ablate_heads_and_run


def main():
    print("=== Check 00 (attn): Attention-head smoke test ===\n")

    model = get_model()
    print("Model loaded.\n")

    prompt_obj = IOI_SOURCE[0]
    prompt, io_name, s_name = prompt_obj.text, prompt_obj.io_name, prompt_obj.s_name
    print(f"Prompt : {prompt!r}")
    print(f"IO={io_name}, S={s_name}\n")

    d_full = logit_diff(model, prompt, io_name, s_name)
    print(f"D_full = {d_full:.4f}")

    K = 10
    range_normal = logit_diff_direction_grad(model, prompt, io_name, s_name)
    circuit = build_attn_circuit(model, prompt, range_normal, k=20, use_cache=False)
    top_layer, top_head, top_ch = circuit[0]
    print(f"\nTop attributed head: layer={top_layer}, head={top_head}, c_h={top_ch:.4f}")
    print(f"\nTop-{K} heads:")
    for l, h, c in circuit[:K]:
        print(f"  L{l}H{h}: c_h={c:.3f}")

    print("\nComputing mean head outputs over neutral corpus ...")
    mean_head_outputs = compute_mean_head_outputs(model)

    # Ablate full top-K set jointly (matches how faith_S is measured downstream)
    head_set = {(l, h) for l, h, _ in circuit[:K]}
    d_ablated = ablate_heads_and_run(model, prompt, head_set, mean_head_outputs, io_name, s_name)
    delta_d = d_full - d_ablated

    print(f"\nD_full             = {d_full:.4f}")
    print(f"D_ablated (top-{K}) = {d_ablated:.4f}")
    print(f"ΔD                 = {delta_d:.4f}")

    goes = delta_d > 1.0

    if goes:
        verdict = f"GO — top-{K} joint ablation dropped D by {delta_d:.3f} logits"
    else:
        verdict = f"NO-GO — ablation effect too small or wrong direction (ΔD={delta_d:.4f})"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/00_attn_smoke.json", "w") as f:
        json.dump({
            "d_full": d_full, "d_ablated": d_ablated, "delta_d": delta_d,
            "top_ch": top_ch, "K": K,
            "top_layer": top_layer, "top_head": top_head,
        }, f, indent=2)

    return goes


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

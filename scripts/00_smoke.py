"""Check 00 — Tooling smoke test (GATE).

Reproduce one feature attribution on a single IOI prompt, ablate the top
causal feature, and verify D moves in the predicted direction.

VERDICT: GO   if ablation moves D opposite to c_n's sign (expected causal direction)
         NO-GO if the intervention has no / wrong effect — report and stop
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch
torch.manual_seed(42)

from pilot.model import get_model, logit_diff, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.features import build_feature_circuit
from pilot.transcoders import load_transcoders
from pilot.intervene import ablate_and_run, run_with_transcoders, compute_mean_activations

def main():
    print("=== Check 00: Tooling smoke test ===\n")

    model = get_model()
    print("Model loaded.")

    print("Loading transcoders …")
    transcoders = load_transcoders()
    print(f"Loaded {len(transcoders)} transcoders.\n")

    prompt_obj = IOI_SOURCE[0]
    prompt, io_name, s_name = prompt_obj.text, prompt_obj.io_name, prompt_obj.s_name
    print(f"Prompt : {prompt!r}")
    print(f"IO={io_name}, S={s_name}\n")

    # Baseline D (original model, for reference only)
    d_orig = logit_diff(model, prompt, io_name, s_name)
    # Transcoder baseline D — all MLPs replaced; this is what attribution is computed against
    d_full = run_with_transcoders(model, transcoders, prompt, io_name, s_name)
    print(f"D_orig (original MLP) = {d_orig:.4f}")
    print(f"D_full (TC baseline)  = {d_full:.4f}")

    # Feature attribution — true gradient computed in TC framework so attribution
    # and ablation are in the same activation space.
    range_normal = logit_diff_direction_grad(model, prompt, io_name, s_name)
    circuit = build_feature_circuit(model, transcoders, prompt, range_normal, k=20, use_cache=False, tc_active=True)
    top_layer, top_feat, top_cn = circuit[0]
    print(f"\nTop causal feature: layer={top_layer}, feat_idx={top_feat}, c_n={top_cn:.4f}")
    print(f"Predicted sign of contribution to D: {'positive' if top_cn > 0 else 'negative'}")

    # Mean activations for ablation baseline
    print("\nComputing mean activations over neutral corpus …")
    mean_acts = compute_mean_activations(model, transcoders)

    # Ablate just the top feature; compare against TC baseline (not original MLP)
    feature_set = {(top_layer, top_feat)}
    d_ablated = ablate_and_run(model, transcoders, prompt, feature_set, mean_acts, io_name, s_name)
    delta_d = d_full - d_ablated
    print(f"\nD_full (TC baseline) = {d_full:.4f}")
    print(f"D_ablated            = {d_ablated:.4f}")
    print(f"ΔD (full - ablated)  = {delta_d:.4f}")
    print(f"c_n sign matches ΔD sign: {(top_cn > 0) == (delta_d > 0)}")

    # Sign check: ablating a feature with positive c_n should DECREASE D (delta_d > 0)
    sign_correct = (top_cn > 0 and delta_d > 0) or (top_cn < 0 and delta_d < 0)
    # Require a non-trivial effect (larger than floating-point noise)
    nontrivial = abs(delta_d) > 1e-4

    # Note: D_orig and D_full (TC baseline) differ substantially because replacing
    # all 12 MLPs simultaneously introduces compounding approximation errors.
    # We verify attribution within the TC framework (consistent baseline), not
    # against the original model D.
    if sign_correct and nontrivial:
        verdict = f"GO — ablation moved D by {delta_d:.4f} in predicted direction (c_n={top_cn:.3f})"
    elif sign_correct and not nontrivial:
        verdict = f"NO-GO — correct sign but effect below noise floor (ΔD={delta_d:.2e}); attribution signal too weak"
    else:
        verdict = f"NO-GO — ablation moved D in WRONG direction (ΔD={delta_d:.4f}, c_n={top_cn:.3f}); IxG attribution unreliable"

    print(f"\nVERDICT: {verdict}")

    # Save result for pytest
    os.makedirs("results", exist_ok=True)
    import json
    with open("results/00_smoke.json", "w") as f:
        json.dump({
            "d_orig": d_orig, "d_full": d_full, "d_ablated": d_ablated,
            "delta_d": delta_d, "top_cn": top_cn, "sign_correct": sign_correct,
            "top_layer": top_layer, "top_feat": top_feat,
        }, f, indent=2)

    return sign_correct and nontrivial

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

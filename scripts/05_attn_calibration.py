"""Check 05 (attn) — In-sample calibration.

On source prompts, ablate each head in S individually.
Correlate realized per-head ΔD against source c_h (Spearman ρ).

VERDICT:
  GO     rho > 0.4 — c_h predicts realized effect sizes
  NO-GO  rho near zero — attribution scores are directional only
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)

from pilot.model import get_model, logit_diff, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.attn_circuit import build_attn_circuit
from pilot.freeze import build_consensus_set
from pilot.attn_ablate import compute_mean_head_outputs, ablate_heads_and_run
from pilot.metrics import spearman_rho

K = 20


def main():
    print("=== Check 05 (attn): In-sample calibration ===\n")
    model = get_model()
    mean_head_outputs = compute_mean_head_outputs(model)

    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_attn_circuit(model, p.text, rn, k=K))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"Consensus head set S: {len(S)} heads\n")

    # Average c_h per head across source prompts
    ch_sum: dict = {}
    ch_count: dict = {}
    for circuit in circuits:
        for layer, head, ch in circuit:
            key = (layer, head)
            if key in S:
                ch_sum[key] = ch_sum.get(key, 0.0) + ch
                ch_count[key] = ch_count.get(key, 0) + 1
    ch_mean = {k: ch_sum[k] / ch_count[k] for k in S}

    predicted_effects = []
    realized_effects = []

    for p in IOI_SOURCE:
        d_full = logit_diff(model, p.text, p.io_name, p.s_name)
        for head_key in S:
            c_h = ch_mean[head_key]
            d_ablated = ablate_heads_and_run(
                model, p.text, {head_key}, mean_head_outputs, p.io_name, p.s_name
            )
            realized_delta = d_full - d_ablated
            predicted_effects.append(c_h)
            realized_effects.append(realized_delta)

    rho, pval = spearman_rho(predicted_effects, realized_effects)
    print(f"Spearman rho = {rho:.3f}  (p={pval:.4f})")
    print(f"N pairs: {len(predicted_effects)}")

    if rho > 0.4:
        verdict = f"GO — Spearman rho={rho:.3f} > 0.4; c_h predicts realized ΔD"
    elif rho > 0.1:
        verdict = f"INVESTIGATE — weak correlation rho={rho:.3f}; c_h is noisy predictor"
    else:
        verdict = f"NO-GO — rho={rho:.3f} near zero; drop quantitative axis"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/05_attn_calibration.json", "w") as f:
        json.dump({
            "rho": rho, "pval": pval, "n_pairs": len(predicted_effects),
            "predicted": predicted_effects, "realized": realized_effects,
        }, f, indent=2)


if __name__ == "__main__":
    main()

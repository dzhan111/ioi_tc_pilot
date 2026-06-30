"""Check 05 — In-sample calibration.

On source prompts (easiest case), ablate each feature in S individually.
Correlate realized per-feature delta-D against source c_n (Spearman rho).

VERDICT:
  GO     rho clearly positive (> 0.4)
  NO-GO  rho near zero → drop 'predicted vs realized' axis; keep only binary Faith_S
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.features import build_feature_circuit
from pilot.freeze import build_consensus_set
from pilot.intervene import ablate_and_run, run_with_transcoders, compute_mean_activations
from pilot.metrics import spearman_rho
from pilot.transcoders import load_transcoders

K = 20

def main():
    print("=== Check 05: In-sample calibration ===\n")
    model = get_model()
    transcoders = load_transcoders()
    mean_acts = compute_mean_activations(model, transcoders)

    # Build circuits and consensus set
    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_feature_circuit(model, transcoders, p.text, rn, k=K, tc_active=True))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"Consensus set S: {len(S)} features\n")

    # Build lookup: feature -> c_n averaged over source prompts
    cn_sum: dict = {}
    cn_count: dict = {}
    for circuit in circuits:
        for layer, feat_idx, cn in circuit:
            key = (layer, feat_idx)
            if key in S:
                cn_sum[key] = cn_sum.get(key, 0.0) + cn
                cn_count[key] = cn_count.get(key, 0) + 1
    cn_mean = {k: cn_sum[k] / cn_count[k] for k in S}

    predicted_effects = []
    realized_effects = []

    for p in IOI_SOURCE:
        # TC baseline so realized delta-D is measured within the attribution framework
        d_full = run_with_transcoders(model, transcoders, p.text, p.io_name, p.s_name)
        for feat_key in S:
            c_n = cn_mean[feat_key]
            d_ablated = ablate_and_run(
                model, transcoders, p.text, {feat_key}, mean_acts, p.io_name, p.s_name
            )
            realized_delta = d_full - d_ablated
            predicted_effects.append(c_n)
            realized_effects.append(realized_delta)

    rho, pval = spearman_rho(predicted_effects, realized_effects)
    print(f"Spearman rho = {rho:.3f}  (p={pval:.4f})")
    print(f"N pairs: {len(predicted_effects)}")

    if rho > 0.4:
        verdict = f"GO — Spearman rho={rho:.3f} > 0.4; c_n predicts realized delta-D; quantitative axis is valid"
    elif rho > 0.1:
        verdict = f"INVESTIGATE — weak correlation rho={rho:.3f}; c_n is a noisy predictor; report with caution"
    else:
        verdict = f"NO-GO — rho={rho:.3f} near zero; drop 'predicted vs realized' axis; keep only binary Faith_S identity"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/05_calibration.json", "w") as f:
        json.dump({"rho": rho, "pval": pval, "n_pairs": len(predicted_effects),
                   "predicted": predicted_effects[:50], "realized": realized_effects[:50]}, f, indent=2)

if __name__ == "__main__":
    main()

"""Check 03 — Cross-task specificity control.

Apply the frozen IOI consensus set S to 5 greater-than prompts.
Faith_S should be near zero (S is task-specific, not generic).

VERDICT:
  GO          cross-task Faith_S near zero / much smaller than on-task
  INVESTIGATE large cross-task effect → S is generic; restrict to IOI-specific features
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE, GT_CONTROL
from pilot.features import build_feature_circuit
from pilot.freeze import build_consensus_set
from pilot.intervene import ablate_gt_and_run, run_gt_with_transcoders, compute_mean_activations
from pilot.transcoders import load_transcoders

K = 20

def main():
    print("=== Check 03: Cross-task specificity control ===\n")
    model = get_model()
    transcoders = load_transcoders()
    mean_acts = compute_mean_activations(model, transcoders)

    # Build IOI consensus set S
    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_feature_circuit(model, transcoders, p.text, rn, k=K, tc_active=True))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"IOI consensus set S: {len(S)} features\n")

    cross_task_faiths = []

    for gp in GT_CONTROL:
        # Use TC baseline for consistency with ablation framework
        d_full = run_gt_with_transcoders(
            model, transcoders, gp.text, gp.high_tok, gp.low_tok
        )

        d_ablated = ablate_gt_and_run(
            model, transcoders, gp.text, S, mean_acts, gp.high_tok, gp.low_tok
        )
        # Faith_S on GT task; should be near 0 if IOI features don't matter
        from pilot.metrics import faith_s
        fs = faith_s(d_full=d_full, d_s=d_ablated, d_corrupt=0.0)
        cross_task_faiths.append(fs)
        print(f"  {gp.text!r}")
        print(f"    D_full={d_full:.3f}  D_ablated={d_ablated:.3f}  Faith_S={fs:.3f}")

    mean_faith = float(np.mean(cross_task_faiths))
    print(f"\nMean cross-task Faith_S: {mean_faith:.3f}")

    if abs(mean_faith) < 0.1:
        verdict = f"GO — cross-task Faith_S={mean_faith:.3f} near zero; IOI set S is task-specific"
    elif abs(mean_faith) < 0.3:
        verdict = f"INVESTIGATE — moderate cross-task Faith_S={mean_faith:.3f}; S may include generic features"
    else:
        verdict = f"INVESTIGATE — large cross-task Faith_S={mean_faith:.3f}; S not IOI-specific; restrict and re-run"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/03_specificity.json", "w") as f:
        json.dump({"cross_task_faiths": cross_task_faiths, "mean_faith": mean_faith, "S_size": len(S)}, f, indent=2)

if __name__ == "__main__":
    main()

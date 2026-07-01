"""Check 03 (attn) — Cross-task specificity.

Apply IOI consensus head set S to 5 Greater-Than prompts.
Faith_S should be near zero — IOI heads should not matter for GT.

VERDICT:
  GO          cross-task Faith_S near zero
  INVESTIGATE large cross-task effect — S includes generic heads
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE, GT_CONTROL
from pilot.attn_circuit import build_attn_circuit
from pilot.freeze import build_consensus_set
from pilot.attn_ablate import compute_mean_head_outputs, ablate_heads_gt_and_run
from pilot.metrics import faith_s

K = 20


def main():
    print("=== Check 03 (attn): Cross-task specificity ===\n")
    model = get_model()
    mean_head_outputs = compute_mean_head_outputs(model)

    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_attn_circuit(model, p.text, rn, k=K))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"IOI consensus head set S: {len(S)} heads\n")

    cross_task_faiths = []

    for gp in GT_CONTROL:
        with torch.no_grad():
            tokens = model.to_tokens(gp.text)
            logits = model(tokens)[0, -1]
        high_id = model.to_single_token(gp.high_tok)
        low_id = model.to_single_token(gp.low_tok)
        d_full = (logits[high_id] - logits[low_id]).item()

        d_ablated = ablate_heads_gt_and_run(
            model, gp.text, S, mean_head_outputs, gp.high_tok, gp.low_tok
        )
        fs = faith_s(d_full=d_full, d_s=d_ablated, d_corrupt=0.0)
        cross_task_faiths.append(fs)
        print(f"  {gp.text!r}")
        print(f"    D_full={d_full:.3f}  D_ablated={d_ablated:.3f}  Faith_S={fs:.3f}")

    mean_faith = float(np.mean(cross_task_faiths))
    print(f"\nMean cross-task Faith_S: {mean_faith:.3f}")

    if abs(mean_faith) < 0.1:
        verdict = f"GO — cross-task Faith_S={mean_faith:.3f} ≈ 0; S is IOI-specific"
    elif abs(mean_faith) < 0.3:
        verdict = f"INVESTIGATE — moderate cross-task Faith_S={mean_faith:.3f}; S may include generic heads"
    else:
        verdict = f"INVESTIGATE — large cross-task Faith_S={mean_faith:.3f}; S not IOI-specific"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/03_attn_specificity.json", "w") as f:
        json.dump({
            "cross_task_faiths": cross_task_faiths,
            "mean_faith": mean_faith, "S_size": len(S),
        }, f, indent=2)


if __name__ == "__main__":
    main()

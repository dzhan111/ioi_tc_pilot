"""Check 01 (attn) — Attention-head circuit stability.

Build head circuits on 10 IOI source prompts. Report pairwise Jaccard
of top-k head sets and top-1 head overlap.

VERDICT:
  GO          median Jaccard > 0.3 — stable circuit
  INVESTIGATE only top-1 stable; top-k noisy
  NO-GO       near-zero overlap even for top heads
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch
torch.manual_seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.attn_circuit import build_attn_circuit, head_set_from_circuit
from pilot.freeze import build_consensus_set, jaccard_diagnostics

K = 20


def main():
    print("=== Check 01 (attn): Attention-head circuit stability ===\n")
    model = get_model()

    circuits = []
    for i, p in enumerate(IOI_SOURCE):
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuit = build_attn_circuit(model, p.text, rn, k=K)
        circuits.append(circuit)
        top = circuit[0]
        print(f"Prompt {i:02d}: top head L{top[0]}H{top[1]}  c_h={top[2]:.3f}")

    diag = jaccard_diagnostics(circuits)
    print(f"\nPairwise Jaccard (top-{K} head sets):")
    print(f"  median = {diag['median_topk_jaccard']:.3f}")
    print(f"  mean   = {diag['mean_topk_jaccard']:.3f}")
    print(f"  min    = {diag['min_topk_jaccard']:.3f}")
    print(f"  max    = {diag['max_topk_jaccard']:.3f}")
    print(f"\nTop-1 head pairwise Jaccard: median = {diag['median_top1_jaccard']:.3f}")

    S = build_consensus_set(circuits, min_hits=2)
    print(f"\nConsensus set size (min_hits=2): {len(S)}")
    for layer, head in sorted(S):
        print(f"  L{layer}H{head}")

    med_jac = diag['median_topk_jaccard']
    top1_jac = diag['median_top1_jaccard']

    if med_jac > 0.3:
        verdict = f"GO — median Jaccard={med_jac:.3f} > 0.3; {len(S)}-head consensus circuit"
    elif top1_jac > 0.5:
        verdict = f"INVESTIGATE — top-k Jaccard={med_jac:.3f} low but top-1={top1_jac:.3f}; tighten k"
    else:
        verdict = f"NO-GO — Jaccard={med_jac:.3f}, top-1={top1_jac:.3f}; head circuit unstable"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/01_attn_stability.json", "w") as f:
        json.dump({
            **diag, "consensus_size": len(S), "k": K,
            "consensus_heads": [list(h) for h in sorted(S)],
        }, f, indent=2)


if __name__ == "__main__":
    main()

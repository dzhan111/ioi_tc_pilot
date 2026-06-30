"""Check 01 — Feature-set stability (GATE).

Build feature circuits on 10 IOI source prompts.
Report pairwise Jaccard of top-k feature sets and top-1 feature overlap.

VERDICT:
  GO          median top-k Jaccard > 0.3 OR top features recur across most prompts
  INVESTIGATE only top-1 features are stable (tighten k, proceed with caution)
  NO-GO       near-zero overlap even for top features → freeze-a-circuit premise breaks
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch
torch.manual_seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE
from pilot.features import build_feature_circuit
from pilot.freeze import build_consensus_set, jaccard_diagnostics
from pilot.transcoders import load_transcoders

K = 20  # top-k features per prompt

def main():
    print("=== Check 01: Feature-set stability ===\n")
    model = get_model()
    transcoders = load_transcoders()

    circuits = []
    for i, p in enumerate(IOI_SOURCE):
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuit = build_feature_circuit(model, transcoders, p.text, rn, k=K, tc_active=True)
        circuits.append(circuit)
        top = circuit[0]
        print(f"Prompt {i:02d}: top feature L{top[0]}/F{top[1]} c_n={top[2]:.3f}")

    diag = jaccard_diagnostics(circuits)
    print(f"\nPairwise Jaccard (top-{K} sets):")
    print(f"  median = {diag['median_topk_jaccard']:.3f}")
    print(f"  mean   = {diag['mean_topk_jaccard']:.3f}")
    print(f"  min    = {diag['min_topk_jaccard']:.3f}")
    print(f"  max    = {diag['max_topk_jaccard']:.3f}")
    print(f"\nTop-1 feature pairwise Jaccard:")
    print(f"  median = {diag['median_top1_jaccard']:.3f}")

    # Consensus set: features appearing in ≥ 2 prompts
    S = build_consensus_set(circuits, min_hits=2)
    print(f"\nConsensus set size (min_hits=2): {len(S)}")

    # Verdict
    med_jac = diag['median_topk_jaccard']
    top1_jac = diag['median_top1_jaccard']

    if med_jac > 0.3:
        verdict = f"GO — median Jaccard={med_jac:.3f} > 0.3; clear consensus set of {len(S)} features"
    elif top1_jac > 0.5:
        verdict = f"INVESTIGATE — top-k Jaccard={med_jac:.3f} low but top-1 Jaccard={top1_jac:.3f} OK; tighten k"
    else:
        verdict = f"NO-GO — median Jaccard={med_jac:.3f}, top-1 Jaccard={top1_jac:.3f}; feature set is unstable; pivot to per-prompt framing"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/01_stability.json", "w") as f:
        json.dump({**diag, "consensus_size": len(S), "k": K,
                   "consensus_features": [list(f) for f in S]}, f, indent=2)

if __name__ == "__main__":
    main()

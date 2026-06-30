"""Check 04 — Lazy-baseline comparison.

On the 5 paraphrases, compare Faith_S for attribution feature set S
vs magnitude-ranked features (top-k by activation) at the same k.

Not a hard gate, but flag clearly if magnitude matches / beats S.
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from pilot.model import get_model, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE, IOI_PARAPHRASE
from pilot.features import build_feature_circuit, get_feature_activations
from pilot.freeze import build_consensus_set
from pilot.intervene import ablate_and_run, run_with_transcoders, compute_mean_activations
from pilot.metrics import faith_s, top_k_by_magnitude  # noqa: F401
from pilot.transcoders import load_transcoders

K = 20

def main():
    print("=== Check 04: Lazy-baseline comparison ===\n")
    model = get_model()
    transcoders = load_transcoders()
    mean_acts = compute_mean_activations(model, transcoders)

    # IOI consensus set S (attribution-ranked)
    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_feature_circuit(model, transcoders, p.text, rn, k=K, tc_active=True))
    S = build_consensus_set(circuits, min_hits=2)
    k_s = len(S)
    print(f"Attribution set S size: {k_s}\n")

    attr_faiths, mag_faiths = [], []

    for p in IOI_PARAPHRASE:
        # TC baseline for consistent Faith_S
        d_full = run_with_transcoders(model, transcoders, p.text, p.io_name, p.s_name)

        # Attribution set ablation
        d_attr = ablate_and_run(model, transcoders, p.text, S, mean_acts, p.io_name, p.s_name)

        # Magnitude baseline: top-k features by activation at final position
        layer_acts = get_feature_activations(model, transcoders, p.text)
        S_mag = top_k_by_magnitude(layer_acts, k=k_s)
        d_mag = ablate_and_run(model, transcoders, p.text, S_mag, mean_acts, p.io_name, p.s_name)

        # D_corrupt = 0 (neutral baseline: no preference between IO and S)
        d_corrupt = 0.0

        fs_attr = faith_s(d_full, d_attr, d_corrupt)
        fs_mag = faith_s(d_full, d_mag, d_corrupt)
        attr_faiths.append(fs_attr)
        mag_faiths.append(fs_mag)

        print(f"  {p.text[:50]!r}")
        print(f"    D_full={d_full:.3f}  D_attr={d_attr:.3f}  D_mag={d_mag:.3f}")
        print(f"    Faith_S(attr)={fs_attr:.3f}  Faith_S(mag)={fs_mag:.3f}")

    mean_attr = float(np.nanmean(attr_faiths))
    mean_mag = float(np.nanmean(mag_faiths))
    gap = mean_attr - mean_mag
    print(f"\nMean Faith_S  attribution={mean_attr:.3f}  magnitude={mean_mag:.3f}  gap={gap:.3f}")

    if gap < -0.1:
        note = "FLAG: magnitude baseline matches or beats attribution — the graph may buy nothing over raw activation magnitude"
    elif gap < 0.1:
        note = "NOTE: small gap; attribution and magnitude perform similarly on easy paraphrases"
    else:
        note = "Attribution set outperforms magnitude baseline"

    print(f"\n{note}")
    print(f"\nVERDICT: REPORT — {note} (gap={gap:.3f})")

    os.makedirs("results", exist_ok=True)
    with open("results/04_lazy_baseline.json", "w") as f:
        json.dump({"attr_faiths": attr_faiths, "mag_faiths": mag_faiths,
                   "mean_attr": mean_attr, "mean_mag": mean_mag, "gap": gap}, f, indent=2)

if __name__ == "__main__":
    main()

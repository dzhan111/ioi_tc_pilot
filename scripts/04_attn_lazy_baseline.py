"""Check 04 (attn) — Lazy baseline comparison.

Compare Faith_S for attribution head set S vs magnitude-ranked heads
(top-k by L2 norm of head output) on 5 paraphrases.

Not a hard gate; flag if magnitude matches or beats S.
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)

from pilot.model import get_model, logit_diff, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE, IOI_PARAPHRASE
from pilot.attn_circuit import build_attn_circuit, get_head_output_magnitudes
from pilot.freeze import build_consensus_set
from pilot.attn_ablate import compute_mean_head_outputs, ablate_heads_and_run
from pilot.metrics import faith_s

K = 20
D_CORRUPT = 0.0


def main():
    print("=== Check 04 (attn): Lazy baseline comparison ===\n")
    model = get_model()
    mean_head_outputs = compute_mean_head_outputs(model)

    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_attn_circuit(model, p.text, rn, k=K))
    S = build_consensus_set(circuits, min_hits=2)
    k_s = len(S)
    print(f"Attribution head set S size: {k_s}\n")

    attr_faiths, mag_faiths = [], []

    for p in IOI_PARAPHRASE:
        d_full = logit_diff(model, p.text, p.io_name, p.s_name)

        d_attr = ablate_heads_and_run(model, p.text, S, mean_head_outputs, p.io_name, p.s_name)

        # Magnitude baseline: top-k_s heads by L2 norm of head output at final position
        magnitudes = get_head_output_magnitudes(model, p.text)
        top_mag_heads = set(sorted(magnitudes, key=magnitudes.get, reverse=True)[:k_s])
        d_mag = ablate_heads_and_run(model, p.text, top_mag_heads, mean_head_outputs, p.io_name, p.s_name)

        fs_attr = faith_s(d_full, d_attr, D_CORRUPT)
        fs_mag = faith_s(d_full, d_mag, D_CORRUPT)
        attr_faiths.append(fs_attr)
        mag_faiths.append(fs_mag)

        print(f"  {p.text[:55]!r}")
        print(f"    D_full={d_full:.3f}  D_attr={d_attr:.3f}  D_mag={d_mag:.3f}")
        print(f"    Faith_S(attr)={fs_attr:.3f}  Faith_S(mag)={fs_mag:.3f}")

    mean_attr = float(np.nanmean(attr_faiths))
    mean_mag = float(np.nanmean(mag_faiths))
    gap = mean_attr - mean_mag
    print(f"\nMean Faith_S  attribution={mean_attr:.3f}  magnitude={mean_mag:.3f}  gap={gap:.3f}")

    if gap < -0.1:
        note = "FLAG: magnitude baseline matches or beats attribution — gradient alignment adds no value"
    elif gap < 0.1:
        note = "NOTE: small gap; attribution and magnitude perform similarly"
    else:
        note = "Attribution set outperforms magnitude baseline"

    print(f"\n{note}")
    print(f"\nVERDICT: REPORT — {note} (gap={gap:.3f})")

    os.makedirs("results", exist_ok=True)
    with open("results/04_attn_lazy_baseline.json", "w") as f:
        json.dump({
            "attr_faiths": attr_faiths, "mag_faiths": mag_faiths,
            "mean_attr": mean_attr, "mean_mag": mean_mag, "gap": gap,
        }, f, indent=2)


if __name__ == "__main__":
    main()

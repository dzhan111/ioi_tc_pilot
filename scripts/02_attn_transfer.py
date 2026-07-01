"""Check 02 (attn) — Attention-head transfer to paraphrases.

Freeze consensus head set S from source prompts.
On 5 easy paraphrases, mean-ablate S and measure Faith_S.
Compare against a random-heads baseline of the same size.

VERDICT:
  GO          Faith_S(S) meaningfully above Faith_S(random)
  INVESTIGATE modest gap; check whether S heads fire on paraphrases
  NO-GO       gap ≤ 0; S does not outperform random
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'transcoder_circuits'))

import torch, numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from pilot.model import get_model, logit_diff, logit_diff_direction_grad
from pilot.prompts import IOI_SOURCE, IOI_PARAPHRASE
from pilot.attn_circuit import build_attn_circuit
from pilot.freeze import build_consensus_set
from pilot.attn_ablate import compute_mean_head_outputs, ablate_heads_and_run
from pilot.metrics import faith_s

K = 20
D_CORRUPT = 0.0


def main():
    print("=== Check 02 (attn): Transfer to paraphrases ===\n")
    model = get_model()
    mean_head_outputs = compute_mean_head_outputs(model)

    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_attn_circuit(model, p.text, rn, k=K))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"Consensus head set S: {len(S)} heads")
    for layer, head in sorted(S):
        print(f"  L{layer}H{head}")
    print()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    random_S = set()
    while len(random_S) < len(S):
        random_S.add((random.randint(0, n_layers - 1), random.randint(0, n_heads - 1)))

    attr_faiths, rand_faiths = [], []

    for p in IOI_PARAPHRASE:
        d_full = logit_diff(model, p.text, p.io_name, p.s_name)
        d_s = ablate_heads_and_run(model, p.text, S, mean_head_outputs, p.io_name, p.s_name)
        d_rand = ablate_heads_and_run(model, p.text, random_S, mean_head_outputs, p.io_name, p.s_name)

        fs_s = faith_s(d_full=d_full, d_s=d_s, d_corrupt=D_CORRUPT)
        fs_rand = faith_s(d_full=d_full, d_s=d_rand, d_corrupt=D_CORRUPT)
        attr_faiths.append(fs_s)
        rand_faiths.append(fs_rand)

        print(f"  {p.text[:55]!r}")
        print(f"    D_full={d_full:.3f}  D_S={d_s:.3f}  D_rand={d_rand:.3f}")
        print(f"    Faith_S(attr)={fs_s:.3f}  Faith_S(rand)={fs_rand:.3f}")

    mean_attr = float(np.mean(attr_faiths))
    mean_rand = float(np.mean(rand_faiths))
    gap = mean_attr - mean_rand
    print(f"\nMean Faith_S  attr={mean_attr:.3f}  rand={mean_rand:.3f}  gap={gap:.3f}")

    if gap > 0.15:
        verdict = f"GO — Faith_S(S)={mean_attr:.3f} >> Faith_S(rand)={mean_rand:.3f} (gap={gap:.3f})"
    elif gap > 0.05:
        verdict = f"INVESTIGATE — modest gap={gap:.3f}; check head activations on paraphrases"
    else:
        verdict = f"NO-GO — gap={gap:.3f}; S does not outperform random on paraphrases"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/02_attn_transfer.json", "w") as f:
        json.dump({
            "attr_faiths": attr_faiths, "rand_faiths": rand_faiths,
            "mean_attr": mean_attr, "mean_rand": mean_rand,
            "gap": gap, "S_size": len(S),
        }, f, indent=2)


if __name__ == "__main__":
    main()

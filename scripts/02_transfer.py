"""Check 02 — Minimal transfer (GATE).

Freeze consensus set S from source prompts.
On 5 easy paraphrases, mean-ablate S and measure Faith_S.
Compare against a random-features baseline of the same size.

Faith_S = (D_full - D_S) / D_full   (D_corrupt=0: neutral baseline)
  → 1.0  ablating S completely destroys the IOI preference
  → 0.0  ablating S has no effect

VERDICT:
  GO          Faith_S meaningfully above random baseline on paraphrases
  INVESTIGATE S features inactive on paraphrases (tokenization / layer issue)
  NO-GO       no transfer at all → central claim fails
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
from pilot.features import build_feature_circuit
from pilot.freeze import build_consensus_set
from pilot.intervene import ablate_and_run, run_with_transcoders, compute_mean_activations
from pilot.metrics import faith_s  # noqa: F401
from pilot.transcoders import load_transcoders

K = 20
D_CORRUPT = 0.0   # neutral: model has no preference between IO and S

def main():
    print("=== Check 02: Minimal transfer ===\n")
    model = get_model()
    transcoders = load_transcoders()
    mean_acts = compute_mean_activations(model, transcoders)

    # Build consensus set from source prompts
    circuits = []
    for p in IOI_SOURCE:
        rn = logit_diff_direction_grad(model, p.text, p.io_name, p.s_name)
        circuits.append(build_feature_circuit(model, transcoders, p.text, rn, k=K, tc_active=True))
    S = build_consensus_set(circuits, min_hits=2)
    print(f"Consensus set S: {len(S)} features\n")

    # Random baseline of same size (random layer/feature_idx pairs)
    n_layers, d_sae = len(transcoders), transcoders[0].cfg.d_sae
    random_S = set()
    random.seed(42)
    while len(random_S) < len(S):
        random_S.add((random.randint(0, n_layers - 1), random.randint(0, d_sae - 1)))

    attr_faiths, rand_faiths = [], []

    for p in IOI_PARAPHRASE:
        # Use TC baseline so Faith_S is measured within the transcoder framework
        d_full = run_with_transcoders(model, transcoders, p.text, p.io_name, p.s_name)
        d_s = ablate_and_run(model, transcoders, p.text, S, mean_acts, p.io_name, p.s_name)
        d_rand = ablate_and_run(model, transcoders, p.text, random_S, mean_acts, p.io_name, p.s_name)

        fs_s = faith_s(d_full=d_full, d_s=d_s, d_corrupt=D_CORRUPT)
        fs_rand = faith_s(d_full=d_full, d_s=d_rand, d_corrupt=D_CORRUPT)
        attr_faiths.append(fs_s)
        rand_faiths.append(fs_rand)

        print(f"  {p.text[:50]!r}")
        print(f"    D_full={d_full:.3f}  D_S={d_s:.3f}  D_rand={d_rand:.3f}")
        print(f"    Faith_S(attr)={fs_s:.3f}  Faith_S(rand)={fs_rand:.3f}")

    mean_attr = float(np.mean(attr_faiths))
    mean_rand = float(np.mean(rand_faiths))
    gap = mean_attr - mean_rand
    print(f"\nMean Faith_S  attr={mean_attr:.3f}  rand={mean_rand:.3f}  gap={gap:.3f}")

    if gap > 0.15:
        verdict = f"GO — Faith_S(S)={mean_attr:.3f} > Faith_S(rand)={mean_rand:.3f} (gap={gap:.3f}); S transfers to paraphrases"
    elif gap > 0.05:
        verdict = f"INVESTIGATE — modest gap={gap:.3f}; check whether S features fire on paraphrases"
    else:
        verdict = f"NO-GO — gap={gap:.3f}; S does not outperform random on paraphrases; central claim fails"

    print(f"\nVERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    with open("results/02_transfer.json", "w") as f:
        json.dump({"attr_faiths": attr_faiths, "rand_faiths": rand_faiths,
                   "mean_attr": mean_attr, "mean_rand": mean_rand,
                   "gap": gap, "S_size": len(S)}, f, indent=2)

if __name__ == "__main__":
    main()

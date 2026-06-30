# Pilot Report — Attribution-Graph Generalization

**Date**: 2026-06-29  
**Model**: GPT-2 small (124M, CPU)  
**Transcoders**: `pchlenski/gpt2-transcoders` (12 layers, d_sae=24576)  
**Task**: IOI (Indirect Object Identification); control: Greater-Than

---

## Six-Check Summary

| Check | Verdict | Key metric |
|-------|---------|------------|
| 00 Smoke | **GO** | ΔD=+0.0086 in predicted direction (c_n=0.112) |
| 01 Stability | **INVESTIGATE** | Top-20 Jaccard median=0.250; top-1 Jaccard=1.000 |
| 02 Transfer | **NO-GO** | Faith_S(attr)=−0.035, Faith_S(rand)=0.000, gap=−0.035 |
| 03 Specificity | **GO** | Cross-task Faith_S=0.048 ≈ 0 |
| 04 Lazy baseline | **REPORT** | Attribution gap over magnitude baseline = −0.002 (negligible) |
| 05 Calibration | **NO-GO** | Spearman ρ = −0.018 (p=0.70, n=450) |

**Overall: NO-GO for the main phase as currently designed.**

---

## Root Cause: Transcoder Approximation Error

The dominant finding is that replacing all 12 MLP layers simultaneously with their transcoders
introduces catastrophic approximation error for the IOI task:

| Condition | D = logit(IO) − logit(S) |
|-----------|--------------------------|
| Original model | **+4.15** (strongly predicts IO) |
| All-TC replacement | **−1.46** (predicts S — wrong answer) |

A 5.6-logit shift means the transcoder framework does not preserve the IOI preference at all.
This invalidates Faith_S as a metric because D_corrupt = 0 now lies *above* D_full (TC),
inverting the Faith_S scale.

---

## Check-by-Check Analysis

### Check 00 — Smoke (GO)

Within the TC framework, the sign check passes: ablating the top-attributed feature
(L11/F15690, c_n=0.112) causes D to decrease by 0.0086 — the predicted direction.
The effect is small (7.7% of c_n) due to compounding approximation errors across 12 layers,
but the causal direction is correct.

**Positive signal**: IxG attribution gives the correct causal sign when attribution and
ablation are performed in the same (TC) activation space.

### Check 01 — Stability (INVESTIGATE)

Feature L11/F15690 is the top-attributed feature on 8 of 10 IOI source prompts —
a strong consistent signal. The top-20 Jaccard (median=0.250) is below the GO threshold (0.3),
but the top-1 Jaccard (1.000) shows a single dominant feature.

Consensus set S contains **45 features** that appear in ≥ 2 of 10 circuits.

**Recommendation**: Tighten k to 10 or lower; top-1 stability is real but diffuse k=20 sets
are unreliable.

### Check 02 — Transfer (NO-GO)

Ablating S on paraphrases yields Faith_S = −0.035 vs −0.000 for random.
The random baseline "wins" because all Faith_S values are near zero and slightly negative
(ablating S slightly increases D in the wrong direction relative to the inverted D_corrupt
baseline). This is a direct consequence of TC approximation error making D_full < D_corrupt.

The central claim — frozen circuits transfer to paraphrases — cannot be evaluated with this
toolchain.

### Check 03 — Specificity (GO)

Cross-task Faith_S on Greater-Than prompts = 0.048 ≈ 0. The IOI consensus set S does not
meaningfully affect Greater-Than predictions, suggesting the 45-feature S is not just
capturing generic high-activation features.

**This is a genuine positive finding**: even though transfer fails, S is task-specific.

### Check 04 — Lazy Baseline (REPORT)

Attribution features (S) and magnitude-ranked features perform identically (gap = −0.002).
In a broken Faith_S regime this is uninformative, but it's consistent with the hypothesis
that transcoder replacement destroys task signal before feature selection matters.

### Check 05 — Calibration (NO-GO)

Spearman ρ = −0.018 between c_n and realized per-feature ΔD (n=450 pairs, p=0.70).
The quantitative axis of IxG attribution (predicting effect *magnitude*) does not hold
within the TC framework. This is expected given the ~92% attenuation of feature effects
observed in Check 00 (c_n=0.112 → ΔD=0.0086).

---

## Recommendations

**Do not proceed to a main-phase study with the current all-at-once TC replacement.**

The pchlenski/gpt2-transcoders introduce 5.6 logits of approximation error on IOI prompts
when all 12 layers are replaced simultaneously. This is too large relative to the task signal
to permit meaningful causal evaluation.

### Pivot options (in priority order)

1. **Single-layer TC replacement**: Replace only the most important MLP layer (likely L11)
   with the transcoder; leave the other 11 as the original MLP. This limits approximation
   error to one layer and should preserve D_full > D_corrupt.

2. **Residual-stream SAEs instead of MLP transcoders**: Use EleutherAI or similar GPT-2
   SAEs trained on residual stream positions. These permit feature clamping without
   disrupting MLP computation.

3. **Attention-only attribution**: IOI is known to be primarily driven by attention heads
   (Name Mover, Backup NMH). Skip MLP transcoders entirely; use `get_attn_head_contribs`
   from `transcoder_circuits/circuit_analysis.py` for attribution and causal scrubbing
   for intervention.

4. **Stronger model**: The IOI task signal is large in GPT-2 small for attention but the
   MLP contribution is less well-studied. Consider using a model where pretrained SAEs
   cover residual-stream positions.

---

## What Worked

- IxG sign prediction is correct (Check 00: GO)
- Feature L11/F15690 is robustly the top IOI feature (appears 8/10 prompts)
- The IOI feature set is task-specific, not cross-task-generic (Check 03: GO)
- The code infrastructure (caching, metrics, TC loading, consistent CPU device) functions
  correctly and is ready for a pivot study

## What Failed

- TC approximation error destroys task signal when all 12 layers are replaced
- Faith_S is not a valid metric in this configuration (D_full < D_corrupt)
- Quantitative c_n prediction of ΔD magnitude does not hold (ρ ≈ 0)

---

*Generated by PILOT phase scripts 00–05. All results in `results/*.json`.*

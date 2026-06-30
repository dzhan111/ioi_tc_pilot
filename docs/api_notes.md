# API Notes — Verified Tooling

Date verified: 2026-06-29  
Model: GPT-2 small (124M, CPU only)

---

## Decision: transcoder_circuits over circuit-tracer

| Tool | GPT-2 small? | Notes |
|---|---|---|
| `circuit-tracer` (safety-research) | **No** | Only Gemma-2, Llama-3.2, Qwen-3, GPT-OSS 20B |
| `transcoder_circuits` (Dunefsky) | **Yes** | Pretrained weights at `pchlenski/gpt2-transcoders` |
| SAELens | Not used | Fallback if transcoder API broke; not needed |

**Choice**: `transcoder_circuits` vendored at `vendor/transcoder_circuits/`.  
No pip package — installed via `sys.path`.

---

## Transcoder weights

Repo: `pchlenski/gpt2-transcoders` (HuggingFace)  
Files: `final_sparse_autoencoder_gpt2-small_blocks.{0..11}.ln2.hook_normalized_24576.pt`  
Layers: 12 (one per transformer block)  
Feature dimension: **d_sae = 24576** per layer  
Downloaded to: `transcoders/` (gitignored)

---

## SparseAutoencoder (transcoder variant)

```python
from sae_training.sparse_autoencoder import SparseAutoencoder

tc = SparseAutoencoder.load_from_pretrained("transcoders/final_...pt").eval()
```

### Forward pass signature
```python
sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid = tc(x)
```
- `x`: input tensor, shape `(batch, seq, d_model)` — **post-LN2 activations**
- `sae_out` `(batch, seq, d_model)`: transcoder's approximation of MLP output
- `feature_acts` `(batch, seq, d_sae)`: sparse feature activations (ReLU)

### Key parameters
```python
tc.W_enc    # (d_model=768, d_sae=24576)  encoder weights
tc.b_enc    # (d_sae,)                    encoder bias
tc.W_dec    # (d_sae=24576, d_model=768)  decoder weights
tc.b_dec    # (d_model,)                  pre-encoder bias (subtracted from input)
tc.b_dec_out # (d_model,)                 output bias (added to decoded output)
tc.cfg.hook_point  # 'blocks.{layer}.ln2.hook_normalized'
tc.cfg.hook_point_layer  # int layer index
tc.cfg.d_sae  # 24576
```

### Forward math (eval mode)
```
sae_in       = x - b_dec
hidden_pre   = sae_in @ W_enc + b_enc
feature_acts = ReLU(hidden_pre)
sae_out      = feature_acts @ W_dec + b_dec_out
```

---

## transformer_lens (version 1.11.0)

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", fold_ln=False,
    center_writing_weights=False, center_unembed=False)
```

### Hook point names (verified against GPT-2 small)
```python
from transformer_lens.utils import get_act_name

get_act_name('normalized', layer, 'ln2')  # post-LN2 normalized input → transcoder input
get_act_name('resid_pre', layer)           # residual stream before layer
get_act_name('resid_mid', layer)           # after attention, before MLP
get_act_name('resid_post', layer)          # after MLP
get_act_name('mlp_out', layer)             # MLP output
get_act_name('pattern', layer)             # attention pattern
get_act_name('v', layer)                   # value vectors
```

### Run with cache
```python
_, cache = model.run_with_cache(tokens)
# tokens shape: (1, seq_len)
```

### Logit difference
```python
logits = model(tokens)[0, -1]        # (d_vocab,) at final position
D = logits[io_token_id] - logits[s_token_id]

# Direction in residual stream space:
range_normal = model.W_U[:, io_id] - model.W_U[:, s_id]  # (d_model,)
# model.W_U shape: (d_model, d_vocab); logit[v] = resid @ W_U[:,v]
```

### MLP replacement (for ablation)
```python
model.blocks[layer].mlp = MyWrapper(...)  # replaces MLP; receives post-LN2 input
# Always restore originals in a try/finally block
```

---

## IxG attribution (c_n values)

From `transcoder_circuits/circuit_analysis.py`:

```python
from transcoder_circuits.circuit_analysis import get_transcoder_ixg

ixg, feature_acts = get_transcoder_ixg(
    transcoder=tc,
    cache=cache,
    range_normal=range_normal,   # (d_model,) — direction to attribute toward
    input_layer=layer,
    input_token_idx=pos,         # token position (can be negative)
    return_numpy=False,          # keep as torch.Tensor
    is_transcoder_post_ln=True,  # True for GPT-2 small transcoders
    return_feature_activs=True,
)
# ixg shape: (d_sae,)
# ixg[n] = c_n = (W_dec[n] · range_normal) * feature_acts[n]
```

**Note**: `c_n` is a first-order (IxG) approximation; it does NOT include the LN2 scaling constant.  
The `get_top_transcoder_features` function applies LN correction to the `vector` field (used for backtracking) but the `contrib` value it reports also uses the raw IxG.

---

## Mean ablation

Implemented in `src/pilot/intervene.py` via `_AblationWrapper`.

For a set of features `{(layer, feat_idx)}`:
```
original MLP output  = feature_acts @ W_dec + b_dec_out
correction           = (mean_acts[feat_indices] - feature_acts[..., feat_indices]) @ W_dec[feat_indices]
ablated output       = original + correction
```

Mean activations computed from 50 neutral English sentences, pooled over all positions.

---

## Known limitations / TODOs

- `is_transcoder_post_ln=True` is hardcoded for all layers — correct for the `pchlenski` weights but should be verified per-layer if using different checkpoints.
- LN2 scaling approximation in IxG attribution may underestimate contributions at layers with high LN variance. If calibration (check 05) fails, this is a candidate cause.
- `SparseAutoencoder.forward()` computes MSE loss on every call even in eval mode — minor CPU overhead, not a bug.
- The `geom_median` submodule in `sae_training/` is only needed for training; safely unused here.

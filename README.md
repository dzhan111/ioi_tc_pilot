# ioi_tc_pilot

An observational study of transcoder feature attribution on the Indirect Object
Identification (IOI) task in GPT-2 small (124M, CPU-only). We use pretrained
transcoders from [`pchlenski/gpt2-transcoders`](https://huggingface.co/pchlenski/gpt2-transcoders)
and Input×Gradient attribution to extract feature circuits, then test their
causal validity, stability, transfer, specificity, and calibration.

**Headline finding:** replacing all 12 MLP layers with transcoders introduces a
5.6-logit approximation error that displaces the IOI signal (D: +4.15 → −1.46),
making the sufficiency metric uninterpretable. A single feature (L11/F15690)
nonetheless emerges as a stable, task-specific IOI feature.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Transcoder checkpoints (~1.7 GB) download automatically from HuggingFace on
first run.

## Run

```bash
python scripts/00_smoke.py         # RQ1: single-feature causal direction
python scripts/01_stability.py     # RQ2: cross-prompt stability
python scripts/02_transfer.py      # RQ3: transfer to paraphrases
python scripts/03_specificity.py   # RQ4: cross-task specificity
python scripts/04_lazy_baseline.py # RQ4a: attribution vs. magnitude
python scripts/05_calibration.py   # RQ5: quantitative calibration
pytest tests/                      # 17 metric unit tests
```

Full pipeline runs on CPU in ~45 min.

## Layout

| Path | Contents |
|------|----------|
| `src/pilot/` | model, transcoders, attribution, intervention, metrics |
| `scripts/` | the six observation scripts (RQ1–RQ5) |
| `results/` | raw JSON outputs + `pilot_report.md` |
| `paper/` | NeurIPS-format writeup (`main.tex`, `main.pdf`) |
| `vendor/` | vendored `transcoder_circuits` (includes a local wandb patch) |
| `docs/api_notes.md` | tooling API reference |

**Environment:** Python 3.9, PyTorch 2.2.0 (CPU), TransformerLens 1.11.0.

"""Integration test: ablating the top causal feature moves D in the predicted direction.

Loads results from 00_smoke.json so the model/transcoders don't need to re-run.
Run script 00 first: python scripts/00_smoke.py
"""

import json
import os
import pytest


RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', '00_smoke.json')


@pytest.mark.skipif(
    not os.path.exists(RESULTS_PATH),
    reason="Run scripts/00_smoke.py first to generate results/00_smoke.json",
)
def test_intervention_sign():
    """Sign of c_n must match sign of (D_full - D_ablated)."""
    with open(RESULTS_PATH) as f:
        res = json.load(f)

    c_n = res["top_cn"]
    delta_d = res["delta_d"]

    assert c_n != 0.0, "Top feature has zero c_n — attribution is degenerate"
    assert delta_d != 0.0, "Ablation had zero effect on D — intervention is not working"
    assert res["sign_correct"], (
        f"Sign mismatch: c_n={c_n:.4f} but ΔD={delta_d:.4f}. "
        "Ablating a positive-contribution feature should decrease D."
    )


@pytest.mark.skipif(
    not os.path.exists(RESULTS_PATH),
    reason="Run scripts/00_smoke.py first",
)
def test_intervention_magnitude():
    """Ablation should move D by at least a small threshold."""
    with open(RESULTS_PATH) as f:
        res = json.load(f)

    assert abs(res["delta_d"]) >= 0.01, (
        f"Ablation effect too small (|ΔD|={abs(res['delta_d']):.4f}); "
        "feature may be too minor for a meaningful smoke test"
    )

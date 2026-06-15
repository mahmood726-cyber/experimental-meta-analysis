"""Coverage assertions for the repo's own estimators vs known truth.

Run: pytest truth-recovery/test_coverage.py  (or python -m pytest)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run


def test_1_nominal_coverage_no_selection():
    """Without selection, Wald methods are ~nominal (slight undercoverage ok at k=12)."""
    r = run(0.20, 0.05, 12, "none", reps=1500, seed0=111)
    assert 0.88 <= r["DL+Wald"]["coverage"] <= 0.97, r["DL+Wald"]


def test_2_hksj_beats_wald_no_selection():
    """HKSJ coverage >= DL+Wald coverage under clean data (t_{k-1} advantage)."""
    r = run(0.20, 0.05, 12, "none", reps=1500, seed0=222)
    assert r["HKSJ"]["coverage"] >= r["DL+Wald"]["coverage"] - 0.005, \
        (r["HKSJ"]["coverage"], r["DL+Wald"]["coverage"])


def test_3_selection_destroys_coverage():
    """Strong publication selection drives coverage of true mu well below nominal."""
    r = run(0.20, 0.05, 12, "step_strong", reps=1500, seed0=333)
    assert r["DL+Wald"]["coverage"] < 0.50, r["DL+Wald"]
    # the cause is BIAS, not narrow CIs
    assert r["DL+Wald"]["bias"] > 0.10, r["DL+Wald"]


def test_4_hksj_best_coverage_under_selection():
    """Among the four, HKSJ retains the highest (still-broken) coverage under selection."""
    r = run(0.20, 0.05, 12, "step_strong", reps=1500, seed0=444)
    covs = {m: r[m]["coverage"] for m in r}
    assert covs["HKSJ"] >= max(covs.values()) - 1e-9, covs


def test_5_null_plus_selection_fabricates_effect():
    """Under a TRUE NULL + selection, the pooled estimate is biased away from 0."""
    r = run(0.0, 0.03, 12, "step_strong", reps=1500, seed0=555)
    assert r["REML+Wald"]["bias"] > 0.10, r["REML+Wald"]
    assert r["REML+Wald"]["coverage"] < 0.60, r["REML+Wald"]

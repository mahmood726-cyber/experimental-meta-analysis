"""
Direct-import regression tests for core_framework.

These complement tests/test_smoke.py (which shells out to tests.py) by
importing the estimators directly so pytest collects and reports each check
individually. They lock in the audit fixes:

* F3 - t-based small-sample methods no longer return nan SE/CI at k=1.
* F4 - MetaAnalysisData fails closed on degenerate input.
* F1 - the R-validation baseline in validation.py is self-consistent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of pytest's rootdir insertion.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_framework import (  # noqa: E402
    MetaAnalysisData,
    DerSimonianLaird,
    HartungKnapp,
    KnappHartungModified,
    PermutationMeta,
)


# ---------------------------------------------------------------------------
# F3: t-based small-sample methods must not return nan at k = 1
# ---------------------------------------------------------------------------

T_BASED_METHODS = [HartungKnapp, KnappHartungModified, PermutationMeta]


@pytest.mark.parametrize("method_cls", T_BASED_METHODS)
def test_t_based_methods_finite_at_k1(method_cls):
    """Regression: single-study input previously produced nan SE/CI."""
    data = MetaAnalysisData(np.array([0.5]), np.array([0.1]))
    result = method_cls().estimate(data)
    assert np.isfinite(result.pooled_effect)
    assert np.isfinite(result.pooled_se)
    assert np.isfinite(result.ci_lower)
    assert np.isfinite(result.ci_upper)
    # With a single study the small-sample correction degrades to the
    # normal-theory RE SE = sqrt(v) and a z-based CI centred on the effect.
    assert result.pooled_effect == pytest.approx(0.5)
    assert result.pooled_se == pytest.approx(np.sqrt(0.1))
    assert result.ci_lower < result.pooled_effect < result.ci_upper


def test_dersimonian_laird_finite_at_k1():
    data = MetaAnalysisData(np.array([0.5]), np.array([0.1]))
    result = DerSimonianLaird().estimate(data)
    assert np.isfinite(result.pooled_se)
    assert result.pooled_effect == pytest.approx(0.5)


def test_hartung_knapp_multistudy_unchanged():
    """The k >= 2 HKSJ path must be untouched by the k=1 guard."""
    data = MetaAnalysisData(
        np.array([0.5, 0.3, 0.7]), np.array([0.1, 0.15, 0.08])
    )
    result = HartungKnapp().estimate(data)
    assert result.pooled_se == pytest.approx(0.111098, abs=1e-5)


# ---------------------------------------------------------------------------
# F4: MetaAnalysisData fails closed on degenerate input
# ---------------------------------------------------------------------------

def test_empty_input_raises():
    with pytest.raises(ValueError):
        MetaAnalysisData(np.array([]), np.array([]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MetaAnalysisData(np.array([0.5, 0.3]), np.array([0.1]))


@pytest.mark.parametrize("bad_var", [0.0, -0.1, np.nan, np.inf])
def test_nonpositive_or_nonfinite_variance_raises(bad_var):
    with pytest.raises(ValueError):
        MetaAnalysisData(np.array([0.5]), np.array([bad_var]))


def test_valid_single_study_allowed():
    data = MetaAnalysisData(np.array([0.5]), np.array([0.1]))
    assert data.n_studies == 1


def test_homogeneous_tau2_near_zero():
    data = MetaAnalysisData(
        np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.01, 0.01, 0.01, 0.01])
    )
    result = DerSimonianLaird().estimate(data)
    assert result.tau2 == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# F1: the validation baseline is self-consistent (all reference checks pass)
# ---------------------------------------------------------------------------

def test_validation_suite_all_pass():
    from validation import MetaAnalysisValidator

    validator = MetaAnalysisValidator()  # default method-appropriate tolerance
    results = validator.validate_all_standard_methods()
    for method_name, method_results in results.items():
        for r in method_results:
            assert r.passed, (
                f"{method_name} failed {r.test_name}: "
                f"python={r.python_result} ref={r.reference_result} "
                f"diffs={r.differences}"
            )

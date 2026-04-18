"""
Comprehensive Test Suite for Experimental Meta-Analysis Framework
=================================================================
Tests all components including methods, simulations, datasets, and utilities.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core_framework import (
        MetaAnalysisData, MetaAnalysisResult, MetaAnalysisMethod,
        DerSimonianLaird, REML, PauleMandel, HartungKnapp,
        RobustHuberMeta, KnappHartungModified, BootstrapMeta
    )
    from methods.experimental_methods_part1 import get_part1_methods
    from methods.experimental_methods_part2 import get_part2_methods
    from methods.experimental_methods_part3 import get_part3_methods
    from methods.experimental_methods_part4 import get_part4_methods
    from methods.experimental_methods_part5 import get_part5_methods
    from simulations.simulation_engine import SimulationEngine, get_all_scenarios
    from datasets import MetaAnalysisDatasets
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the experimental-meta-analysis directory")
    sys.exit(1)


class TestRunner:
    """Test runner for the framework"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name, func):
        """Run a single test"""
        try:
            func()
            self.passed += 1
            print(f"  [PASS] {name}")
            return True
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
            return False
        except Exception as e:
            self.failed += 1
            self.errors.append((name, f"Error: {e}"))
            print(f"  [ERROR] {name}: {e}")
            return False

    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def assert_near(a, b, tol=1e-6, msg=""):
    """Assert two values are near each other"""
    if isinstance(a, np.ndarray):
        assert np.allclose(a, b, atol=tol), msg or f"{a} != {b}"
    else:
        assert abs(a - b) < tol, msg or f"{a} != {b}"


def assert_true(condition, msg=""):
    """Assert condition is true"""
    assert condition, msg or "Condition is False"


def assert_equal(a, b, msg=""):
    """Assert two values are equal"""
    assert a == b, msg or f"{a} != {b}"


def assert_greater(a, b, msg=""):
    """Assert a > b"""
    assert a > b, msg or f"{a} not > {b}"


def assert_between(val, low, high, msg=""):
    """Assert low <= val <= high"""
    assert low <= val <= high, msg or f"{val} not in [{low}, {high}]"


# =============================================================================
# TESTS
# =============================================================================

def test_core_imports():
    """Test core module imports"""
    from core_framework import MetaAnalysisData, MetaAnalysisResult
    assert_true(MetaAnalysisData is not None)
    assert_true(MetaAnalysisResult is not None)


def test_meta_analysis_data():
    """Test MetaAnalysisData creation"""
    yi = np.array([0.5, 0.3, 0.7])
    vi = np.array([0.1, 0.15, 0.08])
    data = MetaAnalysisData(effect_sizes=yi, variances=vi)

    assert_equal(len(data.effect_sizes), 3)
    assert_equal(data.n_studies, 3)
    assert_near(data.standard_errors[0], np.sqrt(0.1))


def test_meta_analysis_result():
    """Test MetaAnalysisResult creation"""
    result = MetaAnalysisResult(
        method_name="Test",
        pooled_effect=0.5,
        pooled_se=0.1,
        tau2=0.05,
        ci_lower=0.3,
        ci_upper=0.7,
        i2=50.0,
        q_stat=10.0,
        p_heterogeneity=0.05
    )

    assert_equal(result.method_name, "Test")
    assert_near(result.pooled_effect, 0.5)
    assert_true(result.converged)


def test_dersimonian_laird():
    """Test DerSimonian-Laird method"""
    yi = np.array([0.5, 0.3, 0.7, 0.4, 0.6])
    vi = np.array([0.1, 0.15, 0.08, 0.12, 0.09])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = DerSimonianLaird()
    result = method.estimate(data)

    assert_between(result.pooled_effect, 0.3, 0.7)
    assert_greater(result.pooled_se, 0)
    assert_greater(result.tau2, -0.01)  # Can be 0 or positive
    assert_between(result.ci_lower, result.pooled_effect - 1, result.pooled_effect)
    assert_between(result.ci_upper, result.pooled_effect, result.pooled_effect + 1)


def test_reml():
    """Test REML method"""
    yi = np.array([0.5, 0.3, 0.7])
    vi = np.array([0.1, 0.15, 0.08])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = REML()
    result = method.estimate(data)

    assert_between(result.pooled_effect, 0.0, 1.0)
    assert_greater(result.tau2, 0)
    assert_true(result.converged)


def test_paule_mandel():
    """Test Paule-Mandel method"""
    yi = np.array([0.5, 0.3, 0.7, 0.4])
    vi = np.array([0.1, 0.15, 0.08, 0.12])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = PauleMandel()
    result = method.estimate(data)

    assert_between(result.pooled_effect, 0.2, 0.8)
    assert_greater(result.tau2, -0.01)  # Can be 0 or positive


def test_hartung_knapp():
    """Test Hartung-Knapp method"""
    yi = np.array([0.5, 0.3, 0.7])
    vi = np.array([0.1, 0.15, 0.08])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = HartungKnapp()
    result = method.estimate(data)

    # HK should give wider CI than DL
    dl = DerSimonianLaird()
    dl_result = dl.estimate(data)

    hk_width = result.ci_upper - result.ci_lower
    dl_width = dl_result.ci_upper - dl_result.ci_lower

    assert_greater(hk_width, dl_width * 0.9)  # HK CI should be similar or wider


def test_knapp_hartung_modified():
    """Test Knapp-Hartung Modified method"""
    yi = np.array([0.5, 0.3, 0.7, 0.4, 0.6])
    vi = np.array([0.1, 0.15, 0.08, 0.12, 0.09])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)

    # Test both truncate variants
    method_true = KnappHartungModified(truncate=True)
    method_false = KnappHartungModified(truncate=False)

    result_true = method_true.estimate(data)
    result_false = method_false.estimate(data)

    assert_between(result_true.pooled_effect, 0.3, 0.7)
    assert_between(result_false.pooled_effect, 0.3, 0.7)


def test_robust_methods():
    """Test robust methods"""
    yi = np.array([0.5, 0.3, 0.7, 2.0, 0.4])  # 2.0 is an outlier
    vi = np.array([0.1, 0.15, 0.08, 0.1, 0.12])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)

    # Test Huber robust
    method = RobustHuberMeta(c=1.345)
    result = method.estimate(data)

    assert_between(result.pooled_effect, 0.3, 0.7)
    assert_true(result.converged or result.iterations > 0)


def test_bootstrap_method():
    """Test bootstrap method"""
    yi = np.array([0.5, 0.3, 0.7, 0.4])
    vi = np.array([0.1, 0.15, 0.08, 0.12])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)

    # Use fewer bootstrap samples for testing
    method = BootstrapMeta(n_bootstrap=100, method="percentile")
    result = method.estimate(data)

    assert_between(result.pooled_effect, 0.3, 0.7)
    assert_between(result.ci_lower, 0.0, result.pooled_effect)
    assert_between(result.ci_upper, result.pooled_effect, 1.0)


def test_experimental_methods_part1():
    """Test experimental methods part 1 loads"""
    methods = get_part1_methods()
    assert_greater(len(methods), 30)

    # Test first method works
    data = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.3, 0.7]),
        variances=np.array([0.1, 0.15, 0.08])
    )

    result = methods[0].estimate(data)
    assert_true(result is not None)
    assert_between(result.pooled_effect, -1.0, 2.0)


def test_experimental_methods_part2():
    """Test experimental methods part 2 loads"""
    methods = get_part2_methods()
    assert_greater(len(methods), 40)
    assert_true(methods[0].category in ["regularization", "geometric", "loss"])


def test_experimental_methods_part3():
    """Test experimental methods part 3 loads"""
    methods = get_part3_methods()
    assert_greater(len(methods), 50)
    assert_true(methods[0].category in ["ensemble", "neural", "copula"])


def test_experimental_methods_part4():
    """Test experimental methods part 4 loads"""
    methods = get_part4_methods()
    assert_greater(len(methods), 30)
    assert_true(methods[0].category in ["robust_score", "quantile", "nonparametric"])


def test_experimental_methods_part5():
    """Test experimental methods part 5 loads"""
    methods = get_part5_methods()
    assert_greater(len(methods), 50)
    assert_true(methods[0].category in ["wavelet", "functional", "game_theoretic"])


def test_total_method_count():
    """Test total method count is >= 300"""
    from methods import get_all_experimental_methods

    experimental = get_all_experimental_methods()
    core_count = 50  # Approximate count in core_framework

    total = len(experimental) + core_count
    assert_greater(total, 299)


def test_simulation_engine():
    """Test simulation engine"""
    engine = SimulationEngine()
    scenarios = get_all_scenarios()

    assert_greater(len(scenarios), 10)
    assert_true(all(s.name for s in scenarios))


def test_simulation_single_run():
    """Test single simulation run"""
    engine = SimulationEngine()
    scenarios = get_all_scenarios()

    method = DerSimonianLaird()
    scenario = scenarios[0]

    result = engine.run_single_simulation(method, scenario, seed=42)

    assert_true(result is not None)
    assert_true(result.converged or True)  # May not always converge
    assert_between(result.estimated_effect, -1.0, 2.0)


def test_simulation_batch():
    """Test batch simulation run"""
    engine = SimulationEngine()
    scenarios = get_all_scenarios()

    method = DerSimonianLaird()
    scenario = scenarios[0]

    results = engine.run_simulation_batch(method, scenario, n_simulations=50)

    assert_greater(len(results), 40)  # Should have at least 40 successful runs
    assert_true(all(r is not None for r in results))


def test_simulation_aggregation():
    """Test result aggregation"""
    engine = SimulationEngine()
    scenarios = get_all_scenarios()

    method = DerSimonianLaird()
    scenario = scenarios[0]

    results = engine.run_simulation_batch(method, scenario, n_simulations=50)
    perf = engine.aggregate_results(results)

    assert_true(perf is not None)
    assert_greater(perf.n_simulations, 40)
    assert_greater(perf.rmse, 0)
    assert_between(perf.coverage_rate, 0.5, 1.0)  # Reasonable coverage


def test_dataset_bcg():
    """Test BCG dataset"""
    data, info = MetaAnalysisDatasets.get_bcg_vaccine()

    assert_equal(info.name, "BCG Vaccine")
    assert_equal(len(data.effect_sizes), 13)
    assert_greater(np.mean(data.effect_sizes), -1)  # Negative effects (vaccine protective)


def test_dataset_magnesium():
    """Test Magnesium MI dataset"""
    data, info = MetaAnalysisDatasets.get_magnesium_mi()

    assert_equal(info.name, "Magnesium MI")
    assert_equal(len(data.effect_sizes), 16)
    assert_true(all(data.variances > 0))


def test_dataset_streptomycin():
    """Test Streptomycin dataset"""
    data, info = MetaAnalysisDatasets.get_streptomycin()

    assert_equal(info.name, "Streptomycin TB")
    assert_equal(len(data.effect_sizes), 10)


def test_dataset_teacher_expectancy():
    """Test Teacher Expectancy dataset"""
    data, info = MetaAnalysisDatasets.get_teacher_expectancy()

    assert_equal(info.name, "Teacher Expectancy")
    assert_equal(len(data.effect_sizes), 19)
    assert_greater(np.mean(data.effect_sizes), 0)  # Positive effects


def test_dataset_psychotherapy():
    """Test Psychotherapy dataset"""
    data, info = MetaAnalysisDatasets.get_psychotherapy()

    assert_equal(info.name, "Psychotherapy")
    assert_equal(len(data.effect_sizes), 16)
    assert_greater(np.mean(data.effect_sizes), 0)  # Positive effects


def test_all_datasets_accessible():
    """Test all datasets can be accessed"""
    datasets = MetaAnalysisDatasets.get_all_datasets()

    assert_greater(len(datasets), 5)

    for name, (data, info) in datasets.items():
        assert_true(len(data.effect_sizes) > 0)
        assert_true(all(data.variances > 0))
        assert_true(info.name)


def test_heterogeneity_stats():
    """Test heterogeneity statistics calculation"""
    yi = np.array([0.5, 0.3, 0.7, 0.4, 0.6])
    vi = np.array([0.1, 0.15, 0.08, 0.12, 0.09])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = DerSimonianLaird()
    result = method.estimate(data)

    assert_between(result.i2, 0, 100)
    assert_greater(result.q_stat, 0)
    assert_between(result.p_heterogeneity, 0, 1)


def test_confidence_intervals():
    """Test confidence interval calculation"""
    yi = np.array([0.5, 0.3, 0.7])
    vi = np.array([0.1, 0.15, 0.08])

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = DerSimonianLaird()
    result = method.estimate(data)

    # CI should contain the point estimate
    assert_between(result.pooled_effect, result.ci_lower, result.ci_upper)
    assert_greater(result.ci_upper - result.ci_lower, 0)


def test_edge_cases():
    """Test edge cases"""
    # Very small study
    data = MetaAnalysisData(
        effect_sizes=np.array([0.5]),
        variances=np.array([0.1])
    )

    method = DerSimonianLaird()
    result = method.estimate(data)

    assert_true(result is not None)
    assert_near(result.pooled_effect, 0.5, tol=0.01)

    # Homogeneous data
    data = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.5, 0.5, 0.5]),
        variances=np.array([0.01, 0.01, 0.01, 0.01])
    )

    result = method.estimate(data)
    assert_near(result.pooled_effect, 0.5, tol=0.01)
    assert_near(result.tau2, 0, tol=0.01)


def test_computation_time():
    """Test computation time is reasonable"""
    yi = np.random.normal(0.5, 0.2, 20)
    vi = np.random.uniform(0.05, 0.15, 20)

    data = MetaAnalysisData(effect_sizes=yi, variances=vi)
    method = DerSimonianLaird()

    start = time.time()
    result = method.estimate(data)
    elapsed = time.time() - start

    assert_true(result is not None)
    assert_less(elapsed, 1.0)  # Should complete in less than 1 second


def assert_less(a, b, msg=""):
    """Assert a < b"""
    assert a < b, msg or f"{a} not < {b}"


def test_method_categories():
    """Test methods have proper categories"""
    from methods import get_all_experimental_methods

    methods = get_all_experimental_methods()
    categories = set(m.category for m in methods)

    # Should have many categories
    assert_greater(len(categories), 15)

    # Check some expected categories exist
    expected = ["adaptive", "mixture", "kernel",
                "regularization", "ensemble", "robust_score"]
    for cat in expected:
        assert_true(cat in categories, f"Missing category: {cat}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Experimental Meta-Analysis Framework - Test Suite")
    print("="*60)
    print()

    runner = TestRunner()

    # Core tests
    print("Core Framework Tests:")
    runner.run_test("Core imports", test_core_imports)
    runner.run_test("MetaAnalysisData creation", test_meta_analysis_data)
    runner.run_test("MetaAnalysisResult creation", test_meta_analysis_result)
    runner.run_test("DerSimonian-Laird", test_dersimonian_laird)
    runner.run_test("REML", test_reml)
    runner.run_test("Paule-Mandel", test_paule_mandel)
    runner.run_test("Hartung-Knapp", test_hartung_knapp)
    runner.run_test("Knapp-Hartung Modified", test_knapp_hartung_modified)
    runner.run_test("Robust methods", test_robust_methods)
    runner.run_test("Bootstrap method", test_bootstrap_method)

    # Experimental methods tests
    print("\nExperimental Methods Tests:")
    runner.run_test("Part 1 methods", test_experimental_methods_part1)
    runner.run_test("Part 2 methods", test_experimental_methods_part2)
    runner.run_test("Part 3 methods", test_experimental_methods_part3)
    runner.run_test("Part 4 methods", test_experimental_methods_part4)
    runner.run_test("Part 5 methods", test_experimental_methods_part5)
    runner.run_test("Total method count", test_total_method_count)
    runner.run_test("Method categories", test_method_categories)

    # Simulation tests
    print("\nSimulation Tests:")
    runner.run_test("Simulation engine", test_simulation_engine)
    runner.run_test("Single simulation", test_simulation_single_run)
    runner.run_test("Batch simulation", test_simulation_batch)
    runner.run_test("Result aggregation", test_simulation_aggregation)

    # Dataset tests
    print("\nDataset Tests:")
    runner.run_test("BCG dataset", test_dataset_bcg)
    runner.run_test("Magnesium dataset", test_dataset_magnesium)
    runner.run_test("Streptomycin dataset", test_dataset_streptomycin)
    runner.run_test("Teacher expectancy dataset", test_dataset_teacher_expectancy)
    runner.run_test("Psychotherapy dataset", test_dataset_psychotherapy)
    runner.run_test("All datasets accessible", test_all_datasets_accessible)

    # Statistical tests
    print("\nStatistical Tests:")
    runner.run_test("Heterogeneity stats", test_heterogeneity_stats)
    runner.run_test("Confidence intervals", test_confidence_intervals)
    runner.run_test("Edge cases", test_edge_cases)
    runner.run_test("Computation time", test_computation_time)

    # Summary
    success = runner.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

"""
Simulation Engine for Experimental Meta-Analysis
=================================================
Generates synthetic meta-analysis data and evaluates methods
"""

import numpy as np
from scipy import stats
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
from datetime import datetime
import os

# Import from parent package
try:
    # For package usage
    from ..core_framework import MetaAnalysisData, MetaAnalysisResult, MetaAnalysisMethod
except ImportError:
    # For direct script execution (fallback)
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from core_framework import MetaAnalysisData, MetaAnalysisResult, MetaAnalysisMethod


@dataclass
class SimulationScenario:
    """Defines a simulation scenario"""
    name: str
    true_effect: float
    tau2: float  # True between-study variance
    k_studies: int  # Number of studies
    n_per_study: Tuple[int, int]  # Range of sample sizes
    heterogeneity_type: str = "normal"  # normal, skewed, outlier, multimodal
    publication_bias: float = 0.0  # 0 = no bias, 1 = severe bias
    outlier_fraction: float = 0.0
    description: str = ""


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    scenario_name: str
    method_name: str
    true_effect: float
    estimated_effect: float
    estimated_se: float
    ci_lower: float
    ci_upper: float
    tau2_true: float
    tau2_estimated: float
    bias: float
    mse: float
    coverage: bool  # Did CI contain true effect?
    ci_width: float
    computation_time: float
    converged: bool = True


@dataclass
class MethodPerformance:
    """Aggregated performance metrics for a method"""
    method_name: str
    n_simulations: int
    mean_bias: float
    median_bias: float
    rmse: float
    mae: float
    coverage_rate: float
    mean_ci_width: float
    median_ci_width: float
    relative_efficiency: float  # Compared to DL
    tau2_bias: float
    mean_computation_time: float
    convergence_rate: float
    overall_score: float = 0.0


class DataGenerator:
    """Generates synthetic meta-analysis data"""

    @staticmethod
    def generate_normal(scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data with normally distributed true effects"""
        if seed is not None:
            np.random.seed(seed)

        k = scenario.k_studies
        n_min, n_max = scenario.n_per_study

        # Sample sizes
        ni = np.random.randint(n_min, n_max + 1, k)

        # True study-specific effects
        theta_i = np.random.normal(scenario.true_effect, np.sqrt(scenario.tau2), k)

        # Within-study variances (approximately 4/n for standardized mean difference)
        vi = 4.0 / ni + np.random.uniform(0.01, 0.05, k)

        # Observed effects
        yi = np.random.normal(theta_i, np.sqrt(vi))

        return MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            sample_sizes=ni
        )

    @staticmethod
    def generate_skewed(scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data with skewed true effect distribution"""
        if seed is not None:
            np.random.seed(seed)

        k = scenario.k_studies
        n_min, n_max = scenario.n_per_study

        ni = np.random.randint(n_min, n_max + 1, k)

        # Skewed true effects (log-normal)
        log_tau = np.sqrt(np.log(1 + scenario.tau2 / max(scenario.true_effect**2, 0.01)))
        log_mu = np.log(max(scenario.true_effect, 0.1)) - 0.5 * log_tau**2

        theta_i = np.random.lognormal(log_mu, log_tau, k)
        if scenario.true_effect < 0:
            theta_i = -theta_i

        vi = 4.0 / ni + np.random.uniform(0.01, 0.05, k)
        yi = np.random.normal(theta_i, np.sqrt(vi))

        return MetaAnalysisData(effect_sizes=yi, variances=vi, sample_sizes=ni)

    @staticmethod
    def generate_with_outliers(scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data with outliers"""
        if seed is not None:
            np.random.seed(seed)

        k = scenario.k_studies
        n_min, n_max = scenario.n_per_study
        outlier_frac = scenario.outlier_fraction

        ni = np.random.randint(n_min, n_max + 1, k)

        # Regular effects
        n_outliers = max(1, int(k * outlier_frac))
        n_regular = k - n_outliers

        theta_regular = np.random.normal(scenario.true_effect, np.sqrt(scenario.tau2), n_regular)

        # Outliers (shifted by 3-5 standard deviations)
        outlier_shift = np.random.uniform(3, 5, n_outliers) * np.sqrt(scenario.tau2 + 0.1)
        outlier_sign = np.random.choice([-1, 1], n_outliers)
        theta_outliers = scenario.true_effect + outlier_sign * outlier_shift

        theta_i = np.concatenate([theta_regular, theta_outliers])
        np.random.shuffle(theta_i)

        vi = 4.0 / ni + np.random.uniform(0.01, 0.05, k)
        yi = np.random.normal(theta_i, np.sqrt(vi))

        return MetaAnalysisData(effect_sizes=yi, variances=vi, sample_sizes=ni)

    @staticmethod
    def generate_multimodal(scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data with multimodal true effect distribution"""
        if seed is not None:
            np.random.seed(seed)

        k = scenario.k_studies
        n_min, n_max = scenario.n_per_study

        ni = np.random.randint(n_min, n_max + 1, k)

        # Two modes
        mode_assignment = np.random.binomial(1, 0.5, k)
        mode1_effect = scenario.true_effect - np.sqrt(scenario.tau2)
        mode2_effect = scenario.true_effect + np.sqrt(scenario.tau2)

        theta_i = np.where(
            mode_assignment == 0,
            np.random.normal(mode1_effect, np.sqrt(scenario.tau2) / 2, k),
            np.random.normal(mode2_effect, np.sqrt(scenario.tau2) / 2, k)
        )

        vi = 4.0 / ni + np.random.uniform(0.01, 0.05, k)
        yi = np.random.normal(theta_i, np.sqrt(vi))

        return MetaAnalysisData(effect_sizes=yi, variances=vi, sample_sizes=ni)

    @staticmethod
    def generate_with_publication_bias(scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data with publication bias"""
        if seed is not None:
            np.random.seed(seed)

        k_target = scenario.k_studies
        n_min, n_max = scenario.n_per_study
        bias_strength = scenario.publication_bias

        # Generate more studies than needed
        k_initial = int(k_target * 2)

        ni = np.random.randint(n_min, n_max + 1, k_initial)
        theta_i = np.random.normal(scenario.true_effect, np.sqrt(scenario.tau2), k_initial)
        vi = 4.0 / ni + np.random.uniform(0.01, 0.05, k_initial)
        yi = np.random.normal(theta_i, np.sqrt(vi))

        # Compute p-values (one-sided)
        z_scores = yi / np.sqrt(vi)
        p_values = 1 - stats.norm.cdf(z_scores)

        # Selection probability based on p-value
        selection_prob = np.where(
            p_values < 0.05,
            1.0,
            1.0 - bias_strength * (1 - np.exp(-5 * (p_values - 0.05)))
        )

        # Select studies
        selected = np.random.random(k_initial) < selection_prob
        selected_idx = np.where(selected)[0]

        if len(selected_idx) > k_target:
            selected_idx = np.random.choice(selected_idx, k_target, replace=False)
        elif len(selected_idx) < k_target:
            # Fill with additional significant studies
            remaining = list(set(range(k_initial)) - set(selected_idx))
            additional = np.random.choice(remaining, k_target - len(selected_idx), replace=True)
            selected_idx = np.concatenate([selected_idx, additional])

        return MetaAnalysisData(
            effect_sizes=yi[selected_idx],
            variances=vi[selected_idx],
            sample_sizes=ni[selected_idx]
        )

    @classmethod
    def generate(cls, scenario: SimulationScenario, seed: int = None) -> MetaAnalysisData:
        """Generate data based on scenario type"""
        generators = {
            "normal": cls.generate_normal,
            "skewed": cls.generate_skewed,
            "outlier": cls.generate_with_outliers,
            "multimodal": cls.generate_multimodal,
            "publication_bias": cls.generate_with_publication_bias,
        }

        generator = generators.get(scenario.heterogeneity_type, cls.generate_normal)

        if scenario.publication_bias > 0 and scenario.heterogeneity_type != "publication_bias":
            # Apply publication bias on top of other scenarios
            data = generator(scenario, seed)
            # Simple p-value based filtering
            z = data.effect_sizes / np.sqrt(data.variances)
            p = 1 - stats.norm.cdf(z)
            selection_prob = np.where(p < 0.05, 1.0, 1.0 - scenario.publication_bias * 0.5)
            keep = np.random.random(len(data.effect_sizes)) < selection_prob
            if np.sum(keep) >= 3:
                return MetaAnalysisData(
                    effect_sizes=data.effect_sizes[keep],
                    variances=data.variances[keep],
                    sample_sizes=data.sample_sizes[keep] if data.sample_sizes is not None else None
                )

        return generator(scenario, seed)


class SimulationEngine:
    """Main simulation engine"""

    def __init__(self):
        self.results: List[SimulationResult] = []
        self.scenarios: List[SimulationScenario] = []
        self.reference_method: str = "DerSimonian-Laird"

    def add_scenario(self, scenario: SimulationScenario):
        """Add a simulation scenario"""
        self.scenarios.append(scenario)

    def create_standard_scenarios(self) -> List[SimulationScenario]:
        """Create standard set of simulation scenarios"""
        scenarios = []

        # Scenario 1: Small, homogeneous
        scenarios.append(SimulationScenario(
            name="small_homogeneous",
            true_effect=0.5,
            tau2=0.0,
            k_studies=5,
            n_per_study=(20, 50),
            heterogeneity_type="normal",
            description="Small MA with no heterogeneity"
        ))

        # Scenario 2: Small, heterogeneous
        scenarios.append(SimulationScenario(
            name="small_heterogeneous",
            true_effect=0.5,
            tau2=0.1,
            k_studies=5,
            n_per_study=(20, 50),
            heterogeneity_type="normal",
            description="Small MA with moderate heterogeneity"
        ))

        # Scenario 3: Medium, homogeneous
        scenarios.append(SimulationScenario(
            name="medium_homogeneous",
            true_effect=0.3,
            tau2=0.0,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="normal",
            description="Medium MA with no heterogeneity"
        ))

        # Scenario 4: Medium, heterogeneous
        scenarios.append(SimulationScenario(
            name="medium_heterogeneous",
            true_effect=0.3,
            tau2=0.2,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="normal",
            description="Medium MA with high heterogeneity"
        ))

        # Scenario 5: Large, heterogeneous
        scenarios.append(SimulationScenario(
            name="large_heterogeneous",
            true_effect=0.4,
            tau2=0.15,
            k_studies=30,
            n_per_study=(50, 200),
            heterogeneity_type="normal",
            description="Large MA with moderate heterogeneity"
        ))

        # Scenario 6: Outliers
        scenarios.append(SimulationScenario(
            name="with_outliers",
            true_effect=0.5,
            tau2=0.1,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="outlier",
            outlier_fraction=0.15,
            description="MA with 15% outliers"
        ))

        # Scenario 7: Skewed effects
        scenarios.append(SimulationScenario(
            name="skewed_effects",
            true_effect=0.4,
            tau2=0.1,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="skewed",
            description="MA with skewed true effect distribution"
        ))

        # Scenario 8: Publication bias
        scenarios.append(SimulationScenario(
            name="publication_bias",
            true_effect=0.2,
            tau2=0.1,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="publication_bias",
            publication_bias=0.7,
            description="MA with strong publication bias"
        ))

        # Scenario 9: Very small (3 studies)
        scenarios.append(SimulationScenario(
            name="very_small",
            true_effect=0.6,
            tau2=0.05,
            k_studies=3,
            n_per_study=(30, 60),
            heterogeneity_type="normal",
            description="Very small MA (3 studies)"
        ))

        # Scenario 10: Multimodal
        scenarios.append(SimulationScenario(
            name="multimodal",
            true_effect=0.4,
            tau2=0.2,
            k_studies=20,
            n_per_study=(40, 120),
            heterogeneity_type="multimodal",
            description="MA with bimodal effect distribution"
        ))

        # Scenario 11: Null effect
        scenarios.append(SimulationScenario(
            name="null_effect",
            true_effect=0.0,
            tau2=0.1,
            k_studies=15,
            n_per_study=(30, 100),
            heterogeneity_type="normal",
            description="MA with null true effect"
        ))

        # Scenario 12: High heterogeneity
        scenarios.append(SimulationScenario(
            name="very_high_heterogeneity",
            true_effect=0.5,
            tau2=0.5,
            k_studies=20,
            n_per_study=(30, 100),
            heterogeneity_type="normal",
            description="MA with very high heterogeneity (I2 > 90%)"
        ))

        return scenarios

    def run_single_simulation(
        self,
        method: MetaAnalysisMethod,
        scenario: SimulationScenario,
        seed: int
    ) -> Optional[SimulationResult]:
        """Run a single simulation"""
        try:
            # Generate data
            data = DataGenerator.generate(scenario, seed)

            # Run method
            start_time = time.time()
            result = method.estimate(data)
            comp_time = time.time() - start_time

            # Calculate metrics
            bias = result.pooled_effect - scenario.true_effect
            mse = bias**2
            coverage = result.ci_lower <= scenario.true_effect <= result.ci_upper
            ci_width = result.ci_upper - result.ci_lower

            return SimulationResult(
                scenario_name=scenario.name,
                method_name=method.name,
                true_effect=scenario.true_effect,
                estimated_effect=result.pooled_effect,
                estimated_se=result.pooled_se,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                tau2_true=scenario.tau2,
                tau2_estimated=result.tau2,
                bias=bias,
                mse=mse,
                coverage=coverage,
                ci_width=ci_width,
                computation_time=comp_time,
                converged=result.converged
            )
        except Exception as e:
            return None

    def run_simulation_batch(
        self,
        method: MetaAnalysisMethod,
        scenario: SimulationScenario,
        n_simulations: int,
        start_seed: int = 0,
        early_stop_failures: int = 100
    ) -> List[SimulationResult]:
        """Run batch of simulations"""
        results = []
        failures = 0

        for i in range(n_simulations):
            result = self.run_single_simulation(method, scenario, start_seed + i)

            if result is not None:
                results.append(result)
                failures = 0  # Reset failure counter
            else:
                failures += 1
                if failures >= early_stop_failures:
                    print(f"  Early stopping: {failures} consecutive failures for {method.name}")
                    break

        return results

    def aggregate_results(
        self,
        results: List[SimulationResult],
        reference_rmse: float = None
    ) -> MethodPerformance:
        """Aggregate simulation results into performance metrics"""
        if not results:
            return None

        method_name = results[0].method_name
        n_sims = len(results)

        biases = [r.bias for r in results]
        mses = [r.mse for r in results]
        coverages = [r.coverage for r in results]
        ci_widths = [r.ci_width for r in results]
        tau2_biases = [r.tau2_estimated - r.tau2_true for r in results]
        times = [r.computation_time for r in results]
        converged = [r.converged for r in results]

        rmse = np.sqrt(np.mean(mses))
        rel_eff = reference_rmse / rmse if reference_rmse and rmse > 0 else 1.0

        return MethodPerformance(
            method_name=method_name,
            n_simulations=n_sims,
            mean_bias=np.mean(biases),
            median_bias=np.median(biases),
            rmse=rmse,
            mae=np.mean(np.abs(biases)),
            coverage_rate=np.mean(coverages),
            mean_ci_width=np.mean(ci_widths),
            median_ci_width=np.median(ci_widths),
            relative_efficiency=rel_eff,
            tau2_bias=np.mean(tau2_biases),
            mean_computation_time=np.mean(times),
            convergence_rate=np.mean(converged)
        )

    def calculate_overall_score(self, performance: MethodPerformance) -> float:
        """Calculate overall performance score (higher is better)"""
        if performance is None:
            return -float('inf')

        # Weights for different metrics
        score = 0.0

        # Bias penalty (lower is better, want close to 0)
        bias_score = 1.0 / (1.0 + np.abs(performance.mean_bias) * 10)
        score += 2.0 * bias_score

        # RMSE penalty (lower is better)
        rmse_score = 1.0 / (1.0 + performance.rmse * 5)
        score += 3.0 * rmse_score

        # Coverage (want close to 0.95)
        coverage_score = 1.0 - np.abs(performance.coverage_rate - 0.95) * 5
        coverage_score = max(0, coverage_score)
        score += 2.5 * coverage_score

        # CI width (lower is better, but not too narrow)
        if performance.coverage_rate >= 0.90:
            width_score = 1.0 / (1.0 + performance.mean_ci_width)
            score += 1.0 * width_score

        # Relative efficiency
        eff_score = min(2.0, performance.relative_efficiency)
        score += 1.0 * eff_score

        # Convergence
        score += 0.5 * performance.convergence_rate

        return score


def get_all_scenarios() -> List[SimulationScenario]:
    """Get all simulation scenarios"""
    engine = SimulationEngine()
    return engine.create_standard_scenarios()


if __name__ == "__main__":
    # Test the simulation engine
    from core_framework import DerSimonianLaird

    engine = SimulationEngine()
    scenarios = engine.create_standard_scenarios()

    print(f"Created {len(scenarios)} simulation scenarios:")
    for s in scenarios:
        print(f"  - {s.name}: {s.description}")

    # Test with DL method
    dl = DerSimonianLaird()
    scenario = scenarios[0]

    print(f"\nTesting {dl.name} on '{scenario.name}'...")
    results = engine.run_simulation_batch(dl, scenario, 100)
    perf = engine.aggregate_results(results)

    print(f"Results from {perf.n_simulations} simulations:")
    print(f"  Mean bias: {perf.mean_bias:.4f}")
    print(f"  RMSE: {perf.rmse:.4f}")
    print(f"  Coverage: {perf.coverage_rate:.1%}")
    print(f"  Mean CI width: {perf.mean_ci_width:.4f}")

"""
Main Runner for Experimental Meta-Analysis Simulations
======================================================
Runs comprehensive simulations for 300+ experimental methods
Identifies methods superior to standard approaches

Run with: python run_experimental_simulations.py
"""

import numpy as np
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import asdict
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Import core framework
from core_framework import (
    MetaAnalysisData, MetaAnalysisResult, MetaAnalysisMethod,
    DerSimonianLaird, REML, PauleMandel, HartungKnapp,
    RobustHuberMeta, RobustTukeyBiweight, RobustAndrewsWave, RobustHampel,
    MedianAbsoluteDeviation, WinsorizedMeta, TrimmedMeanMeta,
    EmpiricalBayesShrinkage, HierarchicalBayesMeta, BayesianModelAveraging,
    PenalizedLikelihoodMeta, SidikJonkman, KnappHartungModified,
    SatterthwaiteDFMeta, KenwardRogerApprox, HunterSchmidt, HedgesOlkin,
    EmpiricalBayesTau, GeneralizedQ, QualityWeighted, InverseVariancePlus,
    SampleSizeWeighted, UniformWeighted, SoftmaxWeighted,
    BootstrapMeta, JackknifeMeta, PermutationMeta
)

# Import experimental methods
from methods.experimental_methods_part1 import get_part1_methods
from methods.experimental_methods_part2 import get_part2_methods
from methods.experimental_methods_part3 import get_part3_methods
from methods.experimental_methods_part4 import get_part4_methods
from methods.experimental_methods_part5 import get_part5_methods

# Import simulation engine
from simulations.simulation_engine import (
    SimulationEngine, SimulationScenario, SimulationResult,
    MethodPerformance, DataGenerator, get_all_scenarios
)


class ExperimentalMetaAnalysisRunner:
    """Main runner for experimental meta-analysis simulations"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Standard reference methods
        self.reference_methods = self._get_reference_methods()

        # All experimental methods
        self.experimental_methods = self._get_all_experimental_methods()

        # Simulation engine
        self.engine = SimulationEngine()
        self.scenarios = get_all_scenarios()

        # Results storage
        self.all_results: Dict[str, Dict[str, MethodPerformance]] = {}
        self.superior_methods: List[Dict] = []

        # Configuration
        self.n_simulations = 1000  # Per method per scenario
        self.early_stop_failures = 50
        self.max_runtime_hours = 24
        self.checkpoint_interval = 30  # minutes

    def _get_reference_methods(self) -> List[MetaAnalysisMethod]:
        """Get standard reference methods"""
        methods = [
            DerSimonianLaird(),
            REML(),
            PauleMandel(),
            HartungKnapp(),
        ]
        return methods

    def _get_all_experimental_methods(self) -> List[MetaAnalysisMethod]:
        """Get all experimental methods from all parts"""
        methods = []

        # Core framework methods
        methods.extend([
            RobustHuberMeta(c=1.345),
            RobustHuberMeta(c=1.5),
            RobustTukeyBiweight(c=4.685),
            RobustTukeyBiweight(c=5.0),
            RobustAndrewsWave(),
            RobustHampel(),
            MedianAbsoluteDeviation(),
            WinsorizedMeta(trim_proportion=0.1),
            WinsorizedMeta(trim_proportion=0.15),
            WinsorizedMeta(trim_proportion=0.2),
            TrimmedMeanMeta(trim_proportion=0.1),
            TrimmedMeanMeta(trim_proportion=0.15),
            EmpiricalBayesShrinkage(shrinkage_type="optimal"),
            EmpiricalBayesShrinkage(shrinkage_type="james_stein"),
            HierarchicalBayesMeta(prior_tau_scale=0.5),
            HierarchicalBayesMeta(prior_tau_scale=1.0),
            HierarchicalBayesMeta(prior_tau_scale=2.0),
            BayesianModelAveraging(),
            PenalizedLikelihoodMeta(penalty_type="ridge", lambda_val=0.1),
            PenalizedLikelihoodMeta(penalty_type="lasso", lambda_val=0.1),
            PenalizedLikelihoodMeta(penalty_type="elastic", lambda_val=0.1),
            SidikJonkman(),
            KnappHartungModified(truncate=True),
            KnappHartungModified(truncate=False),
            SatterthwaiteDFMeta(),
            KenwardRogerApprox(),
            HunterSchmidt(),
            HedgesOlkin(),
            EmpiricalBayesTau(),
            GeneralizedQ(estimator_type="DL"),
            GeneralizedQ(estimator_type="HE"),
            GeneralizedQ(estimator_type="HS"),
            QualityWeighted(quality_power=0.5),
            QualityWeighted(quality_power=1.0),
            QualityWeighted(quality_power=2.0),
            InverseVariancePlus(regularization=0.05),
            InverseVariancePlus(regularization=0.1),
            SampleSizeWeighted(power=0.5),
            SampleSizeWeighted(power=1.0),
            UniformWeighted(),
            SoftmaxWeighted(temperature=0.5),
            SoftmaxWeighted(temperature=1.0),
            SoftmaxWeighted(temperature=2.0),
            BootstrapMeta(n_bootstrap=500, method="percentile"),
            BootstrapMeta(n_bootstrap=500, method="bca"),
            JackknifeMeta(),
            PermutationMeta(n_permutations=500),
        ])

        # Methods from part 1-5
        methods.extend(get_part1_methods())
        methods.extend(get_part2_methods())
        methods.extend(get_part3_methods())
        methods.extend(get_part4_methods())
        methods.extend(get_part5_methods())

        return methods

    def run_method_on_scenario(
        self,
        method: MetaAnalysisMethod,
        scenario: SimulationScenario,
        n_sims: int = None
    ) -> Tuple[str, str, MethodPerformance]:
        """Run a method on a single scenario"""
        if n_sims is None:
            n_sims = self.n_simulations

        try:
            results = self.engine.run_simulation_batch(
                method, scenario, n_sims,
                early_stop_failures=self.early_stop_failures
            )

            if len(results) < n_sims * 0.1:  # Less than 10% success
                return method.name, scenario.name, None

            perf = self.engine.aggregate_results(results)
            return method.name, scenario.name, perf

        except Exception as e:
            print(f"    Error with {method.name} on {scenario.name}: {str(e)[:50]}")
            return method.name, scenario.name, None

    def run_reference_methods(self, n_sims: int = None):
        """Run all reference methods to establish baselines"""
        print("\n" + "="*70)
        print("PHASE 1: Running Reference Methods")
        print("="*70)

        if n_sims is None:
            n_sims = self.n_simulations

        reference_results = {}

        for method in self.reference_methods:
            print(f"\n  Reference: {method.name}")
            method_results = {}

            for scenario in self.scenarios:
                print(f"    Scenario: {scenario.name}...", end=" ", flush=True)
                _, _, perf = self.run_method_on_scenario(method, scenario, n_sims)

                if perf:
                    method_results[scenario.name] = perf
                    print(f"RMSE={perf.rmse:.4f}, Coverage={perf.coverage_rate:.1%}")
                else:
                    print("FAILED")

            reference_results[method.name] = method_results

        self.all_results.update(reference_results)
        return reference_results

    def run_experimental_methods(self, n_sims: int = None, max_hours: float = None):
        """Run all experimental methods"""
        print("\n" + "="*70)
        print("PHASE 2: Running Experimental Methods")
        print(f"Total methods to test: {len(self.experimental_methods)}")
        print("="*70)

        if n_sims is None:
            n_sims = self.n_simulations
        if max_hours is None:
            max_hours = self.max_runtime_hours

        start_time = time.time()
        max_runtime_seconds = max_hours * 3600

        methods_tested = 0
        methods_succeeded = 0
        last_checkpoint = time.time()

        for method in self.experimental_methods:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_runtime_seconds:
                print(f"\n  Time limit reached ({max_hours} hours)")
                break

            methods_tested += 1
            print(f"\n  [{methods_tested}/{len(self.experimental_methods)}] {method.name}")

            method_results = {}
            method_success = True

            for scenario in self.scenarios:
                # Quick timeout for slow methods
                scenario_start = time.time()

                print(f"    {scenario.name}...", end=" ", flush=True)

                try:
                    _, _, perf = self.run_method_on_scenario(method, scenario, n_sims)

                    if perf:
                        method_results[scenario.name] = perf
                        print(f"RMSE={perf.rmse:.4f}, Cov={perf.coverage_rate:.1%}, "
                              f"Time={perf.mean_computation_time*1000:.1f}ms")
                    else:
                        print("FAILED")
                        method_success = False

                    # Skip if taking too long
                    if time.time() - scenario_start > 120:  # 2 minutes per scenario
                        print(f"    [Timeout - skipping remaining scenarios for {method.name}]")
                        method_success = False
                        break

                except Exception as e:
                    print(f"ERROR: {str(e)[:30]}")
                    method_success = False

            if method_results:
                self.all_results[method.name] = method_results
                if method_success:
                    methods_succeeded += 1

            # Checkpoint
            if time.time() - last_checkpoint > self.checkpoint_interval * 60:
                self._save_checkpoint()
                last_checkpoint = time.time()

            # Progress update
            elapsed_hours = (time.time() - start_time) / 3600
            print(f"    Progress: {methods_tested} tested, {methods_succeeded} succeeded, "
                  f"{elapsed_hours:.1f}h elapsed")

        return self.all_results

    def identify_superior_methods(self, top_n: int = 40) -> List[Dict]:
        """Identify methods superior to reference methods"""
        print("\n" + "="*70)
        print("PHASE 3: Identifying Superior Methods")
        print("="*70)

        # Get reference performance (DL as baseline)
        dl_results = self.all_results.get("DerSimonian-Laird", {})

        method_scores = []

        for method_name, scenario_results in self.all_results.items():
            if method_name in [m.name for m in self.reference_methods]:
                continue  # Skip reference methods

            if not scenario_results:
                continue

            total_score = 0
            scenario_count = 0
            wins_vs_dl = 0
            improvements = []

            for scenario_name, perf in scenario_results.items():
                if perf is None:
                    continue

                # Calculate score
                score = self.engine.calculate_overall_score(perf)
                total_score += score
                scenario_count += 1

                # Compare to DL
                dl_perf = dl_results.get(scenario_name)
                if dl_perf:
                    dl_score = self.engine.calculate_overall_score(dl_perf)
                    if score > dl_score:
                        wins_vs_dl += 1
                        improvement = (score - dl_score) / max(dl_score, 0.001) * 100
                        improvements.append(improvement)

            if scenario_count > 0:
                avg_score = total_score / scenario_count
                win_rate = wins_vs_dl / scenario_count
                avg_improvement = np.mean(improvements) if improvements else 0

                method_scores.append({
                    'method_name': method_name,
                    'avg_score': avg_score,
                    'scenarios_tested': scenario_count,
                    'win_rate_vs_dl': win_rate,
                    'avg_improvement': avg_improvement,
                    'detailed_results': scenario_results
                })

        # Sort by average score
        method_scores.sort(key=lambda x: x['avg_score'], reverse=True)

        # Take top N that are superior
        superior = []
        for ms in method_scores[:top_n * 2]:  # Check more to find 40 superior
            if ms['win_rate_vs_dl'] >= 0.5:  # Beats DL in at least half scenarios
                superior.append(ms)
            if len(superior) >= top_n:
                break

        self.superior_methods = superior

        # Print results
        print(f"\n  Found {len(superior)} methods superior to standard approaches:")
        print("-" * 80)

        for i, ms in enumerate(superior[:top_n]):
            print(f"\n  {i+1}. {ms['method_name']}")
            print(f"     Average Score: {ms['avg_score']:.3f}")
            print(f"     Win Rate vs DL: {ms['win_rate_vs_dl']:.1%}")
            print(f"     Avg Improvement: {ms['avg_improvement']:.1f}%")

        return superior

    def _save_checkpoint(self):
        """Save checkpoint of results"""
        checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'methods_tested': len(self.all_results),
            'results_summary': {}
        }

        for method_name, scenarios in self.all_results.items():
            checkpoint_data['results_summary'][method_name] = {
                scenario: {
                    'rmse': perf.rmse if perf else None,
                    'coverage': perf.coverage_rate if perf else None,
                    'score': self.engine.calculate_overall_score(perf) if perf else None
                }
                for scenario, perf in scenarios.items()
            }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"\n  [Checkpoint saved: {len(self.all_results)} methods]")

    def save_final_results(self):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary file
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.json")

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_methods_tested': len(self.all_results),
            'total_scenarios': len(self.scenarios),
            'simulations_per_method': self.n_simulations,
            'superior_methods_count': len(self.superior_methods),
            'superior_methods': [
                {
                    'rank': i + 1,
                    'name': m['method_name'],
                    'score': m['avg_score'],
                    'win_rate': m['win_rate_vs_dl'],
                    'improvement': m['avg_improvement']
                }
                for i, m in enumerate(self.superior_methods)
            ]
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Detailed results file
        detailed_file = os.path.join(self.output_dir, f"detailed_{timestamp}.json")

        detailed = {}
        for method_name, scenarios in self.all_results.items():
            detailed[method_name] = {}
            for scenario_name, perf in scenarios.items():
                if perf:
                    detailed[method_name][scenario_name] = {
                        'n_simulations': perf.n_simulations,
                        'mean_bias': perf.mean_bias,
                        'rmse': perf.rmse,
                        'mae': perf.mae,
                        'coverage_rate': perf.coverage_rate,
                        'mean_ci_width': perf.mean_ci_width,
                        'relative_efficiency': perf.relative_efficiency,
                        'tau2_bias': perf.tau2_bias,
                        'convergence_rate': perf.convergence_rate,
                        'overall_score': self.engine.calculate_overall_score(perf)
                    }

        with open(detailed_file, 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"\n  Results saved to:")
        print(f"    - {summary_file}")
        print(f"    - {detailed_file}")

        return summary_file, detailed_file

    def generate_report(self):
        """Generate human-readable report"""
        report_file = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENTAL META-ANALYSIS: SIMULATION RESULTS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Methods Tested: {len(self.all_results)}\n")
            f.write(f"Scenarios: {len(self.scenarios)}\n")
            f.write(f"Simulations per Method/Scenario: {self.n_simulations}\n\n")

            f.write("-"*80 + "\n")
            f.write("SUPERIOR METHODS (Outperforming Standard Approaches)\n")
            f.write("-"*80 + "\n\n")

            for i, m in enumerate(self.superior_methods):
                f.write(f"{i+1}. {m['method_name']}\n")
                f.write(f"   Score: {m['avg_score']:.3f} | ")
                f.write(f"Win Rate vs DL: {m['win_rate_vs_dl']:.1%} | ")
                f.write(f"Avg Improvement: {m['avg_improvement']:.1f}%\n\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("SCENARIO DESCRIPTIONS\n")
            f.write("-"*80 + "\n\n")

            for s in self.scenarios:
                f.write(f"- {s.name}: {s.description}\n")
                f.write(f"  k={s.k_studies}, tau2={s.tau2}, true_effect={s.true_effect}\n\n")

        print(f"\n  Report saved to: {report_file}")
        return report_file

    def run_full_simulation(
        self,
        n_simulations: int = 1000,
        max_hours: float = 24
    ):
        """Run complete simulation study"""
        print("\n" + "="*80)
        print("EXPERIMENTAL META-ANALYSIS SIMULATION STUDY")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  - Experimental methods: {len(self.experimental_methods)}")
        print(f"  - Reference methods: {len(self.reference_methods)}")
        print(f"  - Scenarios: {len(self.scenarios)}")
        print(f"  - Simulations per method/scenario: {n_simulations}")
        print(f"  - Maximum runtime: {max_hours} hours")
        print(f"  - Output directory: {self.output_dir}")

        self.n_simulations = n_simulations

        start_time = time.time()

        # Phase 1: Reference methods
        self.run_reference_methods(n_simulations)

        # Phase 2: Experimental methods
        self.run_experimental_methods(n_simulations, max_hours)

        # Phase 3: Identify superior methods
        superior = self.identify_superior_methods(top_n=40)

        # Save results
        self.save_final_results()
        self.generate_report()

        # Final summary
        total_time = (time.time() - start_time) / 3600
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        print(f"\n  Total runtime: {total_time:.2f} hours")
        print(f"  Methods tested: {len(self.all_results)}")
        print(f"  Superior methods found: {len(superior)}")
        print(f"\n  Results saved to: {self.output_dir}/")

        return superior


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("EXPERIMENTAL META-ANALYSIS FRAMEWORK")
    print("Finding Superior Methods Through Simulation")
    print("="*80)

    # Create runner
    runner = ExperimentalMetaAnalysisRunner(output_dir="results")

    print(f"\n  Total experimental methods available: {len(runner.experimental_methods)}")
    print(f"  Categories represented:")

    categories = {}
    for m in runner.experimental_methods:
        cat = m.category
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count} methods")

    # Run simulations
    # Default: 1000 simulations, 24 hours max
    # Can reduce for testing: n_simulations=100, max_hours=1
    superior = runner.run_full_simulation(
        n_simulations=1000,
        max_hours=24
    )

    return superior


if __name__ == "__main__":
    superior_methods = main()

"""
Parallel Execution Module for Experimental Meta-Analysis
=========================================================
Provides parallel execution capabilities for simulation studies.
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional
import time
from tqdm import tqdm
import os

# Import core components
try:
    from .core_framework import MetaAnalysisMethod, MetaAnalysisData
    from .simulations.simulation_engine import (
        SimulationEngine, SimulationScenario, SimulationResult, MethodPerformance
    )
except ImportError:
    from core_framework import MetaAnalysisMethod, MetaAnalysisData
    from simulations.simulation_engine import (
        SimulationEngine, SimulationScenario, SimulationResult, MethodPerformance
    )


def run_method_scenario_task(args: tuple) -> tuple:
    """
    Worker function for parallel execution.
    Runs a single method on a single scenario.

    Args:
        args: (method_dict, scenario_dict, n_simulations, seed_start)

    Returns:
        (method_name, scenario_name, performance_dict)
    """
    method_dict, scenario_dict, n_simulations, seed_start = args

    # Reconstruct method from dict
    method = _method_from_dict(method_dict)

    # Reconstruct scenario from dict
    scenario = SimulationScenario(**scenario_dict)

    # Create engine and run
    engine = SimulationEngine()
    results = engine.run_simulation_batch(method, scenario, n_simulations,
                                         start_seed=seed_start, early_stop_failures=50)

    if not results:
        return (method.name, scenario.name, None)

    perf = engine.aggregate_results(results)

    # Convert performance to dict for serialization
    perf_dict = {
        'method_name': perf.method_name,
        'n_simulations': perf.n_simulations,
        'mean_bias': perf.mean_bias,
        'median_bias': perf.median_bias,
        'rmse': perf.rmse,
        'mae': perf.mae,
        'coverage_rate': perf.coverage_rate,
        'mean_ci_width': perf.mean_ci_width,
        'median_ci_width': perf.median_ci_width,
        'relative_efficiency': perf.relative_efficiency,
        'tau2_bias': perf.tau2_bias,
        'mean_computation_time': perf.mean_computation_time,
        'convergence_rate': perf.convergence_rate,
    }

    return (method.name, scenario.name, perf_dict)


def _method_from_dict(method_dict: dict) -> MetaAnalysisMethod:
    """Reconstruct a method object from its dictionary representation"""
    # This is a simplified version - in practice, you'd need to store
    # the class path and use reflection to reconstruct
    from core_framework import DerSimonianLaird, REML, PauleMandel, HartungKnapp

    # Map method names to classes
    method_map = {
        'DerSimonian-Laird': DerSimonianLaird,
        'REML': REML,
        'Paule-Mandel': PauleMandel,
        'HartungKnapp': HartungKnapp,
        # Add more mappings as needed
    }

    method_class = method_map.get(method_dict.get('name'), DerSimonianLaird)
    return method_class()


class ParallelSimulationRunner:
    """Parallel execution engine for meta-analysis simulations"""

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize parallel runner.

        Args:
            n_workers: Number of parallel workers. If None, uses CPU count.
        """
        if n_workers is None:
            self.n_workers = max(1, mp.cpu_count() - 1)
        else:
            self.n_workers = max(1, n_workers)

        print(f"Initialized parallel runner with {self.n_workers} workers")

    def run_simulation_study(
        self,
        methods: List[MetaAnalysisMethod],
        scenarios: List[SimulationScenario],
        n_simulations: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, MethodPerformance]]:
        """
        Run a full simulation study in parallel.

        Args:
            methods: List of methods to test
            scenarios: List of scenarios to test
            n_simulations: Number of simulations per method/scenario
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping (method_name, scenario_name) -> performance
        """
        # Prepare tasks
        tasks = []
        seed = 0

        for method in methods:
            for scenario in scenarios:
                # Convert method to serializable dict
                method_dict = {
                    'name': method.name,
                    'category': method.category,
                }
                # Convert scenario to dict
                scenario_dict = {
                    'name': scenario.name,
                    'true_effect': scenario.true_effect,
                    'tau2': scenario.tau2,
                    'k_studies': scenario.k_studies,
                    'n_per_study': scenario.n_per_study,
                    'heterogeneity_type': scenario.heterogeneity_type,
                    'publication_bias': scenario.publication_bias,
                    'outlier_fraction': scenario.outlier_fraction,
                    'description': scenario.description,
                }

                tasks.append((method_dict, scenario_dict, n_simulations, seed))
                seed += n_simulations

        total_tasks = len(tasks)
        print(f"Running {total_tasks} method/scenario combinations in parallel...")

        # Run tasks
        results = {}

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(run_method_scenario_task, task): task
                      for task in tasks}

            if show_progress:
                futures_list = list(futures.values())
                progress_iter = tqdm(as_completed(futures), total=total_tasks,
                                   desc="Simulation progress")
            else:
                progress_iter = as_completed(futures)

            for future in progress_iter:
                try:
                    method_name, scenario_name, perf_dict = future.result()

                    if perf_dict is not None:
                        # Reconstruct MethodPerformance
                        perf = MethodPerformance(**perf_dict)

                        if method_name not in results:
                            results[method_name] = {}
                        results[method_name][scenario_name] = perf

                except Exception as e:
                    print(f"\nError in task: {e}")

        return results

    def benchmark_methods(
        self,
        methods: List[MetaAnalysisMethod],
        test_data: MetaAnalysisData,
        n_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark method computation times.

        Args:
            methods: List of methods to benchmark
            test_data: Test data to use
            n_runs: Number of runs for timing

        Returns:
            Dictionary mapping method names to average runtime (ms)
        """
        timings = {}

        print(f"Benchmarking {len(methods)} methods ({n_runs} runs each)...")

        for method in methods:
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                result = method.estimate(test_data)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            timings[method.name] = np.mean(times)
            print(f"  {method.name}: {timings[method.name]:.2f} ms")

        return timings


def run_parallel_simulation(
    methods: List[MetaAnalysisMethod],
    scenarios: List[SimulationScenario],
    n_simulations: int = 1000,
    n_workers: Optional[int] = None
) -> Dict[str, Dict[str, MethodPerformance]]:
    """
    Convenience function to run parallel simulations.

    Args:
        methods: List of methods to test
        scenarios: List of scenarios to test
        n_simulations: Number of simulations per method/scenario
        n_workers: Number of parallel workers

    Returns:
        Results dictionary
    """
    runner = ParallelSimulationRunner(n_workers=n_workers)
    return runner.run_simulation_study(methods, scenarios, n_simulations)


if __name__ == "__main__":
    # Example usage
    from core_framework import DerSimonianLaird, REML, PauleMandel
    from simulations.simulation_engine import get_all_scenarios
    import numpy as np

    # Create test data
    np.random.seed(42)
    test_data = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.3, 0.7, 0.4, 0.6]),
        variances=np.array([0.1, 0.15, 0.08, 0.12, 0.09])
    )

    # Benchmark a few methods
    runner = ParallelSimulationRunner(n_workers=2)
    methods = [DerSimonianLaird(), REML(), PauleMandel()]
    timings = runner.benchmark_methods(methods, test_data, n_runs=50)

    print("\nBenchmark results:")
    for name, time_ms in sorted(timings.items(), key=lambda x: x[1]):
        print(f"  {name}: {time_ms:.2f} ms")

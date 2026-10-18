"""
Parallel Execution Module for Experimental Meta-Analysis
=========================================================
Provides parallel execution capabilities for simulation studies.
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import List, Dict, Optional
import time
from tqdm import tqdm

# Import core components
try:
    from .core_framework import MetaAnalysisMethod, MetaAnalysisData
    from .simulations.simulation_engine import (
        SimulationEngine, SimulationScenario, MethodPerformance
    )
except ImportError:
    from core_framework import MetaAnalysisMethod, MetaAnalysisData
    from simulations.simulation_engine import (
        SimulationEngine, SimulationScenario, MethodPerformance
    )


def run_method_scenario_task(args: tuple) -> tuple:
    """
    Worker function for parallel execution.
    Runs a single method on a single scenario.

    Args:
        args: (method_instance, scenario_dict, n_simulations, seed_start)

    Returns:
        (method_name, scenario_name, performance_dict)
    """
    method, scenario_dict, n_simulations, seed_start = args
    if not isinstance(method, MetaAnalysisMethod):
        raise TypeError("Worker requires a MetaAnalysisMethod instance")

    # Reconstruct scenario from dict
    scenario = SimulationScenario(**scenario_dict)

    # Create engine and run
    engine = SimulationEngine()
    results = engine.run_simulation_batch(method, scenario, n_simulations,
                                         start_seed=seed_start, early_stop_failures=50)

    if len(results) != n_simulations:
        raise RuntimeError(
            f"{method.name} / {scenario.name}: completed {len(results)} of "
            f"{n_simulations} simulations; refusing partial performance metrics"
        )

    perf = engine.aggregate_results(results)

    return (method.name, scenario.name, asdict(perf))


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
        if isinstance(n_simulations, bool) or not isinstance(n_simulations, int) or n_simulations < 1:
            raise ValueError("n_simulations must be a positive integer")
        if not methods or not scenarios:
            raise ValueError("methods and scenarios must be non-empty")
        if len({method.name for method in methods}) != len(methods):
            raise ValueError("Method names must be unique")
        if len({scenario.name for scenario in scenarios}) != len(scenarios):
            raise ValueError("Scenario names must be unique")

        # ProcessPoolExecutor pickles the actual estimator, preserving its class
        # and constructor settings instead of guessing from a display name.
        tasks = []
        seed = 0
        for method in methods:
            for scenario in scenarios:
                tasks.append((method, asdict(scenario), n_simulations, seed))
                seed += n_simulations

        total_tasks = len(tasks)
        print(f"Running {total_tasks} method/scenario combinations in parallel...")

        # Run tasks
        results = {method.name: {} for method in methods}

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(run_method_scenario_task, task): task
                      for task in tasks}

            if show_progress:
                progress_iter = tqdm(as_completed(futures), total=total_tasks,
                                   desc="Simulation progress")
            else:
                progress_iter = as_completed(futures)

            for future in progress_iter:
                try:
                    method_name, scenario_name, perf_dict = future.result()

                    results[method_name][scenario_name] = MethodPerformance(**perf_dict)
                except Exception as exc:
                    method, scenario_dict, _, _ = futures[future]
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Parallel task failed for {method.name} / {scenario_dict['name']}"
                    ) from exc

        # Completion order depends on scheduling; public result order does not.
        return {method.name: {scenario.name: results[method.name][scenario.name]
                              for scenario in scenarios} for method in methods}


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

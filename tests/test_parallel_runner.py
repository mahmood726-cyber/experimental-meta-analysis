"""Parallel execution must preserve the estimator and fail on partial work.

Scenarios below are explicit, deterministic software-test simulations.
"""
from dataclasses import asdict, fields
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_framework import DerSimonianLaird, RobustHuberMeta
from parallel_runner import ParallelSimulationRunner, run_method_scenario_task
from simulations.simulation_engine import SimulationEngine, SimulationScenario


def scenario():
    return SimulationScenario('test_scenario', 0.2, 0.05, 6, (30, 60))


def test_worker_preserves_nondefault_method_configuration():
    method = RobustHuberMeta(c=2.0)
    name, scenario_name, metrics = run_method_scenario_task((method, asdict(scenario()), 3, 11))
    engine = SimulationEngine()
    reference = engine.aggregate_results(engine.run_simulation_batch(method, scenario(), 3, start_seed=11))
    assert name == method.name
    assert scenario_name == scenario().name
    assert metrics['method_name'] == method.name
    for key in ['n_simulations', 'mean_bias', 'rmse', 'coverage_rate', 'tau2_bias', 'mean_ci_width']:
        assert metrics[key] == pytest.approx(getattr(reference, key), abs=1e-6)


def test_real_process_pool_preserves_methods_and_matches_worker():
    methods = [DerSimonianLaird(), RobustHuberMeta(c=2.0)]
    results = ParallelSimulationRunner(n_workers=2).run_simulation_study(methods, [scenario()], 3, show_progress=False)
    assert list(results) == [m.name for m in methods]
    for i, method in enumerate(methods):
        name, _, reference = run_method_scenario_task((method, asdict(scenario()), 3, i * 3))
        actual = results[name][scenario().name]
        assert actual.method_name == method.name
        for metric in fields(actual):
            if metric.name in ['mean_computation_time', 'method_name']:
                continue
            assert getattr(actual, metric.name) == pytest.approx(reference[metric.name], abs=1e-6)


def test_worker_rejects_partial_simulations(monkeypatch):
    monkeypatch.setattr(SimulationEngine, 'run_simulation_batch', lambda *args, **kwargs: [object()])
    with pytest.raises(RuntimeError, match='1 of 3'):
        run_method_scenario_task((DerSimonianLaird(), asdict(scenario()), 3, 0))


def test_real_process_failure_cannot_return_partial_results():
    invalid = scenario()
    invalid.n_per_study = (60, 30)  # Intentional invalid fixture: worker data generation fails.
    with pytest.raises(RuntimeError, match='DerSimonian-Laird.*test_scenario'):
        ParallelSimulationRunner(n_workers=1).run_simulation_study([DerSimonianLaird()], [invalid], 1, show_progress=False)


@pytest.mark.parametrize('n', [0, -1, True, 1.5])
def test_invalid_replication_count_is_rejected(n):
    with pytest.raises(ValueError, match='positive integer'):
        ParallelSimulationRunner(n_workers=1).run_simulation_study([DerSimonianLaird()], [scenario()], n, show_progress=False)


@pytest.mark.parametrize('methods, scenarios', [([], [scenario()]), ([DerSimonianLaird()], [])])
def test_empty_study_is_rejected(methods, scenarios):
    with pytest.raises(ValueError, match='non-empty'):
        ParallelSimulationRunner(n_workers=1).run_simulation_study(methods, scenarios, 1, show_progress=False)


def test_duplicate_method_names_cannot_overwrite_results():
    with pytest.raises(ValueError, match='unique'):
        ParallelSimulationRunner(n_workers=1).run_simulation_study([DerSimonianLaird(), DerSimonianLaird()], [scenario()], 1, show_progress=False)


def test_duplicate_scenario_names_cannot_overwrite_results():
    with pytest.raises(ValueError, match='unique'):
        ParallelSimulationRunner(n_workers=1).run_simulation_study([DerSimonianLaird()], [scenario(), scenario()], 1, show_progress=False)

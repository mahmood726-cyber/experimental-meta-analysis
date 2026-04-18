"""
Simulation Engine Package
=========================
Provides simulation infrastructure for meta-analysis method testing.

Components:
    - SimulationEngine: Main simulation runner
    - DataGenerator: Generates synthetic meta-analysis data
    - SimulationScenario: Defines simulation scenarios
    - get_all_scenarios(): Returns standard scenario set
"""

from .simulation_engine import (
    SimulationEngine,
    SimulationScenario,
    SimulationResult,
    MethodPerformance,
    DataGenerator,
    get_all_scenarios,
)

__all__ = [
    "SimulationEngine",
    "SimulationScenario",
    "SimulationResult",
    "MethodPerformance",
    "DataGenerator",
    "get_all_scenarios",
]

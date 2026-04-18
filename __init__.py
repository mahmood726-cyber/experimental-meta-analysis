"""
Experimental Meta-Analysis Framework
====================================
A comprehensive framework for implementing and testing 300+ meta-analysis methods.

Usage:
    >>> from experimental_meta_analysis import DerSimonianLaird, MetaAnalysisData
    >>> import numpy as np
    >>> data = MetaAnalysisData(effect_sizes=np.array([0.5, 0.3, 0.7]),
    ...                         variances=np.array([0.1, 0.15, 0.08]))
    >>> method = DerSimonianLaird()
    >>> result = method.estimate(data)
    >>> print(f"Pooled effect: {result.pooled_effect:.4f}")

Main components:
    - core_framework: Base classes and standard methods
    - methods: Experimental method implementations (5 parts)
    - simulations: Simulation engine and scenarios
    - visualization: Plotting and visualization tools
    - parallel_runner: Parallel execution for simulations
"""

__version__ = "1.1.0"
__author__ = "Experimental Meta-Analysis Research"

# Core imports for convenience
from .core_framework import (
    MetaAnalysisData,
    MetaAnalysisResult,
    MetaAnalysisMethod,
    DerSimonianLaird,
    REML,
    PauleMandel,
    HartungKnapp,
    RobustHuberMeta,
    RobustTukeyBiweight,
    RobustAndrewsWave,
    RobustHampel,
    MedianAbsoluteDeviation,
    WinsorizedMeta,
    TrimmedMeanMeta,
    EmpiricalBayesShrinkage,
    HierarchicalBayesMeta,
    BayesianModelAveraging,
    PenalizedLikelihoodMeta,
    SidikJonkman,
    KnappHartungModified,
    SatterthwaiteDFMeta,
    KenwardRogerApprox,
    HunterSchmidt,
    HedgesOlkin,
    EmpiricalBayesTau,
    GeneralizedQ,
    QualityWeighted,
    InverseVariancePlus,
    SampleSizeWeighted,
    UniformWeighted,
    SoftmaxWeighted,
    BootstrapMeta,
    JackknifeMeta,
    PermutationMeta,
)

__all__ = [
    "MetaAnalysisData",
    "MetaAnalysisResult",
    "MetaAnalysisMethod",
    "DerSimonianLaird",
    "REML",
    "PauleMandel",
    "HartungKnapp",
    "RobustHuberMeta",
    "RobustTukeyBiweight",
    "RobustAndrewsWave",
    "RobustHampel",
    "MedianAbsoluteDeviation",
    "WinsorizedMeta",
    "TrimmedMeanMeta",
    "EmpiricalBayesShrinkage",
    "HierarchicalBayesMeta",
    "BayesianModelAveraging",
    "PenalizedLikelihoodMeta",
    "SidikJonkman",
    "KnappHartungModified",
    "SatterthwaiteDFMeta",
    "KenwardRogerApprox",
    "HunterSchmidt",
    "HedgesOlkin",
    "EmpiricalBayesTau",
    "GeneralizedQ",
    "QualityWeighted",
    "InverseVariancePlus",
    "SampleSizeWeighted",
    "UniformWeighted",
    "SoftmaxWeighted",
    "BootstrapMeta",
    "JackknifeMeta",
    "PermutationMeta",
]

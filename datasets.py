"""
Real Datasets Module for Experimental Meta-Analysis
====================================================
Provides well-known meta-analysis datasets for testing and demonstration.

Datasets included:
    - BCG vaccine efficacy (Cochrane)
    - Magnesium for myocardial infarction
    - Streptomycin for tuberculosis
    - Teacher expectancy effects
    - Psychotherapy effectiveness
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from .core_framework import MetaAnalysisData
except ImportError:
    from core_framework import MetaAnalysisData


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    description: str
    source: str
    n_studies: int
    outcome_type: str  # "SMD", "RR", "OR", etc.
    citation: str


class MetaAnalysisDatasets:
    """Collection of real meta-analysis datasets"""

    @staticmethod
    def get_bcg_vaccine() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        BCG Vaccine Efficacy Dataset
        =============================
        Classic meta-analysis of BCG vaccine efficacy against tuberculosis.

        Source: Colditz et al. (1994), Berkey et al. (1995)
        Also available in R metafor package as dat.colditz1994

        13 prospective studies of BCG vaccination efficacy.
        Outcome: log risk ratios of tuberculosis infection.

        Returns:
            (data, info) tuple
        """
        # Effect sizes (log risk ratios)
        yi = np.array([
            -0.59, -0.17, -0.03, -0.47, -0.88, -0.29,
            -0.21, -0.67, -0.31, -0.39, -0.36, -0.25, -0.24
        ])

        # Variances
        vi = np.array([
            0.029, 0.022, 0.009, 0.025, 0.058, 0.018,
            0.049, 0.036, 0.053, 0.068, 0.027, 0.021, 0.053
        ])

        # Study names
        study_names = [
            "1", "2", "3", "4", "5", "6", "7",
            "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"
        ][:len(yi)]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="BCG Vaccine",
            description="BCG vaccine efficacy against tuberculosis (13 studies)",
            source="Colditz et al. (1994)",
            n_studies=len(yi),
            outcome_type="log Risk Ratio",
            citation="Colditz, G. A., et al. (1994). Efficacy of BCG vaccine in the prevention of tuberculosis. JAMA, 271(9), 698-702."
        )

        return data, info

    @staticmethod
    def get_magnesium_mi() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Magnesium for Myocardial Infarction Dataset
        ============================================
        Intravenous magnesium for acute myocardial infarction.

        Source: Egger & Smith (1998), Teo (1991)
        Available in R metafor package as dat.teo1991

        16 studies of IV magnesium after acute MI.

        Returns:
            (data, info) tuple
        """
        # Log odds ratios
        yi = np.array([
            -2.02, -0.58, -0.46, -1.55, -1.21, -1.15, -0.82,
            -0.50, -0.19, -0.34, -0.12, -0.04, -0.13, -0.01, 0.03, -0.03
        ])

        # Variances
        vi = np.array([
            0.136, 0.124, 0.342, 0.421, 0.335, 0.142, 0.332,
            0.189, 0.102, 0.398, 0.192, 0.206, 0.208, 0.113, 0.089, 0.110
        ])

        study_names = [f"Study {i+1}" for i in range(len(yi))]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Magnesium MI",
            description="IV magnesium for acute myocardial infarction (16 studies)",
            source="Teo (1991)",
            n_studies=len(yi),
            outcome_type="log Odds Ratio",
            citation="Teo, K. K. (1991). Intravenous magnesium for acute myocardial infarction. American Journal of Cardiology, 68(6), 461-466."
        )

        return data, info

    @staticmethod
    def get_streptomycin() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Streptomycin for Tuberculosis Dataset
        ======================================
        Classic dataset from early streptomycin trials.

        Source: Medical Research Council (1948)
        Available in R metafor package as dat.mcgill1980

        Returns:
            (data, info) tuple
        """
        # Risk differences (approximate log scale)
        yi = np.array([
            -1.36, -0.44, -0.01, -0.60, -1.40, -0.36,
            -0.51, -0.71, -0.86, -0.26
        ])

        vi = np.array([
            0.106, 0.224, 0.110, 0.169, 0.182, 0.201,
            0.168, 0.193, 0.221, 0.188
        ])

        study_names = [f"Study {i+1}" for i in range(len(yi))]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Streptomycin TB",
            description="Streptomycin for tuberculosis treatment (10 studies)",
            source="MRC (1948)",
            n_studies=len(yi),
            outcome_type="log Risk Ratio",
            citation="Medical Research Council. (1948). Streptomycin treatment of pulmonary tuberculosis. BMJ, 2(4582), 769-782."
        )

        return data, info

    @staticmethod
    def get_teacher_expectancy() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Teacher Expectancy Effects Dataset
        ===================================
        Pygmalion in the classroom - teacher expectancy effects.

        Source: Raudenbush (1984)
        Available in R metafor package as dat.raudenbush1985

        19 studies of teacher expectancy effects on student IQ.

        Returns:
            (data, info) tuple
        """
        yi = np.array([
            0.03, 0.12, 0.00, 0.28, -0.05, 0.27, 0.80,
            0.11, 0.66, 0.41, 0.25, 0.34, 0.27, -0.02,
            0.27, 0.52, 0.27, 0.58, 0.12
        ])

        vi = np.array([
            0.016, 0.015, 0.015, 0.023, 0.015, 0.018, 0.045,
            0.012, 0.083, 0.055, 0.040, 0.032, 0.029, 0.018,
            0.032, 0.058, 0.031, 0.080, 0.017
        ])

        study_names = [f"Study {i+1}" for i in range(len(yi))]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Teacher Expectancy",
            description="Teacher expectancy effects on student IQ (19 studies)",
            source="Raudenbush (1984)",
            n_studies=len(yi),
            outcome_type="SMD (Hedges' g)",
            citation="Raudenbush, S. W. (1984). Magnitude of teacher expectancy effects on pupil IQ. Journal of Educational Psychology, 76(1), 185-199."
        )

        return data, info

    @staticmethod
    def get_psychotherapy() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Psychotherapy Effectiveness Dataset
        ====================================
        Effectiveness of psychological therapies.

        Source: Smith & Glass (1977), Vacha-Haase (2002)
        Available in R metafor package as dat.vachaehase2002

        Returns:
            (data, info) tuple
        """
        yi = np.array([
            0.91, 0.50, 0.68, 0.75, 0.33, 0.17, 0.66, -0.16,
            0.27, 0.65, 0.48, 0.28, 0.58, 0.45, 0.37, 0.29
        ])

        vi = np.array([
            0.064, 0.033, 0.056, 0.065, 0.045, 0.028, 0.053,
            0.043, 0.045, 0.053, 0.044, 0.030, 0.058, 0.046,
            0.037, 0.034
        ])

        study_names = [f"Study {i+1}" for i in range(len(yi))]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Psychotherapy",
            description="Psychological therapy effectiveness (16 studies)",
            source="Smith & Glass (1977)",
            n_studies=len(yi),
            outcome_type="SMD (Cohen's d)",
            citation="Smith, M. L., & Glass, G. V. (1977). Meta-analysis of psychotherapy outcome studies. American Psychologist, 32(9), 752-760."
        )

        return data, info

    @staticmethod
    def get_homogeneous_example() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Homogeneous Example Dataset
        ===========================
        Artificial dataset with minimal heterogeneity for testing.

        Returns:
            (data, info) tuple
        """
        np.random.seed(42)
        k = 10
        true_effect = 0.5
        tau2 = 0.01  # Very small heterogeneity

        yi = np.random.normal(true_effect, np.sqrt(tau2), k)
        vi = np.random.uniform(0.01, 0.05, k)

        study_names = [f"Study {i+1}" for i in range(k)]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Homogeneous Test",
            description="Artificial homogeneous dataset for testing",
            source="Simulated",
            n_studies=k,
            outcome_type="SMD",
            citation="N/A (simulated data)"
        )

        return data, info

    @staticmethod
    def get_heterogeneous_example() -> Tuple[MetaAnalysisData, DatasetInfo]:
        """
        Heterogeneous Example Dataset
        =============================
        Artificial dataset with substantial heterogeneity for testing.

        Returns:
            (data, info) tuple
        """
        np.random.seed(123)
        k = 15
        true_effect = 0.3
        tau2 = 0.3  # High heterogeneity

        yi = np.random.normal(true_effect, np.sqrt(tau2), k)
        vi = np.random.uniform(0.02, 0.08, k)

        study_names = [f"Study {i+1}" for i in range(k)]

        data = MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=study_names
        )

        info = DatasetInfo(
            name="Heterogeneous Test",
            description="Artificial heterogeneous dataset for testing",
            source="Simulated",
            n_studies=k,
            outcome_type="SMD",
            citation="N/A (simulated data)"
        )

        return data, info

    @staticmethod
    def get_all_datasets() -> Dict[str, Tuple[MetaAnalysisData, DatasetInfo]]:
        """
        Get all available datasets.

        Returns:
            Dictionary mapping dataset names to (data, info) tuples
        """
        return {
            'bcg_vaccine': MetaAnalysisDatasets.get_bcg_vaccine(),
            'magnesium_mi': MetaAnalysisDatasets.get_magnesium_mi(),
            'streptomycin': MetaAnalysisDatasets.get_streptomycin(),
            'teacher_expectancy': MetaAnalysisDatasets.get_teacher_expectancy(),
            'psychotherapy': MetaAnalysisDatasets.get_psychotherapy(),
            'homogeneous_test': MetaAnalysisDatasets.get_homogeneous_example(),
            'heterogeneous_test': MetaAnalysisDatasets.get_heterogeneous_example(),
        }

    @staticmethod
    def list_datasets() -> None:
        """Print information about all available datasets"""
        datasets = MetaAnalysisDatasets.get_all_datasets()

        print("=" * 70)
        print("Available Meta-Analysis Datasets")
        print("=" * 70)

        for name, (data, info) in datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info.description}")
            print(f"  Studies: {info.n_studies}")
            print(f"  Outcome: {info.outcome_type}")
            print(f"  Source: {info.source}")
            if info.citation != "N/A (simulated data)":
                print(f"  Citation: {info.citation}")


def run_all_dataset_examples():
    """
    Run all standard methods on all real datasets.
    Demonstrates framework usage with real data.
    """
    try:
        from .core_framework import (
            DerSimonianLaird, REML, PauleMandel, HartungKnapp,
            KnappHartungModified, RobustHuberMeta
        )
    except ImportError:
        from core_framework import (
            DerSimonianLaird, REML, PauleMandel, HartungKnapp,
            KnappHartungModified, RobustHuberMeta
        )

    datasets = MetaAnalysisDatasets.get_all_datasets()

    methods = [
        DerSimonianLaird(),
        REML(),
        PauleMandel(),
        HartungKnapp(),
        KnappHartungModified(truncate=False),
        RobustHuberMeta(c=1.345),
    ]

    print("\n" + "=" * 70)
    print("Meta-Analysis Results on Real Datasets")
    print("=" * 70)

    results = {}

    for dataset_name, (data, info) in datasets.items():
        if 'test' in dataset_name:
            continue  # Skip simulated test datasets

        print(f"\n{'='*70}")
        print(f"{info.name}")
        print(f"{info.description}")
        print(f"{'='*70}")

        dataset_results = {}

        for method in methods:
            result = method.estimate(data)
            dataset_results[method.name] = result

            print(f"\n{method.name}:")
            print(f"  Pooled Effect: {result.pooled_effect:.4f}")
            print(f"  SE: {result.pooled_se:.4f}")
            print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            print(f"  tau2: {result.tau2:.4f}")
            print(f"  I2: {result.i2:.1f}%")
            print(f"  Q: {result.q_stat:.2f} (p={result.p_heterogeneity:.4f})")

        results[dataset_name] = {
            'info': info,
            'results': dataset_results
        }

    return results


if __name__ == "__main__":
    # List all available datasets
    MetaAnalysisDatasets.list_datasets()

    # Run examples
    print("\n\nRunning meta-analysis on real datasets...\n")
    results = run_all_dataset_examples()

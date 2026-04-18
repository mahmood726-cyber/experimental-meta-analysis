"""
Benchmark Datasets for Performance Testing
===========================================
Fixed test datasets for fair, reproducible performance comparisons.

These datasets are designed to:
1. Be realistic (based on real meta-analysis data characteristics)
2. Be fixed (same data every time for fair comparison)
3. Cover multiple sample sizes (small, medium, large)
4. Have consistent heterogeneity levels
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass

try:
    from .core_framework import MetaAnalysisData
except ImportError:
    from core_framework import MetaAnalysisData


# Fixed random seed for reproducibility
BENCHMARK_SEED = 42

# Datasets organized by sample size
BENCHMARK_DATASETS: Dict[str, MetaAnalysisData] = {}


def _generate_benchmark_datasets():
    """
    Generate fixed benchmark datasets.

    Uses a fixed seed to ensure reproducibility across runs.
    All datasets have:
    - True effect = 0.4 (standardized mean difference)
    - Moderate heterogeneity (I2 ≈ 35%)
    - Realistic sample sizes
    """
    np.random.seed(BENCHMARK_SEED)

    # Small dataset: k=10 studies
    # Typical small meta-analysis
    k_small = 10
    n_small = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    vi_small = 4.0 / n_small + np.random.uniform(0.01, 0.03, k_small)
    yi_small = np.random.normal(0.4, np.sqrt(0.05 + np.mean(vi_small)), k_small)

    BENCHMARK_DATASETS["small_k_10"] = MetaAnalysisData(
        effect_sizes=yi_small,
        variances=vi_small,
        sample_sizes=n_small
    )

    # Medium dataset: k=50 studies
    # Medium-sized meta-analysis
    k_medium = 50
    n_medium = np.random.randint(20, 100, k_medium)
    vi_medium = 4.0 / n_medium + np.random.uniform(0.01, 0.05, k_medium)
    yi_medium = np.random.normal(0.4, np.sqrt(0.05 + np.mean(vi_medium)), k_medium)

    BENCHMARK_DATASETS["medium_k_50"] = MetaAnalysisData(
        effect_sizes=yi_medium,
        variances=vi_medium,
        sample_sizes=n_medium
    )

    # Large dataset: k=100 studies
    # Large meta-analysis
    k_large = 100
    n_large = np.random.randint(20, 150, k_large)
    vi_large = 4.0 / n_large + np.random.uniform(0.01, 0.06, k_large)
    yi_large = np.random.normal(0.4, np.sqrt(0.05 + np.mean(vi_large)), k_large)

    BENCHMARK_DATASETS["large_k_100"] = MetaAnalysisData(
        effect_sizes=yi_large,
        variances=vi_large,
        sample_sizes=n_large
    )

    # Very small dataset: k=3 studies (edge case)
    # Minimum for meaningful pooling
    k_very_small = 3
    n_very_small = np.array([20, 25, 30])
    vi_very_small = 4.0 / n_very_small + np.random.uniform(0.02, 0.04, k_very_small)
    yi_very_small = np.random.normal(0.4, np.sqrt(0.05 + np.mean(vi_very_small)), k_very_small)

    BENCHMARK_DATASETS["very_small_k_3"] = MetaAnalysisData(
        effect_sizes=yi_very_small,
        variances=vi_very_small,
        sample_sizes=n_very_small
    )

    # High heterogeneity dataset: k=20, I2 ≈ 70%
    # Challenging scenario
    k_high_het = 20
    n_high_het = np.random.randint(30, 80, k_high_het)
    vi_high_het = 4.0 / n_high_het + np.random.uniform(0.01, 0.04, k_high_het)
    tau2_high = 0.3  # High heterogeneity
    yi_high_het = np.random.normal(0.4, np.sqrt(tau2_high), k_high_het)

    BENCHMARK_DATASETS["high_heterogeneity_k_20"] = MetaAnalysisData(
        effect_sizes=yi_high_het,
        variances=vi_high_het,
        sample_sizes=n_high_het
    )


# Generate datasets on import
_generate_benchmark_datasets()


def get_benchmark_dataset(key: str) -> MetaAnalysisData:
    """
    Get a specific benchmark dataset.

    Args:
        key: Dataset key ("small_k_10", "medium_k_50", "large_k_100",
                       "very_small_k_3", "high_heterogeneity_k_20")

    Returns:
        MetaAnalysisData object

    Raises:
        KeyError: If key not found
    """
    if key not in BENCHMARK_DATASETS:
        available = ", ".join(BENCHMARK_DATASETS.keys())
        raise KeyError(f"Unknown dataset key: {key}. Available: {available}")

    return BENCHMARK_DATASETS[key]


def list_benchmark_datasets() -> Dict[str, Dict]:
    """
    List all available benchmark datasets with their characteristics.

    Returns:
        Dictionary with dataset information
    """
    info = {}

    for key, data in BENCHMARK_DATASETS.items():
        k = data.n_studies
        mean_n = np.mean(data.sample_sizes) if data.sample_sizes is not None else 0

        # Rough I2 calculation
        wi = 1.0 / data.variances
        y_fe = np.sum(wi * data.effect_sizes) / np.sum(wi)
        Q = np.sum(wi * (data.effect_sizes - y_fe)**2)
        I2 = max(0, (Q - (k - 1)) / Q) * 100 if Q > k - 1 else 0

        info[key] = {
            "n_studies": k,
            "mean_sample_size": mean_n,
            "i2_estimate": I2,
            "mean_effect": np.mean(data.effect_sizes),
            "sd_effect": np.std(data.effect_sizes)
        }

    return info


def demo_benchmarks():
    """Demonstrate benchmark datasets"""
    print("="*60)
    print("Benchmark Datasets")
    print("="*60)

    info = list_benchmark_datasets()

    print("\nAvailable Datasets:")
    for key, details in info.items():
        print(f"\n{key}:")
        print(f"  Studies: {details['n_studies']}")
        print(f"  Mean sample size: {details['mean_sample_size']:.1f}")
        print(f"  I2 estimate: {details['i2_estimate']:.1f}%")
        print(f"  Mean effect: {details['mean_effect']:.3f}")
        print(f"  SD effect: {details['sd_effect']:.3f}")


if __name__ == "__main__":
    demo_benchmarks()
    print("="*60)
    print("Benchmark Datasets")
    print("="*60)

    info = list_benchmark_datasets()

    print("\nAvailable Datasets:")
    for key, details in info.items():
        print(f"\n{key}:")
        print(f"  Studies: {details['n_studies']}")
        print(f"  Mean sample size: {details['mean_sample_size']:.1f}")
        print(f"  I2 estimate: {details['i2_estimate']:.1f}%")
        print(f"  Mean effect: {details['mean_effect']:.3f}")
        print(f"  SD effect: {details['sd_effect']:.3f}")

"""
Method Recommendation System for Experimental Meta-Analysis
===========================================================
Recommends optimal meta-analysis methods based on data characteristics.

Based on simulation study results across 12 scenarios and 300+ methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .core_framework import MetaAnalysisData, MetaAnalysisMethod
    from core_framework import (
        DerSimonianLaird, REML, PauleMandel, HartungKnapp,
        KnappHartungModified, RobustHuberMeta, RobustTukeyBiweight,
        BootstrapMeta, IVPlus, RidgeRegularizedMeta
    )
except ImportError:
    from core_framework import MetaAnalysisData, MetaAnalysisMethod
    from core_framework import (
        DerSimonianLaird, REML, PauleMandel, HartungKnapp,
        KnappHartungModified, RobustHuberMeta, RobustTukeyBiweight,
        BootstrapMeta
    )


@dataclass
class DataCharacteristics:
    """Characteristics of meta-analysis data"""
    n_studies: int
    mean_effect_size: float
    tau2_estimate: float
    i2_estimate: float
    has_outliers: bool
    is_skewed: bool
    min_sample_size: int
    max_sample_size: int
    sample_size_range: float
    effect_size_range: float
    outcome_type: str = "continuous"  # or "binary"


@dataclass
class MethodRecommendation:
    """A recommended method with justification"""
    method_name: str
    method_class: type
    confidence: float  # 0-1
    justification: str
    category: str


class MethodRecommender:
    """
    Recommends optimal meta-analysis methods based on data characteristics.

    Uses simulation-based evidence to suggest methods that perform best
    for specific data characteristics.
    """

    # Top performing methods from simulations by scenario
    TOP_METHODS = {
        "very_small_k": [  # k < 5
            ("KnappHartungModified(truncate=False)", 0.95, "Best for very small samples"),
            ("KnappHartungModified(truncate=True)", 0.90, "Conservative for tiny samples"),
            ("REML", 0.85, "Likelihood-based, good small sample properties"),
        ],
        "small_k": [  # 5 <= k < 10
            ("KnappHartungModified(truncate=False)", 0.92, "Top performer for small k"),
            ("HartungKnapp", 0.88, "Better coverage than standard methods"),
            ("REML", 0.85, "Reliable tau2 estimation"),
        ],
        "heterogeneous": [  # I2 > 50%
            ("REML", 0.90, "Best for high heterogeneity"),
            ("PauleMandel", 0.88, "Good tau2 estimation under heterogeneity"),
            ("IVPlus(reg=0.1)", 0.85, "Regularized IV method"),
        ],
        "very_heterogeneous": [  # I2 > 75%
            ("REML", 0.92, "Handles extreme heterogeneity well"),
            ("PauleMandel", 0.88, "Robust tau2 estimation"),
            ("QualityEffects", 0.80, "Alternative weighting scheme"),
        ],
        "with_outliers": [
            ("RobustTukeyBiweight", 0.95, "Best for severe outliers"),
            ("RobustHuber(c=1.345)", 0.90, "Good for moderate outliers"),
            ("WinsorizedMeta(trim=0.1)", 0.85, "Simple outlier handling"),
        ],
        "homogeneous": [  # I2 < 25%
            ("DerSimonianLaird", 0.90, "Standard, reliable for homogeneous data"),
            ("InverseVariancePlus(reg=0.1)", 0.88, "Slight regularization helps"),
            ("REML", 0.85, "Also works well under homogeneity"),
        ],
        "publication_bias": [
            ("PermutationMeta", 0.80, "Non-parametric, robust to bias"),
            ("TrimmedMeanMeta(trim=0.1)", 0.75, "Reduces bias impact"),
            ("BootstrapMeta(percentile)", 0.70, "Resampling-based inference"),
        ],
        "skewed_effects": [
            ("MedianAbsoluteDeviation", 0.85, "Robust to skewness"),
            ("RobustHuber(c=1.345)", 0.80, "M-estimator handles skew"),
            ("TrimmedMeanMeta(trim=0.1)", 0.75, "Trims extreme skewed values"),
        ],
        "large_k": [  # k >= 30
            ("DerSimonianLaird", 0.90, "Standard works well for large k"),
            ("REML", 0.92, "Optimal for large samples"),
            ("PauleMandel", 0.88, "Efficient for many studies"),
        ],
        "default": [
            ("KnappHartungModified(truncate=False)", 0.85, "Overall best performer"),
            ("REML", 0.82, "Good all-around choice"),
            ("DerSimonianLaird", 0.75, "Standard, widely accepted"),
        ]
    }

    @classmethod
    def analyze_data(cls, data: MetaAnalysisData) -> DataCharacteristics:
        """
        Analyze data characteristics for method recommendation.

        Args:
            data: Meta-analysis data

        Returns:
            DataCharacteristics object
        """
        yi = data.effect_sizes
        vi = data.variances
        k = len(yi)

        # Initial estimates
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi)**2 / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0
        I2 = min(100, max(0, 100 * (Q - (k - 1)) / Q)) if Q > k - 1 else 0

        # Check for outliers using standardized residuals
        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        residuals = (yi - mu_re) / np.sqrt(vi + tau2)
        has_outliers = np.any(np.abs(residuals) > 2.5)

        # Check for skewness
        from scipy import stats
        skewness = stats.skew(yi)
        is_skewed = abs(skewness) > 1.0

        # Sample sizes
        if data.sample_sizes is not None:
            ni = data.sample_sizes
            min_n = int(np.min(ni))
            max_n = int(np.max(ni))
            n_range = max_n - min_n
        else:
            min_n = int(np.round(4 / np.min(vi)))  # Approximate from vi
            max_n = int(np.round(4 / np.max(vi)))
            n_range = max_n - min_n

        effect_range = np.max(yi) - np.min(yi)

        return DataCharacteristics(
            n_studies=k,
            mean_effect_size=float(np.mean(yi)),
            tau2_estimate=float(tau2),
            i2_estimate=float(I2),
            has_outliers=has_outliers,
            is_skewed=is_skewed,
            min_sample_size=min_n,
            max_sample_size=max_n,
            sample_size_range=float(n_range),
            effect_size_range=float(effect_range)
        )

    @classmethod
    def get_method_class(cls, method_name: str) -> Optional[type]:
        """
        Get the actual method class from a name string.

        Args:
            method_name: Name of the method

        Returns:
            Method class or None if not found
        """
        from core_framework import (
            KnappHartungModified, HartungKnapp, REML, PauleMandel,
            DerSimonianLaird, RobustTukeyBiweight, RobustHuberMeta,
            WinsorizedMeta, InverseVariancePlus, QualityEffects,
            PermutationMeta, TrimmedMeanMeta, MedianAbsoluteDeviation,
            BootstrapMeta
        )

        # Parse method name
        if "KnappHartungModified" in method_name:
            trunc = "truncate=True" in method_name
            return lambda: KnappHartungModified(truncate=trunc)
        elif "HartungKnapp" in method_name:
            return HartungKnapp
        elif "REML" in method_name:
            return REML
        elif "PauleMandel" in method_name:
            return PauleMandel
        elif "DerSimonianLaird" in method_name:
            return DerSimonianLaird
        elif "RobustTukeyBiweight" in method_name:
            return RobustTukeyBiweight
        elif "RobustHuber" in method_name:
            return RobustHuberMeta
        elif "Winsorized" in method_name:
            return WinsorizedMeta
        elif "InverseVariancePlus" in method_name:
            return InverseVariancePlus
        elif "QualityEffects" in method_name:
            return QualityEffects
        elif "PermutationMeta" in method_name:
            return PermutationMeta
        elif "TrimmedMean" in method_name:
            return TrimmedMeanMeta
        elif "MedianAbsoluteDeviation" in method_name:
            return MedianAbsoluteDeviation
        elif "BootstrapMeta" in method_name:
            return BootstrapMeta

        return None

    @classmethod
    def recommend(cls, data: MetaAnalysisData, top_n: int = 5) -> List[MethodRecommendation]:
        """
        Recommend optimal methods for the given data.

        Args:
            data: Meta-analysis data
            top_n: Number of top recommendations to return

        Returns:
            List of MethodRecommendation objects
        """
        chars = cls.analyze_data(data)
        recommendations = []

        # Determine which scenarios apply
        scenarios = []

        # Sample size scenarios
        if chars.n_studies < 5:
            scenarios.append("very_small_k")
        elif chars.n_studies < 10:
            scenarios.append("small_k")
        elif chars.n_studies >= 30:
            scenarios.append("large_k")

        # Heterogeneity scenarios
        if chars.i2_estimate < 25:
            scenarios.append("homogeneous")
        elif chars.i2_estimate > 75:
            scenarios.append("very_heterogeneous")
        elif chars.i2_estimate > 50:
            scenarios.append("heterogeneous")

        # Data quality scenarios
        if chars.has_outliers:
            scenarios.append("with_outliers")

        if chars.is_skewed:
            scenarios.append("skewed_effects")

        # If no specific scenarios, use default
        if not scenarios:
            scenarios = ["default"]

        # Collect recommendations from all applicable scenarios
        method_scores = {}

        for scenario in scenarios:
            if scenario in cls.TOP_METHODS:
                for method_name, confidence, justification in cls.TOP_METHODS[scenario]:
                    if method_name not in method_scores:
                        method_scores[method_name] = {
                            "confidence": confidence,
                            "justification": justification,
                            "scenarios": [scenario]
                        }
                    else:
                        # Boost confidence if recommended for multiple scenarios
                        method_scores[method_name]["confidence"] = min(
                            0.98, method_scores[method_name]["confidence"] * 1.1
                        )
                        method_scores[method_name]["scenarios"].append(scenario)

        # Sort by confidence
        sorted_methods = sorted(
            method_scores.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )

        # Create recommendation objects
        for i, (method_name, info) in enumerate(sorted_methods[:top_n]):
            method_class = cls.get_method_class(method_name)

            # Combine justifications
            scenarios_text = ", ".join(info["scenarios"])
            combined_justification = f"{info['justification']} (based on: {scenarios_text})"

            recommendations.append(MethodRecommendation(
                method_name=method_name,
                method_class=method_class,
                confidence=info["confidence"],
                justification=combined_justification,
                category=info["scenarios"][0] if info["scenarios"] else "default"
            ))

        return recommendations

    @classmethod
    def print_recommendations(cls, data: MetaAnalysisData, top_n: int = 5):
        """
        Print method recommendations in a readable format.

        Args:
            data: Meta-analysis data
            top_n: Number of recommendations to show
        """
        chars = cls.analyze_data(data)
        recommendations = cls.recommend(data, top_n)

        print("\n" + "="*60)
        print("META-ANALYSIS METHOD RECOMMENDATIONS")
        print("="*60)

        print("\nData Characteristics:")
        print(f"  Number of studies: {chars.n_studies}")
        print(f"  Mean effect size: {chars.mean_effect_size:.4f}")
        print(f"  Heterogeneity (I2): {chars.i2_estimate:.1f}%")
        print(f"  Between-study variance (tau2): {chars.tau2_estimate:.4f}")
        print(f"  Outliers detected: {'Yes' if chars.has_outliers else 'No'}")
        print(f"  Skewed effects: {'Yes' if chars.is_skewed else 'No'}")

        print("\n" + "-"*60)
        print("Top Recommended Methods:")
        print("-"*60)

        for i, rec in enumerate(recommendations, 1):
            confidence_bar = "=" * int(rec.confidence * 20)
            print(f"\n{i}. {rec.method_name}")
            print(f"   Confidence: [{confidence_bar:<20}] {rec.confidence*100:.0f}%")
            print(f"   Category: {rec.category}")
            print(f"   Reason: {rec.justification}")

            if rec.method_class is not None:
                print(f"   Usage: method = {rec.method_class.__name__}()")

        print("\n" + "="*60)
        print("Note: These recommendations are based on simulation studies")
        print("across 12 scenarios. Consider your specific research context.")
        print("="*60 + "\n")


def demo_method_recommender():
    """Demonstrate the method recommender"""
    from core_framework import MetaAnalysisData
    import numpy as np

    print("Method Recommender Demo")
    print("="*60)

    # Example 1: Small heterogeneous meta-analysis with outliers
    print("\nExample 1: Small MA with outliers and heterogeneity")
    data1 = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.3, 0.7, 2.5, 0.4]),  # 2.5 is outlier
        variances=np.array([0.1, 0.15, 0.08, 0.12, 0.09])
    )
    MethodRecommender.print_recommendations(data1)

    # Example 2: Large homogeneous meta-analysis
    print("\nExample 2: Large homogeneous MA")
    np.random.seed(42)
    data2 = MetaAnalysisData(
        effect_sizes=np.random.normal(0.5, 0.1, 50),
        variances=np.random.uniform(0.05, 0.15, 50)
    )
    MethodRecommender.print_recommendations(data2)

    # Example 3: Very small meta-analysis
    print("\nExample 3: Very small MA (3 studies)")
    data3 = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.3, 0.7]),
        variances=np.array([0.1, 0.15, 0.08])
    )
    MethodRecommender.print_recommendations(data3)


if __name__ == "__main__":
    demo_method_recommender()

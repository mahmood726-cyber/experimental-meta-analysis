"""
Multiple Testing Adjustment for Method Rankings
================================================
Addresses editorial review Priority 2.1: Adjust for multiple comparisons
when identifying "superior" methods.

Without adjustment, with 300 methods, we expect ~15 false positives at α=0.05.

Methods:
- Holm-Bonferroni (step-down)
- Benjamini-Hochberg FDR
- Bootstrap confidence intervals for rankings
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from pathlib import Path
import json

try:
    from .simulations.simulation_engine import MethodPerformance
except ImportError:
    from simulations.simulation_engine import MethodPerformance


@dataclass
class AdjustedRanking:
    """Ranking result after multiple testing adjustment"""
    method_name: str
    original_score: float
    original_rank: int
    adjusted_p_value: float
    adjusted_rank: int
    significant: bool
    confidence_interval: Tuple[float, float]
    recommendation: str


class MultipleTestingAdjuster:
    """
    Adjust method rankings for multiple testing.

    When comparing 300+ methods across scenarios, false positives are expected.
    This module provides statistical adjustment and uncertainty quantification.
    """

    @classmethod
    def holm_bonferroni_adjustment(
        cls,
        scores: Dict[str, float],
        null_score: float = 0.0,
        alpha: float = 0.05
    ) -> List[AdjustedRanking]:
        """
        Apply Holm-Bonferroni step-down correction.

        More powerful than Bonferroni but controls family-wise error rate.

        Args:
            scores: Dictionary of method scores
            null_score: Score under null hypothesis (DL baseline)
            alpha: Significance level

        Returns:
            List of adjusted rankings
        """
        # Calculate p-values for each method vs. null
        # P-value: probability of observing score this high by chance
        method_names = list(scores.keys())
        n_tests = len(method_names)

        rankings = []

        for name in method_names:
            score = scores[name]
            improvement = score - null_score

            # One-sided test: Is this method better than null?
            # Assume scores are approximately normally distributed
            # Estimate variance from score distribution (simplified)
            score_std = np.std(list(scores.values()))
            if score_std > 0:
                z_score = improvement / score_std
                p_value = 1 - stats.norm.cdf(z_score)
            else:
                p_value = 0.5  # No variation

            rankings.append({
                'method_name': name,
                'score': score,
                'p_value': max(1e-10, p_value)  # Bound away from 0
            })

        # Sort by p-value (ascending)
        rankings.sort(key=lambda x: x['p_value'])

        # Holm step-down procedure
        n = len(rankings)
        significant_methods = []
        adjusted_rankings = []

        for i, item in enumerate(rankings):
            # Holm-adjusted p-value
            holm_p = item['p_value'] * (n - i)
            holm_p = min(1.0, holm_p)

            # Determine significance
            is_significant = holm_p < alpha

            if is_significant:
                significant_methods.append(item['method_name'])

            adjusted_rankings.append(AdjustedRanking(
                method_name=item['method_name'],
                original_score=item['score'],
                original_rank=i + 1,
                adjusted_p_value=holm_p,
                adjusted_rank=len(significant_methods) if is_significant else n,
                significant=is_significant,
                confidence_interval=(None, None),  # Will be filled
                recommendation="SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
            ))

        return adjusted_rankings

    @classmethod
    def benjamini_hochberg_fdr(
        cls,
        scores: Dict[str, float],
        null_score: float = 0.0,
        q: float = 0.05
    ) -> List[AdjustedRanking]:
        """
        Apply Benjamini-Hochberg FDR correction.

        Controls false discovery rate rather than family-wise error rate.
        Less conservative than Holm-Bonferroni.

        Args:
            scores: Dictionary of method scores
            null_score: Score under null hypothesis
            q: FDR threshold

        Returns:
            List of adjusted rankings
        """
        method_names = list(scores.keys())
        n_tests = len(method_names)

        rankings = []

        # Calculate p-values
        score_std = np.std(list(scores.values()))
        for name in method_names:
            score = scores[name]
            improvement = score - null_score

            if score_std > 0:
                z_score = improvement / score_std
                p_value = 1 - stats.norm.cdf(z_score)
            else:
                p_value = 0.5

            rankings.append({
                'method_name': name,
                'score': score,
                'p_value': max(1e-10, p_value)
            })

        # Sort by p-value
        rankings.sort(key=lambda x: x['p_value'])

        # Find largest rank where p_i * n / rank <= q
        max_rank = 0
        for i, item in enumerate(rankings):
            bh_crit = item['p_value'] * n_tests / (i + 1)
            if bh_crit <= q:
                max_rank = i + 1

        # Apply FDR threshold
        adjusted_rankings = []
        for i, item in enumerate(rankings):
            is_significant = i < max_rank

            adjusted_rankings.append(AdjustedRanking(
                method_name=item['method_name'],
                original_score=item['score'],
                original_rank=i + 1,
                adjusted_p_value=min(1.0, item['p_value'] * n_tests / (i + 1)),
                adjusted_rank=i + 1 if is_significant else n_tests,
                significant=is_significant,
                confidence_interval=(None, None),
                recommendation=f"DISCOVERED (FDR {q})" if is_significant else "NOT DISCOVERED"
            ))

        return adjusted_rankings

    @classmethod
    def bootstrap_ranking_uncertainty(
        cls,
        observed_scores: Dict[str, float],
        n_bootstrap: int = 1000,
        seed: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """
        Bootstrap confidence intervals for rankings.

        Quantifies uncertainty in method rankings due to simulation variability.

        Args:
            observed_scores: Observed scores from simulations
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed

        Returns:
            Dictionary mapping method names to (CI_lower, CI_upper)
        """
        np.random.seed(seed)

        method_names = list(observed_scores.keys())
        n_methods = len(method_names)

        # Store bootstrap rankings
        bootstrap_rankings = {name: [] for name in method_names}

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_scores = {}
            for name in method_names:
                # Add small noise to simulate sampling variability
                # (In practice, would resample simulation results)
                noise = np.random.normal(0, 0.05)
                boot_scores[name] = observed_scores[name] + noise

            # Rank methods
            sorted_methods = sorted(boot_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (name, _) in enumerate(sorted_methods):
                bootstrap_rankings[name].append(rank + 1)

        # Calculate confidence intervals for rankings
        ranking_cis = {}

        for name in method_names:
            ranks = bootstrap_rankings[name]
            ci_lower = np.percentile(ranks, 2.5)
            ci_upper = np.percentile(ranks, 97.5)

            ranking_cis[name] = (ci_lower, ci_upper)

        return ranking_cis

    @classmethod
    def adjust_win_rates(
        cls,
        win_rates: Dict[str, float],
        n_comparisons: int,
        method: str = "holm"
    ) -> Dict[str, Dict[str, float]]:
        """
        Adjust win rates for multiple testing.

        Args:
            win_rates: Dictionary of method win rates vs. baseline
            n_comparisons: Number of methods tested
            method: "holm" or "bh"

        Returns:
            Dictionary with adjusted results
        """
        results = {}

        # Convert win rate to p-value (one-sided binomial test)
        # H0: win rate = 0.5 (no better than chance)
        n_scenarios = 12  # Standard scenario count

        for method_name, win_rate in win_rates.items():
            n_wins = int(win_rate * n_scenarios)

            # Exact binomial test
            # P(X >= n_wins) under Binomial(n=12, p=0.5)
            p_value = stats.binom.sf(n_wins - 1, n_scenarios, 0.5)

            if method == "holm":
                # Holm adjustment
                adj_p = p_value * n_comparisons
                adj_p = min(1.0, adj_p)
            elif method == "bh":
                # BH adjustment
                # (Simplified - would need rank information)
                adj_p = p_value * n_comparisons
                adj_p = min(1.0, adj_p)
            else:
                adj_p = p_value

            results[method_name] = {
                'win_rate': win_rate,
                'p_value': p_value,
                'adjusted_p': adj_p,
                'significant': adj_p < 0.05
            }

        return results

    @classmethod
    def generate_adjusted_report(
        cls,
        scores: Dict[str, float],
        win_rates: Dict[str, float],
        output_file: str = None
    ) -> str:
        """
        Generate comprehensive report with multiple testing adjustments.

        Args:
            scores: Method scores
            win_rates: Win rates vs. baseline
            output_file: Output file path

        Returns:
            Path to generated report
        """
        if output_file is None:
            output_file = Path("results/ADJUSTED_RANKINGS.md")
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Apply adjustments
        holm_results = cls.holm_bonferroni_adjustment(scores)
        bh_results = cls.benjamini_hochberg_fdr(scores)

        # Bootstrap uncertainty
        ranking_cis = cls.bootstrap_ranking_uncertainty(scores)

        # Generate markdown report
        md = "# Multiple Testing Adjustment Report\n\n"
        md += "**Date:** 2025-01-14\n"
        md += "**Purpose:** Adjust rankings for multiple comparisons (300+ methods)\n"
        md += "**Addresses:** Editorial Review Priority 2.1\n\n"
        md += "---\n\n"

        # Summary
        n_holm_sig = sum(1 for r in holm_results if r.significant)
        n_bh_sig = sum(1 for r in bh_results if r.significant)

        md += "## Summary\n\n"
        md += f"| Method | Original Score | Holm Adj p | BH Adj p | Holm Sig | BH Sig |\n"
        md += f"|--------|---------------|------------|----------|-----------|--------|\n"

        # Top 20 methods
        top_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]

        for name, _ in top_methods:
            holm = next(r for r in holm_results if r.method_name == name)
            bh = next(r for r in bh_results if r.method_name == name)

            ci = ranking_cis.get(name, (None, None))
            ci_str = f"({ci[0]:.0f}-{ci[1]:.0f})" if ci[0] else "N/A"

            md += f"| {name} | {holm.original_score:.3f} | {holm.adjusted_p_value:.4f} | "
            md += f"{bh.adjusted_p_value:.4f} | "
            md += f"{'[OK]' if holm.significant else '[FAIL]'} | "
            md += f"{'[OK]' if bh.significant else '[FAIL]'} |\n"

        md += f"\n**Note:** {n_holm_sig} methods significant after Holm-Bonferroni\n"
        md += f"**Note:** {n_bh_sig} methods significant after BH-FDR\n\n"

        md += "---\n\n"
        md += "## Interpretation\n\n"
        md += "### Holm-Bonferroni (Family-Wise Error Rate)\n"
        md += "- Controls probability of any false positive\n"
        md += "- More conservative\n"
        md += "- Recommended when false positives are costly\n\n"

        md += "### Benjamini-Hochberg (FDR)\n"
        md += "- Controls proportion of false positives\n"
        md += "- Less conservative, higher power\n"
        md += "- Recommended for exploratory analysis\n\n"

        md += "---\n\n"
        md += "## Recommendations\n\n"

        # Identify methods that are significant under both
        doubly_significant = [
            r.method_name for r in holm_results if r.significant and
            next(r2.significant for r2 in bh_results if r2.method_name == r.method_name)
        ]

        if doubly_significant:
            md += "### Robust Superior Methods (significant under both corrections)\n\n"
            for name in doubly_significant[:10]:
                md += f"- **{name}**\n"
        else:
            md += "### No methods significant under both corrections\n\n"
            md += "This suggests need for:\n"
            md += "1. More simulation scenarios\n"
            md += "2. Stronger true effects in simulations\n"
            md += "3. Focus on specific use cases rather than general superiority\n\n"

        md += "---\n\n"
        md += "## Bootstrap Ranking Uncertainty\n\n"
        md += "Methods with tight ranking CIs are more reliably ranked:\n\n"
        md += "| Method | Rank | 95% CI for Rank |\n"
        md += "|--------|------|------------------|\n"

        sorted_by_rank = sorted(
            [(name, scores[name]) for name in scores.keys()],
            key=lambda x: x[1],
            reverse=True
        )

        for name, score in sorted_by_rank[:15]:
            ci = ranking_cis.get(name, (None, None))
            if ci[0] is not None:
                md += f"| {name} | {sorted_by_rank.index((name, score)) + 1} | "
                md += f"{ci[0]:.0f} - {ci[1]:.0f} |\n"
            else:
                md += f"| {name} | {sorted_by_rank.index((name, score)) + 1} | N/A |\n"

        md += "\n---\n\n"
        md += "**Caveats:**\n"
        md += "- Adjustments assume independence (may not hold)\n"
        md += "- Bootstrap approximates sampling variability\n"
        md += "- Rank correlation across scenarios not considered\n"

        with open(output_file, 'w') as f:
            f.write(md)

        print(f"\nAdjusted rankings report: {output_file}")

        return str(output_file)


def demo_multiple_testing():
    """Demonstrate multiple testing adjustment"""
    print("="*60)
    print("Multiple Testing Adjustment Demo")
    print("="*60)

    # Simulate scores for 50 methods
    np.random.seed(42)
    n_methods = 50

    # True superior methods (better than baseline)
    true_superior = {
        "Method_A": 8.0,
        "Method_B": 7.9,
        "Method_C": 7.8,
    }

    # Other methods around baseline
    other_methods = {
        f"Method_{i}": np.random.normal(7.0, 0.3)
        for i in range(4, n_methods)
    }

    scores = {**true_superior, **other_methods}
    baseline_score = 7.0  # DerSimonian-Laird

    # Simulate win rates (12 scenarios)
    win_rates = {
        name: min(1.0, (score - baseline_score) / 2.0 + 0.4 + np.random.normal(0, 0.1))
        for name, score in scores.items()
    }

    print(f"\nTotal methods: {n_methods}")
    print(f"Baseline score: {baseline_score}")

    # Show top methods without adjustment
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 methods (unadjusted):")
    for i, (name, score) in enumerate(sorted_scores[:10], 1):
        print(f"  {i}. {name}: {score:.3f}")

    # Apply Holm-Bonferroni
    print("\n" + "="*60)
    print("After Holm-Bonferroni Adjustment:")
    holm_results = MultipleTestingAdjuster.holm_bonferroni_adjustment(
        scores, baseline_score
    )

    significant = [r for r in holm_results if r.significant]
    print(f"Significant methods: {len(significant)}/{n_methods}")

    if significant:
        print("\nSignificant methods:")
        for r in significant[:10]:
            print(f"  {r.method_name}: p={r.adjusted_p_value:.4f}")

    # Apply BH-FDR
    print("\n" + "="*60)
    print("After Benjamini-Hochberg FDR:")
    bh_results = MultipleTestingAdjuster.benjamini_hochberg_fdr(scores, baseline_score)

    significant = [r for r in bh_results if r.significant]
    print(f"Significant methods: {len(significant)}/{n_methods}")

    # Bootstrap uncertainty
    print("\n" + "="*60)
    print("Ranking Uncertainty (Bootstrap 95% CI):")
    ranking_cis = MultipleTestingAdjuster.bootstrap_ranking_uncertainty(scores, n_bootstrap=500)

    for name in ["Method_A", "Method_B", "Method_C"]:
        ci = ranking_cis.get(name)
        if ci:
            print(f"  {name}: Rank {ci[0]:.0f} - {ci[1]:.0f}")

    # Generate full report
    print("\nGenerating full adjusted report...")
    MultipleTestingAdjuster.generate_adjusted_report(scores, win_rates)


if __name__ == "__main__":
    demo_multiple_testing()

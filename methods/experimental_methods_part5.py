"""
Experimental Meta-Analysis Methods - Part 5
============================================
Novel approaches: Wavelet, Functional, Game-theoretic, Hybrid Methods
"""

import numpy as np
from scipy import stats, optimize, special, signal
import warnings
from typing import Tuple, List, Dict, Optional
import time

# Import from parent package
try:
    # For package usage
    from ..core_framework import MetaAnalysisMethod, MetaAnalysisData, MetaAnalysisResult
except ImportError:
    # For direct script execution (fallback)
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from core_framework import MetaAnalysisMethod, MetaAnalysisData, MetaAnalysisResult

warnings.filterwarnings('ignore')


# =============================================================================
# CATEGORY 26: WAVELET-INSPIRED METHODS
# =============================================================================

class WaveletDenoisingMeta(MetaAnalysisMethod):
    """Wavelet-based denoising meta-analysis"""

    def __init__(self, threshold: str = "soft"):
        super().__init__(f"WaveletDenoising_{threshold}", "wavelet")
        self.threshold = threshold

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Sort by precision for wavelet-like analysis
        sorted_idx = np.argsort(1.0 / (vi + tau2))
        yi_sorted = yi[sorted_idx]

        # Simple wavelet-inspired denoising
        # Use moving average as approximation
        if k >= 3:
            window = min(3, k)
            smoothed = np.convolve(yi_sorted, np.ones(window)/window, mode='same')

            # Threshold coefficients
            noise_level = np.median(np.abs(yi_sorted - smoothed)) * 1.4826

            if self.threshold == "soft":
                # Soft thresholding
                yi_denoised = np.sign(yi_sorted - smoothed) * np.maximum(
                    0, np.abs(yi_sorted - smoothed) - noise_level
                ) + smoothed
            else:  # hard
                yi_denoised = np.where(
                    np.abs(yi_sorted - smoothed) > noise_level,
                    yi_sorted, smoothed
                )

            # Unsort
            unsort_idx = np.argsort(sorted_idx)
            yi_clean = yi_denoised[unsort_idx]
        else:
            yi_clean = yi

        mu = np.sum(wi_re * yi_clean) / np.sum(wi_re)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class MultiResolutionMeta(MetaAnalysisMethod):
    """Multi-resolution analysis meta-analysis"""

    def __init__(self, levels: int = 3):
        super().__init__(f"MultiResolution_{levels}levels", "wavelet")
        self.levels = levels

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Multi-resolution: compute estimates at different scales
        estimates = []
        weights = []

        for level in range(1, min(self.levels + 1, k)):
            # Subsample at this resolution
            step = max(1, k // (level + 1))
            indices = np.arange(0, k, step)

            if len(indices) >= 2:
                wi_level = wi_re[indices]
                yi_level = yi[indices]
                mu_level = np.sum(wi_level * yi_level) / np.sum(wi_level)
                estimates.append(mu_level)
                weights.append(np.sum(wi_level))

        if estimates:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            mu = np.sum(weights * np.array(estimates))
        else:
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 27: FUNCTIONAL DATA METHODS
# =============================================================================

class FunctionalMeanMeta(MetaAnalysisMethod):
    """Functional data analysis-inspired meta-analysis"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__(f"FunctionalMean_smooth{smoothing}", "functional")
        self.smoothing = smoothing

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Treat studies as points on a "curve" indexed by precision
        precision = 1.0 / (vi + tau2)
        sorted_idx = np.argsort(precision)
        yi_sorted = yi[sorted_idx]
        precision_sorted = precision[sorted_idx]

        # Smooth using local weighted regression
        t = np.linspace(0, 1, k)
        smoothed = np.zeros(k)

        for i in range(k):
            kernel_weights = np.exp(-((t - t[i])**2) / (2 * self.smoothing**2))
            smoothed[i] = np.sum(kernel_weights * yi_sorted) / np.sum(kernel_weights)

        # Functional mean: integrate over the curve
        mu = np.trapz(smoothed * precision_sorted, t) / np.trapz(precision_sorted, t)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class BSplineMeta(MetaAnalysisMethod):
    """B-spline smoothed meta-analysis"""

    def __init__(self, n_knots: int = 4):
        super().__init__(f"BSpline_{n_knots}knots", "functional")
        self.n_knots = n_knots

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Simple polynomial spline approximation
        precision = 1.0 / (vi + tau2)
        precision_norm = (precision - np.min(precision)) / (np.max(precision) - np.min(precision) + 1e-10)

        # Fit polynomial
        degree = min(self.n_knots - 1, k - 1)
        try:
            coeffs = np.polyfit(precision_norm, yi, degree, w=wi_re)
            fitted = np.polyval(coeffs, precision_norm)
            mu = np.sum(wi_re * fitted) / np.sum(wi_re)
        except:
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 28: GAME-THEORETIC METHODS
# =============================================================================

class ShapleyValueMeta(MetaAnalysisMethod):
    """Shapley value-based meta-analysis"""

    def __init__(self):
        super().__init__("ShapleyValue", "game_theory")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Shapley value: marginal contribution of each study
        # Simplified: contribution to reducing variance
        shapley_values = np.zeros(k)

        total_precision = np.sum(wi_re)
        for i in range(k):
            # Marginal contribution
            precision_without_i = total_precision - wi_re[i]
            if precision_without_i > 0:
                shapley_values[i] = wi_re[i] / total_precision
            else:
                shapley_values[i] = 1.0 / k

        shapley_values = shapley_values / np.sum(shapley_values)
        mu = np.sum(shapley_values * yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=shapley_values,
            computation_time=time.time() - start_time
        )


class NashEquilibriumMeta(MetaAnalysisMethod):
    """Nash equilibrium-inspired meta-analysis"""

    def __init__(self, iterations: int = 50):
        super().__init__(f"NashEquilibrium_{iterations}iter", "game_theory")
        self.iterations = iterations

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Nash-like dynamics: studies "compete" for influence
        weights = np.ones(k) / k

        for _ in range(self.iterations):
            # Current estimate
            mu = np.sum(weights * yi)

            # Update weights based on "payoff" (inverse squared error)
            payoffs = 1.0 / ((yi - mu)**2 + vi + tau2 + 1e-10)
            payoffs = payoffs * wi_re  # Scale by precision

            # Normalize
            weights = payoffs / np.sum(payoffs)

        mu = np.sum(weights * yi)
        se = np.sqrt(np.sum(weights**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=weights,
            computation_time=time.time() - start_time
        )


class CoreMeta(MetaAnalysisMethod):
    """Game-theoretic core-based meta-analysis"""

    def __init__(self):
        super().__init__("GameCore", "game_theory")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Core: stable allocation where no coalition can improve
        # Use nucleolus-like approach: minimize maximum "unhappiness"
        def coalition_value(subset_mask):
            if np.sum(subset_mask) == 0:
                return 0
            wi_sub = wi_re[subset_mask]
            yi_sub = yi[subset_mask]
            return 1.0 / np.sqrt(np.sum(wi_sub))  # Information gain

        # Approximate core allocation
        weights = wi_re.copy()

        # Adjust for coalition stability
        for i in range(k):
            # Check contribution to precision
            contribution = wi_re[i] / np.sum(wi_re)
            # Adjust based on how central the study is
            centrality = 1.0 / (1 + np.abs(yi[i] - mu_fe) / (np.std(yi) + 1e-10))
            weights[i] = wi_re[i] * (0.5 + 0.5 * centrality)

        weights = weights / np.sum(weights)
        mu = np.sum(weights * yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=weights,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 29: HYBRID METHODS
# =============================================================================

class HybridRobustBayesMeta(MetaAnalysisMethod):
    """Hybrid robust-Bayesian meta-analysis"""

    def __init__(self, robustness: float = 0.5):
        super().__init__(f"HybridRobustBayes_{robustness}", "hybrid")
        self.robustness = robustness

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Robust estimate (Huber-like)
        residuals = yi - np.median(yi)
        mad = np.median(np.abs(residuals)) * 1.4826
        huber_weights = np.where(np.abs(residuals) / (mad + 1e-10) <= 1.345,
                                 1.0, 1.345 / (np.abs(residuals) / (mad + 1e-10)))

        wi_robust = wi_re * huber_weights
        mu_robust = np.sum(wi_robust * yi) / np.sum(wi_robust)

        # Bayesian estimate (shrinkage toward prior)
        prior_mu = 0.0
        prior_precision = 0.1
        posterior_precision = np.sum(wi_re) + prior_precision
        mu_bayes = (np.sum(wi_re * yi) + prior_precision * prior_mu) / posterior_precision

        # Hybrid: weighted combination
        mu = self.robustness * mu_robust + (1 - self.robustness) * mu_bayes

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'mu_robust': mu_robust, 'mu_bayes': mu_bayes},
            computation_time=time.time() - start_time
        )


class AdaptiveHybridMeta(MetaAnalysisMethod):
    """Adaptive hybrid method selecting best approach"""

    def __init__(self):
        super().__init__("AdaptiveHybrid", "hybrid")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Compute multiple estimates
        estimates = {}

        # DL
        estimates['DL'] = np.sum(wi_re * yi) / np.sum(wi_re)

        # Median
        estimates['Median'] = np.median(yi)

        # Weighted median
        sorted_idx = np.argsort(yi)
        cumsum = np.cumsum(wi_re[sorted_idx])
        med_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        estimates['WMedian'] = yi[sorted_idx[min(med_idx, k-1)]]

        # Trimmed mean
        trim = max(1, k // 10)
        if 2 * trim < k:
            sorted_yi = np.sort(yi)
            estimates['Trimmed'] = np.mean(sorted_yi[trim:-trim])
        else:
            estimates['Trimmed'] = estimates['DL']

        # Adaptive selection based on heterogeneity
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        if I2 > 75:  # High heterogeneity
            # Use more robust methods
            weights = {'DL': 0.2, 'Median': 0.3, 'WMedian': 0.3, 'Trimmed': 0.2}
        elif I2 > 50:  # Moderate heterogeneity
            weights = {'DL': 0.4, 'Median': 0.2, 'WMedian': 0.2, 'Trimmed': 0.2}
        else:  # Low heterogeneity
            weights = {'DL': 0.7, 'Median': 0.1, 'WMedian': 0.1, 'Trimmed': 0.1}

        mu = sum(weights[name] * est for name, est in estimates.items())

        se = np.sqrt(1.0 / np.sum(wi_re))
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'component_estimates': estimates, 'weights': weights},
            computation_time=time.time() - start_time
        )


class ConsensusHybridMeta(MetaAnalysisMethod):
    """Consensus-based hybrid meta-analysis"""

    def __init__(self):
        super().__init__("ConsensusHybrid", "hybrid")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Multiple estimators
        estimates = []

        # DL
        estimates.append(np.sum(wi_re * yi) / np.sum(wi_re))

        # REML approximation
        def neg_reml(t2):
            if t2 < 0:
                return 1e10
            w = 1.0 / (vi + t2)
            m = np.sum(w * yi) / np.sum(w)
            return 0.5 * (np.sum(np.log(vi + t2)) + np.log(np.sum(w)) + np.sum(w * (yi - m)**2))

        result = optimize.minimize_scalar(neg_reml, bounds=(0, 10), method='bounded')
        tau2_reml = max(0, result.x)
        wi_reml = 1.0 / (vi + tau2_reml)
        estimates.append(np.sum(wi_reml * yi) / np.sum(wi_reml))

        # Median
        estimates.append(np.median(yi))

        # Harmonic mean of positive values
        yi_pos = np.abs(yi) + 0.01
        hm = 1.0 / np.mean(1.0 / yi_pos)
        if np.mean(yi) < 0:
            hm = -hm
        estimates.append(hm)

        # Consensus: weighted by how close to each other
        estimates = np.array(estimates)
        distances = np.abs(estimates.reshape(-1, 1) - estimates.reshape(1, -1))
        consensus_weights = 1.0 / (1 + np.sum(distances, axis=1))
        consensus_weights = consensus_weights / np.sum(consensus_weights)

        mu = np.sum(consensus_weights * estimates)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 30: SPECIAL PURPOSE METHODS
# =============================================================================

class SmallStudyAdjustedMeta(MetaAnalysisMethod):
    """Adjusted for small study effects"""

    def __init__(self, adjustment: str = "precision"):
        super().__init__(f"SmallStudyAdj_{adjustment}", "special")
        self.adjustment = adjustment

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Adjust for small study effects
        se_total = np.sqrt(vi + tau2)

        if self.adjustment == "precision":
            # Weight smaller studies less
            precision_rank = stats.rankdata(wi_re) / k
            adjustment_weights = precision_rank
        elif self.adjustment == "egger":
            # Egger-style adjustment
            intercept = np.sum(wi_re * yi * se_total) / np.sum(wi_re * se_total)
            slope = (np.sum(wi_re * yi) / np.sum(wi_re) - intercept) / np.mean(se_total)
            adjustment_weights = 1.0 / (1 + np.abs(yi - (intercept + slope * se_total)))
        else:
            adjustment_weights = np.ones(k)

        wi_adj = wi_re * adjustment_weights
        wi_adj = wi_adj / np.sum(wi_adj) * np.sum(wi_re)

        mu = np.sum(wi_adj * yi) / np.sum(wi_adj)
        se = np.sqrt(1.0 / np.sum(wi_adj))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_adj,
            computation_time=time.time() - start_time
        )


class InfluenceReducedMeta(MetaAnalysisMethod):
    """Influence-reduced meta-analysis"""

    def __init__(self, threshold: float = 2.0):
        super().__init__(f"InfluenceReduced_th{threshold}", "special")
        self.threshold = threshold

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu = np.sum(wi_re * yi) / np.sum(wi_re)

        # Cook's distance-like influence
        influence = np.zeros(k)
        for i in range(k):
            wi_loo = np.delete(wi_re, i)
            yi_loo = np.delete(yi, i)
            mu_loo = np.sum(wi_loo * yi_loo) / np.sum(wi_loo)
            influence[i] = (mu - mu_loo)**2 * np.sum(wi_re)

        # Reduce influence of high-influence studies
        influence_threshold = np.median(influence) + self.threshold * np.std(influence)
        reduction_factor = np.where(influence > influence_threshold,
                                   influence_threshold / (influence + 1e-10),
                                   1.0)

        wi_reduced = wi_re * reduction_factor
        mu_reduced = np.sum(wi_reduced * yi) / np.sum(wi_reduced)

        se = np.sqrt(1.0 / np.sum(wi_reduced))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_reduced, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_reduced,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_reduced,
            additional_info={'high_influence_studies': int(np.sum(influence > influence_threshold))},
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 31: CROSS-VALIDATION METHODS
# =============================================================================

class LOOCVMeta(MetaAnalysisMethod):
    """Leave-one-out cross-validation meta-analysis"""

    def __init__(self, combination: str = "mean"):
        super().__init__(f"LOOCV_{combination}", "cross_validation")
        self.combination = combination

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # LOO estimates
        loo_estimates = []
        for i in range(k):
            wi_loo = np.delete(wi_re, i)
            yi_loo = np.delete(yi, i)
            mu_loo = np.sum(wi_loo * yi_loo) / np.sum(wi_loo)
            loo_estimates.append(mu_loo)

        loo_estimates = np.array(loo_estimates)

        if self.combination == "mean":
            mu = np.mean(loo_estimates)
        elif self.combination == "median":
            mu = np.median(loo_estimates)
        elif self.combination == "weighted":
            mu = np.sum(wi_re * loo_estimates) / np.sum(wi_re)
        else:
            mu = np.mean(loo_estimates)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class KFoldCVMeta(MetaAnalysisMethod):
    """K-fold cross-validation meta-analysis"""

    def __init__(self, k_folds: int = 5):
        super().__init__(f"KFoldCV_{k_folds}fold", "cross_validation")
        self.k_folds = k_folds

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # K-fold estimates
        fold_size = max(1, k // self.k_folds)
        fold_estimates = []

        indices = np.arange(k)
        np.random.shuffle(indices)

        for fold in range(self.k_folds):
            start_idx = fold * fold_size
            end_idx = min((fold + 1) * fold_size, k)
            test_idx = indices[start_idx:end_idx]
            train_idx = np.setdiff1d(indices, test_idx)

            if len(train_idx) > 0:
                wi_train = wi_re[train_idx]
                yi_train = yi[train_idx]
                mu_fold = np.sum(wi_train * yi_train) / np.sum(wi_train)
                fold_estimates.append(mu_fold)

        mu = np.mean(fold_estimates) if fold_estimates else np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 32: ORDER STATISTIC METHODS
# =============================================================================

class OrderStatisticMeta(MetaAnalysisMethod):
    """Order statistic-based meta-analysis"""

    def __init__(self, quantile: float = 0.5):
        super().__init__(f"OrderStat_q{quantile}", "order_statistics")
        self.quantile = quantile

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Order statistic
        sorted_idx = np.argsort(yi)
        target_idx = int(self.quantile * (k - 1))
        mu = yi[sorted_idx[target_idx]]

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class MidrangeMeta(MetaAnalysisMethod):
    """Midrange meta-analysis"""

    def __init__(self):
        super().__init__("Midrange", "order_statistics")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Midrange
        mu = (np.max(yi) + np.min(yi)) / 2

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class InterquartileMeanMeta(MetaAnalysisMethod):
    """Interquartile mean meta-analysis"""

    def __init__(self):
        super().__init__("InterquartileMean", "order_statistics")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Interquartile mean
        sorted_yi = np.sort(yi)
        q1_idx = int(0.25 * k)
        q3_idx = int(0.75 * k)
        if q3_idx > q1_idx:
            mu = np.mean(sorted_yi[q1_idx:q3_idx])
        else:
            mu = np.median(yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 33: SIGNAL PROCESSING METHODS
# =============================================================================

class MovingAverageMeta(MetaAnalysisMethod):
    """Moving average meta-analysis"""

    def __init__(self, window: int = 3):
        super().__init__(f"MovingAvg_w{window}", "signal_processing")
        self.window = window

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Sort by precision
        sorted_idx = np.argsort(wi_re)[::-1]
        yi_sorted = yi[sorted_idx]

        # Moving average
        window = min(self.window, k)
        smoothed = np.convolve(yi_sorted, np.ones(window)/window, mode='valid')
        mu = np.mean(smoothed)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class ExponentialSmoothingMeta(MetaAnalysisMethod):
    """Exponential smoothing meta-analysis"""

    def __init__(self, alpha: float = 0.3):
        super().__init__(f"ExpSmooth_a{alpha}", "signal_processing")
        self.alpha = alpha

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Sort by precision
        sorted_idx = np.argsort(wi_re)[::-1]
        yi_sorted = yi[sorted_idx]

        # Exponential smoothing
        smoothed = np.zeros(k)
        smoothed[0] = yi_sorted[0]
        for i in range(1, k):
            smoothed[i] = self.alpha * yi_sorted[i] + (1 - self.alpha) * smoothed[i-1]

        mu = smoothed[-1]

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class MedianFilterMeta(MetaAnalysisMethod):
    """Median filter meta-analysis"""

    def __init__(self, window: int = 3):
        super().__init__(f"MedianFilter_w{window}", "signal_processing")
        self.window = window

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Sort by precision
        sorted_idx = np.argsort(wi_re)[::-1]
        yi_sorted = yi[sorted_idx]

        # Median filter
        try:
            filtered = signal.medfilt(yi_sorted, min(self.window, k) | 1)  # Must be odd
            mu = np.mean(filtered)
        except:
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 34: HIERARCHICAL CLUSTERING METHODS
# =============================================================================

class ClusterWeightedMeta(MetaAnalysisMethod):
    """Cluster-weighted meta-analysis"""

    def __init__(self, n_clusters: int = 3):
        super().__init__(f"ClusterWeighted_{n_clusters}c", "clustering")
        self.n_clusters = n_clusters

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Simple k-means-like clustering
        n_clust = min(self.n_clusters, k)
        sorted_idx = np.argsort(yi)
        cluster_size = k // n_clust

        cluster_means = []
        cluster_weights = []

        for c in range(n_clust):
            start = c * cluster_size
            end = (c + 1) * cluster_size if c < n_clust - 1 else k
            idx = sorted_idx[start:end]

            wi_c = wi_re[idx]
            yi_c = yi[idx]
            mu_c = np.sum(wi_c * yi_c) / np.sum(wi_c)
            cluster_means.append(mu_c)
            cluster_weights.append(np.sum(wi_c))

        cluster_means = np.array(cluster_means)
        cluster_weights = np.array(cluster_weights)
        cluster_weights = cluster_weights / np.sum(cluster_weights)

        mu = np.sum(cluster_weights * cluster_means)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class OutlierClusterMeta(MetaAnalysisMethod):
    """Outlier-aware cluster meta-analysis"""

    def __init__(self, outlier_threshold: float = 2.0):
        super().__init__(f"OutlierCluster_th{outlier_threshold}", "clustering")
        self.threshold = outlier_threshold

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Identify outliers
        med = np.median(yi)
        mad = np.median(np.abs(yi - med)) * 1.4826
        z_scores = np.abs(yi - med) / (mad + 1e-10)

        inliers = z_scores <= self.threshold

        if np.sum(inliers) >= 2:
            wi_inlier = wi_re[inliers]
            yi_inlier = yi[inliers]
            mu = np.sum(wi_inlier * yi_inlier) / np.sum(wi_inlier)
        else:
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'n_outliers': int(np.sum(~inliers))},
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 35: MODIFIED TAU2 ESTIMATORS
# =============================================================================

class ModifiedREMLMeta(MetaAnalysisMethod):
    """Modified REML estimator"""

    def __init__(self, correction: str = "small_sample"):
        super().__init__(f"ModifiedREML_{correction}", "tau2_estimator")
        self.correction = correction

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # REML tau2
        def neg_reml(t2):
            if t2 < 0:
                return 1e10
            w = 1.0 / (vi + t2)
            m = np.sum(w * yi) / np.sum(w)
            return 0.5 * (np.sum(np.log(vi + t2)) + np.log(np.sum(w)) + np.sum(w * (yi - m)**2))

        result = optimize.minimize_scalar(neg_reml, bounds=(0, 10), method='bounded')
        tau2 = max(0, result.x)

        # Apply correction
        if self.correction == "small_sample" and k <= 10:
            tau2 = tau2 * (k + 2) / k  # Bias correction for small samples
        elif self.correction == "conservative":
            tau2 = tau2 * 1.1  # Conservative adjustment

        wi_re = 1.0 / (vi + tau2)
        mu = np.sum(wi_re * yi) / np.sum(wi_re)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class RobustTau2Meta(MetaAnalysisMethod):
    """Robust tau2 estimator"""

    def __init__(self, method: str = "median"):
        super().__init__(f"RobustTau2_{method}", "tau2_estimator")
        self.method = method

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Robust tau2 estimate
        med = np.median(yi)
        residuals = (yi - med)**2 - vi

        if self.method == "median":
            tau2 = max(0, np.median(residuals))
        elif self.method == "trimmed":
            trim = max(1, k // 10)
            sorted_res = np.sort(residuals)
            tau2 = max(0, np.mean(sorted_res[trim:-trim]) if 2*trim < k else np.median(residuals))
        else:
            tau2 = max(0, np.median(residuals))

        wi_re = 1.0 / (vi + tau2)
        mu = np.sum(wi_re * yi) / np.sum(wi_re)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 36: MOMENT-BASED METHODS
# =============================================================================

class MomentMatchingMeta(MetaAnalysisMethod):
    """Moment matching meta-analysis"""

    def __init__(self, moments: int = 2):
        super().__init__(f"MomentMatch_{moments}m", "moment")
        self.moments = moments

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # First moment: mean
        m1 = np.sum(wi_re * yi) / np.sum(wi_re)

        if self.moments >= 2:
            # Second moment: variance-adjusted
            m2 = np.sum(wi_re * yi**2) / np.sum(wi_re)
            var_estimate = m2 - m1**2

            # Combine with mean
            mu = m1 * (1 - 0.1 * np.sign(var_estimate - tau2))
        else:
            mu = m1

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class LMomentMeta(MetaAnalysisMethod):
    """L-moment meta-analysis"""

    def __init__(self):
        super().__init__("LMoment", "moment")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # L-moments
        sorted_yi = np.sort(yi)
        n = k

        # L1 (mean)
        L1 = np.mean(sorted_yi)

        # L2 (scale)
        if k >= 2:
            b0 = np.mean(sorted_yi)
            b1 = np.sum(np.arange(k) * sorted_yi) / (k * (k - 1))
            L2 = 2 * b1 - b0
        else:
            L2 = 0

        # Use L1 as estimate (robust location)
        mu = L1

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 37: DEPTH-BASED METHODS
# =============================================================================

class TukeyDepthMeta(MetaAnalysisMethod):
    """Tukey depth-based meta-analysis"""

    def __init__(self):
        super().__init__("TukeyDepth", "depth")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Tukey depth: proportion of data on each side
        depths = np.zeros(k)
        for i in range(k):
            left = np.sum(yi < yi[i])
            right = np.sum(yi > yi[i])
            depths[i] = min(left, right) / k + 0.5 / k

        # Use depth-weighted mean
        depth_weights = depths / np.sum(depths)
        mu = np.sum(depth_weights * yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class ZonoidDepthMeta(MetaAnalysisMethod):
    """Zonoid depth-based meta-analysis"""

    def __init__(self):
        super().__init__("ZonoidDepth", "depth")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Zonoid depth approximation
        center = np.mean(yi)
        spread = np.std(yi)

        if spread > 0:
            z_scores = np.abs(yi - center) / spread
            depths = 1.0 / (1 + z_scores)
        else:
            depths = np.ones(k)

        depth_weights = depths * wi_re
        depth_weights = depth_weights / np.sum(depth_weights)
        mu = np.sum(depth_weights * yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


# =============================================================================
# COLLECT ALL METHODS FROM THIS MODULE
# =============================================================================

def get_part5_methods():
    """Return all experimental methods from this module"""
    methods = []

    # Wavelet methods
    for thresh in ["soft", "hard"]:
        methods.append(WaveletDenoisingMeta(threshold=thresh))
    for levels in [2, 3, 4]:
        methods.append(MultiResolutionMeta(levels=levels))

    # Functional methods
    for smooth in [0.05, 0.1, 0.2]:
        methods.append(FunctionalMeanMeta(smoothing=smooth))
    for knots in [3, 4, 5]:
        methods.append(BSplineMeta(n_knots=knots))

    # Game-theoretic methods
    methods.append(ShapleyValueMeta())
    for iters in [30, 50, 100]:
        methods.append(NashEquilibriumMeta(iterations=iters))
    methods.append(CoreMeta())

    # Hybrid methods
    for robust in [0.3, 0.5, 0.7]:
        methods.append(HybridRobustBayesMeta(robustness=robust))
    methods.append(AdaptiveHybridMeta())
    methods.append(ConsensusHybridMeta())

    # Special purpose
    for adj in ["precision", "egger"]:
        methods.append(SmallStudyAdjustedMeta(adjustment=adj))
    for th in [1.5, 2.0, 2.5]:
        methods.append(InfluenceReducedMeta(threshold=th))

    # Cross-validation methods (Category 31)
    for comb in ["mean", "median", "weighted"]:
        methods.append(LOOCVMeta(combination=comb))
    for folds in [3, 5, 10]:
        methods.append(KFoldCVMeta(k_folds=folds))

    # Order statistic methods (Category 32)
    for q in [0.25, 0.5, 0.75]:
        methods.append(OrderStatisticMeta(quantile=q))
    methods.append(MidrangeMeta())
    methods.append(InterquartileMeanMeta())

    # Signal processing methods (Category 33)
    for w in [3, 5, 7]:
        methods.append(MovingAverageMeta(window=w))
    for a in [0.2, 0.3, 0.5]:
        methods.append(ExponentialSmoothingMeta(alpha=a))
    for w in [3, 5]:
        methods.append(MedianFilterMeta(window=w))

    # Clustering methods (Category 34)
    for c in [2, 3, 4]:
        methods.append(ClusterWeightedMeta(n_clusters=c))
    for th in [1.5, 2.0, 2.5]:
        methods.append(OutlierClusterMeta(outlier_threshold=th))

    # Modified tau2 estimators (Category 35)
    for corr in ["small_sample", "conservative"]:
        methods.append(ModifiedREMLMeta(correction=corr))
    for meth in ["median", "trimmed"]:
        methods.append(RobustTau2Meta(method=meth))

    # Moment-based methods (Category 36)
    for m in [1, 2, 3]:
        methods.append(MomentMatchingMeta(moments=m))
    methods.append(LMomentMeta())

    # Depth-based methods (Category 37)
    methods.append(TukeyDepthMeta())
    methods.append(ZonoidDepthMeta())

    return methods


if __name__ == "__main__":
    methods = get_part5_methods()
    print(f"Part 5 contains {len(methods)} experimental methods")
    for m in methods:
        print(f"  - {m.name} ({m.category})")

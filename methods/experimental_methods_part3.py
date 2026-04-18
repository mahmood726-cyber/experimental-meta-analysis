"""
Experimental Meta-Analysis Methods - Part 3
============================================
Novel approaches: Ensemble, Neural-inspired, Copula, Extreme Value, Density
"""

import numpy as np
from scipy import stats, optimize, special
from scipy.integrate import quad
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
# CATEGORY 15: ENSEMBLE METHODS
# =============================================================================

class EnsembleStackingMeta(MetaAnalysisMethod):
    """Stacking ensemble of multiple meta-analysis methods"""

    def __init__(self, base_methods: List[str] = ["DL", "REML", "PM"]):
        super().__init__(f"EnsembleStacking_{len(base_methods)}", "ensemble")
        self.base_methods = base_methods

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        estimates = []
        se_estimates = []

        # DL estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2_dl = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_dl = 1.0 / (vi + tau2_dl)
        mu_dl = np.sum(wi_dl * yi) / np.sum(wi_dl)
        se_dl = np.sqrt(1.0 / np.sum(wi_dl))
        estimates.append(mu_dl)
        se_estimates.append(se_dl)

        # REML estimate
        def neg_reml(tau2):
            if tau2 < 0:
                return 1e10
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)
            ll = -0.5 * (np.sum(np.log(vi + tau2)) + np.log(np.sum(wi)) +
                        np.sum(wi * (yi - mu)**2))
            return -ll

        result = optimize.minimize_scalar(neg_reml, bounds=(0, 10), method='bounded')
        tau2_reml = max(0, result.x)
        wi_reml = 1.0 / (vi + tau2_reml)
        mu_reml = np.sum(wi_reml * yi) / np.sum(wi_reml)
        se_reml = np.sqrt(1.0 / np.sum(wi_reml))
        estimates.append(mu_reml)
        se_estimates.append(se_reml)

        # PM estimate
        def pm_eq(tau2):
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)
            return np.sum(wi * (yi - mu)**2) - (k - 1)

        try:
            if pm_eq(0) <= 0:
                tau2_pm = 0.0
            else:
                tau2_pm = optimize.brentq(pm_eq, 0, 100)
        except:
            tau2_pm = tau2_dl

        wi_pm = 1.0 / (vi + tau2_pm)
        mu_pm = np.sum(wi_pm * yi) / np.sum(wi_pm)
        se_pm = np.sqrt(1.0 / np.sum(wi_pm))
        estimates.append(mu_pm)
        se_estimates.append(se_pm)

        estimates = np.array(estimates)
        se_estimates = np.array(se_estimates)

        # Stack: inverse-variance weighted combination
        stack_weights = 1.0 / se_estimates**2
        stack_weights = stack_weights / np.sum(stack_weights)

        mu_stack = np.sum(stack_weights * estimates)
        se_stack = np.sqrt(np.sum(stack_weights**2 * se_estimates**2))

        tau2 = np.mean([tau2_dl, tau2_reml, tau2_pm])
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_stack, se_stack)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_stack,
            pooled_se=se_stack,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'base_estimates': estimates.tolist()},
            computation_time=time.time() - start_time
        )


class BaggingMeta(MetaAnalysisMethod):
    """Bootstrap aggregating meta-analysis"""

    def __init__(self, n_estimators: int = 50):
        super().__init__(f"Bagging_{n_estimators}", "ensemble")
        self.n_estimators = n_estimators

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        bag_estimates = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(k, k, replace=True)
            yi_b = yi[idx]
            vi_b = vi[idx]

            # DL on bootstrap sample
            wi_b = 1.0 / vi_b
            mu_fe_b = np.sum(wi_b * yi_b) / np.sum(wi_b)
            Q_b = np.sum(wi_b * (yi_b - mu_fe_b)**2)
            c_b = np.sum(wi_b) - np.sum(wi_b**2) / np.sum(wi_b)
            tau2_b = max(0, (Q_b - (k - 1)) / c_b) if c_b > 0 else 0.0

            wi_re_b = 1.0 / (vi_b + tau2_b)
            mu_b = np.sum(wi_re_b * yi_b) / np.sum(wi_re_b)
            bag_estimates.append(mu_b)

        bag_estimates = np.array(bag_estimates)
        mu_bag = np.mean(bag_estimates)
        se_bag = np.std(bag_estimates)

        # Original tau2 for reference
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_bag, se_bag)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_bag,
            pooled_se=se_bag,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class RandomSubspaceMeta(MetaAnalysisMethod):
    """Random subspace ensemble"""

    def __init__(self, n_estimators: int = 50, subsample_frac: float = 0.7):
        super().__init__(f"RandomSubspace_{n_estimators}_{subsample_frac}", "ensemble")
        self.n_estimators = n_estimators
        self.subsample_frac = subsample_frac

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        n_sub = max(3, int(k * self.subsample_frac))
        sub_estimates = []

        for _ in range(self.n_estimators):
            # Random subset
            idx = np.random.choice(k, n_sub, replace=False)
            yi_s = yi[idx]
            vi_s = vi[idx]

            # DL on subset
            wi_s = 1.0 / vi_s
            mu_fe_s = np.sum(wi_s * yi_s) / np.sum(wi_s)
            Q_s = np.sum(wi_s * (yi_s - mu_fe_s)**2)
            c_s = np.sum(wi_s) - np.sum(wi_s**2) / np.sum(wi_s)
            tau2_s = max(0, (Q_s - (n_sub - 1)) / c_s) if c_s > 0 else 0.0

            wi_re_s = 1.0 / (vi_s + tau2_s)
            mu_s = np.sum(wi_re_s * yi_s) / np.sum(wi_re_s)
            sub_estimates.append(mu_s)

        sub_estimates = np.array(sub_estimates)
        mu_rs = np.mean(sub_estimates)
        se_rs = np.std(sub_estimates) / np.sqrt(self.n_estimators)

        # Original tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_rs, se_rs)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_rs,
            pooled_se=se_rs,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class BoostingMeta(MetaAnalysisMethod):
    """Gradient boosting-inspired meta-analysis"""

    def __init__(self, n_rounds: int = 10, learning_rate: float = 0.1):
        super().__init__(f"Boosting_{n_rounds}_lr{learning_rate}", "ensemble")
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initialize with weighted mean
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)

        # DL tau2
        mu_fe = mu
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Boosting rounds
        for _ in range(self.n_rounds):
            # Compute residuals
            residuals = yi - mu

            # Fit to residuals (weighted mean of residuals)
            increment = np.sum(wi_re * residuals) / np.sum(wi_re)

            # Update
            mu = mu + self.learning_rate * increment

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


class MedianEnsembleMeta(MetaAnalysisMethod):
    """Median ensemble of multiple estimators"""

    def __init__(self):
        super().__init__("MedianEnsemble", "ensemble")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        estimates = []

        # DL
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0
        wi_re = 1.0 / (vi + tau2)
        estimates.append(np.sum(wi_re * yi) / np.sum(wi_re))

        # Fixed effect
        estimates.append(mu_fe)

        # Median
        estimates.append(np.median(yi))

        # Trimmed mean
        sorted_yi = np.sort(yi)
        trim = max(1, k // 10)
        if 2 * trim < k:
            estimates.append(np.mean(sorted_yi[trim:-trim]))

        # Weighted median
        sorted_idx = np.argsort(yi)
        cumsum = np.cumsum(wi_re[sorted_idx])
        med_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        estimates.append(yi[sorted_idx[min(med_idx, k-1)]])

        mu_median = np.median(estimates)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_median, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_median,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'component_estimates': estimates},
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 16: NEURAL-INSPIRED METHODS
# =============================================================================

class SigmoidWeightedMeta(MetaAnalysisMethod):
    """Sigmoid activation weighted meta-analysis"""

    def __init__(self, temperature: float = 1.0):
        super().__init__(f"SigmoidWeighted_t{temperature}", "neural")
        self.temperature = temperature

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

        # Sigmoid activation on precision
        precision = 1.0 / (vi + tau2)
        precision_norm = (precision - np.mean(precision)) / (np.std(precision) + 1e-10)

        sigmoid_weights = 1.0 / (1 + np.exp(-precision_norm / self.temperature))
        sigmoid_weights = sigmoid_weights / np.sum(sigmoid_weights)

        mu_sig = np.sum(sigmoid_weights * yi)
        se = np.sqrt(np.sum(sigmoid_weights**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_sig, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_sig,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=sigmoid_weights,
            computation_time=time.time() - start_time
        )


class ReLUWeightedMeta(MetaAnalysisMethod):
    """ReLU-inspired sparse weighting"""

    def __init__(self, threshold: float = 0.0):
        super().__init__(f"ReLUWeighted_th{threshold}", "neural")
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

        # ReLU on standardized weights
        wi_std = (wi_re - np.mean(wi_re)) / (np.std(wi_re) + 1e-10)
        relu_weights = np.maximum(0, wi_std - self.threshold)

        # Ensure we have some weights
        if np.sum(relu_weights) < 1e-10:
            relu_weights = wi_re

        relu_weights = relu_weights / np.sum(relu_weights)
        mu_relu = np.sum(relu_weights * yi)
        se = np.sqrt(np.sum(relu_weights**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_relu, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_relu,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=relu_weights,
            computation_time=time.time() - start_time
        )


class AttentionWeightedMeta(MetaAnalysisMethod):
    """Attention mechanism inspired weighting"""

    def __init__(self, n_heads: int = 4):
        super().__init__(f"Attention_{n_heads}heads", "neural")
        self.n_heads = n_heads

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

        # Multi-head attention simulation
        head_outputs = []

        for h in range(self.n_heads):
            # Different "queries" based on different aspects
            if h == 0:
                query = yi  # Effect size similarity
            elif h == 1:
                query = 1.0 / vi  # Precision similarity
            elif h == 2:
                query = yi * np.sqrt(wi_re)  # Weighted effect
            else:
                query = np.abs(yi - mu_fe)  # Deviation from mean

            # Attention scores (cosine-like similarity)
            scores = query / (np.linalg.norm(query) + 1e-10)
            scores = np.abs(scores)

            # Softmax
            scores = np.exp(scores - np.max(scores))
            attn_weights = scores / np.sum(scores)

            # Weighted sum
            head_outputs.append(np.sum(attn_weights * yi))

        mu_attn = np.mean(head_outputs)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_attn, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_attn,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'head_outputs': head_outputs},
            computation_time=time.time() - start_time
        )


class HopfieldMeta(MetaAnalysisMethod):
    """Hopfield network-inspired associative pooling"""

    def __init__(self, n_iterations: int = 10):
        super().__init__(f"Hopfield_{n_iterations}iter", "neural")
        self.n_iterations = n_iterations

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

        # Hopfield-style associative memory
        # Pattern matrix: studies are patterns, yi are values
        patterns = yi.reshape(-1, 1)
        weights_matrix = wi_re.reshape(-1, 1)

        # Initialize state with weighted mean
        state = mu_fe

        for _ in range(self.n_iterations):
            # Compute similarities
            similarities = np.exp(-0.5 * ((patterns.flatten() - state)**2) * wi_re)
            similarities = similarities / np.sum(similarities)

            # Update state
            state_new = np.sum(similarities * yi)

            if abs(state_new - state) < 1e-8:
                break
            state = state_new

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(state, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=state,
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
# CATEGORY 17: COPULA-BASED METHODS
# =============================================================================

class GaussianCopulaMeta(MetaAnalysisMethod):
    """Gaussian copula meta-analysis"""

    def __init__(self):
        super().__init__("GaussianCopula", "copula")

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

        # Transform to uniform via CDF
        se_total = np.sqrt(vi + tau2)
        u = stats.norm.cdf(yi, mu_fe, se_total)
        u = np.clip(u, 0.001, 0.999)

        # Transform to standard normal
        z = stats.norm.ppf(u)

        # Estimate correlation
        rho = np.corrcoef(z[:-1], z[1:])[0, 1] if k > 1 else 0

        # Adjust weights based on dependence
        wi_re = 1.0 / (vi + tau2)
        dependence_adj = 1.0 - np.abs(rho) * 0.5
        wi_adj = wi_re * dependence_adj

        mu_cop = np.sum(wi_adj * yi) / np.sum(wi_adj)
        se_cop = np.sqrt(1.0 / np.sum(wi_adj))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_cop, se_cop)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_cop,
            pooled_se=se_cop,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_adj,
            additional_info={'estimated_correlation': rho},
            computation_time=time.time() - start_time
        )


class ClaytonCopulaMeta(MetaAnalysisMethod):
    """Clayton copula meta-analysis (lower tail dependence)"""

    def __init__(self, theta: float = 2.0):
        super().__init__(f"ClaytonCopula_th{theta}", "copula")
        self.theta = theta

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

        # Clayton copula gives higher weight to lower tail
        # Transform effect sizes to pseudo-uniform
        ranks = stats.rankdata(yi) / (k + 1)

        # Clayton-inspired weights: emphasize concordance
        theta = self.theta
        clayton_weights = (1 + theta * np.log(ranks + 0.01))**(-1/theta - 1)
        clayton_weights = np.clip(clayton_weights, 0.1, 10)
        clayton_weights = clayton_weights / np.sum(clayton_weights)

        # Combine with inverse variance
        wi_combined = np.sqrt(clayton_weights * wi_re)
        wi_combined = wi_combined / np.sum(wi_combined)

        mu_clay = np.sum(wi_combined * yi)
        se_clay = np.sqrt(np.sum(wi_combined**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_clay, se_clay)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_clay,
            pooled_se=se_clay,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_combined,
            computation_time=time.time() - start_time
        )


class FrankCopulaMeta(MetaAnalysisMethod):
    """Frank copula meta-analysis (symmetric dependence)"""

    def __init__(self, theta: float = 5.0):
        super().__init__(f"FrankCopula_th{theta}", "copula")
        self.theta = theta

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

        # Frank copula weights
        ranks = stats.rankdata(yi) / (k + 1)
        theta = self.theta

        # Frank generator: log((exp(-theta*t) - 1)/(exp(-theta) - 1))
        exp_term = np.exp(-theta * ranks)
        frank_weights = theta * exp_term / (1 - exp_term + 1e-10)
        frank_weights = np.abs(frank_weights)
        frank_weights = frank_weights / np.sum(frank_weights)

        # Combine
        wi_combined = np.sqrt(frank_weights * wi_re)
        wi_combined = wi_combined / np.sum(wi_combined)

        mu_frank = np.sum(wi_combined * yi)
        se_frank = np.sqrt(np.sum(wi_combined**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_frank, se_frank)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_frank,
            pooled_se=se_frank,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_combined,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 18: EXTREME VALUE METHODS
# =============================================================================

class GEVMeta(MetaAnalysisMethod):
    """Generalized Extreme Value meta-analysis"""

    def __init__(self, focus: str = "max"):
        super().__init__(f"GEV_{focus}", "extreme_value")
        self.focus = focus

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

        # GEV-inspired: focus on extreme studies
        if self.focus == "max":
            # Weight toward larger effects
            z_scores = (yi - np.min(yi)) / (np.max(yi) - np.min(yi) + 1e-10)
        else:
            # Weight toward smaller effects
            z_scores = (np.max(yi) - yi) / (np.max(yi) - np.min(yi) + 1e-10)

        # Gumbel-type weighting
        gev_weights = np.exp(-np.exp(-z_scores * 2))
        gev_weights = gev_weights / np.sum(gev_weights)

        # Combine
        wi_combined = np.sqrt(gev_weights * wi_re)
        wi_combined = wi_combined / np.sum(wi_combined)

        mu_gev = np.sum(wi_combined * yi)
        se_gev = np.sqrt(np.sum(wi_combined**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_gev, se_gev)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_gev,
            pooled_se=se_gev,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_combined,
            computation_time=time.time() - start_time
        )


class ParetoMeta(MetaAnalysisMethod):
    """Pareto distribution-based meta-analysis"""

    def __init__(self, alpha: float = 2.0):
        super().__init__(f"Pareto_a{alpha}", "extreme_value")
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

        # Pareto weights based on effect size rank
        ranks = stats.rankdata(yi)
        pareto_weights = (k / ranks) ** self.alpha
        pareto_weights = pareto_weights / np.sum(pareto_weights)

        # Combine
        wi_combined = np.sqrt(pareto_weights * wi_re)
        wi_combined = wi_combined / np.sum(wi_combined)

        mu_pareto = np.sum(wi_combined * yi)
        se_pareto = np.sqrt(np.sum(wi_combined**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_pareto, se_pareto)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pareto,
            pooled_se=se_pareto,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_combined,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 19: DENSITY ESTIMATION METHODS
# =============================================================================

class KDEModeMeta(MetaAnalysisMethod):
    """Kernel density estimation mode meta-analysis"""

    def __init__(self, bandwidth: str = "silverman"):
        super().__init__(f"KDEMode_{bandwidth}", "density")
        self.bandwidth = bandwidth

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

        # Bandwidth selection
        if self.bandwidth == "silverman":
            h = 1.06 * np.std(yi) * k**(-0.2)
        elif self.bandwidth == "scott":
            h = 1.059 * np.std(yi) * k**(-0.2)
        else:
            h = np.std(yi) / 2
        h = max(h, 0.01)

        # KDE
        grid = np.linspace(np.min(yi) - 3*h, np.max(yi) + 3*h, 500)
        density = np.zeros(len(grid))

        for i in range(k):
            kernel = stats.norm.pdf(grid, yi[i], h) * wi_re[i]
            density += kernel

        density /= np.sum(wi_re)

        # Find mode
        mode_idx = np.argmax(density)
        mu_mode = grid[mode_idx]

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_mode, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_mode,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'bandwidth': h},
            computation_time=time.time() - start_time
        )


class MixtureKDEMeta(MetaAnalysisMethod):
    """Mixture of KDEs meta-analysis"""

    def __init__(self, n_components: int = 2):
        super().__init__(f"MixtureKDE_{n_components}", "density")
        self.n_components = n_components

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

        # Initialize components using percentiles
        n_comp = min(self.n_components, k)
        percentiles = np.linspace(0, 100, n_comp + 2)[1:-1]
        mu_comp = np.percentile(yi, percentiles)
        sigma_comp = np.ones(n_comp) * np.std(yi) / n_comp
        pi_comp = np.ones(n_comp) / n_comp

        # EM for mixture KDE
        for _ in range(50):
            # E-step
            resp = np.zeros((k, n_comp))
            for j in range(n_comp):
                resp[:, j] = pi_comp[j] * stats.norm.pdf(yi, mu_comp[j], sigma_comp[j])
            resp = resp / (np.sum(resp, axis=1, keepdims=True) + 1e-10)

            # M-step
            n_j = np.sum(resp, axis=0) + 1e-10
            pi_comp = n_j / k

            for j in range(n_comp):
                mu_comp[j] = np.sum(resp[:, j] * wi_re * yi) / (np.sum(resp[:, j] * wi_re) + 1e-10)
                var_j = np.sum(resp[:, j] * wi_re * (yi - mu_comp[j])**2) / (np.sum(resp[:, j] * wi_re) + 1e-10)
                sigma_comp[j] = np.sqrt(max(var_j, 0.01))

        # Main component
        main_idx = np.argmax(pi_comp)
        mu_mix = mu_comp[main_idx]

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_mix, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_mix,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'component_means': mu_comp.tolist(), 'component_probs': pi_comp.tolist()},
            computation_time=time.time() - start_time
        )


# =============================================================================
# COLLECT ALL METHODS FROM THIS MODULE
# =============================================================================

def get_part3_methods():
    """Return all experimental methods from this module"""
    methods = []

    # Ensemble methods
    methods.append(EnsembleStackingMeta())
    for n in [20, 50, 100]:
        methods.append(BaggingMeta(n_estimators=n))
    for n in [30, 50]:
        for frac in [0.6, 0.7, 0.8]:
            methods.append(RandomSubspaceMeta(n_estimators=n, subsample_frac=frac))
    for n in [5, 10, 20]:
        for lr in [0.05, 0.1, 0.2]:
            methods.append(BoostingMeta(n_rounds=n, learning_rate=lr))
    methods.append(MedianEnsembleMeta())

    # Neural-inspired
    for t in [0.5, 1.0, 2.0]:
        methods.append(SigmoidWeightedMeta(temperature=t))
    for th in [-0.5, 0.0, 0.5]:
        methods.append(ReLUWeightedMeta(threshold=th))
    for nh in [2, 4, 8]:
        methods.append(AttentionWeightedMeta(n_heads=nh))
    for ni in [5, 10, 20]:
        methods.append(HopfieldMeta(n_iterations=ni))

    # Copula-based
    methods.append(GaussianCopulaMeta())
    for th in [1.0, 2.0, 5.0]:
        methods.append(ClaytonCopulaMeta(theta=th))
        methods.append(FrankCopulaMeta(theta=th))

    # Extreme value
    for focus in ["max", "min"]:
        methods.append(GEVMeta(focus=focus))
    for alpha in [1.5, 2.0, 3.0]:
        methods.append(ParetoMeta(alpha=alpha))

    # Density estimation (expanded)
    for bw in ["silverman", "scott"]:
        methods.append(KDEModeMeta(bandwidth=bw))
    for nc in [2, 3, 4, 5]:
        methods.append(MixtureKDEMeta(n_components=nc))

    # Additional ensemble variants
    for n in [10, 30]:
        methods.append(BaggingMeta(n_estimators=n))

    # Additional neural variants
    for t in [0.25, 3.0]:
        methods.append(SigmoidWeightedMeta(temperature=t))

    # Additional copula variants
    for th in [0.5, 3.0]:
        methods.append(ClaytonCopulaMeta(theta=th))
        methods.append(FrankCopulaMeta(theta=th))

    # Additional extreme value
    for alpha in [1.0, 4.0, 5.0]:
        methods.append(ParetoMeta(alpha=alpha))

    return methods


if __name__ == "__main__":
    methods = get_part3_methods()
    print(f"Part 3 contains {len(methods)} experimental methods")
    for m in methods:
        print(f"  - {m.name} ({m.category})")

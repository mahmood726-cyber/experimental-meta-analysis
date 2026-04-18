"""
Experimental Meta-Analysis Framework
=====================================
A comprehensive framework implementing 300+ experimental meta-analysis methods
with simulation infrastructure for identifying superior approaches.

Author: Experimental Meta-Analysis Research
Version: 1.0.0
"""

import numpy as np
from scipy import stats, optimize, special
from scipy.integrate import quad
from scipy.special import gamma as gamma_func, digamma, polygamma
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetaAnalysisData:
    """Container for meta-analysis input data"""
    effect_sizes: np.ndarray  # yi - effect sizes
    variances: np.ndarray     # vi - within-study variances
    sample_sizes: Optional[np.ndarray] = None
    study_names: Optional[List[str]] = None

    @property
    def n_studies(self) -> int:
        return len(self.effect_sizes)

    @property
    def standard_errors(self) -> np.ndarray:
        return np.sqrt(self.variances)

    @property
    def weights_fixed(self) -> np.ndarray:
        return 1.0 / self.variances

    def precision(self) -> np.ndarray:
        return 1.0 / self.variances


@dataclass
class MetaAnalysisResult:
    """Container for meta-analysis results"""
    method_name: str
    pooled_effect: float
    pooled_se: float
    tau2: float  # between-study variance
    ci_lower: float
    ci_upper: float
    i2: float = 0.0  # heterogeneity measure
    q_stat: float = 0.0
    p_heterogeneity: float = 1.0
    weights: Optional[np.ndarray] = None
    converged: bool = True
    iterations: int = 0
    computation_time: float = 0.0
    additional_info: Dict = field(default_factory=dict)


# =============================================================================
# BASE CLASS FOR ALL META-ANALYSIS METHODS
# =============================================================================

class MetaAnalysisMethod(ABC):
    """Abstract base class for all meta-analysis methods"""

    def __init__(self, name: str, category: str, experimental: bool = True):
        self.name = name
        self.category = category
        self.experimental = experimental

    @abstractmethod
    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        """Estimate pooled effect and heterogeneity"""
        pass

    def compute_heterogeneity_stats(self, data: MetaAnalysisData, tau2: float) -> Tuple[float, float, float]:
        """Compute Q, I², and p-value for heterogeneity"""
        yi = data.effect_sizes
        vi = data.variances
        k = len(yi)

        # Fixed-effect pooled estimate for Q
        wi_fe = 1.0 / vi
        mu_fe = np.sum(wi_fe * yi) / np.sum(wi_fe)

        # Q statistic
        Q = np.sum(wi_fe * (yi - mu_fe)**2)

        # I² statistic
        if Q > k - 1:
            I2 = 100 * (Q - (k - 1)) / Q
        else:
            I2 = 0.0

        # p-value
        p_val = 1 - stats.chi2.cdf(Q, k - 1) if k > 1 else 1.0

        return Q, max(0, min(100, I2)), p_val

    def compute_ci(self, effect: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute confidence interval"""
        z = stats.norm.ppf(1 - alpha/2)
        return effect - z * se, effect + z * se


# =============================================================================
# STANDARD REFERENCE METHODS (for comparison)
# =============================================================================

class DerSimonianLaird(MetaAnalysisMethod):
    """DerSimonian-Laird random effects method (standard reference)"""

    def __init__(self):
        super().__init__("DerSimonian-Laird", "standard", experimental=False)

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Fixed-effect estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)

        # Q statistic
        Q = np.sum(wi * (yi - mu_fe)**2)

        # DL tau² estimate
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Random-effects estimate
        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class REML(MetaAnalysisMethod):
    """Restricted Maximum Likelihood method (standard reference)"""

    def __init__(self):
        super().__init__("REML", "standard", experimental=False)

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        def neg_reml_ll(tau2):
            if tau2 < 0:
                return 1e10
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)
            ll = -0.5 * (np.sum(np.log(vi + tau2)) +
                        np.log(np.sum(wi)) +
                        np.sum(wi * (yi - mu)**2))
            return -ll

        # Optimize
        result = optimize.minimize_scalar(neg_reml_ll, bounds=(0, 10), method='bounded')
        tau2 = max(0, result.x)

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            converged=result.success,
            computation_time=time.time() - start_time
        )


class PauleMandel(MetaAnalysisMethod):
    """Paule-Mandel method (standard reference)"""

    def __init__(self):
        super().__init__("Paule-Mandel", "standard", experimental=False)

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        def pm_equation(tau2):
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)
            Q = np.sum(wi * (yi - mu)**2)
            return Q - (k - 1)

        # Check if tau2 = 0 is solution
        if pm_equation(0) <= 0:
            tau2 = 0.0
        else:
            try:
                tau2 = optimize.brentq(pm_equation, 0, 100)
            except:
                tau2 = 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 1: ROBUST ESTIMATORS
# =============================================================================

class RobustHuberMeta(MetaAnalysisMethod):
    """Huber's M-estimator for robust meta-analysis"""

    def __init__(self, c: float = 1.345):
        super().__init__(f"RobustHuber_c{c}", "robust")
        self.c = c

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances

        def huber_weight(r, c):
            return np.where(np.abs(r) <= c, 1.0, c / np.abs(r))

        # Initial estimate
        wi = 1.0 / vi
        mu = np.median(yi)  # Robust start
        tau2 = np.var(yi) / 2

        for iteration in range(100):
            mu_old = mu
            wi_total = 1.0 / (vi + tau2)
            residuals = (yi - mu) * np.sqrt(wi_total)
            rw = huber_weight(residuals, self.c)

            wi_robust = wi_total * rw
            mu = np.sum(wi_robust * yi) / np.sum(wi_robust)

            # Update tau2 using robust Q
            Q_robust = np.sum(wi_robust * (yi - mu)**2)
            c_val = np.sum(wi_robust) - np.sum(wi_robust**2) / np.sum(wi_robust)
            tau2 = max(0, (Q_robust - (len(yi) - 1)) / c_val) if c_val > 0 else tau2

            if abs(mu - mu_old) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_robust))
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
            weights=wi_robust,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class RobustTukeyBiweight(MetaAnalysisMethod):
    """Tukey's biweight M-estimator for meta-analysis"""

    def __init__(self, c: float = 4.685):
        super().__init__(f"TukeyBiweight_c{c}", "robust")
        self.c = c

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances

        def biweight(r, c):
            return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0)

        mu = np.median(yi)
        tau2 = np.var(yi) / 2

        for iteration in range(100):
            mu_old = mu
            wi_total = 1.0 / (vi + tau2)
            residuals = (yi - mu) * np.sqrt(wi_total)
            rw = biweight(residuals, self.c)

            wi_robust = wi_total * rw + 1e-10
            mu = np.sum(wi_robust * yi) / np.sum(wi_robust)

            Q_robust = np.sum(wi_robust * (yi - mu)**2)
            c_val = np.sum(wi_robust) - np.sum(wi_robust**2) / np.sum(wi_robust)
            tau2 = max(0, (Q_robust - (len(yi) - 1)) / c_val) if c_val > 0 else tau2

            if abs(mu - mu_old) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_robust))
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
            weights=wi_robust,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class RobustAndrewsWave(MetaAnalysisMethod):
    """Andrews' wave estimator for robust meta-analysis"""

    def __init__(self, c: float = 1.339 * np.pi):
        super().__init__(f"AndrewsWave", "robust")
        self.c = c

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances

        def andrews_weight(r, c):
            return np.where(np.abs(r) <= c, np.sin(r/c) / (r/c + 1e-10), 0)

        mu = np.median(yi)
        tau2 = np.var(yi) / 2

        for iteration in range(100):
            mu_old = mu
            wi_total = 1.0 / (vi + tau2)
            residuals = (yi - mu) * np.sqrt(wi_total)
            rw = andrews_weight(residuals, self.c)
            rw = np.clip(rw, 0.01, 2)

            wi_robust = wi_total * rw
            mu = np.sum(wi_robust * yi) / np.sum(wi_robust)

            Q_robust = np.sum(wi_robust * (yi - mu)**2)
            c_val = np.sum(wi_robust) - np.sum(wi_robust**2) / np.sum(wi_robust)
            tau2 = max(0, (Q_robust - (len(yi) - 1)) / c_val) if c_val > 0 else tau2

            if abs(mu - mu_old) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_robust))
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
            weights=wi_robust,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class RobustHampel(MetaAnalysisMethod):
    """Hampel's three-part redescending M-estimator"""

    def __init__(self, a: float = 1.7, b: float = 3.4, c: float = 8.5):
        super().__init__(f"Hampel_{a}_{b}_{c}", "robust")
        self.a, self.b, self.c = a, b, c

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        a, b, c = self.a, self.b, self.c

        def hampel_weight(r):
            r_abs = np.abs(r)
            w = np.ones_like(r)
            mask1 = (r_abs > a) & (r_abs <= b)
            mask2 = (r_abs > b) & (r_abs <= c)
            mask3 = r_abs > c
            w[mask1] = a / r_abs[mask1]
            w[mask2] = a * (c - r_abs[mask2]) / (r_abs[mask2] * (c - b))
            w[mask3] = 0
            return w

        mu = np.median(yi)
        tau2 = np.var(yi) / 2

        for iteration in range(100):
            mu_old = mu
            wi_total = 1.0 / (vi + tau2)
            residuals = (yi - mu) * np.sqrt(wi_total)
            rw = hampel_weight(residuals)

            wi_robust = wi_total * rw + 1e-10
            mu = np.sum(wi_robust * yi) / np.sum(wi_robust)

            Q_robust = np.sum(wi_robust * (yi - mu)**2)
            c_val = np.sum(wi_robust) - np.sum(wi_robust**2) / np.sum(wi_robust)
            tau2 = max(0, (Q_robust - (len(yi) - 1)) / c_val) if c_val > 0 else tau2

            if abs(mu - mu_old) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_robust))
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
            weights=wi_robust,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class MedianAbsoluteDeviation(MetaAnalysisMethod):
    """MAD-based robust meta-analysis estimator"""

    def __init__(self):
        super().__init__("MAD_Robust", "robust")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances

        # Weighted median as initial estimate
        wi = 1.0 / vi
        sorted_idx = np.argsort(yi)
        yi_sorted = yi[sorted_idx]
        wi_sorted = wi[sorted_idx]
        cumsum = np.cumsum(wi_sorted)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        mu = yi_sorted[min(median_idx, len(yi_sorted)-1)]

        # MAD for tau estimation
        mad = np.median(np.abs(yi - mu))
        tau2 = max(0, (mad * 1.4826)**2 - np.median(vi))

        # Final weighted estimate
        wi_re = 1.0 / (vi + tau2)
        mu_final = np.sum(wi_re * yi) / np.sum(wi_re)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_final, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_final,
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


class WinsorizedMeta(MetaAnalysisMethod):
    """Winsorized meta-analysis with adaptive truncation"""

    def __init__(self, trim_proportion: float = 0.1):
        super().__init__(f"Winsorized_{trim_proportion}", "robust")
        self.trim = trim_proportion

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Winsorize effect sizes
        n_trim = int(k * self.trim)
        sorted_idx = np.argsort(yi)
        yi_winsor = yi.copy()
        if n_trim > 0:
            lower_val = yi[sorted_idx[n_trim]]
            upper_val = yi[sorted_idx[-(n_trim+1)]]
            yi_winsor = np.clip(yi, lower_val, upper_val)

        # DL-style estimation on winsorized data
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi_winsor) / np.sum(wi)
        Q = np.sum(wi * (yi_winsor - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi_winsor) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class TrimmedMeanMeta(MetaAnalysisMethod):
    """Trimmed mean meta-analysis"""

    def __init__(self, trim_proportion: float = 0.1):
        super().__init__(f"TrimmedMean_{trim_proportion}", "robust")
        self.trim = trim_proportion

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        n_trim = int(k * self.trim)
        if n_trim > 0 and 2*n_trim < k:
            sorted_idx = np.argsort(yi)
            keep_idx = sorted_idx[n_trim:k-n_trim]
            yi_trim = yi[keep_idx]
            vi_trim = vi[keep_idx]
        else:
            yi_trim, vi_trim = yi, vi

        # Standard DL on trimmed data
        wi = 1.0 / vi_trim
        mu_fe = np.sum(wi * yi_trim) / np.sum(wi)
        Q = np.sum(wi * (yi_trim - mu_fe)**2)
        k_trim = len(yi_trim)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k_trim - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi_trim + tau2)
        mu_re = np.sum(wi_re * yi_trim) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 2: BAYESIAN & SHRINKAGE
# =============================================================================

class EmpiricalBayesShrinkage(MetaAnalysisMethod):
    """Empirical Bayes shrinkage estimator"""

    def __init__(self, shrinkage_type: str = "optimal"):
        super().__init__(f"EBShrinkage_{shrinkage_type}", "bayesian")
        self.shrinkage_type = shrinkage_type

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Estimate tau2 using DL
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Shrinkage factors
        if self.shrinkage_type == "optimal":
            B = vi / (vi + tau2)
        elif self.shrinkage_type == "james_stein":
            B = max(0, 1 - (k - 3) * vi / np.sum((yi - mu_fe)**2))
        else:
            B = vi / (vi + tau2)

        # Shrunk estimates
        yi_shrunk = mu_fe + (1 - B) * (yi - mu_fe)

        # Final pooled estimate
        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi_shrunk) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class HierarchicalBayesMeta(MetaAnalysisMethod):
    """Approximate hierarchical Bayes meta-analysis"""

    def __init__(self, prior_tau_scale: float = 1.0):
        super().__init__(f"HierarchicalBayes_scale{prior_tau_scale}", "bayesian")
        self.prior_scale = prior_tau_scale

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Approximate posterior using grid search
        tau2_grid = np.linspace(0, 5 * np.var(yi), 100)
        log_posterior = np.zeros(len(tau2_grid))

        for i, tau2 in enumerate(tau2_grid):
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)

            # Likelihood
            log_lik = -0.5 * np.sum(np.log(vi + tau2) + (yi - mu)**2 / (vi + tau2))

            # Half-Cauchy prior on tau
            tau = np.sqrt(tau2) if tau2 > 0 else 0.001
            log_prior = np.log(2 / (np.pi * self.prior_scale * (1 + (tau/self.prior_scale)**2)))

            log_posterior[i] = log_lik + log_prior

        # Posterior mean tau2
        posterior = np.exp(log_posterior - np.max(log_posterior))
        posterior /= np.sum(posterior)
        tau2 = np.sum(posterior * tau2_grid)

        # Posterior mean effect
        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class BayesianModelAveraging(MetaAnalysisMethod):
    """Bayesian model averaging across heterogeneity models"""

    def __init__(self):
        super().__init__("BayesianModelAveraging", "bayesian")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Model 1: Fixed effects (tau2 = 0)
        wi_fe = 1.0 / vi
        mu_fe = np.sum(wi_fe * yi) / np.sum(wi_fe)
        ll_fe = -0.5 * np.sum(np.log(vi) + (yi - mu_fe)**2 / vi)
        bic_fe = -2 * ll_fe + 1 * np.log(k)  # 1 parameter

        # Model 2: Random effects with DL tau2
        Q = np.sum(wi_fe * (yi - mu_fe)**2)
        c = np.sum(wi_fe) - np.sum(wi_fe**2) / np.sum(wi_fe)
        tau2_dl = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2_dl)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        ll_re = -0.5 * np.sum(np.log(vi + tau2_dl) + (yi - mu_re)**2 / (vi + tau2_dl))
        bic_re = -2 * ll_re + 2 * np.log(k)  # 2 parameters

        # Model weights (BIC approximation)
        bic_min = min(bic_fe, bic_re)
        w_fe = np.exp(-0.5 * (bic_fe - bic_min))
        w_re = np.exp(-0.5 * (bic_re - bic_min))
        w_total = w_fe + w_re
        w_fe, w_re = w_fe / w_total, w_re / w_total

        # Model-averaged estimate
        mu_avg = w_fe * mu_fe + w_re * mu_re
        se_fe = np.sqrt(1.0 / np.sum(wi_fe))
        se_re = np.sqrt(1.0 / np.sum(wi_re))
        se_avg = np.sqrt(w_fe * se_fe**2 + w_re * se_re**2 +
                        w_fe * w_re * (mu_fe - mu_re)**2)
        tau2_avg = w_re * tau2_dl

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2_avg)
        ci_lo, ci_hi = self.compute_ci(mu_avg, se_avg)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_avg,
            pooled_se=se_avg,
            tau2=tau2_avg,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'weight_fe': w_fe, 'weight_re': w_re},
            computation_time=time.time() - start_time
        )


class PenalizedLikelihoodMeta(MetaAnalysisMethod):
    """Penalized likelihood estimator with various penalties"""

    def __init__(self, penalty_type: str = "ridge", lambda_val: float = 0.1):
        super().__init__(f"PenalizedML_{penalty_type}_{lambda_val}", "bayesian")
        self.penalty_type = penalty_type
        self.lambda_val = lambda_val

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        def penalized_nll(params):
            mu, log_tau2 = params
            tau2 = np.exp(log_tau2)
            wi = 1.0 / (vi + tau2)

            nll = 0.5 * np.sum(np.log(vi + tau2) + (yi - mu)**2 / (vi + tau2))

            if self.penalty_type == "ridge":
                penalty = self.lambda_val * (mu**2 + tau2)
            elif self.penalty_type == "lasso":
                penalty = self.lambda_val * (np.abs(mu) + np.sqrt(tau2))
            elif self.penalty_type == "elastic":
                penalty = self.lambda_val * (0.5 * (mu**2 + tau2) + 0.5 * (np.abs(mu) + np.sqrt(tau2)))
            else:
                penalty = 0

            return nll + penalty

        # Initial values
        mu0 = np.mean(yi)
        tau2_0 = np.var(yi)

        result = optimize.minimize(penalized_nll, [mu0, np.log(tau2_0 + 0.01)],
                                  method='L-BFGS-B')

        mu_est = result.x[0]
        tau2_est = np.exp(result.x[1])

        wi_re = 1.0 / (vi + tau2_est)
        se_est = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2_est)
        ci_lo, ci_hi = self.compute_ci(mu_est, se_est)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_est,
            pooled_se=se_est,
            tau2=tau2_est,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            converged=result.success,
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 3: SMALL SAMPLE CORRECTIONS
# =============================================================================

class HartungKnapp(MetaAnalysisMethod):
    """Hartung-Knapp-Sidik-Jonkman method with t-distribution"""

    def __init__(self):
        super().__init__("HartungKnapp", "small_sample", experimental=False)

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # First get DL estimates
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)

        # HKSJ correction
        q_adj = np.sum(wi_re * (yi - mu_re)**2) / (k - 1)
        se_hksj = np.sqrt(q_adj / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        # t-distribution CI
        t_crit = stats.t.ppf(0.975, k - 1)
        ci_lo = mu_re - t_crit * se_hksj
        ci_hi = mu_re + t_crit * se_hksj

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_hksj,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class SidikJonkman(MetaAnalysisMethod):
    """Sidik-Jonkman variance estimator"""

    def __init__(self):
        super().__init__("SidikJonkman", "small_sample")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initial tau2 from unweighted variance
        mu_uw = np.mean(yi)
        tau2 = np.sum((yi - mu_uw)**2) / k

        # Iterate
        for _ in range(100):
            wi = 1.0 / (vi + tau2)
            mu = np.sum(wi * yi) / np.sum(wi)
            tau2_new = np.sum((yi - mu)**2) / k

            if abs(tau2_new - tau2) < 1e-8:
                break
            tau2 = tau2_new

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class KnappHartungModified(MetaAnalysisMethod):
    """Modified Knapp-Hartung with truncation"""

    def __init__(self, truncate: bool = True):
        super().__init__(f"KnappHartungMod_trunc{truncate}", "small_sample")
        self.truncate = truncate

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL estimates
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)

        # Modified HKSJ
        q_adj = np.sum(wi_re * (yi - mu_re)**2) / (k - 1)

        if self.truncate:
            q_adj = max(1.0, q_adj)

        se_mod = np.sqrt(q_adj / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        t_crit = stats.t.ppf(0.975, k - 1)
        ci_lo = mu_re - t_crit * se_mod
        ci_hi = mu_re + t_crit * se_mod

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_mod,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class SatterthwaiteDFMeta(MetaAnalysisMethod):
    """Satterthwaite degrees of freedom approximation"""

    def __init__(self):
        super().__init__("SatterthwaiteDF", "small_sample")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL estimates
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        # Satterthwaite df
        total_var = vi + tau2
        df_satt = np.sum(total_var)**2 / np.sum(total_var**2)
        df_satt = max(1, min(k - 1, df_satt))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        t_crit = stats.t.ppf(0.975, df_satt)
        ci_lo = mu_re - t_crit * se_re
        ci_hi = mu_re + t_crit * se_re

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            additional_info={'df_satterthwaite': df_satt},
            computation_time=time.time() - start_time
        )


class KenwardRogerApprox(MetaAnalysisMethod):
    """Kenward-Roger approximation for small samples"""

    def __init__(self):
        super().__init__("KenwardRoger", "small_sample")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

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
        tau2 = max(0, result.x)

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)

        # Kenward-Roger adjustment
        V_inv = np.diag(wi_re)
        X = np.ones((k, 1))
        bread = np.linalg.inv(X.T @ V_inv @ X)

        # Approximate adjustment factor
        P = V_inv - V_inv @ X @ bread @ X.T @ V_inv
        W = 2 * np.trace(P @ P)

        if W > 0:
            adjustment = k / max(1, k - 1 - W / k)
        else:
            adjustment = 1.0

        se_kr = np.sqrt(adjustment * bread[0, 0])

        # Approximate df
        df_kr = 2 * bread[0, 0]**2 / max(1e-10, np.var(wi_re) / k)
        df_kr = max(1, min(k - 1, df_kr))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        t_crit = stats.t.ppf(0.975, df_kr)
        ci_lo = mu_re - t_crit * se_kr
        ci_hi = mu_re + t_crit * se_kr

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_kr,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            additional_info={'df_kr': df_kr, 'adjustment': adjustment},
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 4: ALTERNATIVE TAU² ESTIMATORS
# =============================================================================

class HunterSchmidt(MetaAnalysisMethod):
    """Hunter-Schmidt variance estimator"""

    def __init__(self):
        super().__init__("HunterSchmidt", "tau_estimator")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Sample-size weighted average
        if data.sample_sizes is not None:
            ni = data.sample_sizes
        else:
            ni = 1.0 / vi  # Use precision as proxy

        mu = np.sum(ni * yi) / np.sum(ni)

        # Hunter-Schmidt variance
        var_yi = np.sum(ni * (yi - mu)**2) / np.sum(ni)
        var_e = np.sum(ni * vi) / np.sum(ni)
        tau2 = max(0, var_yi - var_e)

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class HedgesOlkin(MetaAnalysisMethod):
    """Hedges-Olkin maximum likelihood estimator"""

    def __init__(self):
        super().__init__("HedgesOlkin", "tau_estimator")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        def neg_ll(params):
            mu, tau2 = params
            if tau2 < 0:
                return 1e10
            wi = 1.0 / (vi + tau2)
            ll = -0.5 * np.sum(np.log(vi + tau2) + (yi - mu)**2 / (vi + tau2))
            return -ll

        mu0 = np.mean(yi)
        tau2_0 = np.var(yi)

        result = optimize.minimize(neg_ll, [mu0, tau2_0], method='L-BFGS-B',
                                  bounds=[(None, None), (0, None)])

        mu_re, tau2 = result.x
        tau2 = max(0, tau2)

        wi_re = 1.0 / (vi + tau2)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            converged=result.success,
            computation_time=time.time() - start_time
        )


class EmpiricalBayesTau(MetaAnalysisMethod):
    """Empirical Bayes tau² estimator (Morris)"""

    def __init__(self):
        super().__init__("EmpiricalBayesTau", "tau_estimator")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Iterative EB estimation
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)
        tau2 = np.var(yi)

        for _ in range(100):
            wi_re = 1.0 / (vi + tau2)
            mu_new = np.sum(wi_re * yi) / np.sum(wi_re)

            # EB tau² update
            num = np.sum((yi - mu_new)**2 - vi)
            tau2_new = max(0, num / k)

            if abs(tau2_new - tau2) < 1e-8 and abs(mu_new - mu) < 1e-8:
                break

            mu, tau2 = mu_new, tau2_new

        wi_re = 1.0 / (vi + tau2)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class GeneralizedQ(MetaAnalysisMethod):
    """Generalized Q method for tau² estimation"""

    def __init__(self, estimator_type: str = "DL"):
        super().__init__(f"GeneralizedQ_{estimator_type}", "tau_estimator")
        self.estimator_type = estimator_type

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Generalized Q estimation with different weighting schemes
        if self.estimator_type == "DL":
            wi0 = 1.0 / vi
        elif self.estimator_type == "HE":
            wi0 = np.ones(k)
        elif self.estimator_type == "HS":
            wi0 = 1.0 / vi**2
        else:
            wi0 = 1.0 / vi

        mu0 = np.sum(wi0 * yi) / np.sum(wi0)
        Q = np.sum(wi0 * (yi - mu0)**2)

        # Iterative tau² estimation
        tau2 = 0.0
        for _ in range(100):
            wi = 1.0 / (vi + tau2)
            a = np.sum(wi)
            b = np.sum(wi**2)
            c = a - b / a

            mu = np.sum(wi * yi) / a
            Q_gen = np.sum(wi * (yi - mu)**2)

            tau2_new = max(0, (Q_gen - (k - 1)) / c) if c > 0 else 0

            if abs(tau2_new - tau2) < 1e-8:
                break
            tau2 = tau2_new

        wi_re = 1.0 / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 5: NOVEL WEIGHTING SCHEMES
# =============================================================================

class QualityWeighted(MetaAnalysisMethod):
    """Quality-effects model with study quality weights"""

    def __init__(self, quality_power: float = 1.0):
        super().__init__(f"QualityEffects_pow{quality_power}", "weighting")
        self.quality_power = quality_power

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Generate quality weights based on precision (as proxy)
        # Higher precision = higher quality
        quality = (1.0 / vi) / np.max(1.0 / vi)
        quality_weights = quality ** self.quality_power

        # DL tau²
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Quality-adjusted weights
        wi_re = quality_weights / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class InverseVariancePlus(MetaAnalysisMethod):
    """Enhanced inverse variance with regularization"""

    def __init__(self, regularization: float = 0.1):
        super().__init__(f"IVPlus_reg{regularization}", "weighting")
        self.reg = regularization

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau²
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Regularized weights
        vi_reg = vi + self.reg * np.median(vi)
        wi_re = 1.0 / (vi_reg + tau2)

        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class SampleSizeWeighted(MetaAnalysisMethod):
    """Sample size weighted meta-analysis"""

    def __init__(self, power: float = 0.5):
        super().__init__(f"SampleSizeWeighted_pow{power}", "weighting")
        self.power = power

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Use precision as sample size proxy
        if data.sample_sizes is not None:
            ni = data.sample_sizes
        else:
            ni = 1.0 / vi

        ni_scaled = (ni / np.mean(ni)) ** self.power

        # DL tau²
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Sample-size adjusted weights
        wi_re = ni_scaled / (vi + tau2)
        mu_re = np.sum(wi_re * yi) / np.sum(wi_re)
        se_re = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class UniformWeighted(MetaAnalysisMethod):
    """Unweighted (uniform weights) meta-analysis"""

    def __init__(self):
        super().__init__("UniformWeighted", "weighting")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Simple mean
        mu = np.mean(yi)
        se = np.sqrt(np.var(yi) / k + np.mean(vi))

        # Estimate tau2 for reference
        tau2 = max(0, np.var(yi) - np.mean(vi))

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
            weights=np.ones(k) / k,
            computation_time=time.time() - start_time
        )


class SoftmaxWeighted(MetaAnalysisMethod):
    """Softmax-based adaptive weighting"""

    def __init__(self, temperature: float = 1.0):
        super().__init__(f"Softmax_temp{temperature}", "weighting")
        self.temperature = temperature

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # DL tau²
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Softmax weights based on precision
        precision = 1.0 / (vi + tau2)
        log_weights = precision / self.temperature
        log_weights = log_weights - np.max(log_weights)
        wi_softmax = np.exp(log_weights)
        wi_softmax = wi_softmax / np.sum(wi_softmax)

        mu_re = np.sum(wi_softmax * yi)

        # SE estimation
        se_re = np.sqrt(np.sum(wi_softmax**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_re, se_re)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_re,
            pooled_se=se_re,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_softmax,
            computation_time=time.time() - start_time
        )


# =============================================================================
# EXPERIMENTAL METHODS - CATEGORY 6: BOOTSTRAP & RESAMPLING
# =============================================================================

class BootstrapMeta(MetaAnalysisMethod):
    """Bootstrap meta-analysis"""

    def __init__(self, n_bootstrap: int = 1000, method: str = "percentile"):
        super().__init__(f"Bootstrap_{method}_{n_bootstrap}", "resampling")
        self.n_bootstrap = n_bootstrap
        self.method = method

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Original DL estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_orig = np.sum(wi_re * yi) / np.sum(wi_re)

        # Bootstrap
        boot_estimates = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(k, k, replace=True)
            yi_b = yi[idx]
            vi_b = vi[idx]

            wi_b = 1.0 / vi_b
            mu_fe_b = np.sum(wi_b * yi_b) / np.sum(wi_b)
            Q_b = np.sum(wi_b * (yi_b - mu_fe_b)**2)
            c_b = np.sum(wi_b) - np.sum(wi_b**2) / np.sum(wi_b)
            tau2_b = max(0, (Q_b - (k - 1)) / c_b) if c_b > 0 else 0.0

            wi_re_b = 1.0 / (vi_b + tau2_b)
            mu_b = np.sum(wi_re_b * yi_b) / np.sum(wi_re_b)
            boot_estimates.append(mu_b)

        boot_estimates = np.array(boot_estimates)
        se_boot = np.std(boot_estimates)

        if self.method == "percentile":
            ci_lo = np.percentile(boot_estimates, 2.5)
            ci_hi = np.percentile(boot_estimates, 97.5)
        elif self.method == "bca":
            # Bias-corrected and accelerated
            z0 = stats.norm.ppf(np.mean(boot_estimates < mu_orig))

            # Jackknife for acceleration
            jack = np.zeros(k)
            for i in range(k):
                idx = np.concatenate([np.arange(i), np.arange(i+1, k)])
                yi_j = yi[idx]
                vi_j = vi[idx]
                wi_j = 1.0 / vi_j
                jack[i] = np.sum(wi_j * yi_j) / np.sum(wi_j)

            jack_mean = np.mean(jack)
            a = np.sum((jack_mean - jack)**3) / (6 * np.sum((jack_mean - jack)**2)**1.5 + 1e-10)

            z_alpha1, z_alpha2 = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
            alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha1) / (1 - a * (z0 + z_alpha1)))
            alpha2 = stats.norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))

            ci_lo = np.percentile(boot_estimates, 100 * alpha1)
            ci_hi = np.percentile(boot_estimates, 100 * alpha2)
        else:  # basic
            ci_lo = 2 * mu_orig - np.percentile(boot_estimates, 97.5)
            ci_hi = 2 * mu_orig - np.percentile(boot_estimates, 2.5)

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_orig,
            pooled_se=se_boot,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class JackknifeMeta(MetaAnalysisMethod):
    """Jackknife meta-analysis for bias correction"""

    def __init__(self):
        super().__init__("Jackknife", "resampling")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Original estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_orig = np.sum(wi_re * yi) / np.sum(wi_re)

        # Jackknife estimates
        jack_estimates = []
        for i in range(k):
            idx = np.concatenate([np.arange(i), np.arange(i+1, k)])
            yi_j = yi[idx]
            vi_j = vi[idx]

            wi_j = 1.0 / vi_j
            mu_fe_j = np.sum(wi_j * yi_j) / np.sum(wi_j)
            Q_j = np.sum(wi_j * (yi_j - mu_fe_j)**2)
            c_j = np.sum(wi_j) - np.sum(wi_j**2) / np.sum(wi_j)
            tau2_j = max(0, (Q_j - (k - 2)) / c_j) if c_j > 0 else 0.0

            wi_re_j = 1.0 / (vi_j + tau2_j)
            mu_j = np.sum(wi_re_j * yi_j) / np.sum(wi_re_j)
            jack_estimates.append(mu_j)

        jack_estimates = np.array(jack_estimates)

        # Bias-corrected estimate
        mu_jack = k * mu_orig - (k - 1) * np.mean(jack_estimates)

        # Jackknife SE
        pseudo = k * mu_orig - (k - 1) * jack_estimates
        se_jack = np.sqrt(np.var(pseudo) / k)

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_jack, se_jack)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_jack,
            pooled_se=se_jack,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class PermutationMeta(MetaAnalysisMethod):
    """Permutation-based inference for meta-analysis"""

    def __init__(self, n_permutations: int = 1000):
        super().__init__(f"Permutation_{n_permutations}", "resampling")
        self.n_perm = n_permutations

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Original estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_orig = np.sum(wi_re * yi) / np.sum(wi_re)
        se_orig = np.sqrt(1.0 / np.sum(wi_re))

        # Permutation for CI
        perm_estimates = []
        centered_yi = yi - np.mean(yi)  # Center around 0 for permutation

        for _ in range(self.n_perm):
            signs = np.random.choice([-1, 1], k)
            yi_p = mu_orig + signs * np.abs(centered_yi)

            mu_p = np.sum(wi_re * yi_p) / np.sum(wi_re)
            perm_estimates.append(mu_p)

        perm_estimates = np.array(perm_estimates)

        # SE from permutation variance
        se_perm = np.std(perm_estimates)

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)

        # Use t-distribution for small samples
        t_crit = stats.t.ppf(0.975, k - 1)
        ci_lo = mu_orig - t_crit * se_perm
        ci_hi = mu_orig + t_crit * se_perm

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_orig,
            pooled_se=se_perm,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_re,
            computation_time=time.time() - start_time
        )


# Save to be continued in part 2...
if __name__ == "__main__":
    print("Core framework loaded successfully")
    print("Standard methods: DerSimonianLaird, REML, PauleMandel, HartungKnapp")
    print("Experimental categories: robust, bayesian, small_sample, tau_estimator, weighting, resampling")

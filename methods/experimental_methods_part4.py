"""
Experimental Meta-Analysis Methods - Part 4
============================================
Novel approaches: Robust Score, Exponential Family, Quantile-based,
                  Nonparametric, Spectral Methods
"""

import numpy as np
from scipy import stats, optimize, special, linalg
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
# CATEGORY 20: ROBUST SCORE METHODS
# =============================================================================

class RobustScoreMeta(MetaAnalysisMethod):
    """Robust score function meta-analysis"""

    def __init__(self, score_type: str = "sign"):
        super().__init__(f"RobustScore_{score_type}", "robust_score")
        self.score_type = score_type

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

        def score_function(mu):
            residuals = yi - mu
            if self.score_type == "sign":
                scores = np.sign(residuals)
            elif self.score_type == "wilcoxon":
                ranks = stats.rankdata(np.abs(residuals))
                scores = np.sign(residuals) * ranks
            elif self.score_type == "normal_score":
                ranks = stats.rankdata(residuals)
                expected = stats.norm.ppf(ranks / (k + 1))
                scores = expected
            else:
                scores = residuals

            return np.sum(wi_re * scores)

        # Find root of score equation
        try:
            result = optimize.brentq(score_function, np.min(yi) - 2*np.std(yi),
                                    np.max(yi) + 2*np.std(yi))
            mu_rs = result
        except:
            mu_rs = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_rs, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_rs,
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


class MEstimatorMeta(MetaAnalysisMethod):
    """M-estimator with general psi function"""

    def __init__(self, psi_type: str = "logistic"):
        super().__init__(f"MEstimator_{psi_type}", "robust_score")
        self.psi_type = psi_type

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

        def psi_function(r):
            if self.psi_type == "logistic":
                return np.tanh(r)
            elif self.psi_type == "fair":
                c = 1.4
                return r / (1 + np.abs(r) / c)
            elif self.psi_type == "talwar":
                c = 2.795
                return np.where(np.abs(r) <= c, r, 0)
            else:
                return r

        # Iteratively solve
        mu = np.median(yi)
        scale = np.median(np.abs(yi - mu)) * 1.4826

        for _ in range(100):
            r = (yi - mu) / (scale + 1e-10)
            psi_r = psi_function(r)
            wi_psi = wi_re * psi_r / (r + 1e-10)
            wi_psi = np.clip(wi_psi, 0.01, np.max(wi_re) * 10)

            mu_new = np.sum(wi_psi * yi) / np.sum(wi_psi)

            if abs(mu_new - mu) < 1e-8:
                break
            mu = mu_new

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


class SEstimatorMeta(MetaAnalysisMethod):
    """S-estimator for high breakdown"""

    def __init__(self, breakdown: float = 0.5):
        super().__init__(f"SEstimator_bd{breakdown}", "robust_score")
        self.breakdown = breakdown

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

        # S-estimator: minimize robust scale
        def rho_bisquare(r, c=4.685):
            return np.where(np.abs(r) <= c,
                           (1 - (1 - (r/c)**2)**3) / 6,
                           1/6)

        def objective(params):
            mu, log_scale = params
            scale = np.exp(log_scale)
            r = (yi - mu) / (scale + 1e-10)
            return np.sum(wi_re * rho_bisquare(r))

        # Initialize
        mu0 = np.median(yi)
        scale0 = np.median(np.abs(yi - mu0)) * 1.4826

        try:
            result = optimize.minimize(objective, [mu0, np.log(scale0 + 0.01)],
                                      method='Nelder-Mead')
            mu_s = result.x[0]
        except:
            mu_s = mu_fe

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_s, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_s,
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


# =============================================================================
# CATEGORY 21: EXPONENTIAL FAMILY METHODS
# =============================================================================

class ExponentialFamilyMeta(MetaAnalysisMethod):
    """Exponential family meta-analysis"""

    def __init__(self, family: str = "gaussian"):
        super().__init__(f"ExpFamily_{family}", "exponential_family")
        self.family = family

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

        if self.family == "gaussian":
            # Standard random effects
            mu = np.sum(wi_re * yi) / np.sum(wi_re)
        elif self.family == "laplace":
            # Median (L1)
            sorted_idx = np.argsort(yi)
            cumsum = np.cumsum(wi_re[sorted_idx])
            med_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            mu = yi[sorted_idx[min(med_idx, k-1)]]
        elif self.family == "poisson_like":
            # Log-linear model
            yi_pos = yi - np.min(yi) + 0.1
            log_yi = np.log(yi_pos)
            mu_log = np.sum(wi_re * log_yi) / np.sum(wi_re)
            mu = np.exp(mu_log) + np.min(yi) - 0.1
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
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class NaturalParameterMeta(MetaAnalysisMethod):
    """Natural parameter space meta-analysis"""

    def __init__(self):
        super().__init__("NaturalParameter", "exponential_family")

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

        # Natural parameters: eta = mu/sigma^2
        eta = yi / (vi + tau2)
        sum_precision = np.sum(1.0 / (vi + tau2))

        # Sufficient statistics
        sum_eta = np.sum(eta)

        mu_np = sum_eta / sum_precision
        se = np.sqrt(1.0 / sum_precision)

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_np, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_np,
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
# CATEGORY 22: QUANTILE-BASED METHODS
# =============================================================================

class QuantileRegressionMeta(MetaAnalysisMethod):
    """Quantile regression meta-analysis"""

    def __init__(self, tau: float = 0.5):
        super().__init__(f"QuantileRegression_tau{tau}", "quantile")
        self.tau = tau

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

        # Quantile regression: minimize check function
        tau_q = self.tau

        def check_loss(mu):
            residuals = yi - mu
            loss = np.where(residuals >= 0,
                           tau_q * residuals,
                           (tau_q - 1) * residuals)
            return np.sum(wi_re * loss)

        result = optimize.minimize_scalar(check_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_qr = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_qr, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_qr,
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


class InterquartileMeta(MetaAnalysisMethod):
    """Interquartile range meta-analysis"""

    def __init__(self):
        super().__init__("InterquartileMean", "quantile")

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

        # Weighted quartiles
        sorted_idx = np.argsort(yi)
        yi_sorted = yi[sorted_idx]
        wi_sorted = wi_re[sorted_idx]
        cumsum = np.cumsum(wi_sorted)
        total = cumsum[-1]

        q1_idx = np.searchsorted(cumsum, 0.25 * total)
        q3_idx = np.searchsorted(cumsum, 0.75 * total)

        q1 = yi_sorted[min(q1_idx, k-1)]
        q3 = yi_sorted[min(q3_idx, k-1)]

        # Interquartile mean
        iq_mask = (yi >= q1) & (yi <= q3)
        if np.sum(iq_mask) > 0:
            wi_iq = wi_re[iq_mask]
            yi_iq = yi[iq_mask]
            mu_iq = np.sum(wi_iq * yi_iq) / np.sum(wi_iq)
        else:
            mu_iq = np.median(yi)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_iq, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_iq,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class TrimeanMeta(MetaAnalysisMethod):
    """Tukey's trimean meta-analysis"""

    def __init__(self):
        super().__init__("Trimean", "quantile")

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

        # Weighted quartiles
        sorted_idx = np.argsort(yi)
        yi_sorted = yi[sorted_idx]
        wi_sorted = wi_re[sorted_idx]
        cumsum = np.cumsum(wi_sorted)
        total = cumsum[-1]

        q1_idx = np.searchsorted(cumsum, 0.25 * total)
        q2_idx = np.searchsorted(cumsum, 0.50 * total)
        q3_idx = np.searchsorted(cumsum, 0.75 * total)

        q1 = yi_sorted[min(q1_idx, k-1)]
        q2 = yi_sorted[min(q2_idx, k-1)]
        q3 = yi_sorted[min(q3_idx, k-1)]

        # Trimean: (Q1 + 2*Q2 + Q3) / 4
        mu_tm = (q1 + 2 * q2 + q3) / 4

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_tm, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_tm,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class MidhingeMeta(MetaAnalysisMethod):
    """Midhinge (average of Q1 and Q3) meta-analysis"""

    def __init__(self):
        super().__init__("Midhinge", "quantile")

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

        # Weighted quartiles
        sorted_idx = np.argsort(yi)
        yi_sorted = yi[sorted_idx]
        wi_sorted = wi_re[sorted_idx]
        cumsum = np.cumsum(wi_sorted)
        total = cumsum[-1]

        q1_idx = np.searchsorted(cumsum, 0.25 * total)
        q3_idx = np.searchsorted(cumsum, 0.75 * total)

        q1 = yi_sorted[min(q1_idx, k-1)]
        q3 = yi_sorted[min(q3_idx, k-1)]

        # Midhinge: (Q1 + Q3) / 2
        mu_mh = (q1 + q3) / 2

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_mh, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_mh,
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
# CATEGORY 23: NONPARAMETRIC METHODS
# =============================================================================

class HodgesLehmannMeta(MetaAnalysisMethod):
    """Hodges-Lehmann estimator meta-analysis"""

    def __init__(self):
        super().__init__("HodgesLehmann", "nonparametric")

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

        # Hodges-Lehmann: median of pairwise averages
        pairwise_means = []
        for i in range(k):
            for j in range(i, k):
                pairwise_means.append((yi[i] + yi[j]) / 2)

        mu_hl = np.median(pairwise_means)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_hl, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_hl,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class WilcoxonLocationMeta(MetaAnalysisMethod):
    """Wilcoxon-type location estimator"""

    def __init__(self):
        super().__init__("WilcoxonLocation", "nonparametric")

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

        # Wilcoxon signed-rank type estimator
        # Find mu that minimizes sum of signed ranks
        def wilcoxon_criterion(mu):
            deviations = yi - mu
            ranks = stats.rankdata(np.abs(deviations))
            signed_ranks = np.sign(deviations) * ranks * wi_re
            return np.abs(np.sum(signed_ranks))

        result = optimize.minimize_scalar(wilcoxon_criterion,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_wil = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_wil, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_wil,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class RankBasedMeta(MetaAnalysisMethod):
    """Rank-based meta-analysis"""

    def __init__(self, transform: str = "van_der_waerden"):
        super().__init__(f"RankBased_{transform}", "nonparametric")
        self.transform = transform

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

        # Rank transformation
        ranks = stats.rankdata(yi)

        if self.transform == "van_der_waerden":
            # Normal scores
            scores = stats.norm.ppf(ranks / (k + 1))
        elif self.transform == "expected_normal":
            # Expected order statistics
            scores = stats.norm.ppf((ranks - 0.5) / k)
        elif self.transform == "blom":
            scores = stats.norm.ppf((ranks - 0.375) / (k + 0.25))
        else:
            scores = ranks / (k + 1)

        # Map back to effect scale
        score_mu = np.sum(wi_re * scores) / np.sum(wi_re)
        effect_mu = np.sum(wi_re * yi) / np.sum(wi_re)

        # Adjust estimate using rank information
        rank_adjustment = score_mu * np.std(yi)
        mu_rb = effect_mu + 0.1 * rank_adjustment  # Moderate adjustment

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_rb, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_rb,
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
# CATEGORY 24: SPECTRAL METHODS
# =============================================================================

class SpectralClusteringMeta(MetaAnalysisMethod):
    """Spectral clustering-based meta-analysis"""

    def __init__(self, n_clusters: int = 2):
        super().__init__(f"SpectralClustering_{n_clusters}", "spectral")
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

        # Build similarity matrix
        diff_matrix = np.abs(yi.reshape(-1, 1) - yi.reshape(1, -1))
        sigma = np.median(diff_matrix) + 1e-10
        similarity = np.exp(-diff_matrix**2 / (2 * sigma**2))

        # Degree matrix and Laplacian
        D = np.diag(np.sum(similarity, axis=1))
        L = D - similarity

        # Eigen decomposition
        try:
            eigenvalues, eigenvectors = linalg.eigh(L)
            # Use second smallest eigenvector for clustering
            n_clust = min(self.n_clusters, k)
            fiedler = eigenvectors[:, 1] if k > 1 else np.zeros(k)

            # Simple clustering based on sign
            cluster_labels = (fiedler >= np.median(fiedler)).astype(int)

            # Weighted estimate from largest cluster
            cluster_sizes = [np.sum(cluster_labels == 0), np.sum(cluster_labels == 1)]
            main_cluster = np.argmax(cluster_sizes)
            mask = cluster_labels == main_cluster

            wi_clust = wi_re[mask]
            yi_clust = yi[mask]
            mu_spec = np.sum(wi_clust * yi_clust) / np.sum(wi_clust)
        except:
            mu_spec = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_spec, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_spec,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class PCAWeightedMeta(MetaAnalysisMethod):
    """PCA-weighted meta-analysis"""

    def __init__(self, n_components: int = 1):
        super().__init__(f"PCAWeighted_{n_components}comp", "spectral")
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

        # Create feature matrix: [yi, precision, yi*precision]
        precision = 1.0 / (vi + tau2)
        X = np.column_stack([
            (yi - np.mean(yi)) / (np.std(yi) + 1e-10),
            (precision - np.mean(precision)) / (np.std(precision) + 1e-10),
        ])

        # PCA
        try:
            cov = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal component weights
            pc1 = eigenvectors[:, -1]
            pca_weights = np.abs(X @ pc1)
            pca_weights = pca_weights / np.sum(pca_weights)

            # Combine with inverse variance
            wi_pca = np.sqrt(pca_weights * wi_re)
            wi_pca = wi_pca / np.sum(wi_pca)

            mu_pca = np.sum(wi_pca * yi)
        except:
            mu_pca = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_pca, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pca,
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
# CATEGORY 25: CONVEX OPTIMIZATION METHODS
# =============================================================================

class ConvexOptMeta(MetaAnalysisMethod):
    """Convex optimization meta-analysis"""

    def __init__(self, norm_type: str = "L2"):
        super().__init__(f"ConvexOpt_{norm_type}", "convex")
        self.norm_type = norm_type

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

        if self.norm_type == "L2":
            # Standard weighted least squares
            mu = np.sum(wi_re * yi) / np.sum(wi_re)
        elif self.norm_type == "L1":
            # Weighted L1 minimization
            def l1_objective(mu):
                return np.sum(wi_re * np.abs(yi - mu))
            result = optimize.minimize_scalar(l1_objective,
                                             bounds=(np.min(yi), np.max(yi)),
                                             method='bounded')
            mu = result.x
        elif self.norm_type == "Linf":
            # Minimize maximum weighted residual
            def linf_objective(mu):
                return np.max(wi_re * np.abs(yi - mu))
            result = optimize.minimize_scalar(linf_objective,
                                             bounds=(np.min(yi), np.max(yi)),
                                             method='bounded')
            mu = result.x
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
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class MinimaxMeta(MetaAnalysisMethod):
    """Minimax meta-analysis"""

    def __init__(self):
        super().__init__("Minimax", "convex")

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

        # Minimax: minimize maximum loss
        def max_loss(mu):
            losses = wi_re * (yi - mu)**2
            return np.max(losses)

        result = optimize.minimize_scalar(max_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_mm = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_mm, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_mm,
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

def get_part4_methods():
    """Return all experimental methods from this module"""
    methods = []

    # Robust score methods
    for score_type in ["sign", "wilcoxon", "normal_score"]:
        methods.append(RobustScoreMeta(score_type=score_type))
    for psi_type in ["logistic", "fair", "talwar"]:
        methods.append(MEstimatorMeta(psi_type=psi_type))
    for bd in [0.1, 0.25, 0.5]:
        methods.append(SEstimatorMeta(breakdown=bd))

    # Exponential family (expanded)
    for family in ["gaussian", "laplace", "poisson_like"]:
        methods.append(ExponentialFamilyMeta(family=family))
    methods.append(NaturalParameterMeta())

    # Quantile-based (expanded)
    for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
        methods.append(QuantileRegressionMeta(tau=tau))
    methods.append(InterquartileMeta())
    methods.append(TrimeanMeta())
    methods.append(MidhingeMeta())

    # Nonparametric (expanded)
    methods.append(HodgesLehmannMeta())
    methods.append(WilcoxonLocationMeta())
    for transform in ["van_der_waerden", "expected_normal", "blom"]:
        methods.append(RankBasedMeta(transform=transform))

    # Spectral methods (expanded)
    for nc in [2, 3, 4]:
        methods.append(SpectralClusteringMeta(n_clusters=nc))
    for nc in [1, 2, 3]:
        methods.append(PCAWeightedMeta(n_components=nc))

    # Convex optimization (expanded)
    for norm in ["L1", "L2", "Linf"]:
        methods.append(ConvexOptMeta(norm_type=norm))
    methods.append(MinimaxMeta())

    return methods


if __name__ == "__main__":
    methods = get_part4_methods()
    print(f"Part 4 contains {len(methods)} experimental methods")
    for m in methods:
        print(f"  - {m.name} ({m.category})")

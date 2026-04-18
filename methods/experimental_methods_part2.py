"""
Experimental Meta-Analysis Methods - Part 2
============================================
Novel approaches: Regularization, Geometric, Loss Functions, Publication Bias
"""

import numpy as np
from scipy import stats, optimize
from scipy.linalg import sqrtm
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
# CATEGORY 11: REGULARIZATION METHODS
# =============================================================================

class RidgeRegularizedMeta(MetaAnalysisMethod):
    """Ridge-regularized meta-analysis"""

    def __init__(self, lambda_val: float = 0.1):
        super().__init__(f"RidgeRegularized_{lambda_val}", "regularization")
        self.lambda_val = lambda_val

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

        # Ridge-regularized estimate
        wi_re = 1.0 / (vi + tau2)

        # Ridge penalty shrinks toward 0
        reg_wi = wi_re + self.lambda_val
        mu_ridge = np.sum(wi_re * yi) / (np.sum(wi_re) + self.lambda_val * k)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_ridge, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_ridge,
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


class LassoRegularizedMeta(MetaAnalysisMethod):
    """LASSO-regularized meta-analysis for sparsity"""

    def __init__(self, lambda_val: float = 0.1):
        super().__init__(f"LassoRegularized_{lambda_val}", "regularization")
        self.lambda_val = lambda_val

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

        # LASSO: soft thresholding
        def soft_threshold(x, t):
            return np.sign(x) * max(0, np.abs(x) - t)

        # Coordinate descent
        mu = mu_fe
        for _ in range(100):
            gradient = np.sum(wi_re * (mu - yi))
            hessian = np.sum(wi_re)

            mu_unpenalized = mu - gradient / hessian
            mu_new = soft_threshold(mu_unpenalized, self.lambda_val / hessian)

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
            weights=wi_re,
            computation_time=time.time() - start_time
        )


class ElasticNetMeta(MetaAnalysisMethod):
    """Elastic net regularized meta-analysis"""

    def __init__(self, lambda_val: float = 0.1, alpha: float = 0.5):
        super().__init__(f"ElasticNet_l{lambda_val}_a{alpha}", "regularization")
        self.lambda_val = lambda_val
        self.alpha = alpha  # Mix of L1 and L2

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

        # Elastic net optimization
        def objective(mu):
            loss = 0.5 * np.sum(wi_re * (yi - mu)**2)
            l1_pen = self.alpha * self.lambda_val * np.abs(mu)
            l2_pen = 0.5 * (1 - self.alpha) * self.lambda_val * mu**2
            return loss + l1_pen + l2_pen

        result = optimize.minimize_scalar(objective,
                                         bounds=(np.min(yi) - np.std(yi), np.max(yi) + np.std(yi)),
                                         method='bounded')
        mu_en = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_en, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_en,
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


class TikhonovRegularized(MetaAnalysisMethod):
    """Tikhonov regularization with prior"""

    def __init__(self, prior_mean: float = 0.0, prior_precision: float = 1.0):
        super().__init__(f"Tikhonov_pm{prior_mean}_pp{prior_precision}", "regularization")
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision

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

        # Tikhonov solution with prior
        sum_wi = np.sum(wi_re)
        sum_wi_yi = np.sum(wi_re * yi)

        mu_tik = (sum_wi_yi + self.prior_precision * self.prior_mean) / (sum_wi + self.prior_precision)
        se = np.sqrt(1.0 / (sum_wi + self.prior_precision))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_tik, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_tik,
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


class GroupLassoMeta(MetaAnalysisMethod):
    """Group LASSO for grouped meta-analysis"""

    def __init__(self, lambda_val: float = 0.1):
        super().__init__(f"GroupLasso_{lambda_val}", "regularization")
        self.lambda_val = lambda_val

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

        # Group studies by precision quartiles
        precision = 1.0 / (vi + tau2)
        quartiles = np.percentile(precision, [25, 50, 75])
        groups = np.digitize(precision, quartiles)

        # Group-wise weighted average with L2 penalty
        group_means = []
        group_weights = []

        for g in range(4):
            mask = groups == g
            if np.sum(mask) > 0:
                wi_g = wi_re[mask]
                yi_g = yi[mask]
                group_means.append(np.sum(wi_g * yi_g) / np.sum(wi_g))
                group_weights.append(np.sum(wi_g))

        # Weighted combination with group penalty
        group_means = np.array(group_means)
        group_weights = np.array(group_weights)
        group_weights = group_weights / np.sum(group_weights)

        mu_gl = np.sum(group_weights * group_means)

        # Apply soft-thresholding to group effect
        penalty_factor = self.lambda_val / (np.sum(wi_re) + 1e-10)
        if np.abs(mu_gl) > penalty_factor:
            mu_gl = np.sign(mu_gl) * (np.abs(mu_gl) - penalty_factor)
        else:
            mu_gl = 0.0

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_gl, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_gl,
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
# CATEGORY 12: GEOMETRIC METHODS
# =============================================================================

class GeometricMeanMeta(MetaAnalysisMethod):
    """Geometric mean meta-analysis (for positive effects)"""

    def __init__(self):
        super().__init__("GeometricMean", "geometric")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Shift to positive if needed
        min_y = np.min(yi)
        if min_y <= 0:
            yi_shifted = yi - min_y + 1
            shift = min_y - 1
        else:
            yi_shifted = yi
            shift = 0

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Weighted geometric mean
        log_yi = np.log(yi_shifted)
        mu_log = np.sum(wi_re * log_yi) / np.sum(wi_re)
        mu_geo = np.exp(mu_log) + shift

        # SE via delta method
        se_log = np.sqrt(1.0 / np.sum(wi_re))
        se = mu_geo * se_log  # Delta method

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_geo, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_geo,
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


class HarmonicMeanMeta(MetaAnalysisMethod):
    """Harmonic mean meta-analysis"""

    def __init__(self):
        super().__init__("HarmonicMean", "geometric")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Shift to positive
        min_y = np.min(yi)
        if min_y <= 0:
            yi_shifted = yi - min_y + 0.1
            shift = min_y - 0.1
        else:
            yi_shifted = yi
            shift = 0

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # Weighted harmonic mean
        inv_yi = 1.0 / yi_shifted
        inv_mean = np.sum(wi_re * inv_yi) / np.sum(wi_re)
        mu_harm = 1.0 / inv_mean + shift

        # SE via delta method
        se_inv = np.sqrt(1.0 / np.sum(wi_re))
        se = se_inv / inv_mean**2

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_harm, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_harm,
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


class PowerMeanMeta(MetaAnalysisMethod):
    """Power mean (generalized mean) meta-analysis"""

    def __init__(self, power: float = 2.0):
        super().__init__(f"PowerMean_p{power}", "geometric")
        self.power = power

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
        wi_norm = wi_re / np.sum(wi_re)

        p = self.power
        if p == 0:
            # Geometric mean
            mu_pm = np.exp(np.sum(wi_norm * np.log(np.abs(yi) + 0.001)))
        else:
            # Power mean
            yi_pow = np.sign(yi) * np.abs(yi)**p
            mean_pow = np.sum(wi_norm * yi_pow)
            mu_pm = np.sign(mean_pow) * np.abs(mean_pow)**(1/p)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_pm, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pm,
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


class LehmerMeanMeta(MetaAnalysisMethod):
    """Lehmer mean meta-analysis"""

    def __init__(self, power: float = 2.0):
        super().__init__(f"LehmerMean_p{power}", "geometric")
        self.power = power

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

        p = self.power
        yi_abs = np.abs(yi) + 0.001

        # Lehmer mean: sum(w * y^p) / sum(w * y^(p-1))
        numerator = np.sum(wi_re * np.sign(yi) * yi_abs**p)
        denominator = np.sum(wi_re * yi_abs**(p-1))
        mu_lehmer = numerator / (denominator + 1e-10)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_lehmer, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_lehmer,
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


class ContraharmonicMean(MetaAnalysisMethod):
    """Contraharmonic mean meta-analysis"""

    def __init__(self):
        super().__init__("ContraharmonicMean", "geometric")

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

        # Contraharmonic: sum(w * y^2) / sum(w * y)
        numerator = np.sum(wi_re * yi**2)
        denominator = np.sum(wi_re * yi)
        mu_ch = numerator / (denominator + 1e-10) if np.abs(denominator) > 1e-10 else mu_fe

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_ch, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_ch,
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
# CATEGORY 13: ALTERNATIVE LOSS FUNCTIONS
# =============================================================================

class HuberLossMeta(MetaAnalysisMethod):
    """Huber loss meta-analysis"""

    def __init__(self, delta: float = 1.345):
        super().__init__(f"HuberLoss_d{delta}", "loss_function")
        self.delta = delta

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
        delta = self.delta

        def huber_loss(mu):
            r = (yi - mu) * np.sqrt(wi_re)
            loss = np.where(np.abs(r) <= delta,
                           0.5 * r**2,
                           delta * (np.abs(r) - 0.5 * delta))
            return np.sum(loss)

        result = optimize.minimize_scalar(huber_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_huber = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_huber, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_huber,
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


class QuantileLossMeta(MetaAnalysisMethod):
    """Quantile loss (pinball loss) meta-analysis"""

    def __init__(self, quantile: float = 0.5):
        super().__init__(f"QuantileLoss_q{quantile}", "loss_function")
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
        q = self.quantile

        def pinball_loss(mu):
            r = yi - mu
            loss = np.where(r >= 0, q * r, (q - 1) * r)
            return np.sum(wi_re * loss)

        result = optimize.minimize_scalar(pinball_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_quantile = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_quantile, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_quantile,
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


class LogCoshLossMeta(MetaAnalysisMethod):
    """Log-cosh loss meta-analysis (smooth L1)"""

    def __init__(self):
        super().__init__("LogCoshLoss", "loss_function")

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

        def logcosh_loss(mu):
            r = (yi - mu) * np.sqrt(wi_re)
            return np.sum(np.log(np.cosh(r)))

        result = optimize.minimize_scalar(logcosh_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_logcosh = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_logcosh, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_logcosh,
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


class CauchyLossMeta(MetaAnalysisMethod):
    """Cauchy loss (Lorentzian) meta-analysis"""

    def __init__(self, scale: float = 1.0):
        super().__init__(f"CauchyLoss_s{scale}", "loss_function")
        self.scale = scale

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
        s = self.scale

        def cauchy_loss(mu):
            r = (yi - mu) * np.sqrt(wi_re)
            return np.sum(np.log(1 + (r / s)**2))

        result = optimize.minimize_scalar(cauchy_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_cauchy = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_cauchy, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_cauchy,
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


class GemanMcClureLoss(MetaAnalysisMethod):
    """Geman-McClure loss meta-analysis"""

    def __init__(self, scale: float = 1.0):
        super().__init__(f"GemanMcClure_s{scale}", "loss_function")
        self.scale = scale

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
        s = self.scale

        def gm_loss(mu):
            r = (yi - mu) * np.sqrt(wi_re)
            return np.sum(r**2 / (1 + r**2 / s**2))

        result = optimize.minimize_scalar(gm_loss,
                                         bounds=(np.min(yi), np.max(yi)),
                                         method='bounded')
        mu_gm = result.x

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_gm, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_gm,
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
# CATEGORY 14: PUBLICATION BIAS ADJUSTED
# =============================================================================

class TrimAndFillMeta(MetaAnalysisMethod):
    """Trim and fill publication bias adjustment"""

    def __init__(self, side: str = "right"):
        super().__init__(f"TrimAndFill_{side}", "publication_bias")
        self.side = side

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initial DL estimate
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)
        mu_0 = np.sum(wi_re * yi) / np.sum(wi_re)

        # Determine which side to trim
        if self.side == "right":
            extreme_idx = np.argsort(yi)[::-1]
        else:
            extreme_idx = np.argsort(yi)

        # Estimate number of missing studies (L0 estimator)
        deviations = yi - mu_0
        if self.side == "right":
            n_pos = np.sum(deviations > 0)
            n_neg = np.sum(deviations < 0)
            k0 = max(0, n_pos - n_neg)
        else:
            n_pos = np.sum(deviations > 0)
            n_neg = np.sum(deviations < 0)
            k0 = max(0, n_neg - n_pos)

        # Trim most extreme studies
        k0 = min(k0, k // 2)

        if k0 > 0:
            keep_idx = extreme_idx[k0:]
            yi_trim = yi[keep_idx]
            vi_trim = vi[keep_idx]

            # Fill in symmetric studies
            trim_yi = yi[extreme_idx[:k0]]
            trim_vi = vi[extreme_idx[:k0]]
            fill_yi = 2 * mu_0 - trim_yi
            fill_vi = trim_vi

            yi_filled = np.concatenate([yi_trim, fill_yi])
            vi_filled = np.concatenate([vi_trim, fill_vi])
        else:
            yi_filled = yi
            vi_filled = vi

        # Re-estimate with filled data
        wi_filled = 1.0 / vi_filled
        mu_fe_f = np.sum(wi_filled * yi_filled) / np.sum(wi_filled)
        Q_f = np.sum(wi_filled * (yi_filled - mu_fe_f)**2)
        c_f = np.sum(wi_filled) - np.sum(wi_filled**2) / np.sum(wi_filled)
        tau2_f = max(0, (Q_f - (len(yi_filled) - 1)) / c_f) if c_f > 0 else 0.0

        wi_re_f = 1.0 / (vi_filled + tau2_f)
        mu_tf = np.sum(wi_re_f * yi_filled) / np.sum(wi_re_f)
        se_tf = np.sqrt(1.0 / np.sum(wi_re_f))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2_f)
        ci_lo, ci_hi = self.compute_ci(mu_tf, se_tf)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_tf,
            pooled_se=se_tf,
            tau2=tau2_f,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'studies_filled': k0, 'total_studies': len(yi_filled)},
            computation_time=time.time() - start_time
        )


class SelectionModelMeta(MetaAnalysisMethod):
    """Simple selection model for publication bias"""

    def __init__(self, selection_type: str = "one_sided"):
        super().__init__(f"SelectionModel_{selection_type}", "publication_bias")
        self.selection_type = selection_type

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)
        se = np.sqrt(vi)

        # Compute p-values (one-sided)
        z = yi / se
        if self.selection_type == "one_sided":
            pvals = 1 - stats.norm.cdf(z)
        else:
            pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))

        # Weight by inverse of selection probability
        # Assume studies with p < 0.05 have probability 1, others have lower
        selection_weights = np.where(pvals < 0.05, 1.0, np.exp(-2 * (pvals - 0.05)))

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Selection-adjusted weights
        wi_re = 1.0 / (vi + tau2)
        wi_adj = wi_re / selection_weights  # Upweight non-significant

        mu_sel = np.sum(wi_adj * yi) / np.sum(wi_adj)
        se_sel = np.sqrt(1.0 / np.sum(wi_adj))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_sel, se_sel)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_sel,
            pooled_se=se_sel,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_adj,
            computation_time=time.time() - start_time
        )


class PETMeta(MetaAnalysisMethod):
    """Precision Effect Test (PET) for publication bias"""

    def __init__(self):
        super().__init__("PET", "publication_bias")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)
        se = np.sqrt(vi)

        # PET: regress effect sizes on standard errors
        # yi = b0 + b1*se_i + error
        X = np.column_stack([np.ones(k), se])
        W = np.diag(1.0 / vi)

        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ yi
            beta = np.linalg.solve(XtWX, XtWy)
            mu_pet = beta[0]  # Intercept is bias-corrected estimate

            # SE of intercept
            cov_beta = np.linalg.inv(XtWX)
            se_pet = np.sqrt(cov_beta[0, 0])
        except:
            wi = 1.0 / vi
            mu_pet = np.sum(wi * yi) / np.sum(wi)
            se_pet = np.sqrt(1.0 / np.sum(wi))

        # DL tau2 for reference
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_pet, se_pet)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pet,
            pooled_se=se_pet,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class PEESEMeta(MetaAnalysisMethod):
    """Precision Effect Estimate with Standard Error (PEESE)"""

    def __init__(self):
        super().__init__("PEESE", "publication_bias")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)
        se = np.sqrt(vi)

        # PEESE: regress on variance (se^2)
        X = np.column_stack([np.ones(k), vi])
        W = np.diag(1.0 / vi)

        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ yi
            beta = np.linalg.solve(XtWX, XtWy)
            mu_peese = beta[0]

            cov_beta = np.linalg.inv(XtWX)
            se_peese = np.sqrt(cov_beta[0, 0])
        except:
            wi = 1.0 / vi
            mu_peese = np.sum(wi * yi) / np.sum(wi)
            se_peese = np.sqrt(1.0 / np.sum(wi))

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_peese, se_peese)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_peese,
            pooled_se=se_peese,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class VevaetalMeta(MetaAnalysisMethod):
    """Vevea-Hedges weight function selection model (simplified)"""

    def __init__(self, steps: List[float] = [0.05, 0.10]):
        super().__init__(f"VeveaHedges_steps{len(steps)}", "publication_bias")
        self.steps = steps

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)
        se = np.sqrt(vi)

        # Compute one-sided p-values
        z = yi / se
        pvals = 1 - stats.norm.cdf(z)

        # Step function weights based on p-value intervals
        steps = [0] + self.steps + [1.0]
        n_intervals = len(steps) - 1

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Estimate weights for each interval via method of moments
        interval_counts = []
        interval_effects = []

        for i in range(n_intervals):
            mask = (pvals >= steps[i]) & (pvals < steps[i+1])
            if np.sum(mask) > 0:
                interval_counts.append(np.sum(mask))
                wi_int = 1.0 / (vi[mask] + tau2)
                interval_effects.append(np.sum(wi_int * yi[mask]) / np.sum(wi_int))

        # Adjust weights inversely proportional to interval size
        # (fewer studies in an interval suggests suppression)
        expected_per_interval = k / n_intervals
        interval_weights = np.array([expected_per_interval / max(1, c) for c in interval_counts])
        interval_weights = interval_weights / np.sum(interval_weights)

        if len(interval_effects) > 0:
            mu_vh = np.sum(interval_weights * interval_effects)
        else:
            mu_vh = mu_fe

        wi_re = 1.0 / (vi + tau2)
        se_vh = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_vh, se_vh)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_vh,
            pooled_se=se_vh,
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

def get_part2_methods():
    """Return all experimental methods from this module"""
    methods = []

    # Regularization
    for lam in [0.01, 0.1, 0.5, 1.0]:
        methods.append(RidgeRegularizedMeta(lambda_val=lam))
        methods.append(LassoRegularizedMeta(lambda_val=lam))

    for lam in [0.1, 0.5]:
        for alpha in [0.25, 0.5, 0.75]:
            methods.append(ElasticNetMeta(lambda_val=lam, alpha=alpha))

    for pm in [0.0, 0.5]:
        for pp in [0.1, 1.0, 10.0]:
            methods.append(TikhonovRegularized(prior_mean=pm, prior_precision=pp))

    for lam in [0.1, 0.5]:
        methods.append(GroupLassoMeta(lambda_val=lam))

    # Geometric methods
    methods.append(GeometricMeanMeta())
    methods.append(HarmonicMeanMeta())
    for p in [-1.0, 0.5, 2.0, 3.0]:
        methods.append(PowerMeanMeta(power=p))
    for p in [1.5, 2.0, 3.0]:
        methods.append(LehmerMeanMeta(power=p))
    methods.append(ContraharmonicMean())

    # Loss functions
    for delta in [1.0, 1.345, 2.0]:
        methods.append(HuberLossMeta(delta=delta))
    for q in [0.25, 0.5, 0.75]:
        methods.append(QuantileLossMeta(quantile=q))
    methods.append(LogCoshLossMeta())
    for s in [0.5, 1.0, 2.0]:
        methods.append(CauchyLossMeta(scale=s))
        methods.append(GemanMcClureLoss(scale=s))

    # Publication bias
    for side in ["left", "right"]:
        methods.append(TrimAndFillMeta(side=side))
    for sel_type in ["one_sided", "two_sided"]:
        methods.append(SelectionModelMeta(selection_type=sel_type))
    methods.append(PETMeta())
    methods.append(PEESEMeta())
    methods.append(VevaetalMeta(steps=[0.05]))
    methods.append(VevaetalMeta(steps=[0.05, 0.10]))
    methods.append(VevaetalMeta(steps=[0.025, 0.05, 0.10]))

    return methods


if __name__ == "__main__":
    methods = get_part2_methods()
    print(f"Part 2 contains {len(methods)} experimental methods")
    for m in methods:
        print(f"  - {m.name} ({m.category})")

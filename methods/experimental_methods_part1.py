"""
Experimental Meta-Analysis Methods - Part 1
============================================
Novel approaches: Adaptive, Mixture Models, Kernel-based, Information-theoretic
"""

import numpy as np
from scipy import stats, optimize, special
from scipy.integrate import quad
from scipy.spatial.distance import cdist
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
# CATEGORY 7: ADAPTIVE METHODS
# =============================================================================

class AdaptiveWeightMeta(MetaAnalysisMethod):
    """Adaptive weight adjustment based on leverage"""

    def __init__(self, sensitivity: float = 1.0):
        super().__init__(f"AdaptiveWeight_sens{sensitivity}", "adaptive")
        self.sensitivity = sensitivity

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initial DL
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Iterative adaptive weighting
        for _ in range(50):
            wi_re = 1.0 / (vi + tau2)
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

            # Leverage and influence
            h = wi_re / np.sum(wi_re)
            residuals = yi - mu
            influence = np.abs(residuals) * np.sqrt(wi_re)

            # Adaptive adjustment
            adjustment = 1.0 / (1 + self.sensitivity * influence / np.median(influence))
            wi_adapt = wi_re * adjustment

            mu_new = np.sum(wi_adapt * yi) / np.sum(wi_adapt)

            # Update tau2
            Q_new = np.sum(wi_adapt * (yi - mu_new)**2)
            c_new = np.sum(wi_adapt) - np.sum(wi_adapt**2) / np.sum(wi_adapt)
            tau2 = max(0, (Q_new - (k - 1)) / c_new) if c_new > 0 else tau2

            if abs(mu_new - mu) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_adapt))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_new, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_new,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_adapt,
            computation_time=time.time() - start_time
        )


class IterativelyReweighted(MetaAnalysisMethod):
    """Iteratively reweighted least squares meta-analysis"""

    def __init__(self, loss: str = "huber"):
        super().__init__(f"IRLS_{loss}", "adaptive")
        self.loss = loss

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        def weight_function(r, loss_type):
            r_abs = np.abs(r) + 1e-10
            if loss_type == "huber":
                c = 1.345
                return np.where(r_abs <= c, 1.0, c / r_abs)
            elif loss_type == "cauchy":
                c = 2.385
                return 1.0 / (1 + (r / c)**2)
            elif loss_type == "welsch":
                c = 2.985
                return np.exp(-(r / c)**2)
            elif loss_type == "fair":
                c = 1.4
                return 1.0 / (1 + r_abs / c)
            else:
                return np.ones_like(r)

        # Initial estimate
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)
        tau2 = np.var(yi) / 2

        for iteration in range(100):
            mu_old = mu
            wi_total = 1.0 / (vi + tau2)
            scale = np.median(np.abs(yi - mu)) * 1.4826

            if scale < 1e-10:
                scale = np.std(yi) + 1e-10

            r = (yi - mu) / scale
            rw = weight_function(r, self.loss)

            wi_irls = wi_total * rw
            mu = np.sum(wi_irls * yi) / np.sum(wi_irls)

            # Update tau2
            Q = np.sum(wi_irls * (yi - mu)**2)
            c = np.sum(wi_irls) - np.sum(wi_irls**2) / np.sum(wi_irls)
            tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else tau2

            if abs(mu - mu_old) < 1e-8:
                break

        se = np.sqrt(1.0 / np.sum(wi_irls))
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
            weights=wi_irls,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class OutlierAdaptiveMeta(MetaAnalysisMethod):
    """Outlier-adaptive meta-analysis with automatic detection"""

    def __init__(self, outlier_threshold: float = 2.5):
        super().__init__(f"OutlierAdaptive_thresh{outlier_threshold}", "adaptive")
        self.threshold = outlier_threshold

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initial estimate
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        for _ in range(20):
            wi_re = 1.0 / (vi + tau2)
            mu = np.sum(wi_re * yi) / np.sum(wi_re)

            # Standardized residuals
            se_total = np.sqrt(vi + tau2)
            z_scores = (yi - mu) / se_total

            # Soft downweighting for outliers
            outlier_weights = np.where(
                np.abs(z_scores) > self.threshold,
                (self.threshold / np.abs(z_scores))**2,
                1.0
            )

            wi_adapt = wi_re * outlier_weights
            mu_new = np.sum(wi_adapt * yi) / np.sum(wi_adapt)

            # Update tau2
            Q_new = np.sum(wi_adapt * (yi - mu_new)**2)
            c_new = np.sum(wi_adapt) - np.sum(wi_adapt**2) / np.sum(wi_adapt)
            tau2 = max(0, (Q_new - (k - 1)) / c_new) if c_new > 0 else tau2

            if abs(mu_new - mu) < 1e-8:
                break
            mu = mu_new

        se = np.sqrt(1.0 / np.sum(wi_adapt))
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
            weights=wi_adapt,
            additional_info={'outliers_detected': np.sum(outlier_weights < 0.5)},
            computation_time=time.time() - start_time
        )


class SequentialAdaptive(MetaAnalysisMethod):
    """Sequential adaptive learning meta-analysis"""

    def __init__(self, learning_rate: float = 0.1):
        super().__init__(f"SequentialAdaptive_lr{learning_rate}", "adaptive")
        self.lr = learning_rate

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Sort by precision (proxy for chronological in some contexts)
        sorted_idx = np.argsort(-1.0/vi)
        yi_sorted = yi[sorted_idx]
        vi_sorted = vi[sorted_idx]

        # Sequential update
        mu = yi_sorted[0]
        tau2 = 0.0
        cumulative_weight = 1.0 / vi_sorted[0]

        for i in range(1, k):
            prior_var = 1.0 / cumulative_weight + tau2
            likelihood_var = vi_sorted[i]

            # Bayesian update
            posterior_var = 1.0 / (1.0/prior_var + 1.0/likelihood_var)
            mu_new = posterior_var * (mu/prior_var + yi_sorted[i]/likelihood_var)

            # Adaptive tau2 update
            prediction_error = (yi_sorted[i] - mu)**2
            expected_var = prior_var + likelihood_var
            tau2 = tau2 + self.lr * max(0, prediction_error - expected_var)

            mu = mu_new
            cumulative_weight += 1.0 / (vi_sorted[i] + tau2)

        # Final estimate with all data
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


# =============================================================================
# CATEGORY 8: MIXTURE MODELS
# =============================================================================

class GaussianMixtureMeta(MetaAnalysisMethod):
    """Gaussian mixture model meta-analysis"""

    def __init__(self, n_components: int = 2):
        super().__init__(f"GaussianMixture_{n_components}comp", "mixture")
        self.n_components = n_components

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initialize mixture components
        n_comp = min(self.n_components, k)
        quantiles = np.linspace(0, 1, n_comp + 2)[1:-1]
        mu_components = np.percentile(yi, 100 * quantiles)
        tau2_components = np.ones(n_comp) * np.var(yi) / n_comp
        pi_components = np.ones(n_comp) / n_comp

        # EM algorithm
        for _ in range(100):
            # E-step: compute responsibilities
            resp = np.zeros((k, n_comp))
            for j in range(n_comp):
                total_var = vi + tau2_components[j]
                resp[:, j] = pi_components[j] * stats.norm.pdf(yi, mu_components[j], np.sqrt(total_var))

            resp_sum = np.sum(resp, axis=1, keepdims=True) + 1e-10
            resp = resp / resp_sum

            # M-step
            n_j = np.sum(resp, axis=0) + 1e-10
            pi_components = n_j / k

            for j in range(n_comp):
                wi_j = resp[:, j] / (vi + tau2_components[j])
                mu_components[j] = np.sum(wi_j * yi) / np.sum(wi_j)

                weighted_ss = np.sum(resp[:, j] * (yi - mu_components[j])**2)
                weighted_vi = np.sum(resp[:, j] * vi)
                tau2_components[j] = max(0, (weighted_ss - weighted_vi) / n_j[j])

        # Final pooled estimate (weighted by component probabilities)
        mu_pooled = np.sum(pi_components * mu_components)
        tau2_pooled = np.sum(pi_components * tau2_components) + np.sum(pi_components * (mu_components - mu_pooled)**2)

        wi_re = 1.0 / (vi + tau2_pooled)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2_pooled)
        ci_lo, ci_hi = self.compute_ci(mu_pooled, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pooled,
            pooled_se=se,
            tau2=tau2_pooled,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'n_components': n_comp, 'component_means': mu_components.tolist()},
            computation_time=time.time() - start_time
        )


class ContaminatedNormalMeta(MetaAnalysisMethod):
    """Contaminated normal model for outlier-robust estimation"""

    def __init__(self, contamination: float = 0.1, scale_factor: float = 5.0):
        super().__init__(f"ContaminatedNormal_c{contamination}", "mixture")
        self.contamination = contamination
        self.scale_factor = scale_factor

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initialize
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        epsilon = self.contamination
        scale = self.scale_factor

        for _ in range(50):
            # E-step: posterior probability of being from the "good" component
            var_good = vi + tau2
            var_bad = vi + tau2 * scale**2

            lik_good = (1 - epsilon) * stats.norm.pdf(yi, mu, np.sqrt(var_good))
            lik_bad = epsilon * stats.norm.pdf(yi, mu, np.sqrt(var_bad))

            posterior_good = lik_good / (lik_good + lik_bad + 1e-10)

            # M-step
            wi_effective = posterior_good / var_good + (1 - posterior_good) / var_bad
            mu_new = np.sum(wi_effective * yi) / np.sum(wi_effective)

            # Update tau2
            weighted_ss = np.sum(posterior_good * (yi - mu_new)**2 / var_good)
            weighted_ss += np.sum((1 - posterior_good) * (yi - mu_new)**2 / var_bad)

            tau2_new = max(0, np.sum(posterior_good * ((yi - mu_new)**2 - vi)) / np.sum(posterior_good))

            if abs(mu_new - mu) < 1e-8 and abs(tau2_new - tau2) < 1e-8:
                break

            mu, tau2 = mu_new, tau2_new

        se = np.sqrt(1.0 / np.sum(wi_effective))
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
            weights=wi_effective,
            additional_info={'estimated_outliers': np.sum(posterior_good < 0.5)},
            computation_time=time.time() - start_time
        )


class StudentTMixtureMeta(MetaAnalysisMethod):
    """Student-t mixture model for heavy tails"""

    def __init__(self, df: float = 5.0):
        super().__init__(f"StudentTMixture_df{df}", "mixture")
        self.df = df

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Initialize
        wi = 1.0 / vi
        mu = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        nu = self.df

        for iteration in range(100):
            # E-step: expected weights from t-distribution
            total_var = vi + tau2
            delta = (yi - mu)**2 / total_var
            w_t = (nu + 1) / (nu + delta)

            # M-step
            wi_effective = w_t / total_var
            mu_new = np.sum(wi_effective * yi) / np.sum(wi_effective)

            # Update tau2
            weighted_ss = np.sum(w_t * (yi - mu_new)**2)
            weighted_vi = np.sum(w_t * vi)
            tau2_new = max(0, (weighted_ss - weighted_vi) / k)

            if abs(mu_new - mu) < 1e-8:
                break

            mu, tau2 = mu_new, tau2_new

        se = np.sqrt(1.0 / np.sum(wi_effective))
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
            weights=wi_effective,
            iterations=iteration,
            computation_time=time.time() - start_time
        )


class LatentClassMeta(MetaAnalysisMethod):
    """Latent class meta-analysis model"""

    def __init__(self, n_classes: int = 2):
        super().__init__(f"LatentClass_{n_classes}", "mixture")
        self.n_classes = n_classes

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        n_class = min(self.n_classes, k)

        # Initialize with k-means-style clustering
        sorted_yi = np.sort(yi)
        boundaries = np.linspace(0, k, n_class + 1).astype(int)
        mu_class = np.array([np.mean(sorted_yi[boundaries[i]:boundaries[i+1]])
                           for i in range(n_class)])
        tau2_class = np.ones(n_class) * np.var(yi) / n_class
        pi_class = np.ones(n_class) / n_class

        # EM
        for _ in range(100):
            # E-step
            resp = np.zeros((k, n_class))
            for c in range(n_class):
                var_c = vi + tau2_class[c]
                resp[:, c] = pi_class[c] * stats.norm.pdf(yi, mu_class[c], np.sqrt(var_c))

            resp = resp / (np.sum(resp, axis=1, keepdims=True) + 1e-10)

            # M-step
            n_c = np.sum(resp, axis=0) + 1e-10
            pi_class = n_c / k

            for c in range(n_class):
                wi_c = resp[:, c] / (vi + tau2_class[c])
                mu_class[c] = np.sum(wi_c * yi) / np.sum(wi_c)
                tau2_class[c] = max(0, np.sum(resp[:, c] * ((yi - mu_class[c])**2 - vi)) / n_c[c])

        # Pooled estimate
        class_assignment = np.argmax(resp, axis=1)
        main_class = np.argmax(np.bincount(class_assignment, minlength=n_class))
        mu_pooled = mu_class[main_class]
        tau2_pooled = tau2_class[main_class]

        wi_re = 1.0 / (vi + tau2_pooled)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2_pooled)
        ci_lo, ci_hi = self.compute_ci(mu_pooled, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_pooled,
            pooled_se=se,
            tau2=tau2_pooled,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            additional_info={'class_means': mu_class.tolist(), 'class_proportions': pi_class.tolist()},
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 9: KERNEL-BASED METHODS
# =============================================================================

class KernelWeightedMeta(MetaAnalysisMethod):
    """Kernel-weighted meta-analysis with adaptive bandwidth"""

    def __init__(self, kernel: str = "gaussian", bandwidth: str = "silverman"):
        super().__init__(f"KernelWeighted_{kernel}_{bandwidth}", "kernel")
        self.kernel = kernel
        self.bandwidth = bandwidth

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Calculate bandwidth
        if self.bandwidth == "silverman":
            h = 1.06 * np.std(yi) * k**(-0.2)
        elif self.bandwidth == "scott":
            h = 1.059 * np.std(yi) * k**(-0.2)
        else:
            h = np.std(yi) / 2

        h = max(h, 0.01)

        # Kernel functions
        def gaussian_kernel(u):
            return np.exp(-0.5 * u**2)

        def epanechnikov_kernel(u):
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

        def tricube_kernel(u):
            return np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0)

        kernels = {
            "gaussian": gaussian_kernel,
            "epanechnikov": epanechnikov_kernel,
            "tricube": tricube_kernel
        }
        kernel_func = kernels.get(self.kernel, gaussian_kernel)

        # DL for tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Kernel density estimate for effect sizes
        grid = np.linspace(np.min(yi) - 2*h, np.max(yi) + 2*h, 200)
        density = np.zeros(len(grid))

        wi_re = 1.0 / (vi + tau2)
        wi_norm = wi_re / np.sum(wi_re)

        for i in range(k):
            u = (grid - yi[i]) / h
            density += wi_norm[i] * kernel_func(u) / h

        # Find mode
        mu_mode = grid[np.argmax(density)]

        # Kernel weights centered at mode
        kernel_weights = np.zeros(k)
        for i in range(k):
            u = (yi[i] - mu_mode) / h
            kernel_weights[i] = kernel_func(u)

        wi_kernel = wi_re * kernel_weights
        wi_kernel = np.maximum(wi_kernel, 1e-10)

        mu_final = np.sum(wi_kernel * yi) / np.sum(wi_kernel)
        se = np.sqrt(1.0 / np.sum(wi_kernel))

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
            weights=wi_kernel,
            additional_info={'bandwidth': h, 'mode': mu_mode},
            computation_time=time.time() - start_time
        )


class LocalPolynomialMeta(MetaAnalysisMethod):
    """Local polynomial regression for meta-analysis"""

    def __init__(self, degree: int = 1, bandwidth: float = None):
        super().__init__(f"LocalPolynomial_deg{degree}", "kernel")
        self.degree = degree
        self.bandwidth = bandwidth

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Get bandwidth
        h = self.bandwidth if self.bandwidth else np.std(yi) * k**(-0.2)
        h = max(h, 0.01)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        # Local polynomial fit at the weighted median
        wi_re = 1.0 / (vi + tau2)
        sorted_idx = np.argsort(yi)
        cumsum = np.cumsum(wi_re[sorted_idx])
        med_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        x0 = yi[sorted_idx[min(med_idx, k-1)]]

        # Kernel weights
        kernel_w = np.exp(-0.5 * ((yi - x0) / h)**2)

        # Build design matrix
        X = np.column_stack([yi**p for p in range(self.degree + 1)])
        X0 = np.array([x0**p for p in range(self.degree + 1)])

        # Weighted least squares
        W = np.diag(kernel_w * wi_re)
        try:
            beta = np.linalg.solve(X.T @ W @ X + 1e-6 * np.eye(self.degree + 1), X.T @ W @ yi)
            mu_local = X0 @ beta
        except:
            mu_local = np.sum(wi_re * yi) / np.sum(wi_re)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_local, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_local,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            computation_time=time.time() - start_time
        )


class NadarayaWatsonMeta(MetaAnalysisMethod):
    """Nadaraya-Watson kernel regression meta-analysis"""

    def __init__(self, bandwidth: float = None):
        super().__init__("NadarayaWatson", "kernel")
        self.bandwidth = bandwidth

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        h = self.bandwidth if self.bandwidth else np.std(yi) * k**(-0.2)
        h = max(h, 0.01)

        # DL tau2
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        wi_re = 1.0 / (vi + tau2)

        # NW estimate at the center of mass
        center = np.sum(wi_re * yi) / np.sum(wi_re)

        # Kernel weights
        K = np.exp(-0.5 * ((yi - center) / h)**2)
        wi_nw = K * wi_re
        mu_nw = np.sum(wi_nw * yi) / np.sum(wi_nw)

        se = np.sqrt(1.0 / np.sum(wi_re))
        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_nw, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_nw,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_nw,
            computation_time=time.time() - start_time
        )


# =============================================================================
# CATEGORY 10: INFORMATION-THEORETIC METHODS
# =============================================================================

class MaximumEntropyMeta(MetaAnalysisMethod):
    """Maximum entropy meta-analysis"""

    def __init__(self):
        super().__init__("MaximumEntropy", "information")

    def estimate(self, data: MetaAnalysisData) -> MetaAnalysisResult:
        start_time = time.time()
        yi, vi = data.effect_sizes, data.variances
        k = len(yi)

        # Maximum entropy subject to moment constraints
        def neg_entropy(w):
            w = np.clip(w, 1e-10, 1)
            return np.sum(w * np.log(w))

        def constraint_sum(w):
            return np.sum(w) - 1

        def constraint_variance(w, target_var):
            mu_w = np.sum(w * yi)
            return np.sum(w * (yi - mu_w)**2) - target_var

        # Start with uniform weights
        w0 = np.ones(k) / k

        # DL tau2 for target variance
        wi = 1.0 / vi
        mu_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - mu_fe)**2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / c) if c > 0 else 0.0

        target_var = np.mean(vi) + tau2

        # Constrained optimization
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
        ]

        bounds = [(1e-10, 1) for _ in range(k)]

        try:
            result = optimize.minimize(neg_entropy, w0, method='SLSQP',
                                      bounds=bounds, constraints=constraints)
            w_maxent = result.x
            w_maxent = w_maxent / np.sum(w_maxent)
        except:
            w_maxent = w0

        mu_maxent = np.sum(w_maxent * yi)

        # Combine with precision weights
        wi_re = 1.0 / (vi + tau2)
        wi_final = np.sqrt(w_maxent * wi_re)
        wi_final = wi_final / np.sum(wi_final)

        mu_final = np.sum(wi_final * yi)
        se = np.sqrt(np.sum(wi_final**2 * (vi + tau2)))

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
            weights=wi_final,
            computation_time=time.time() - start_time
        )


class KLDivergenceMeta(MetaAnalysisMethod):
    """KL-divergence minimizing meta-analysis"""

    def __init__(self):
        super().__init__("KLDivergence", "information")

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

        # Find mu that minimizes total KL divergence to all studies
        def total_kl(mu):
            sigma2 = vi + tau2
            # KL(N(yi, vi) || N(mu, sigma2)) for each study
            kl = np.sum(0.5 * ((yi - mu)**2 / sigma2 + np.log(sigma2 / vi) - 1 + vi / sigma2))
            return kl

        result = optimize.minimize_scalar(total_kl,
                                         bounds=(np.min(yi) - 2*np.std(yi), np.max(yi) + 2*np.std(yi)),
                                         method='bounded')
        mu_kl = result.x

        wi_re = 1.0 / (vi + tau2)
        se = np.sqrt(1.0 / np.sum(wi_re))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_kl, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_kl,
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


class MutualInformationMeta(MetaAnalysisMethod):
    """Mutual information weighted meta-analysis"""

    def __init__(self):
        super().__init__("MutualInformation", "information")

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

        # Information content of each study
        # Higher precision = more information
        total_var = vi + tau2
        information = -np.log(np.sqrt(2 * np.pi * total_var))  # Negative entropy

        # Normalize to weights
        info_weights = np.exp(information - np.max(information))
        info_weights = info_weights / np.sum(info_weights)

        # Combine with inverse variance
        wi_re = 1.0 / (vi + tau2)
        wi_combined = np.sqrt(info_weights * wi_re)
        wi_combined = wi_combined / np.sum(wi_combined)

        mu_mi = np.sum(wi_combined * yi)
        se = np.sqrt(np.sum(wi_combined**2 * (vi + tau2)))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_mi, se)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_mi,
            pooled_se=se,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=wi_combined,
            computation_time=time.time() - start_time
        )


class FisherInformationMeta(MetaAnalysisMethod):
    """Fisher information weighted meta-analysis"""

    def __init__(self):
        super().__init__("FisherInformation", "information")

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

        # Fisher information: I(mu) = 1/variance
        total_var = vi + tau2
        fisher_info = 1.0 / total_var

        # Fisher-optimal pooled estimate
        mu_fisher = np.sum(fisher_info * yi) / np.sum(fisher_info)
        se_fisher = np.sqrt(1.0 / np.sum(fisher_info))

        Q_stat, I2, p_het = self.compute_heterogeneity_stats(data, tau2)
        ci_lo, ci_hi = self.compute_ci(mu_fisher, se_fisher)

        return MetaAnalysisResult(
            method_name=self.name,
            pooled_effect=mu_fisher,
            pooled_se=se_fisher,
            tau2=tau2,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            i2=I2,
            q_stat=Q_stat,
            p_heterogeneity=p_het,
            weights=fisher_info,
            computation_time=time.time() - start_time
        )


# =============================================================================
# COLLECT ALL METHODS FROM THIS MODULE
# =============================================================================

def get_part1_methods():
    """Return all experimental methods from this module"""
    methods = []

    # Adaptive methods
    for sens in [0.5, 1.0, 2.0]:
        methods.append(AdaptiveWeightMeta(sensitivity=sens))

    for loss in ["huber", "cauchy", "welsch", "fair"]:
        methods.append(IterativelyReweighted(loss=loss))

    for thresh in [2.0, 2.5, 3.0]:
        methods.append(OutlierAdaptiveMeta(outlier_threshold=thresh))

    for lr in [0.05, 0.1, 0.2]:
        methods.append(SequentialAdaptive(learning_rate=lr))

    # Mixture models
    for n_comp in [2, 3, 4]:
        methods.append(GaussianMixtureMeta(n_components=n_comp))

    for contam in [0.05, 0.1, 0.15]:
        methods.append(ContaminatedNormalMeta(contamination=contam))

    for df in [3.0, 5.0, 10.0]:
        methods.append(StudentTMixtureMeta(df=df))

    for n_class in [2, 3]:
        methods.append(LatentClassMeta(n_classes=n_class))

    # Kernel methods
    for kernel in ["gaussian", "epanechnikov", "tricube"]:
        for bw in ["silverman", "scott"]:
            methods.append(KernelWeightedMeta(kernel=kernel, bandwidth=bw))

    for degree in [1, 2, 3]:
        methods.append(LocalPolynomialMeta(degree=degree))

    methods.append(NadarayaWatsonMeta())

    # Information-theoretic
    methods.append(MaximumEntropyMeta())
    methods.append(KLDivergenceMeta())
    methods.append(MutualInformationMeta())
    methods.append(FisherInformationMeta())

    return methods


if __name__ == "__main__":
    methods = get_part1_methods()
    print(f"Part 1 contains {len(methods)} experimental methods")
    for m in methods:
        print(f"  - {m.name} ({m.category})")

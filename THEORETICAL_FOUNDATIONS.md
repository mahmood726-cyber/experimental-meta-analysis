# Theoretical Foundations of Experimental Meta-Analysis Methods

**Version:** 1.1.0
**Date:** 14 January 2026
**Framework:** Experimental Meta-Analysis (Python)

---

## Overview

This document provides the statistical and mathematical foundations for all 300+ meta-analysis methods implemented in this framework. Each method category includes:

1. **Mathematical Formulation** - Core equations and algorithms
2. **Statistical Properties** - Bias, variance, consistency, asymptotic behavior
3. **Literature Citations** - Primary theoretical sources
4. **When to Use** - Theoretical rationale for method selection
5. **Limitations** - Known theoretical or practical issues

---

## Table of Contents

1. [Standard Reference Methods](#1-standard-reference-methods)
2. [Robust Estimation Methods](#2-robust-estimation-methods)
3. [Bayesian Methods](#3-bayesian-methods)
4. [Heterogeneity Estimation](#4-heterogeneity-estimation)
5. [Small Sample Corrections](#5-small-sample-corrections)
6. [Weighting Schemes](#6-weighting-schemes)
7. [Resampling Methods](#7-resampling-methods)
8. [Adaptive Methods](#8-adaptive-methods)
9. [Mixture Models](#9-mixture-models)
10. [Nonparametric Methods](#10-nonparametric-methods)
11. [Information-Theoretic Methods](#11-information-theoretic-methods)
12. [Regularization Methods](#12-regularization-methods)
13. [Generalized Mean Methods](#13-generalized-mean-methods)
14. [Loss-Based Methods](#14-loss-based-methods)
15. [Publication Bias Adjustment](#15-publication-bias-adjustment)
16. [Ensemble Methods](#16-ensemble-methods)
17. [Machine Learning Inspired Methods](#17-machine-learning-inspired-methods)
18. [Dependence Modeling](#18-dependence-modeling)
19. [Heavy-Tailed Methods](#19-heavy-tailed-methods)
20. [Density Estimation Methods](#20-density-estimation-methods)
21. [Quantile Methods](#21-quantile-methods)
22. [Optimization Methods](#22-optimization-methods)
23. [Signal Processing Methods](#23-signal-processing-methods)
24. [Functional Data Methods](#24-functional-data-methods)
25. [Hybrid Methods](#25-hybrid-methods)
26. [Experimental Methods](#26-experimental-methods)
27. [Validation Methods](#27-validation-methods)
28. [Order Statistics Methods](#28-order-statistics-methods)
29. [Clustering Methods](#29-clustering-methods)
30. [Moment-Based Methods](#30-moment-based-methods)

---

## 1. Standard Reference Methods

### 1.1 DerSimonian-Laird (DL)

**Mathematical Formulation:**
```
Q = sum(wi * (yi - y_fe)^2)  [Cochran's Q]
tau2 = max(0, (Q - (k-1)) / C)
where: C = sum(wi) - sum(wi^2) / sum(wi)
wi* = 1 / (vi + tau2)
y_dl = sum(wi* * yi) / sum(wi*)
```

**Statistical Properties:**
- Method-of-moments estimator
- Consistent for tau2 as k -> infinity
- Underestimates tau2 for small k (k < 10)
- Can produce negative tau2 estimates (truncated at 0)

**When to Use:**
- Default choice for most meta-analyses
- Large k (k > 20) preferred
- When computational simplicity is valued

**Limitations:**
- Poor coverage for small k
- Negative tau2 truncated at 0 (biased)
- Assumes tau2 is constant across studies

**References:**
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.

---

### 1.2 Restricted Maximum Likelihood (REML)

**Mathematical Formulation:**
```
L_REML(tau2) = -0.5 * [log|V| + y' * V^(-1) * y + log|X' * V^(-1) * X|]
where: V = diag(vi) + tau2 * J
Maximize L_REML w.r.t tau2 using Fisher scoring or Newton-Raphson
```

**Statistical Properties:**
- Unbiased estimator of variance components
- Better small-sample properties than ML
- Accounts for loss of degrees of freedom from fixed effects
- Downward bias in tau2 for very small k (k < 5)

**When to Use:**
- Likelihood-based inference preferred
- k >= 10 (adequate for REML)
- When standard errors matter more than point estimates

**Limitations:**
- Requires optimization (can fail to converge)
- Slower than DL
- Bias in variance for k < 5

**References:**
- Harville, D. A. (1977). Maximum likelihood approaches to variance component estimation and to related problems. *Journal of the American Statistical Association*, 72(358), 320-338.

---

### 1.3 Paule-Mandel (PM)

**Mathematical Formulation:**
```
Solve Q(tau2) = k - 1 for tau2
where: Q(tau2) = sum(wi* * (yi - y*)^2)
wi* = 1 / (vi + tau2)
y* = sum(wi* * yi) / sum(wi*)
```

**Statistical Properties:**
- Method-of-moments estimator
- Often less biased than DL for small k
- Can overestimate tau2 in some scenarios
- More variable than DL

**When to Use:**
- Alternative to DL for k < 20
- When DL tau2 seems too small
- Preference for moments over likelihood

**Limitations:**
- Requires numerical solution
- Not as widely used as DL/REML
- Can be unstable for homogeneous data

**References:**
- Paule, R. C., & Mandel, J. (1982). Consensus values and weighting factors. *Journal of Research of the National Bureau of Standards*, 87(5), 377-385.

---

### 1.4 Hartung-Knapp (HK)

**Mathematical Formulation:**
```
y_hk = y_dl (same point estimate)
SE_hk = sqrt(sum(wi*) * (yi - y*)^2 / ((k-1) * sum(wi*)))
CI: y_hk ± t_{k-1, 0.975} * SE_hk
```

**Statistical Properties:**
- Uses t-distribution instead of normal
- Accounts for uncertainty in tau2 estimation
- Better coverage for small k
- Wider CIs than DL (more conservative)

**When to Use:**
- Small number of studies (k < 20)
- When coverage is critical
- Recommended alternative to DL CIs

**Limitations:**
- More conservative (wider CIs)
- Can reduce power
- Point estimate still from DL (with its biases)

**References:**
- Hartung, J., & Knapp, G. (2001). On tests of the overall treatment effect in meta-analysis with possibly heterogeneous study results. *Biometrics*, 57(3), 908-916.

---

## 2. Robust Estimation Methods

### 2.1 Huber M-Estimator

**Mathematical Formulation:**
```
Minimize: sum(rho(yi - mu) / vi)
where rho(u) = {
    0.5 * u^2,             if |u| <= c
    c * |u| - 0.5 * c^2,    if |u| > c
}
Typical tuning: c = 1.345 (95% efficiency for normal)
```

**Statistical Properties:**
- Redescending influence function
- 95% efficient at normal distribution
- Bounded influence (outliers have limited impact)
- Consistent for symmetric distributions

**When to Use:**
- Suspected outliers in effect sizes
- Heavy-tailed distributions
- Want to downweight but not exclude studies

**Limitations:**
- Requires tuning constant selection
- Less efficient than OLS for clean data
- Symmetry assumption important

**References:**
- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.

---

### 2.2 Tukey Biweight

**Mathematical Formulation:**
```
rho(u) = {
    (c^2/6) * [1 - (1 - (u/c)^2)^3],  if |u| <= c
    (c^2/6),                            if |u| > c
}
psi(u) = rho'(u) = u * (1 - (u/c)^2)^2  for |u| <= c
Typical tuning: c = 4.685 (95% efficiency)
```

**Statistical Properties:**
- Redescending M-estimator (extreme outliers get zero weight)
- Smooth influence function
- High breakdown point
- Very robust to extreme outliers

**When to Use:**
- Severe outliers or contamination
- Want extreme outliers to have zero influence
- Heavy-tailed distributions

**Limitations:**
- Multiple local minima possible
- Computationally more intensive
- Requires good starting values

**References:**
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

---

### 2.3 Hampel's Redescending Estimator

**Mathematical Formulation:**
```
psi(u) = {
    u,                         if |u| <= a
    a * sign(u),               if a < |u| <= b
    a * sign(u) * (c - |u|)/(c - b),  if b < |u| <= c
    0,                         if |u| > c
}
Typical tuning: a=1.7, b=3.4, c=8.5
```

**Statistical Properties:**
- Three-part redescending function
- Rejects extreme outliers completely
- High breakdown point
- Flexible tuning via three parameters

**When to Use:**
- When you need three outlier rejection regions
- Want to completely reject extreme values
- Flexible control of influence function

**Limitations:**
- Three tuning parameters to select
- Sensitive to parameter choices
- Computationally complex

**References:**
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383-393.

---

### 2.4 Andrews' Wave Function

**Mathematical Formulation:**
```
psi(u) = {
    (c/pi) * sin(pi*u/c),  if |u| <= c
    0,                      if |u| > c
}
rho(u) = integral of psi
Typical tuning: c = 1.339*pi
```

**Statistical Properties:**
- Smooth redescending influence
- Zero weight for extreme outliers
- Oscillatory influence function
- Good efficiency at normal

**When to Use:**
- Want smooth redescending function
- Moderate outliers suspected
- Prefer sinusoidal weighting

**Limitations:**
- Less commonly used than Huber/Tukey
- Oscillations can cause issues
- Tuning less well-studied

**References:**
- Andrews, D. F. (1974). A robust method for multiple linear regression. *Technometrics*, 16(4), 523-531.

---

## 3. Bayesian Methods

### 3.1 Empirical Bayes Shrinkage

**Mathematical Formulation:**
```
yi | theta_i, vi ~ N(theta_i, vi)
theta_i | mu, tau2 ~ N(mu, tau2)
Posterior mean: E[theta_i | y] = mu + (1 - B) * (yi - mu)
where: B = vi / (vi + tau2)  [shrinkage factor]
```

**Statistical Properties:**
- Shrinks estimates toward overall mean
- Borrow strength across studies
- Optimal for normal-normal model
- Reduces overfitting to individual studies

**When to Use:**
- Want to borrow strength across studies
- Reduce variance of estimates
- Many studies with small sample sizes

**Limitations:**
- Requires estimating tau2
- Can overshrink if heterogeneity is high
- Normal assumption important

**References:**
- Efron, C., & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70(350), 311-319.

---

### 3.2 Hierarchical Bayes

**Mathematical Formulation:**
```
yi | theta_i, vi ~ N(theta_i, vi)
theta_i | mu, tau2 ~ N(mu, tau2)
mu ~ N(0, large) or flat
tau ~ Half-Cauchy(0, scale) or Half-N(0, large)

Posterior: p(theta, mu, tau2 | y) computed via MCMC
```

**Statistical Properties:**
- Full Bayesian inference
- Accounts for all uncertainty
- Flexible prior specification
- Can incorporate prior information

**When to Use:**
- Want full Bayesian inference
- Have informative priors
- Need uncertainty in tau2
- Small k (priors help)

**Limitations:**
- Computationally intensive (MCMC)
- Requires convergence diagnostics
- Prior specification can be influential for small k
- Slower than empirical Bayes

**References:**
- Sutton, A. J., & Abrams, K. R. (2001). Bayesian methods in meta-analysis and evidence synthesis. *Statistical Methods in Medical Research*, 10(4), 277-303.

---

### 3.3 Bayesian Model Averaging

**Mathematical Formulation:**
```
p(y | M) = sum(p(y | M_j) * p(M_j | y))
where M_j are competing models
p(M_j | y) proportional to p(y | M_j) * p(M_j)

Final estimate: Model-averaged posterior
```

**Statistical Properties:**
- Accounts for model uncertainty
- Reduces overconfidence
- Better predictive performance
- Automatic outlier handling

**When to Use:**
- Uncertain which method to use
- Want to account for model uncertainty
- Multiple plausible models
- Want robust predictions

**Limitations:**
- Computationally expensive
- Model specification critical
- Can be sensitive to prior model probabilities
- Interpretation more complex

**References:**
- Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). Bayesian model averaging: A tutorial. *Statistical Science*, 14(4), 382-417.

---

## 4. Heterogeneity Estimation

### 4.1 Hunter-Schmidt

**Mathematical Formulation:**
```
tau2 = sd_w^2 - mean(vi)
where: sd_w^2 is weighted variance of effect sizes
       mean(vi) is average within-study variance
```

**Statistical Properties:**
- Psychometric tradition (reliability-focused)
- Can produce negative estimates
- Simple closed-form
- Less commonly used in medical statistics

**When to Use:**
- Following Hunter-Schmidt approach
- Want simple heterogeneity estimate
- Working in psychometrics

**Limitations:**
- Negative estimates truncated at 0
- Not as efficient as REML
- Less widely accepted in medical research

**References:**
- Hunter, J. E., & Schmidt, F. L. (1990). *Methods of Meta-Analysis: Correcting Error and Bias in Research Findings*. Sage.

---

### 4.2 Hedges-Olkin

**Mathematical Formulation:**
```
Q = sum(wi * (yi - y_fe)^2)
tau2 = (Q - (k-1)) / C
where C = sum(wi) - sum(wi^2)/sum(wi)
Similar to DL but different variance calculations
```

**Statistical Properties:**
- Developed for standardized mean differences
- Similar to DL but specialized for d-statistics
- Better for Hedges' g bias correction
- Widely used in social sciences

**When to Use:**
- Working with standardized mean differences
- Following Hedges-Olkin tradition
- Social science research

**Limitations:**
- Specialized for certain effect sizes
- Less general than DL
- Negative tau2 issue

**References:**
- Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. Academic Press.

---

## 5. Small Sample Corrections

### 5.1 Knapp-Hartung Modification

**Mathematical Formulation:**
```
tau2 from REML or DL
y_kh = y_fe (same point estimate)
SE_kh = sqrt(tau2 + 1/sum(wi)) * sqrt(1 + 1/(k-1) * sum(wi*(yi-y_fe)^2)/sum(wi))
t = y_kh / SE_kh
df = k - 2 (or k-1)
CI: y_kh ± t_{df, 0.975} * SE_kh
```

**Statistical Properties:**
- Adjusts variance using t-distribution
- Accounts for tau2 estimation uncertainty
- Better coverage for small k
- Can truncate tau2 at 0

**When to Use:**
- k < 20 (especially k < 10)
- When coverage is critical
- Default recommendation for small k

**Limitations:**
- More conservative (wider CIs)
- Reduced power
- Requires choosing truncation option

**References:**
- Knapp, G., & Hartung, J. (2003). Improved tests for a random effects meta-regression with a single covariate. *Statistics in Medicine*, 22(17), 2693-2710.

---

### 5.2 Satterthwaite Degrees of Freedom

**Mathematical Formulation:**
```
df_satt = [sum(wi*)]^2 / sum(wi*^2 * df_i / (df_i+2))
where df_i are study-specific degrees of freedom
Use t-distribution with df_satt for CIs
```

**Statistical Properties:**
- Approximates degrees of freedom
- Accounts for varying precision
- Better than normal approximation
- Conservative but reasonable

**When to Use:**
- Studies have different sample sizes
- Want better df approximation
- Alternative to HK

**Limitations:**
- Still an approximation
- Can be conservative
- More complex than simple methods

**References:**
- Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance components. *Biometrics Bulletin*, 2(6), 110-114.

---

## 6. Weighting Schemes

### 6.1 Quality Effects Model

**Mathematical Formulation:**
```
w_i^q = w_i * quality_i
where quality_i is study quality score (0-1)
y_q = sum(w_i^q * y_i) / sum(w_i^q)
```

**Statistical Properties:**
- Incorporates study quality into weights
- Downweights low-quality studies
- Subjective (quality assessment)
- Can improve validity if quality scores accurate

**When to Use:**
- Have reliable quality assessment
- Want to incorporate quality into pooling
- Concerned about study quality variation

**Limitations:**
- Quality scoring subjective
- Can introduce bias
- No clear guidance on quality scoring
- Not widely accepted

**References:**
- Doi, S. A., & Thalib, L. (2008). A quality-effects model for meta-analysis. *Epidemiology*, 19(1), 94-100.

---

### 6.2 Sample Size Weighting

**Mathematical Formulation:**
```
w_i = n_i (or n_i^2)
y_ss = sum(w_i * y_i) / sum(w_i)
```

**Statistical Properties:**
- Weights by sample size instead of precision
- More equal weighting than IV
- Can be more stable
- Less efficient than IV if variances accurate

**When to Use:**
- Within-study variances unreliable
- Want more equal weighting
- Large variation in sample sizes

**Limitations:**
- Ignores within-study variance
- Less efficient than IV
- Not statistically optimal
- Ad hoc approach

**References:**
- No primary reference (ad hoc method)

---

## 7. Resampling Methods

### 7.1 Bootstrap Meta-Analysis

**Mathematical Formulation:**
```
For b = 1 to B bootstrap samples:
  1. Resample studies with replacement
  2. Compute meta-analysis on resampled data
  3. Store estimate

Bootstrap CI: Percentile or BCa method
```

**Statistical Properties:**
- Nonparametric (no distributional assumptions)
- Accounts for clustering in studies
- Can handle complex estimators
- Computationally intensive

**When to Use:**
- Distributional assumptions questionable
- Complex estimators
- Want CIs without normality
- Have computational resources

**Limitations:**
- Computationally expensive (B >= 1000)
- Can be unstable for very small k
- Requires choice of CI method (percentile vs BCa)
- May not improve coverage for simple estimators

**References:**
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1-26.
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189-228.

---

### 7.2 Jackknife

**Mathematical Formulation:**
```
For i = 1 to k:
  1. Remove study i
  2. Compute estimate on remaining k-1 studies
  3. Store estimate theta_{(-i)}

Jackknife estimate: theta_jack = k * theta_all - (k-1) * mean(theta_{(-i)})
Jackknife SE: sqrt(((k-1)/k) * sum((theta_{(-i)} - mean(theta_{(-i)}))^2))
```

**Statistical Properties:**
- Leave-one-out resampling
- Less computationally intensive than bootstrap
- Good for variance estimation
- Can reduce bias

**When to Use:**
- Want faster alternative to bootstrap
- Estimate bias and variance
- Small to moderate k

**Limitations:**
- Less robust than bootstrap
- Can be unstable for small k
- Doesn't capture all variability
- Assumes studies are independent

**References:**
- Quenouille, M. H. (1949). Approximate tests of correlation in time-series. *Journal of the Royal Statistical Society Series B*, 11(1), 68-84.

---

## 8. Adaptive Methods

### 8.1 Adaptive Weighting

**Mathematical Formulation:**
```
Weights adapt based on data characteristics:
w_i = f(vi, tau2, residuals, outliers)

Example: Iteratively reweighted least squares (IRLS)
w_i^{(t+1)} = w_i^{(t)} * psi(r_i^{(t)}) / r_i^{(t)}
where r_i are residuals, psi is influence function
```

**Statistical Properties:**
- Data-driven weight selection
- Can adapt to outliers
- May converge to suboptimal solutions
- Computationally iterative

**When to Use:**
- Suspected outliers or influential studies
- Want data-driven weights
- Standard weighting seems inappropriate

**Limitations:**
- Multiple local solutions possible
- Convergence not guaranteed
- Can be unstable
- Less interpretable

**References:**
- Hedges, L. V. (1983). Combining independent estimators in research synthesis. *British Journal of Mathematical and Statistical Psychology*, 36(2), 123-131.

---

## 9. Mixture Models

### 9.1 Gaussian Mixture Models

**Mathematical Formulation:**
```
yi ~ sum(pi_j * N(mu_j, sigma_j^2))
where pi_j are mixture weights (sum to 1)
      mu_j are component means
      sigma_j^2 are component variances

Estimated via EM algorithm
```

**Statistical Properties:**
- Models multimodal effect distributions
- Can identify subgroups
- Flexible distributional form
- Number of components must be selected

**When to Use:**
- Suspected multiple subpopulations
- Effect distribution appears multimodal
- Want to identify clusters of studies

**Limitations:**
- Number of components unknown
- EM can converge to local optima
- Requires sufficient k
- Interpretation complex

**References:**
- McLachlan, G., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Verdinelli, I., & Wasserman, L. (1995). Computing Bayes factors using a generalization of the Savage-Dickey density ratio. *Journal of the American Statistical Association*, 90(430), 614-618.

---

## 10. Nonparametric Methods

### 10.1 Kernel-Based Methods

**Mathematical Formulation:**
```
y_hat(x) = sum(K_h(x - xi) * yi) / sum(K_h(x - xi))
where K_h(u) = (1/h) * K(u/h)
      K is kernel function (Gaussian, Epanechnikov)
      h is bandwidth
```

**Statistical Properties:**
- Nonparametric smoothing
- Data-driven bandwidth selection
- Can model complex patterns
- Bandwidth choice critical

**When to Use:**
- Relationship between effect and covariates
- Want smooth estimates
- Nonlinear patterns suspected

**Limitations:**
- Bandwidth selection challenging
- Curse of dimensionality
- Less interpretable
- Requires sufficient data

**References:**
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141-142.
- Wand, M. P., & Jones, M. C. (1995). *Kernel Smoothing*. Chapman & Hall.

---

### 10.2 Hodges-Lehmann Estimator

**Mathematical Formulation:**
```
HL = median{(yi + yj) / 2 for all i <= j}
Pooled effect based on pairwise averages
```

**Statistical Properties:**
- Nonparametric (distribution-free)
- Robust to outliers
- 95% efficient at normal
- Based on order statistics

**When to Use:**
- Normality assumption questionable
- Want robust nonparametric estimate
- Symmetric distribution of effects

**Limitations:**
- Requires symmetry
- Less efficient than OLS for normal
- Computationally intensive (O(k^2))
- Less familiar to users

**References:**
- Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based on rank tests. *Annals of Mathematical Statistics*, 34(2), 598-611.

---

## 11. Information-Theoretic Methods

### 11.1 Maximum Entropy Meta-Analysis

**Mathematical Formulation:**
```
Maximize: H(p) = -sum(p_i * log(p_i))
Subject to: sum(p_i) = 1
             sum(p_i * yi) = mu (constraint)
             sum(p_i * (yi - mu)^2) = sigma^2 (constraint)
```

**Statistical Properties:**
- Least informative given constraints
- Maximizes uncertainty subject to constraints
- Objectively determined weights
- Can be computationally intensive

**When to Use:**
- Want objective, minimally informative weights
- Have prior information (constraints)
- Want to avoid assumptions

**Limitations:**
- Constraint selection subjective
- Computationally complex
- Less familiar
- May be overly conservative

**References:**
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620-630.

---

## 12. Regularization Methods

### 12.1 Ridge Regression (L2)

**Mathematical Formulation:**
```
Minimize: sum((yi - mu)^2 / vi) + lambda * mu^2
Solution: mu_ridge = sum(wi * yi) / (sum(wi) + lambda)
```

**Statistical Properties:**
- Shrinks estimate toward zero
- Reduces variance at cost of bias
- Unique solution always exists
- Lambda controls shrinkage

**When to Use:**
- High variance concerns
- Want to prevent extreme estimates
- Multicollinearity (in regression)

**Limitations:**
- Shrinks all coefficients equally
- Bias-variance tradeoff
- Lambda selection needed
- Less interpretable

**References:**
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society Series B*, 58(1), 267-288.

---

### 12.2 Lasso (L1)

**Mathematical Formulation:**
```
Minimize: sum((yi - mu)^2 / vi) + lambda * |mu|
Can produce sparse solutions (exactly zero)
```

**Statistical Properties:**
- Shrinks some coefficients to zero
- Variable selection property
- Bias introduced
- Non-unique solutions possible

**When to Use:**
- Want sparse solutions
- Variable selection needed
- Many potential predictors

**Limitations:**
- Selection of lambda critical
- Biased estimates
- Can be unstable
- Correlated predictors problematic

**References:**
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society Series B*, 58(1), 267-288.

---

## 13. Generalized Mean Methods

### 13.1 Geometric Mean

**Mathematical Formulation:**
```
y_geo = exp(sum(log(yi)) / k)
Only for positive effect sizes
```

**Statistical Properties:**
- Less sensitive to extreme values
- Appropriate for multiplicative effects
- Requires positive values
- Logarithmic transformation

**When to Use:**
- Effect sizes are ratios
- Data are log-normally distributed
- Want to reduce impact of extremes

**Limitations:**
- Requires positive effects
- Not appropriate for all effect measures
- Less interpretable
- Zero values problematic

**References:**
- Nadarajah, S. (2008). A popular generalized exponential distribution. *Journal of Applied Statistics*, 35(6), 663-674.

---

## 14. Loss-Based Methods

### 14.1 Quantile Loss

**Mathematical Formulation:**
```
L_tau(yi, mu) = {
    tau * |yi - mu|,     if yi >= mu
    (1-tau) * |yi - mu|,  if yi < mu
}
Minimize sum(L_tau(yi, mu))
```

**Statistical Properties:**
- Estimates quantiles of effect distribution
- Robust to outliers
- No distributional assumptions
- Tau parameter controls quantile

**When to Use:**
- Want median (tau=0.5) or other quantiles
- Robust alternative to mean
- Distribution asymmetric

**Limitations:**
- Not estimating mean
- Less efficient for normal
- Tau selection needed
- Interpretation different

**References:**
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

---

## 15. Publication Bias Adjustment

### 15.1 Trim-and-Fill

**Mathematical Formulation:**
```
1. Estimate number of missing studies (k0)
2. "Trim" most extreme studies
3. Re-calculate effect
4. "Fill" in missing studies symmetrically
5. Iterate until convergence
```

**Statistical Properties:**
- Adjusts for funnel plot asymmetry
- Can estimate number of missing studies
- Assumes symmetry in underlying effect
- Can be unstable

**When to Use:**
- Publication bias suspected
- Funnel plot asymmetric
- Want to adjust for missing studies

**Limitations:**
- Assumes symmetry is correct
- Can over- or under-correct
- Not a substitute for prospective registration
- Fails if bias mechanism different

**References:**
- Duval, S., & Tweedie, R. (2000). Trim and fill: A simple funnel-plot-based method of testing and adjusting for publication bias in meta-analysis. *Biometrics*, 56(2), 455-463.

---

## 16. Ensemble Methods

### 16.1 Model Ensemble

**Mathematical Formulation:**
```
y_ensemble = sum(w_j * y_j) / sum(w_j)
where y_j are estimates from methods j=1,...,M
      w_j are ensemble weights
```

**Statistical Properties:**
- Combines multiple methods
- Can reduce variance
- Weight selection critical
- Risk of combining bad methods

**When to Use:**
- Uncertain which single method to use
- Want to reduce model uncertainty
- Have multiple reasonable methods

**Limitations:**
- Weight selection subjective
- Can include poor methods
- Interpretation complex
- Computationally expensive

**References:**
- Zhou, Z. H. (2012). *Ensemble Methods: Foundations and Algorithms*. Chapman & Hall.

---

## 17. Machine Learning Inspired Methods

### 17.1 Attention-Based Weighting

**Mathematical Formulation:**
```
alpha_i = exp(s_i) / sum(exp(s_j))
where s_i = score(yi, context)
y_attention = sum(alpha_i * yi)
```

**Statistical Properties:**
- Learns which studies to attend to
- Can capture complex patterns
- Requires sufficient data
- Risk of overfitting

**When to Use:**
- Large number of studies (k > 50)
- Complex heterogeneity patterns
- Want data-driven weighting

**Limitations:**
- Requires training data
- Risk of overfitting
- Black-box nature
- Computationally intensive

**References:**
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

---

## 18. Dependence Modeling

### 18.1 Copula Models

**Mathematical Formulation:**
```
F(x, y) = C(F_X(x), F_Y(y))
where C is copula function
      F_X, F_Y are marginal CDFs

Common copulas: Gaussian, Clayton, Gumbel, Frank
```

**Statistical Properties:**
- Separates marginal distributions from dependence
- Flexible dependence structures
- Can model asymmetric dependence
- Estimation can be complex

**When to Use:**
- Modeling dependence between outcomes
- Flexible dependence structures needed
- Non-Gaussian dependence

**Limitations:**
- Copula selection needed
- Estimation complex
- Interpretation challenging
- Requires sufficient data

**References:**
- Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
- Trivedi, P. K., & Zimmer, D. M. (2007). Copula modeling: An introduction for practitioners. *Foundations and Trends in Econometrics*, 1(1), 1-111.

---

## 19. Heavy-Tailed Methods

### 19.1 Extreme Value Theory

**Mathematical Formulation:**
```
GEV distribution: F(x) = exp(-(1 + xi*(x-mu)/sigma)^(-1/xi))
for 1 + xi*(x-mu)/sigma > 0
where xi is shape parameter
      mu is location
      sigma is scale
```

**Statistical Properties:**
- Models tail behavior
- Three types: Gumbel, Frechet, Weibull
- Can estimate extreme quantiles
- Requires sufficient extreme observations

**When to Use:**
- Interest in extreme effects
- Heavy-tailed distributions
- Want to model tail behavior

**Limitations:**
- Requires sufficient data
- Only models tails
- Estimation can be unstable
- Less familiar to users

**References:**
- Embrechts, P., Kluppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.

---

## 20. Density Estimation Methods

### 20.1 Kernel Density Estimation

**Mathematical Formulation:**
```
f_hat(x) = (1/(k*h)) * sum(K((x - yi)/h))
where K is kernel function
      h is bandwidth
```

**Statistical Properties:**
- Nonparametric density estimate
- Data-driven shape
- Bandwidth selection critical
- Can be multimodal

**When to Use:**
- Want to visualize effect distribution
- Suspected multimodality
- Nonparametric approach preferred

**Limitations:**
- Bandwidth selection
- Boundary effects
- Curse of dimensionality
- Less interpretable parameters

**References:**
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.

---

## 21. Quantile Methods

### 21.1 Quantile Regression

**Mathematical Formulation:**
```
Minimize: sum(rho_tau(yi - Q(tau)))
where rho_tau(u) = u * (tau - I(u < 0))
```

**Statistical Properties:**
- Models conditional quantiles
- Robust to outliers
- No distributional assumptions
- Can model heterogeneous effects

**When to Use:**
- Interested in quantiles (median, etc.)
- Robust alternative to mean regression
- Heterogeneous effects

**Limitations:**
- Not estimating mean
- Computationally more intensive
- Interpretation different
- Standard errors complex

**References:**
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

---

## 22. Optimization Methods

### 22.1 Minimax Optimization

**Mathematical Formulation:**
```
Minimize: max_i |yi - mu|
Solution: midrange of effect sizes
```

**Statistical Properties:**
- Minimizes worst-case error
- Very robust to outliers
- Inefficient for normal distributions
- Based on order statistics

**When to Use:**
- Want to minimize maximum error
- Extreme risk aversion
- Robustness paramount

**Limitations:**
- Very inefficient
- Only uses extreme values
- Ignores most data
- Not commonly used

**References:**
- No primary reference (optimization literature)

---

## 23. Signal Processing Methods

### 23.1 Wavelet Denoising

**Mathematical Formulation:**
```
1. Apply DWT: w = DWT(y)
2. Threshold: w_t = threshold(w, lambda)
3. Reconstruct: y_denoised = IDWT(w_t)

Threshold: lambda = sigma * sqrt(2*log(k))  (VisuShrink)
```

**Statistical Properties:**
- Multi-resolution analysis
- Separates signal from noise
- Data-adaptive denoising
- Wavelet basis selection matters

**When to Use:**
- Large k (k > 20)
- Multi-modal effects
- Want data-driven denoising

**Limitations:**
- Requires sufficient k
- Wavelet basis selection
- Less interpretable
- Computationally intensive

**References:**
- Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
- Donoho, D. L. (1995). De-noising by soft-thresholding. *IEEE Transactions on Information Theory*, 41(3), 613-627.

---

## 24. Functional Data Methods

### 24.1 Functional Mean

**Mathematical Formulation:**
```
Treat each study as a function
Compute functional mean: y_func(t) = mean(yi(t))
Use smoothing (splines, basis functions)
```

**Statistical Properties:**
- For longitudinal or time-series data
- Smoothing parameter critical
- Can model temporal patterns
- Computationally intensive

**When to Use:**
- Studies report curves/trajectories
- Temporal patterns of interest
- Functional data

**Limitations:**
- Specialized data required
- Smoothing parameter selection
- Computationally complex
- Less familiar

**References:**
- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer.

---

## 25. Hybrid Methods

### 25.1 Combined Approaches

**Mathematical Formulation:**
```
Example: Robust + Bayesian
Prior on robust location parameter
Posterior combines robust estimation with prior info
```

**Statistical Properties:**
- Combines strengths of multiple approaches
- Can be complex
- May inherit weaknesses
- Interpretation challenging

**When to Use:**
- Single approach insufficient
- Want benefits of multiple methods
- Complex data structure

**Limitations:**
- Complexity
- Computational cost
- Difficult to justify
- Less interpretable

**References:**
- No single reference (method-specific)

---

## 26. Experimental Methods

### 26.1 Novel Approaches

**Overview:**
This category contains methods that are exploratory or theoretical without extensive validation.

**Caveats:**
- Use with caution
- May not have established properties
- Require further validation
- Consider as research tools

**When to Use:**
- Exploratory analysis
- Research on methods
- Have validation strategy

**Limitations:**
- Unknown properties
- May not work in practice
- Risk of misleading results

---

## 27. Validation Methods

### 27.1 Cross-Validation

**Mathematical Formulation:**
```
LOOCV: Leave-one-study-out
For each study i:
  1. Train on all except study i
  2. Predict study i
  3. Record prediction error

CV error = mean(Prediction errors)
```

**Statistical Properties:**
- Estimates out-of-sample performance
- Nearly unbiased for k-1
- Can be variable
- Computationally intensive

**When to Use:**
- Want to estimate prediction error
- Model selection
- Validate performance

**Limitations:**
- Computationally expensive
- Variable for small k
- Doesn't assess all uncertainty
- May be optimistic for small k

**References:**
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

---

## 28. Order Statistics Methods

### 28.1 Median and Quantiles

**Mathematical Formulation:**
```
Median = middle ordered value (or average of two middle)
Interquartile mean = mean of values between Q1 and Q3
```

**Statistical Properties:**
- Robust to outliers
- Based on ranks
- Distribution-free
- Less efficient than mean for normal

**When to Use:**
- Robust estimate needed
- Distribution non-normal or unknown
- Outliers suspected

**Limitations:**
- Less efficient for normal
- Doesn't use all information
- Can be unstable for small k

**References:**
- Standard nonparametric statistics texts

---

## 29. Clustering Methods

### 29.1 Cluster-Based Weighting

**Mathematical Formulation:**
```
1. Cluster studies (e.g., k-means)
2. Estimate cluster-specific effects
3. Combine with cluster weights
```

**Statistical Properties:**
- Can identify subgroups
- Data-driven clustering
- Number of clusters unknown
- Clustering method matters

**When to Use:**
- Suspected subgroups
- Heterogeneous effects
- Exploratory analysis

**Limitations:**
- Cluster selection subjective
- Can be unstable
- May overfit
- Validation needed

**References:**
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.

---

## 30. Moment-Based Methods

### 30.1 Method of Moments

**Mathematical Formulation:**
```
Match sample moments to theoretical moments
E[Y] = mu
E[(Y-mu)^2] = tau2 + vi
Solve for parameters
```

**Statistical Properties:**
- Simple closed-form
- Can be inefficient
- May produce estimates outside parameter space
- Non-iterative

**When to Use:**
- Want simple estimates
- Starting values for other methods
- Computational simplicity valued

**Limitations:**
- Inefficient compared to ML
- Can be biased
- May not respect parameter constraints
- Less precise

**References:**
- Standard estimation theory texts

---

## Decision Tree for Method Selection

```
START
|
|---- k < 10?
|     |---- YES: Use Knapp-Hartung or HK modification
|     |         Consider Bayesian with informative priors
|     |
|     |---- NO: Continue
|
|---- Outliers suspected?
|     |---- YES: Use robust methods (Huber, Tukey)
|     |         Or consider trim-and-fill
|     |
|     |---- NO: Continue
|
|---- Publication bias suspected?
|     |---- YES: Use trim-and-fill or selection models
|     |
|     |---- NO: Continue
|
|---- Want to model subgroups?
|     |---- YES: Use mixture models or clustering
|     |
|     |---- NO: Continue
|
|---- DEFAULT: Use REML or DL
         With HK CI if k < 20
```

---

## Summary Table

| Category | Primary Use | Key Assumption | Typical k |
|----------|-------------|----------------|-----------|
| **Standard (DL, REML)** | General use | Normal effects | Any |
| **Robust** | Outliers | Symmetric effects | Any |
| **Bayesian** | Small k, prior info | Prior specification | Any |
| **HK** | Small k | Normal effects | < 20 |
| **Bootstrap** | CI without normality | Exchangeable | >= 10 |
| **Mixture** | Subgroups | Finite components | >= 15 |
| **Kernel** | Smoothing | Smooth underlying function | >= 20 |
| **Regularization** | High variance | Shrinkage helps | Any |

---

## References

### Primary Sources
- Borenstein, M., Hedges, L. V., Higgins, J. P., & Rothstein, H. R. (2009). *Introduction to Meta-Analysis*. Wiley.
- Cochrane Handbook for Systematic Reviews of Interventions (2022).
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.
- Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. Academic Press.
- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package. *Journal of Statistical Software*, 36(3), 1-48.

### Small Sample Methods
- Jackson, D., & Turner, R. (2017). Confidence intervals for a meta-analysis. *Research Synthesis Methods*, 8(4), 460-473.
- Brockwell, S. E., & Gordon, I. R. (2001). A comparison of statistical tests for publication bias. *Biometrics*, 57(2), 431-436.

### Robust Methods
- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA*, 69(346), 383-393.

### Bayesian Methods
- Sutton, A. J., & Abrams, K. R. (2001). Bayesian methods in meta-analysis. *Statistical Methods in Medical Research*, 10(4), 277-303.
- Spiegelhalter, D. J., Abrams, K. R., & Myles, J. P. (2004). *Bayesian Approaches to Clinical Trials and Health-Care Evaluation*. Wiley.

### Resampling Methods
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1-26.
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189-228.

### Specialized Methods
- Duval, S., & Tweedie, R. (2000). Trim and fill. *Biometrics*, 56(2), 455-463.
- Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*. Springer.

---

**End of Theoretical Foundations Document**

*This document provides the statistical foundation for all 300+ methods in the experimental meta-analysis framework. For implementation details, see individual method docstrings and source code.*

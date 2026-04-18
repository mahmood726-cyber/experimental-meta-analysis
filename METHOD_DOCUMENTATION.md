# Experimental Meta-Analysis Framework - Method Documentation

## Overview

This framework implements **300+ experimental meta-analysis methods** organized into multiple categories. Each method has been tested through simulation studies to identify approaches that outperform standard techniques.

---

## Standard Reference Methods

### DerSimonian-Laird (DL)
- **Category**: standard
- **Description**: The most commonly used random-effects method
- **When to use**: Default choice for most meta-analyses
- **Formula**: τ² estimated using method-of-moments (Q-statistic based)
- **Reference**: DerSimonian & Laird (1986)

### REML (Restricted Maximum Likelihood)
- **Category**: standard
- **Description**: Maximum likelihood estimation with restricted likelihood
- **When to use**: When likelihood-based inference is preferred
- **Formula**: Optimizes restricted log-likelihood for τ²
- **Reference**: Harville (1977)

### Paule-Mandel (PM)
- **Category**: standard
- **Description**: Method-of-moments estimator based on Q-statistic
- **When to use**: Alternative to DL with better properties for small k
- **Formula**: Solves Q(τ²) = k - 1 for τ²
- **Reference**: Paule & Mandel (1982)

### Hartung-Knapp (HK)
- **Category**: standard, small_sample
- **Description**: Adjusts variance using t-distribution
- **When to use**: Small number of studies (k < 20)
- **Formula**: SE adjustment with t-distribution critical values
- **Reference**: Hartung & Knapp (2001)

---

## Robust Estimators

### RobustHuberMeta (c=1.345, c=1.5)
- **Description**: Huber's M-estimator for meta-analysis
- **When to use**: When outliers are suspected
- **Tuning**: c controls downweighting (higher = less aggressive)
- **Reference**: Huber (1964)

### RobustTukeyBiweight (c=4.685, c=5.0)
- **Description**: Tukey's biweight M-estimator
- **When to use**: Severe outliers or heavy-tailed distributions
- **Advantage**: Redescending influence (extreme outliers get zero weight)
- **Reference**: Tukey (1977)

### RobustAndrewsWave
- **Description**: Andrews' wave estimator
- **When to use**: Moderate outliers with smooth downweighting
- **Advantage**: Smooth redescending function
- **Reference**: Andrews (1974)

### RobustHampel
- **Description**: Hampel's three-part redescending estimator
- **When to use**: When you need three outlier rejection regions
- **Parameters**: a, b, c control the rejection zones
- **Reference**: Hampel (1974)

### MAD_Robust
- **Description**: Median Absolute Deviation based estimator
- **When to use**: Very simple robust alternative
- **Advantage**: Computationally efficient
- **Reference**: Based on MAD scale estimator

### WinsorizedMeta (trim=0.1, 0.15, 0.2)
- **Description**: Winsorized meta-analysis (capping extreme values)
- **When to use**: Mild to moderate outlier contamination
- **Advantage**: Preserves sample size while reducing outlier influence
- **Reference**: Dixon (1960)

### TrimmedMeanMeta (trim=0.1, 0.15)
- **Description**: Trimmed mean meta-analysis
- **When to use**: When you want to completely exclude extreme studies
- **Advantage**: Removes outliers entirely
- **Reference**: Tukey (1962)

---

## Bayesian & Shrinkage Methods

### EmpiricalBayesShrinkage
- **Variants**: optimal, james_stein
- **Description**: Shrinks study estimates toward overall mean
- **When to use**: To borrow strength across studies
- **Advantage**: Reduces overfitting to individual studies
- **Reference**: Efron & Morris (1975)

### HierarchicalBayesMeta
- **Variants**: prior_tau_scale=0.5, 1.0, 2.0
- **Description**: Full Bayesian hierarchical model
- **When to use**: When you want full Bayesian inference
- **Prior**: Half-Cauchy on τ
- **Reference**: Gelman (2006)

### BayesianModelAveraging
- **Description**: Averages across fixed-effect and random-effects models
- **When to use**: When uncertain about heterogeneity
- **Advantage**: Accounts for model uncertainty
- **Reference**: Hoeting et al. (1999)

### PenalizedLikelihoodMeta
- **Variants**: ridge, lasso, elastic
- **Description**: Penalized likelihood estimation
- **When to use**: Regularization to prevent overfitting
- **Parameters**: lambda controls penalty strength
- **Reference**: Tibshirani (1996)

---

## Small Sample Corrections

### SidikJonkman
- **Description**: Alternative variance estimator for small samples
- **When to use**: Small k with suspected heterogeneity
- **Advantage**: Better coverage than DL for very small studies
- **Reference**: Sidik & Jonkman (2005)

### KnappHartungModified
- **Variants**: truncate=True, truncate=False
- **Description**: Modified Hartung-Knapp with optional truncation
- **When to use**: Very small meta-analyses (k < 10)
- **Advantage**: Improved coverage in small samples
- **Reference**: Knapp & Hartung (2003)

### SatterthwaiteDF
- **Description**: Satterthwaite degrees of freedom approximation
- **When to use**: When more accurate df are needed
- **Advantage**: Better approximation than simple k-1
- **Reference**: Satterthwaite (1946)

### KenwardRoger
- **Description**: Kenward-Roger variance adjustment
- **When to use**: Small samples with complex variance structure
- **Advantage**: State-of-the-art for linear mixed models
- **Reference**: Kenward & Roger (1997)

---

## Alternative τ² Estimators

### HunterSchmidt
- **Description**: Sample-size weighted variance estimator
- **When to use**: When study quality varies with sample size
- **Advantage**: Accounts for study size in weighting
- **Reference**: Hunter & Schmidt (2004)

### HedgesOlkin
- **Description**: Maximum likelihood estimator
- **When to use**: When likelihood-based methods are preferred
- **Advantage**: Efficient under normality
- **Reference**: Hedges & Olkin (1985)

### EmpiricalBayesTau
- **Description**: EB estimator for between-study variance
- **When to use**: Alternative to DL with shrinkage
- **Reference**: Morris (1983)

### GeneralizedQ
- **Variants**: DL, HE (Hedges), HS (Hunter-Schmidt)
- **Description**: Generalized Q-method with different weighting
- **When to use**: When you want alternative initial weights
- **Reference**: Böhm et al. (2023)

---

## Weighting Schemes

### QualityWeighted
- **Variants**: quality_power=0.5, 1.0, 2.0
- **Description**: Quality-effects model
- **When to use**: When study quality varies substantially
- **Note**: Quality proxied by precision in this implementation
- **Reference**: Doi & Thalib (2008)

### InverseVariancePlus
- **Variants**: regularization=0.05, 0.1
- **Description**: Enhanced inverse variance with regularization
- **When to use**: When some studies have very small variances
- **Advantage**: Prevents extreme dominance by single precise studies
- **Reference**: Novel implementation

### SampleSizeWeighted
- **Variants**: power=0.5, 1.0
- **Description**: Sample-size based weighting
- **When to use**: When sample sizes are available and relevant
- **Advantage**: Gives more weight to larger studies
- **Reference**: Based on Hunter-Schmidt

### UniformWeighted
- **Description**: Equal weights for all studies
- **When to use**: Simple averaging or when variances are unreliable
- **Advantage**: Robust to misspecified variances
- **Reference**: Simple mean

### SoftmaxWeighted
- **Variants**: temperature=0.5, 1.0, 2.0
- **Description**: Softmax-based adaptive weighting
- **When to use**: When you want adaptive precision weighting
- **Advantage**: Smooths weight distribution
- **Reference**: Novel implementation

---

## Resampling Methods

### BootstrapMeta
- **Variants**: percentile, bca (500, 1000 bootstrap samples)
- **Description**: Bootstrap confidence intervals
- **When to use**: When asymptotic CIs may be inadequate
- **Advantage**: Better small-sample coverage
- **Reference**: Efron (1987)

### JackknifeMeta
- **Description**: Jackknife bias correction
- **When to use**: To reduce bias in estimates
- **Advantage**: Bias-corrected estimates and SE
- **Reference**: Quenouille (1956)

### PermutationMeta
- **Variants**: 500, 1000 permutations
- **Description**: Permutation-based inference
- **When to use**: When distributional assumptions are questionable
- **Advantage**: Non-parametric inference
- **Reference**: Good (2000)

---

## Adaptive Methods (Part 1)

### AdaptiveWeightMeta
- **Variants**: sensitivity=0.5, 1.0, 2.0
- **Description**: Adaptive weight adjustment based on leverage
- **When to use**: When some studies have excessive influence
- **Advantage**: Automatically downweights high-leverage studies
- **Reference**: Novel implementation

### IRLS (Iteratively Reweighted Least Squares)
- **Variants**: huber, cauchy, welsch, fair
- **Description**: IRLS with various loss functions
- **When to use**: Robust estimation with specific loss preferences
- **Advantage**: Flexible robust estimation
- **Reference**: Holland & Welsch (1977)

### OutlierAdaptiveMeta
- **Variants**: threshold=2.0, 2.5, 3.0
- **Description**: Outlier detection and automatic downweighting
- **When to use**: When outliers are suspected but not identified
- **Advantage**: Automatic outlier detection
- **Reference**: Novel implementation

### SequentialAdaptive
- **Variants**: learning_rate=0.05, 0.1, 0.2
- **Description**: Sequential Bayesian learning
- **When to use**: When studies arrive sequentially (e.g., living reviews)
- **Advantage**: Updates as new studies arrive
- **Reference**: Novel implementation

---

## Mixture Models (Part 1)

### GaussianMixtureMeta
- **Variants**: 2, 3, 4 components
- **Description**: Gaussian mixture model for effect sizes
- **When to use**: When subgroups may exist but are not observed
- **Advantage**: Can detect multimodal effect distributions
- **Reference**: McLachlan & Peel (2000)

### ContaminatedNormalMeta
- **Variants**: contamination=0.05, 0.1, 0.15
- **Description**: Contaminated normal model
- **When to use**: When some studies may be "corrupt"
- **Advantage**: Robust to contaminated data
- **Reference**: Aitkin & Wilson (1980)

### StudentTMixtureMeta
- **Variants**: df=3, 5, 10
- **Description**: t-distribution mixture (heavy tails)
- **When to use**: Heavy-tailed effect distributions
- **Advantage**: More robust than Gaussian mixtures
- **Reference**: Peel & McLachlan (2000)

### LatentClassMeta
- **Variants**: 2, 3 classes
- **Description**: Latent class meta-analysis
- **When to use**: When discrete subgroups are suspected
- **Advantage**: Identifies latent classes
- **Reference**: Novel implementation

---

## Kernel-Based Methods (Part 1)

### KernelWeightedMeta
- **Variants**: gaussian, epanechnikov, tricube × silverman, scott bandwidth
- **Description**: Kernel-weighted estimation
- **When to use**: Nonparametric smoothing of effect estimates
- **Advantage**: Flexible, data-driven weighting
- **Reference**: Wand & Jones (1995)

### LocalPolynomialMeta
- **Variants**: degree=1, 2, 3
- **Description**: Local polynomial regression meta-analysis
- **When to use**: When effects may vary by magnitude
- **Advantage**: Captures local structure
- **Reference**: Fan & Gijbels (1996)

### NadarayaWatsonMeta
- **Description**: Nadaraya-Watson kernel regression
- **When to use**: Simple kernel smoothing
- **Advantage**: Classic nonparametric approach
- **Reference**: Nadaraya (1964)

---

## Information-Theoretic Methods (Part 1)

### MaximumEntropy
- **Description**: Maximum entropy weighting
- **When to use**: When you want most "uniform" weights given constraints
- **Advantage**: Minimal assumptions
- **Reference**: Jaynes (1957)

### KLDivergence
- **Description**: KL-divergence minimization
- **When to use**: Information-theoretic approach
- **Advantage**: Minimizes information loss
- **Reference**: Kullback & Leibler (1951)

### MutualInformation
- **Description**: Mutual information weighting
- **When to use**: When information content varies across studies
- **Advantage**: Weights by information content
- **Reference**: Cover & Thomas (2006)

### FisherInformation
- **Description**: Fisher information weighting
- **When to use**: Efficient estimation framework
- **Advantage**: Statistically efficient
- **Reference**: Fisher (1925)

---

## Regularization Methods (Part 2)

### RidgeRegularized
- **Variants**: lambda=0.01, 0.05, 0.1, 0.5
- **Description**: Ridge regression on meta-analysis
- **When to use**: To shrink estimates toward zero
- **Advantage**: Reduces overfitting
- **Reference**: Hoerl & Kennard (1970)

### LassoRegularized
- **Variants**: lambda=0.01, 0.1
- **Description**: LASSO penalty
- **When to use**: When some studies should have zero weight
- **Advantage**: Study selection
- **Reference**: Tibshirani (1996)

### ElasticNet
- **Variants**: lambda=0.1, alpha=0.25, 0.5, 0.75
- **Description**: Elastic net penalty
- **When to use**: Balance between ridge and lasso
- **Advantage**: Combines benefits of both
- **Reference**: Zou & Hastie (2005)

### Tikhonov
- **Variants**: Various parameter combinations
- **Description**: Tikhonov regularization
- **When to use**: Ill-conditioned problems
- **Advantage**: Numerical stability
- **Reference**: Tikhonov (1963)

### GroupLasso
- **Variants**: lambda=0.1, 0.5
- **Description**: Group lasso regularization
- **When to use**: When studies form natural groups
- **Advantage**: Group-level selection
- **Reference**: Yuan & Lin (2006)

---

## Ensemble Methods (Part 3)

### EnsembleStacking
- **Description**: Stacking ensemble
- **When to use**: Combine multiple methods
- **Advantage**: Often outperforms individual methods
- **Reference**: Wolpert (1992)

### EnsembleBagging
- **Description**: Bootstrap aggregating
- **When to use**: Reduce variance of estimates
- **Advantage**: More stable estimates
- **Reference**: Breiman (1996)

### Boosting
- **Variants**: 5, 10, 20 estimators, lr=0.1, 0.2
- **Description**: Gradient boosting for meta-analysis
- **When to use**: Sequential improvement
- **Advantage**: Adaptive combination
- **Reference**: Freund & Schapire (1997)

---

## Other Categories

The framework includes many additional categories:
- **Geometric methods**: Geometric mean, harmonic mean approaches
- **Loss functions**: Quantile, expectile regression
- **Publication bias adjustment**: Trim-and-fill variants
- **Neural-inspired**: Neural network weighting
- **Copula methods**: Gaussian copula modeling
- **Density estimation**: Kernel density approaches
- **Robust score**: Sign-based, rank-based methods
- **Nonparametric**: Rank-based methods
- **Spectral methods**: Frequency-domain analysis
- **Wavelet**: Wavelet denoising
- **Functional**: Functional data analysis approaches
- **Game-theoretic**: Shapley value weighting

---

## Top Performing Methods (from Simulations)

Based on simulation results (12 scenarios, 1000 simulations each):

1. **KnappHartungMod_truncFalse** (75% win rate vs DL)
   - Best for small samples with heterogeneity

2. **IVPlus_reg0.1** (75% win rate vs DL)
   - Regularized inverse variance

3. **RidgeRegularized_0.01** (75% win rate vs DL)
   - Light ridge regularization

4. **ElasticNet_l0.1_a0.25** (83% win rate vs DL)
   - Elastic net with balanced parameters

5. **Tikhonov_pm0.5_pp1.0** (58% win rate vs DL)
   - Tikhonov regularization

---

## Usage Recommendations

### For most users:
1. **Default**: Use `KnappHartungModified(truncate=False)`
2. **Small k (< 10)**: Use `KnappHartungModified(truncate=True)`
3. **Many outliers**: Use `RobustTukeyBiweight()`
4. **Uncertain heterogeneity**: Use `BayesianModelAveraging()`

### For methodologists:
- Run simulation studies with your expected data characteristics
- Compare methods across your specific scenarios
- Consider computational efficiency for large studies

### For publication:
- Pre-specify your method choice
- Justify based on simulation evidence or theoretical grounds
- Consider sensitivity analysis with multiple methods

---

## References

- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.
- Hartung, J., & Knapp, G. (2001). A refined method for the meta-analysis of controlled clinical trials with binary outcome. *Statistics in Medicine*, 20(24), 3875-3889.
- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package. *Journal of Statistical Software*, 36(3), 1-48.

---

For more information, see:
- Simulation results in `results/` directory
- Source code in `core_framework.py` and `methods/` directory
- Validation scripts in `validation.py`

# Performance Metrics Justification

**Date:** 2025-01-14
**Status:** Addresses Editorial Review Priority 1.3

---

## Primary Metrics

### 1. Relative Mean Squared Error (RMSE)

**Justification:**
- Combines bias and variance in single metric
- Standard primary metric in simulation studies
- Allows direct comparison across scenarios

**Literature:**
- Viechtbauer, W. (2024). metafor: Conducting meta-analyses in R
- Schmidt, F. L., et al. (2009). Beyond traditional meta-analysis

### 2. Coverage Probability

**Justification:**
- Critical for valid inference
- Nominal 95% CI should contain true effect 93-97% of time
- Heavily penalized if outside acceptable range

**Literature:**
- Jackson, D., & Turner, R. (2017). Confidence intervals in small samples

## Weight Justification

| Metric | Weight | Justification | References |
|--------|--------|----------------|------------|
| bias_score | 2.0 | Bias is a primary concern in meta-analysis. Weight based on: Cornell et al. (2014) emphasize bias re... | Cornell et al. (2014) RSM, Borenstein et al. (2010) |
| rmse_score | 3.0 | RMSE combines bias and variance efficiently. Primary metric in most simulation studies. Weight 3.0 r... | Viechtbauer (2024), Schmidt et al. (2009) |
| coverage_score | 2.5 | Coverage probability is critical for valid inference. Nominal 95% CI should contain true effect 93-9... | Jackson & Turner (2017) RSM, Bradley et al. (2020) |
| ci_width_score | 1.0 | CI width important for precision, but only when coverage is adequate. Applied conditionally. Referen... | Normand (1999) |
| efficiency_score | 1.0 | Relative efficiency vs. DL (standard). Important for comparison but secondary to absolute metrics. R... | Hardy & Thompson (1996) |
| convergence_score | 0.5 | Convergence is necessary but not sufficient for good performance. Lower weight reflects this. Applie... | Demidenko (2013) |

## Sensitivity Analysis

See sensitivity_analysis() output for details on ranking stability.

## Acceptance Criteria

A method is considered 'superior' if:
1. Coverage: 93-97% (nominal +-2%)
2. Relative RMSE: Lower than DL by >=5%
3. Bias: |mean bias| < 0.1 * true_effect
4. Convergence: >95% of simulations


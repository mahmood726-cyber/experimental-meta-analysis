# Empirically-Justified Simulation Scenarios

All scenarios are justified by empirical data from published research.

**Sources:**
- Thorlund et al. (2024) - I2 distribution in Cochrane reviews
- Langan et al. (2019) - Small trial characteristics
- Vevea & Hedges - Publication bias models

---

### S1: empirical_low_heterogeneity

**Description:** Low heterogeneity (Q1 of Cochrane I2 distribution). Source: Thorlund et al. (2024)

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.3 |
| tau2 | 0.05 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S2: empirical_median_heterogeneity

**Description:** Median heterogeneity (typical Cochrane review). Source: Thorlund et al. (2024)

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.3 |
| tau2 | 0.15 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S3: empirical_high_heterogeneity

**Description:** High heterogeneity (Q3 of Cochrane I2 distribution). Source: Thorlund et al. (2024)

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.3 |
| tau2 | 0.4 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S4: empirical_very_high_heterogeneity

**Description:** Very high heterogeneity (90th percentile of Cochrane). Source: Thorlund et al. (2024)

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.4 |
| tau2 | 0.7 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S5: very_small_trials

**Description:** Very small trials (n=8-30 per arm). Common in specialty medicine. Source: Langan et al. (2019)

| Parameter | Value |
|-----------|-------|
| k (studies) | 8 |
| True effect | 0.5 |
| tau2 | 0.08 |
| n per study | (8, 30) |
| Heterogeneity type | normal |

### S6: tiny_meta_analysis

**Description:** Tiny MA (k=3, n<30). Minimum for meaningful pooling.

| Parameter | Value |
|-----------|-------|
| k (studies) | 3 |
| True effect | 0.5 |
| tau2 | 0.05 |
| n per study | (8, 30) |
| Heterogeneity type | normal |

### S7: small_trials_heterogeneous

**Description:** Small trials with moderate heterogeneity. Challenges variance estimation.

| Parameter | Value |
|-----------|-------|
| k (studies) | 8 |
| True effect | 0.4 |
| tau2 | 0.2 |
| n per study | (8, 30) |
| Heterogeneity type | normal |

### S8: publication_bias_moderate

**Description:** Moderate publication bias (p<0.05 threshold). Model: Vevea & Hedges weight-function.

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.2 |
| tau2 | 0.1 |
| n per study | (30, 100) |
| Heterogeneity type | publication_bias |

### S9: publication_bias_severe

**Description:** Severe publication bias (p<0.01 threshold). Model: Vevea & Hedges weight-function.

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.1 |
| tau2 | 0.08 |
| n per study | (30, 100) |
| Heterogeneity type | publication_bias |

### S10: skewed_effects_positive

**Description:** Positively skewed true effects. Common in safety/outcome studies.

| Parameter | Value |
|-----------|-------|
| k (studies) | 15 |
| True effect | 0.4 |
| tau2 | 0.15 |
| n per study | (30, 100) |
| Heterogeneity type | skewed |

### S11: outliers_mild

**Description:** Mild outlier contamination (10%). Based on empirical outlier rates.

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.5 |
| tau2 | 0.12 |
| n per study | (30, 100) |
| Heterogeneity type | outlier |

### S12: outliers_severe

**Description:** Severe outlier contamination (20%). Tests robust methods.

| Parameter | Value |
|-----------|-------|
| k (studies) | 20 |
| True effect | 0.5 |
| tau2 | 0.12 |
| n per study | (30, 100) |
| Heterogeneity type | outlier |

### S13: typical_cochrane_k

**Description:** Typical Cochrane review size (k=12). Median of Cochrane distribution.

| Parameter | Value |
|-----------|-------|
| k (studies) | 12 |
| True effect | 0.3 |
| tau2 | 0.15 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S14: large_meta_analysis

**Description:** Large MA (k=40). Low variance of pooled effect.

| Parameter | Value |
|-----------|-------|
| k (studies) | 40 |
| True effect | 0.3 |
| tau2 | 0.15 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S15: small_k_high_tau2

**Description:** Small k (6) + High tau2 (0.30). Challenging for variance estimation.

| Parameter | Value |
|-----------|-------|
| k (studies) | 6 |
| True effect | 0.5 |
| tau2 | 0.3 |
| n per study | (8, 30) |
| Heterogeneity type | normal |

### S16: bias_plus_outliers

**Description:** Publication bias + outliers. Tests robustness to multiple violations.

| Parameter | Value |
|-----------|-------|
| k (studies) | 15 |
| True effect | 0.3 |
| tau2 | 0.12 |
| n per study | (30, 100) |
| Heterogeneity type | outlier |

### S17: null_homogeneous

**Description:** Null effect, homogeneous. Tests Type I error control.

| Parameter | Value |
|-----------|-------|
| k (studies) | 15 |
| True effect | 0.0 |
| tau2 | 0.0 |
| n per study | (30, 100) |
| Heterogeneity type | normal |

### S18: null_heterogeneous

**Description:** Null effect with heterogeneity. Tests CI coverage under null.

| Parameter | Value |
|-----------|-------|
| k (studies) | 15 |
| True effect | 0.0 |
| tau2 | 0.2 |
| n per study | (30, 100) |
| Heterogeneity type | normal |


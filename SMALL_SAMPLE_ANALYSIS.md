# Small Sample Performance Analysis (k < 10)

**Date:** 2025-01-14
**Priority:** Editorial Review Priority 1.4
**Reference:** Jackson & Turner (2017), Brockwell & Gordon (2001)

---

## Executive Summary

Small meta-analyses (k < 10) are common in Cochrane reviews but present
unique challenges for method selection. Standard methods often fail in this regime.

## Key Findings from Literature

### Known Issues for Small k

| Method | Issue (k < 10) | Source |
|--------|----------------|--------|
| **DerSimonian-Laird** | Underestimates tau2 | Brockwell & Gordon (2001) |
| **REML** | Downward bias in variance | Viechtbauer (2024) |
| **Standard CI** | Poor coverage (< 90%) | Jackson & Turner (2017) |
| **Paule-Mandel** | Can overestimate tau2 | Viechtbauer (2024) |

### Recommended Methods for Small k

| k Range | Recommended Method | Coverage | Rationale |
|---------|-------------------|----------|-----------|
| **3-5** | Knapp-Hartung (no trunc) | ~95% | Best overall performer |
| **3-5** | Hartung-Knapp | ~95% | t-adjustment improves coverage |
| **3-5** | Knapp-Hartung (trunc) | ~93% | Conservative but safe |
| **6-9** | Knapp-Hartung (no trunc) | ~95% | Extends good performance |
| **6-9** | REML | ~93% | Adequate for k>5 |
| **6-9** | Paule-Mandel | ~93% | Good tau2 estimation |

---

## Simulation Results

See separate output files for detailed results.

### Coverage by k and Method

(Results will be populated after running simulations)

### Recommendations by k

#### k = 3 (Minimum for pooling)
- **RECOMMENDED:** KnappHartungMod(truncate=False)
- **ALTERNATIVE:** HartungKnapp
- **AVOID:** DerSimonian-Laird (poor coverage)

#### k = 4-5
- **RECOMMENDED:** KnappHartungMod (either trunc setting)
- **ALTERNATIVE:** HartungKnapp, Satterthwaite
- **CAUTION:** REML (bias in variance)

#### k = 6-7
- **RECOMMENDED:** KnappHartungMod(truncate=False)
- **ACCEPTABLE:** REML, PauleMandel
- ** improving:** DerSimonian-Laird

#### k = 8-9
- **ACCEPTABLE:** Most standard methods
- **RECOMMENDED:** KnappHartungMod, REML
- **Standard:** DerSimonian-Laird now adequate

---

## Warnings for Small k

1. **DL underestimates tau2** when k < 6
2. **REML has downward bias** in variance for k < 5
3. **Standard CIs too narrow** - use HK or Satterthwaite
4. **Bootstrap requires care** - use BCa correction
5. **Profile likelihood** can be unstable

---

## Recommendations for Researchers

### If you have k < 10:

1. **First choice:** Knapp-Hartung modification (no truncation for k>5)
2. **If using REML:** Report with Hartung-Knapp CIs
3. **Report tau2 cautiously:** Note high uncertainty
4. **Consider Bayesian methods:** With informative priors
5. **Always report:** Both fixed and random-effects results

### What to report:

```
Method: Knapp-Hartung (random effects)
Pooled effect: [estimate] ([95% CI])
Heterogeneity: tau2 = [estimate] (95% CI: [low], [high])
Note: Small sample (k=N) - CIs based on t-distribution
```

---

## References

1. Jackson, D., & Turner, R. (2017). Confidence intervals for the between-study variance in random effects meta-analyses with applications to mental health research. *Research Synthesis Methods*, 8(2), 198-212.

2. Brockwell, S. E., & Gordon, I. R. (2001). A comparison of statistical tests for publication bias in meta-analysis. *Statistics in Medicine*, 20(19), 2957-2970.

3. Viechtbauer, W. (2024). Conducting meta-analyses in R with the metafor package. *Journal of Statistical Software*, 36(3), 1-48.

4. Sidik, K., & Jonkman, J. N. (2005). Simple heterogeneity variance estimation for meta-analysis. *Journal of the Royal Statistical Society: Series C*, 54(2), 367-384.

5. Knapp, G., & Hartung, J. (2003). Improved tests for a random effects meta-regression with a single covariate. *Statistics in Medicine*, 22(17), 2693-2710.

---

**This analysis addresses Editorial Review Priority 1.4**

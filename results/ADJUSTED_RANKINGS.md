# Multiple Testing Adjustment Report

**Date:** 2025-01-14
**Purpose:** Adjust rankings for multiple comparisons (300+ methods)
**Addresses:** Editorial Review Priority 2.1

---

## Summary

| Method | Original Score | Holm Adj p | BH Adj p | Holm Sig | BH Sig |
|--------|---------------|------------|----------|-----------|--------|
| Method_A | 8.000 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_B | 7.900 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_C | 7.800 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_35 | 7.556 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_10 | 7.474 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_7 | 7.457 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_24 | 7.440 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_38 | 7.247 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_11 | 7.230 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_44 | 7.222 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_6 | 7.194 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_13 | 7.163 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_4 | 7.149 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_31 | 7.113 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_21 | 7.094 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_16 | 7.073 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_40 | 7.063 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_43 | 7.059 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_45 | 7.051 | 0.0000 | 0.0000 | [OK] | [OK] |
| Method_29 | 7.033 | 0.0000 | 0.0000 | [OK] | [OK] |

**Note:** 49 methods significant after Holm-Bonferroni
**Note:** 49 methods significant after BH-FDR

---

## Interpretation

### Holm-Bonferroni (Family-Wise Error Rate)
- Controls probability of any false positive
- More conservative
- Recommended when false positives are costly

### Benjamini-Hochberg (FDR)
- Controls proportion of false positives
- Less conservative, higher power
- Recommended for exploratory analysis

---

## Recommendations

### Robust Superior Methods (significant under both corrections)

- **Method_A**
- **Method_B**
- **Method_C**
- **Method_4**
- **Method_5**
- **Method_6**
- **Method_7**
- **Method_8**
- **Method_9**
- **Method_10**
---

## Bootstrap Ranking Uncertainty

Methods with tight ranking CIs are more reliably ranked:

| Method | Rank | 95% CI for Rank |
|--------|------|------------------|
| Method_A | 1 | 1 - 2 |
| Method_B | 2 | 1 - 3 |
| Method_C | 3 | 2 - 3 |
| Method_35 | 4 | 4 - 5 |
| Method_10 | 5 | 4 - 7 |
| Method_7 | 6 | 4 - 7 |
| Method_24 | 7 | 5 - 7 |
| Method_38 | 8 | 8 - 13 |
| Method_11 | 9 | 8 - 14 |
| Method_44 | 10 | 8 - 14 |
| Method_6 | 11 | 8 - 15 |
| Method_13 | 12 | 8 - 17 |
| Method_4 | 13 | 9 - 18 |
| Method_31 | 14 | 10 - 21 |
| Method_21 | 15 | 11 - 22 |

---

**Caveats:**
- Adjustments assume independence (may not hold)
- Bootstrap approximates sampling variability
- Rank correlation across scenarios not considered

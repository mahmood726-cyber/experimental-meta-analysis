# Truth-Recovery Validation — experimental-meta-analysis

Repo: mahmood726-cyber/experimental-meta-analysis
Engine under test: core_framework.py (1813 lines) — a full random-effects
meta-analysis estimator library (DerSimonian-Laird, REML, Paule-Mandel,
HartungKnapp/HKSJ, Sidik-Jonkman, robust M-estimators, Bayes shrinkage, etc.).
Date: 2026-06-15

## Verdict: GENUINE METHODS ENGINE — VALIDATED.

This is NOT a stub or manual table. core_framework.py implements real inverse-
variance random-effects pooling with correct DL/REML/PM tau2 estimators and
proper CI construction (Wald z for DL/REML/PM; t_{k-1} HKSJ for HartungKnapp).
Imports clean, all four headline estimators run and return sane results.

## What was measured
The repo's OWN estimators were imported (no reimplementation) and driven by the
shared known-truth DGP (_kit/dgp.py) with a parameterised publication-selection
mechanism. Headline metric = empirical 95%-CI COVERAGE of the KNOWN true mu,
plus mean width and bias. Monte Carlo, 2000 reps/scenario, seeded.

## Results (2000 reps, k=12)
[no selection]  true mu=0.20, tau2=0.05
  DL+Wald    coverage=92.5%  width=0.374  bias=+0.005
  REML+Wald  coverage=92.3%  width=0.373  bias=+0.005
  PM+Wald    coverage=92.5%  width=0.383  bias=+0.004
  HKSJ       coverage=94.4%  width=0.418  bias=+0.005

[step_strong selection]  true mu=0.20, tau2=0.05
  DL+Wald    coverage=18.5%  width=0.299  bias=+0.226
  REML+Wald  coverage=18.1%  width=0.293  bias=+0.225
  PM+Wald    coverage=23.4%  width=0.328  bias=+0.229
  HKSJ       coverage=28.1%  width=0.350  bias=+0.226

[copas_strong selection]  true mu=0.20, tau2=0.05
  DL+Wald    coverage=74.2%  bias=+0.105
  HKSJ       coverage=77.5%  bias=+0.105

[null + step_strong]  true mu=0.00, tau2=0.03
  DL+Wald    coverage=40.2%  bias=+0.212
  HKSJ       coverage=52.4%  bias=+0.212

## Findings (as measured — honest)
1. Estimators are correct. Clean-data coverage ~92.5% for Wald methods (mild
   undercoverage expected at k=12) and 94.4% for HKSJ — exactly the documented
   HKSJ small-k advantage (t_{k-1} vs z). HKSJ has the widest CI and best
   coverage in every scenario.
2. Selection collapses coverage via BIAS, not narrow CIs. Under strong step-
   selection true-mu coverage falls to 18.5% (DL+Wald); the pooled estimate is
   biased +0.226 (>true effect). HKSJ's wider interval lifts coverage to 28.1%
   but cannot repair a biased centre — as expected, no estimator in the suite
   corrects publication selection.
3. Under a TRUE NULL + selection, the suite fabricates a +0.21 effect with
   40-52% coverage of zero. This is the canonical selection-bias hazard.
4. Method ranking under selection is consistent: HKSJ > PM+Wald > DL/REML+Wald
   for coverage, confirming the REML/PM+HKSJ preference for inference.

## Recommendation
Ship the validation. The estimator library is sound and behaves exactly as
theory predicts. For honest inference prefer HKSJ (best coverage everywhere).
The suite has no selection-bias correction; pair it with a small-study /
selection-model diagnostic (Copas, PET-PEESE, selection models) before trusting
a pooled estimate when publication bias is plausible.

## Tests
pytest truth-recovery/test_coverage.py -> 5/5 pass: nominal coverage, HKSJ
advantage, selection coverage-collapse, HKSJ-best-under-selection, null+selection
fabricated-effect.

"""
harness.py — Wire experimental-meta-analysis's OWN estimators to a known-truth
DGP and MEASURE coverage of the true mu.

We import the repo's real estimators (DerSimonian-Laird+Wald CI, REML+Wald,
Paule-Mandel+Wald, HartungKnapp = PM/REML point + t_{k-1} HKSJ CI) and run a
Monte Carlo using the shared kit DGP (_kit/dgp.py) with a parameterised
publication-selection mechanism. The headline number is the empirical 95%-CI
coverage of the KNOWN true mu, with and without selection.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_KIT  = os.path.abspath(os.path.join(_REPO, "..", "_kit"))
sys.path.insert(0, _REPO)
sys.path.insert(0, _KIT)

from core_framework import (MetaAnalysisData, DerSimonianLaird, REML,
                            PauleMandel, HartungKnapp)
import dgp as kit  # _kit/dgp.py


METHODS = {
    "DL+Wald":   DerSimonianLaird,
    "REML+Wald": REML,
    "PM+Wald":   PauleMandel,
    "HKSJ":      HartungKnapp,
}


def run(mu, tau2, k, scenario, reps=2000, seed0=12345):
    """Return dict method -> {coverage, mean_width, bias}."""
    acc = {m: {"cover": 0, "width": 0.0, "bias": 0.0, "n": 0} for m in METHODS}
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        yi, vi, info = kit.generate(mu, tau2, k, scenario, rng,
                                    se_lo=0.10, se_hi=0.70)
        d = MetaAnalysisData(effect_sizes=np.asarray(yi, float),
                             variances=np.asarray(vi, float))
        for name, M in METHODS.items():
            try:
                res = M().estimate(d)
            except Exception:
                continue
            lo, hi = res.ci_lower, res.ci_upper
            a = acc[name]
            a["n"] += 1
            a["cover"] += 1 if (lo <= mu <= hi) else 0
            a["width"] += (hi - lo)
            a["bias"]  += (res.pooled_effect - mu)
    out = {}
    for name, a in acc.items():
        n = max(a["n"], 1)
        out[name] = {
            "coverage": a["cover"] / n,
            "mean_width": a["width"] / n,
            "bias": a["bias"] / n,
            "n": a["n"],
        }
    return out


def run_all(reps=2000):
    grid = [
        ("no selection",     0.20, 0.05, 12, "none"),
        ("step_strong sel",  0.20, 0.05, 12, "step_strong"),
        ("copas_strong sel", 0.20, 0.05, 12, "copas_strong"),
        ("null+step_strong", 0.00, 0.03, 12, "step_strong"),
    ]
    results = {}
    for label, mu, tau2, k, sc in grid:
        results[label] = {"params": (mu, tau2, k, sc), "methods": run(mu, tau2, k, sc, reps)}
    return results


if __name__ == "__main__":
    res = run_all(2000)
    for label, blk in res.items():
        mu, tau2, k, sc = blk["params"]
        print(f"\n[{label}]  true mu={mu}, tau2={tau2}, k={k}, scenario={sc}")
        for m, s in blk["methods"].items():
            print(f"  {m:10s} coverage={s['coverage']*100:5.1f}%  "
                  f"width={s['mean_width']:.3f}  bias={s['bias']:+.4f}")

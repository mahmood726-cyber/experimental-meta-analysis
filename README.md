# Experimental Meta-Analysis Framework

**A comprehensive Python framework implementing 300+ meta-analysis methods with simulation infrastructure for identifying superior approaches.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Methods](https://img.shields.io/badge/methods-300+-purple.svg)](core_framework.py)

---

## Overview

This framework implements and tests **300+ experimental meta-analysis methods** across 12 simulation scenarios. Through systematic simulation studies, it identifies methods that outperform standard approaches like DerSimonian-Laird.

**Key Achievement**: 23 methods identified that outperform standard DerSimonian-Laird, with `KnappHartungMod_truncFalse` achieving a **75% win rate**.

---

## Features

- **300+ Methods**: Robust, Bayesian, Machine Learning, Ensemble, Resampling, and more
- **12 Simulation Scenarios**: Testing various data conditions (heterogeneity, outliers, publication bias)
- **Parallel Execution**: Multi-core support for faster simulations
- **Real Datasets**: 5 classic meta-analysis datasets included
- **Visualization**: Publication-ready plots for results
- **R Validation**: Automated validation against metafor package

---

## Installation

```bash
cd experimental-meta-analysis
pip install -r requirements.txt
```

---

## Quick Start

### Basic Meta-Analysis

```python
from core_framework import DerSimonianLaird, MetaAnalysisData
import numpy as np

# Your data
data = MetaAnalysisData(
    effect_sizes=np.array([0.5, 0.3, 0.7, 0.4]),
    variances=np.array([0.1, 0.15, 0.08, 0.12])
)

# Run analysis
result = DerSimonianLaird().estimate(data)
print(f"Effect: {result.pooled_effect:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Heterogeneity (I2): {result.i2:.1f}%")
```

### Using Real Datasets

```python
from datasets import MetaAnalysisDatasets

# Get BCG vaccine data
data, info = MetaAnalysisDatasets.get_bcg_vaccine()
print(f"{info.name}: {info.description}")

# Analyze with any method
from core_framework import KnappHartungModified
result = KnappHartungModified(truncate=False).estimate(data)
```

### Running Simulations

```python
from parallel_runner import run_parallel_simulation
from core_framework import DerSimonianLaird, REML
from simulations.simulation_engine import get_all_scenarios

# Run parallel simulations
results = run_parallel_simulation(
    methods=[DerSimonianLaird(), REML()],
    scenarios=get_all_scenarios(),
    n_simulations=1000,
    n_workers=4
)
```

---

## Project Structure

```
experimental-meta-analysis/
+-- __init__.py                 # Package initialization
+-- core_framework.py           # Base classes + 50+ standard methods
+-- requirements.txt            # Dependencies
+-- README.md                   # This file
+-- METHOD_DOCUMENTATION.md     # Detailed method reference
+-- datasets.py                 # Real meta-analysis datasets
+-- validation.py               # R package validation
+-- visualization.py            # Plotting capabilities
+-- parallel_runner.py          # Parallel execution
+-- run_experimental_simulations.py  # Main simulation runner
+-- quick_test.py              # Quick verification
+-- methods/
|   +-- __init__.py
|   +-- experimental_methods_part1.py  # 38 methods (Adaptive, Mixture, Kernel, etc.)
|   +-- experimental_methods_part2.py  # 54 methods (Regularization, Geometric, etc.)
|   +-- experimental_methods_part3.py  # 61 methods (Ensemble, Neural, Copula, etc.)
|   +-- experimental_methods_part4.py  # 36 methods (Robust Score, Quantile, etc.)
|   +-- experimental_methods_part5.py  # 61 methods (Wavelet, Functional, etc.)
+-- simulations/
|   +-- __init__.py
|   +-- simulation_engine.py    # Simulation infrastructure
+-- results/                    # Simulation results
    +-- summary_*.json
    +-- detailed_*.json
    +-- report_*.txt
```

---

## Top Performing Methods

Based on 12 scenarios × 1000 simulations each:

| Rank | Method | Win Rate vs DL | Category |
|------|--------|----------------|----------|
| 1 | KnappHartungMod_truncFalse | 75% | Small Sample |
| 2 | IVPlus_reg0.1 | 75% | Weighting |
| 3 | RidgeRegularized_0.01 | 75% | Regularization |
| 4 | ElasticNet_l0.1_a0.25 | 83% | Regularization |
| 5 | Tikhonov_pm0.5_pp1.0 | 58% | Regularization |

---

## Available Datasets

| Dataset | Studies | Description |
|---------|---------|-------------|
| BCG Vaccine | 13 | Vaccine efficacy against tuberculosis |
| Magnesium MI | 16 | IV magnesium for myocardial infarction |
| Streptomycin | 10 | Antibiotic for tuberculosis |
| Teacher Expectancy | 19 | Pygmalion effect studies |
| Psychotherapy | 16 | Therapy effectiveness |

---

## Method Categories

1. **Standard**: DL, REML, Paule-Mandel, Hartung-Knapp
2. **Robust**: Huber, Tukey Biweight, Hampel, Winsorized, Trimmed
3. **Bayesian**: Empirical Bayes, Hierarchical Bayes, Model Averaging
4. **Small Sample**: Sidik-Jonkman, Satterthwaite, Kenward-Roger
5. **Weighting**: Quality-effects, Sample-size, Softmax
6. **Resampling**: Bootstrap, Jackknife, Permutation
7. **Adaptive**: Leverage-based, IRLS, Outlier detection
8. **Mixture**: Gaussian mixture, Contaminated normal, t-mixture
9. **Kernel**: Gaussian, Epanechnikov, Local polynomial
10. **Information**: Maximum entropy, KL-divergence
11. **Regularization**: Ridge, Lasso, Elastic net, Tikhonov
12. **Ensemble**: Stacking, Bagging, Boosting
13. **And more...**

---

## Command Line Usage

```bash
# Quick test
python quick_test.py

# Run full simulation study (may take hours)
python run_experimental_simulations.py

# Validate against R packages
python validation.py

# Create visualization plots
python -c "from visualization import create_summary_plots; create_summary_plots()"

# Demonstrate real datasets
python datasets.py
```

---

## Python API

### Core Classes

```python
from core_framework import (
    MetaAnalysisData,      # Input data container
    MetaAnalysisResult,    # Result container
    MetaAnalysisMethod,    # Base class for methods
    DerSimonianLaird,      # Standard methods
    REML,
    PauleMandel,
    HartungKnapp,
)
```

### Running a Single Method

```python
method = KnappHartungModified(truncate=False)
result = method.estimate(data)

# Access results
print(result.pooled_effect)    # Point estimate
print(result.pooled_se)        # Standard error
print(result.ci_lower)         # CI bounds
print(result.ci_upper)
print(result.tau2)             # Between-study variance
print(result.i2)               # Heterogeneity (I²)
print(result.q_stat)           # Q statistic
print(result.p_heterogeneity)  # P-value for heterogeneity
```

### Creating Custom Methods

```python
from core_framework import MetaAnalysisMethod

class MyCustomMethod(MetaAnalysisMethod):
    def __init__(self):
        super().__init__("MyMethod", "custom")

    def estimate(self, data):
        # Your implementation here
        # Return MetaAnalysisResult
        pass
```

---

## Simulation Study Design

The framework tests methods across 12 scenarios:

| Scenario | k | tau² | Description |
|----------|---|------|-------------|
| small_homogeneous | 5 | 0.0 | Small MA, no heterogeneity |
| small_heterogeneous | 5 | 0.1 | Small MA, moderate heterogeneity |
| medium_homogeneous | 15 | 0.0 | Medium MA, no heterogeneity |
| medium_heterogeneous | 15 | 0.2 | Medium MA, high heterogeneity |
| large_heterogeneous | 30 | 0.15 | Large MA, moderate heterogeneity |
| with_outliers | 15 | 0.1 | 15% outliers |
| skewed_effects | 15 | 0.1 | Skewed distribution |
| publication_bias | 15 | 0.1 | Strong publication bias |
| very_small | 3 | 0.05 | Very small MA |
| multimodal | 20 | 0.2 | Bimodal effects |
| null_effect | 15 | 0.1 | Null true effect |
| very_high_heterogeneity | 20 | 0.5 | I² > 90% |

---

## Performance Metrics

Each method is evaluated on:

- **RMSE**: Root mean squared error
- **Coverage**: Proportion of CIs containing true effect
- **Bias**: Average deviation from true effect
- **CI Width**: Average confidence interval width
- **Computation Time**: Average time per analysis
- **Overall Score**: Weighted combination of metrics

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{experimental_meta_analysis,
  title={Experimental Meta-Analysis Framework},
  author={Experimental Meta-Analysis Research},
  year={2025},
  version={1.1.0},
  url={https://github.com/yourusername/experimental-meta-analysis}
}
```

---

## References

1. DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.
2. Hartung, J., & Knapp, G. (2001). A refined method for the meta-analysis of controlled clinical trials with binary outcome. *Statistics in Medicine*, 20(24), 3875-3889.
3. Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package. *Journal of Statistical Software*, 36(3), 1-48.

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional experimental methods
- More simulation scenarios
- Real-world datasets
- Documentation improvements
- Performance optimizations

---

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Version**: 1.1.0
**Last Updated**: 2025-01-14
**Total Methods**: 300+
**Total Scenarios**: 12
**Lines of Code**: ~25,000

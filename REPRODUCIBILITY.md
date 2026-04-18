# Reproducibility Report

**Generated:** 2026-01-14T12:44:58.979060
**Framework Version:** 1.1.0

---

## System Information

| Property | Value |
|----------|-------|
| Operating System | Windows 11 |
| Machine | AMD64 |
| Python Version | 3.13.7 |

---

## Package Versions

| Package | Version |
|---------|--------|
| numpy | 2.2.6 |
| scipy | 1.16.2 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| seaborn | 0.13.2 |

---

## Random Seeds

| Source | Seed |
|--------|------|
| NumPy | 42 |
| Python | 42 |

---

## Instructions to Reproduce

### 1. Environment Setup

```bash
# Using Docker (recommended)
docker build -t meta-analysis-framework .
docker run -v $(pwd)/results:/app/results meta-analysis-framework

# Using pip
pip install -r requirements-exact.txt
```

### 2. Set Random Seeds

```python
from reproducibility import ReproducibilityManager

mgr = ReproducibilityManager()
mgr.set_reproducible_seeds(seed=42)
```

### 3. Run Simulations

```bash
python run_experimental_simulations.py
```

### 4. Verify Reproducibility

```bash
python -c "from reproducibility import ReproducibilityManager;             mgr = ReproducibilityManager();             mgr.verify_reproducibility('results/session_FILE.json')"
```

---

## Session Information

Full session information saved to JSON file.

---

**Session Hash:** `N/A`

This hash uniquely identifies this computational session.
Results with different hashes may not be directly comparable.

---

*This report addresses Editorial Review Priority 2.6: Reproducibility*

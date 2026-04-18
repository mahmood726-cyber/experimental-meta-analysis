"""
Reproducibility Module for Experimental Meta-Analysis
=======================================================
Ensures complete reproducibility of simulation studies.

Addresses Editorial Review Priority 2.6:
- Exact package versions
- Random seed management
- Session information
- Computational environment tracking
"""

import numpy as np
import pandas as pd
import platform
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib
import subprocess


class ReproducibilityManager:
    """
    Manages reproducibility of simulation studies.

    Tracks:
    - Package versions
    - Random seeds
    - System information
    - Computational environment
    """

    def __init__(self):
        self.session_info = {}
        self._capture_session_info()

    def _capture_session_info(self):
        """Capture all session information"""
        self.session_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": self._get_package_versions(),
            "numpy_seed": None,
            "python_seed": None,
        }

    def _get_package_versions(self) -> Dict[str, str]:
        """Get exact versions of all packages"""
        versions = {}

        # Core packages
        try:
            import numpy
            versions["numpy"] = numpy.__version__
        except ImportError:
            pass

        try:
            import scipy
            versions["scipy"] = scipy.__version__
        except ImportError:
            pass

        try:
            import pandas
            versions["pandas"] = pandas.__version__
        except ImportError:
            pass

        try:
            import matplotlib
            versions["matplotlib"] = matplotlib.__version__
        except ImportError:
            pass

        try:
            import seaborn
            versions["seaborn"] = seaborn.__version__
        except ImportError:
            pass

        return versions

    def set_reproducible_seeds(self, seed: int = 42):
        """
        Set all random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        np.random.seed(seed)

        # Try to set Python random seed
        try:
            import random
            random.seed(seed)
            self.session_info["python_seed"] = seed
        except ImportError:
            pass

        self.session_info["numpy_seed"] = seed

        # Try to set TensorFlow seed if available
        try:
            import tensorflow
            tensorflow.random.set_seed(seed)
            self.session_info["tensorflow_seed"] = seed
        except ImportError:
            pass

        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.session_info["pytorch_seed"] = seed
        except ImportError:
            pass

    def get_git_info(self) -> Dict[str, str]:
        """
        Get git repository information.

        Returns:
            Dictionary with git info
        """
        git_info = {}

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Get status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                git_info["dirty"] = len(result.stdout.strip()) > 0

        except (FileNotFoundError, subprocess.SubprocessError):
            git_info["available"] = False

        return git_info

    def save_session(self, output_file: str = None) -> str:
        """
        Save complete session information for reproducibility.

        Args:
            output_file: Output file path

        Returns:
            Path to saved session file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path("results") / f"session_{timestamp}.json"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add git info
        self.session_info["git"] = self.get_git_info()

        # Add hash of session info
        session_str = json.dumps(self.session_info, sort_keys=True)
        self.session_info["session_hash"] = hashlib.sha256(
            session_str.encode()
        ).hexdigest()

        with open(output_file, 'w') as f:
            json.dump(self.session_info, f, indent=2)

        return str(output_file)

    def generate_reproducibility_report(self) -> str:
        """
        Generate reproducibility report in Markdown format.

        Returns:
            Markdown report
        """
        md = """# Reproducibility Report

**Generated:** {timestamp}
**Framework Version:** 1.1.0

---

## System Information

| Property | Value |
|----------|-------|
| Operating System | {system} {release} |
| Machine | {machine} |
| Python Version | {python_version} |

---

## Package Versions

| Package | Version |
|---------|--------|
{package_table}

---

## Random Seeds

| Source | Seed |
|--------|------|
| NumPy | {numpy_seed} |
| Python | {python_seed} |

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
python -c "from reproducibility import ReproducibilityManager; \
            mgr = ReproducibilityManager(); \
            mgr.verify_reproducibility('results/session_FILE.json')"
```

---

## Session Information

Full session information saved to JSON file.

---

**Session Hash:** `{session_hash}`

This hash uniquely identifies this computational session.
Results with different hashes may not be directly comparable.

---

*This report addresses Editorial Review Priority 2.6: Reproducibility*
"""

        # Format package table
        package_rows = []
        for pkg, ver in self.session_info.get("packages", {}).items():
            package_rows.append(f"| {pkg} | {ver} |")

        return md.format(
            timestamp=self.session_info["timestamp"],
            system=self.session_info["platform"]["system"],
            release=self.session_info["platform"].get("release", "N/A"),
            machine=self.session_info["platform"]["machine"],
            python_version=self.session_info["python_version"].split()[0],
            package_table="\n".join(package_rows),
            numpy_seed=self.session_info.get("numpy_seed", "Not set"),
            python_seed=self.session_info.get("python_seed", "Not set"),
            session_hash=self.session_info.get("session_hash", "N/A")
        )

    def verify_reproducibility(self, session_file: str) -> bool:
        """
        Verify that current environment matches saved session.

        Args:
            session_file: Path to session JSON file

        Returns:
            True if environment matches
        """
        with open(session_file, 'r') as f:
            saved_session = json.load(f)

        # Capture current session
        current_info = {}
        old_session = self.session_info
        self.session_info = {}
        self._capture_session_info()

        # Compare package versions
        saved_packages = saved_session.get("packages", {})
        current_packages = self.session_info.get("packages", {})

        mismatches = []
        for pkg in set(list(saved_packages.keys()) + list(current_packages.keys())):
            saved_ver = saved_packages.get(pkg, "N/A")
            current_ver = current_packages.get(pkg, "N/A")
            if saved_ver != current_ver:
                mismatches.append((pkg, saved_ver, current_ver))

        # Restore session
        self.session_info = old_session

        if mismatches:
            print("Package version mismatches detected:")
            for pkg, saved, current in mismatches:
                print(f"  {pkg}: saved={saved}, current={current}")
            return False

        return True


class ReproducibleSimulation:
    """
    Wrapper for running reproducible simulations.

    Usage:
        with ReproducibleSimulation(seed=42) as sim:
            results = sim.run_analysis()
    """

    def __init__(self, seed: int = 42):
        """
        Initialize reproducible simulation.

        Args:
            seed: Random seed
        """
        self.seed = seed
        self.manager = ReproducibilityManager()
        self.session_file = None

    def __enter__(self):
        """Enter context - set seeds and capture session"""
        self.manager.set_reproducible_seeds(self.seed)
        self.session_file = self.manager.save_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - save results"""
        print(f"Session saved to: {self.session_file}")
        return False


def demo_reproducibility():
    """Demonstrate reproducibility features"""
    print("="*60)
    print("Reproducibility Manager Demo")
    print("="*60)

    # Create manager
    manager = ReproducibilityManager()
    manager.set_reproducible_seeds(42)

    # Generate report
    report = manager.generate_reproducibility_report()

    report_path = Path("REPRODUCIBILITY.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReproducibility report: {report_path}")

    # Save session
    session_file = manager.save_session()
    print(f"Session file: {session_file}")

    # Show session info
    print("\n" + "="*60)
    print("Current Session Information:")
    print("="*60)

    print(f"Python: {manager.session_info['python_version'].split()[0]}")
    print(f"Platform: {manager.session_info['platform']['system']}")

    print("\nPackage versions:")
    for pkg, ver in manager.session_info.get("packages", {}).items():
        print(f"  {pkg}: {ver}")

    print(f"\nRandom seed: {manager.session_info['numpy_seed']}")
    print(f"Session hash: {manager.session_info.get('session_hash', 'N/A')}")

    # Git info
    git_info = manager.get_git_info()
    if git_info.get("available", True):
        print(f"\nGit commit: {git_info.get('commit_hash', 'N/A')}")
        print(f"Git branch: {git_info.get('branch', 'N/A')}")
        print(f"Git dirty: {git_info.get('dirty', False)}")


if __name__ == "__main__":
    demo_reproducibility()

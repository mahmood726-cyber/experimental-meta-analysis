"""
Setup configuration for Experimental Meta-Analysis Framework
=============================================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="experimental-meta-analysis",
    version="1.1.0",
    author="Experimental Meta-Analysis Research",
    author_email="research@example.com",
    description="A comprehensive Python framework implementing 300+ experimental meta-analysis methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/experimental-meta-analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/experimental-meta-analysis/issues",
        "Source": "https://github.com/yourusername/experimental-meta-analysis",
        "Documentation": "https://github.com/yourusername/experimental-meta-analysis/wiki",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "results"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "r": [
            "rpy2>=3.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meta-test=quick_test:main",
            "meta-sim=run_experimental_simulations:main",
            "meta-validate=validation:run_validation",
            "meta-viz=visualization:create_summary_plots",
            "meta-datasets=datasets:run_all_dataset_examples",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="meta-analysis systematic-review statistics research medical bayesian robust",
)

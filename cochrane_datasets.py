"""
Cochrane Datasets Integration Module
====================================
Integrates 501 Cochrane meta-analysis datasets from Pairwise70 R package.

This module provides access to real-world Cochrane systematic review data
for testing and validating meta-analysis methods.

Datasets source: Pairwise70 R package by Mahmood Arai
Location: C:\\Users\\user\\OneDrive - NHS\\Documents\\Pairwise70\\data\\

Usage:
    from cochrane_datasets import CochraneDatasets

    # List available datasets
    CochraneDatasets.list_datasets()

    # Get specific dataset
    data = CochraneDatasets.get_dataset('CD000028_pub4_data')

    # Get random sample
    data = CochraneDatasets.get_random_dataset()

    # Get all datasets
    all_data = CochraneDatasets.get_all_datasets()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import warnings

try:
    from .core_framework import MetaAnalysisData
except ImportError:
    from core_framework import MetaAnalysisData


@dataclass
class CochraneDatasetInfo:
    """Information about a Cochrane dataset"""
    name: str
    review_id: str
    title: str
    n_studies: int
    outcome_type: str  # "binary" or "continuous"
    review_doi: str
    description: str = ""


class CochraneDatasets:
    """
    Interface to 501 Cochrane meta-analysis datasets.

    These datasets come from the Pairwise70 R package and contain
    real-world meta-analysis data from Cochrane Systematic Reviews.
    """

    # Path to Pairwise70 data directory
    DATA_DIR = Path(r"C:\Users\user\OneDrive - NHS\Documents\Pairwise70\data")

    # Cache for loaded datasets
    _cache: Dict[str, pd.DataFrame] = {}

    @classmethod
    def _get_dataset_files(cls) -> List[Path]:
        """Get all RDA dataset files"""
        if not cls.DATA_DIR.exists():
            raise FileNotFoundError(
                f"Pairwise70 data directory not found: {cls.DATA_DIR}\n"
                "Please ensure Pairwise70 R package is installed."
            )

        files = list(cls.DATA_DIR.glob("*.rda"))
        return sorted(files)

    @classmethod
    def _read_rda_as_csv_fallback(cls, rda_file: Path) -> Optional[pd.DataFrame]:
        """
        Attempt to read RDA file by converting to CSV format.

        Since RDA files are binary R format, we need to either:
        1. Use rpy2 (if available)
        2. Use pre-converted CSV files
        3. Generate synthetic data based on the structure

        For now, we'll use the structure information to generate
        sample data that matches the format.
        """
        # Extract dataset name
        dataset_name = rda_file.stem

        # For demonstration, create sample data based on known structure
        # In production, you would either:
        # - Use rpy2 to read the RDA file directly
        # - Pre-convert all RDA files to CSV
        # - Use an R script to export all to CSV

        # Check if we have a CSV version
        csv_file = rda_file.with_suffix('.csv')
        if csv_file.exists():
            return pd.read_csv(csv_file)

        # Otherwise, return None and let the caller know
        return None

    @classmethod
    def get_all_csv_converted(cls, force_convert: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get all datasets, converting from RDA to CSV if needed.

        This method uses an R script to convert RDA files to CSV format
        for Python compatibility.

        Args:
            force_convert: Force re-conversion even if CSV exists

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        rda_files = cls._get_dataset_files()

        # First, check if we have a master CSV index
        index_file = cls.DATA_DIR.parent / "data_index.csv"
        if index_file.exists() and not force_convert:
            index_df = pd.read_csv(index_file)
            for _, row in index_df.iterrows():
                csv_path = cls.DATA_DIR / f"{row['dataset_name']}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        datasets[row['dataset_name']] = df
                    except Exception as e:
                        warnings.warn(f"Could not load {csv_path}: {e}")

            return datasets

        # Generate R script to convert all RDA files to CSV
        r_script = f"""
        # Convert Pairwise70 RDA files to CSV format
        library(data.table)

        # Set working directory
        setwd("{cls.DATA_DIR.as_posix()}")

        # Get all RDA files
        rda_files <- list.files(pattern = "\\.rda$")

        # Create index
        index_data <- data.table()

        for (rda_file in rda_files) {{
            try {{
                # Load the RDA file
                load(rda_file)
                # Get the dataset name (variable name in RDA)
                dataset_name <- gsub("\\.rda$", "", rda_file)

                # Get the data object
                data_obj <- get(dataset_name)

                # Convert to data.table if not already
                if (!is.data.table(data_obj)) {{
                    data_obj <- as.data.table(data_obj)
                }}

                # Save as CSV
                csv_file <- gsub("\\.rda$", ".csv", rda_file)
                fwrite(data_obj, csv_file)

                # Add to index
                n_studies <- nrow(data_obj)
                outcome_type <- "unknown"

                # Try to determine outcome type from columns
                if ("Experimental.cases" %in% names(data_obj) &&
                    "Control.cases" %in% names(data_obj)) {{
                    outcome_type <- "binary"
                }} else if ("Experimental.mean" %in% names(data_obj) &&
                          "Control.mean" %in% names(data_obj)) {{
                    outcome_type <- "continuous"
                }}

                review_doi <- ifelse("review_doi" %in% names(data_obj),
                                    as.character(data_obj$review_doi[1]), "NA")

                index_data <- rbind(index_data, data.table(
                    dataset_name = dataset_name,
                    n_studies = n_studies,
                    outcome_type = outcome_type,
                    review_doi = review_doi
                ))

                cat(sprintf("Converted: %s (%d studies, %s)\\n",
                          dataset_name, n_studies, outcome_type))
            }} catch (e) {{
                cat(sprintf("Error converting %s: %s\\n", rda_file, e$message))
            }}
        }}

        # Save index
        fwrite(index_data, "{index_file.as_posix()}")
        cat(sprintf("\\nConversion complete: %d datasets\\n", nrow(index_data))
        """

        # Save R script
        r_script_path = cls.DATA_DIR.parent / "convert_to_csv.R"
        with open(r_script_path, 'w') as f:
            f.write(r_script)

        print(f"\nR script created: {r_script_path}")
        print(f"\nTo convert RDA files to CSV, run in R:")
        print(f'  source("{r_script_path.as_posix()}")')
        print(f"\nOr from command line:")
        print(f'  Rscript "{r_script_path.as_posix()}"')

        return datasets

    @classmethod
    def convert_rda_to_csv(cls) -> bool:
        """
        Convert RDA files to CSV format using R.

        This requires R to be installed with the data.table package.

        Returns:
            True if conversion was successful
        """
        import subprocess
        import sys

        # First, create the conversion script
        cls.get_all_csv_converted(force_convert=True)

        r_script_path = cls.DATA_DIR.parent / "convert_to_csv.R"

        try:
            # Try to run R script
            result = subprocess.run(
                [sys.executable.parent / "Scripts" / "Rscript.exe" if sys.platform == "win32" else "Rscript",
                 str(r_script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            return result.returncode == 0

        except FileNotFoundError:
            print("\nR not found. Please install R or convert files manually.")
            print(f"R script location: {r_script_path}")
            return False
        except Exception as e:
            print(f"Error running R: {e}")
            return False

    @classmethod
    def _load_csv_dataset(cls, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a dataset from CSV file"""
        csv_file = cls.DATA_DIR / f"{dataset_name}.csv"

        if not csv_file.exists():
            return None

        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            warnings.warn(f"Error loading {csv_file}: {e}")
            return None

    @classmethod
    def get_dataset(cls, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific Cochrane dataset by name.

        Args:
            dataset_name: Name of the dataset (e.g., 'CD000028_pub4_data')

        Returns:
            DataFrame with the dataset or None if not found
        """
        # Check cache first
        if dataset_name in cls._cache:
            return cls._cache[dataset_name]

        # Try to load from CSV
        df = cls._load_csv_dataset(dataset_name)

        if df is not None:
            cls._cache[dataset_name] = df
            return df

        return None

    @classmethod
    def list_datasets(cls, limit: Optional[int] = None) -> List[CochraneDatasetInfo]:
        """
        List available Cochrane datasets.

        Args:
            limit: Maximum number of datasets to list

        Returns:
            List of CochraneDatasetInfo objects
        """
        rda_files = cls._get_dataset_files()
        datasets = []

        for rda_file in rda_files[:limit] if limit else rda_files:
            dataset_name = rda_file.stem

            # Try to get info from CSV if available
            csv_file = rda_file.with_suffix('.csv')
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, nrows=1)
                    n_studies = len(pd.read_csv(csv_file))

                    # Try to get metadata
                    review_doi = df.get('review_doi', [''])[0] if 'review_doi' in df.columns else ''
                    review_title = df.get('review_title', [''])[0] if 'review_title' in df.columns else ''

                    # Determine outcome type
                    outcome_type = "unknown"
                    if 'Experimental.cases' in df.columns and 'Control.cases' in df.columns:
                        outcome_type = "binary (OR/RR)"
                    elif 'Experimental.mean' in df.columns and 'Control.mean' in df.columns:
                        outcome_type = "continuous (SMD)"

                    datasets.append(CochraneDatasetInfo(
                        name=dataset_name,
                        review_id=dataset_name.split('_')[0],
                        title=review_title,
                        n_studies=n_studies,
                        outcome_type=outcome_type,
                        review_doi=review_doi,
                        description=f"Cochrane review {dataset_name.split('_')[0]}"
                    ))
                except Exception:
                    # Fallback to basic info
                    datasets.append(CochraneDatasetInfo(
                        name=dataset_name,
                        review_id=dataset_name.split('_')[0],
                        title="",
                        n_studies=0,
                        outcome_type="unknown",
                        review_doi=""
                    ))

        return datasets

    @classmethod
    def get_random_dataset(cls) -> Optional[pd.DataFrame]:
        """Get a random dataset for testing"""
        rda_files = cls._get_dataset_files()
        if not rda_files:
            return None

        import random
        random_file = random.choice(rda_files)
        dataset_name = random_file.stem

        return cls.get_dataset(dataset_name)

    @classmethod
    def calculate_effect_sizes(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate effect sizes and variances from raw data.

        Handles both binary and continuous outcomes.

        Args:
            df: DataFrame with raw study data

        Returns:
            (effect_sizes, variances) tuple
        """
        yi = []
        vi = []

        # Check for binary outcome data
        if 'Experimental.cases' in df.columns and 'Control.cases' in df.columns:
            for _, row in df.iterrows():
                a = row.get('Experimental.cases', 0)  # Events in treatment
                n1 = row.get('Experimental.N', 1)      # Total in treatment
                c = row.get('Control.cases', 0)        # Events in control
                n2 = row.get('Control.N', 1)           # Total in control

                # Avoid division by zero
                if n1 > 0 and n2 > 0:
                    # Calculate log odds ratio
                    if a < n1 and c < n2:  # Avoid 0 or n counts
                        # Add 0.5 for continuity correction
                        lor = np.log(((a + 0.5) * (n2 - c + 0.5)) / ((n1 - a + 0.5) * (c + 0.5)))
                        # Variance of log OR
                        var_lor = 1/(a + 0.5) + 1/(n1 - a + 0.5) + 1/(c + 0.5) + 1/(n2 - c + 0.5)
                        yi.append(lor)
                        vi.append(var_lor)

        # Check for continuous outcome data
        elif 'Experimental.mean' in df.columns and 'Control.mean' in df.columns:
            for _, row in df.iterrows():
                m1 = row.get('Experimental.mean', 0)
                sd1 = row.get('Experimental.SD', 1)
                n1 = row.get('Experimental.N', 1)
                m2 = row.get('Control.mean', 0)
                sd2 = row.get('Control.SD', 1)
                n2 = row.get('Control.N', 1)

                # Calculate standardized mean difference (Hedges' g)
                if n1 > 0 and n2 > 0 and sd1 > 0 and sd2 > 0:
                    # Pooled SD
                    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

                    # Hedges' g (with bias correction)
                    g = (m1 - m2) / pooled_sd

                    # Bias correction for small samples
                    correction = 1 - 3 / (4 * (n1 + n2) - 9)
                    g = g * correction

                    # Variance of g
                    var_g = (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))

                    yi.append(g)
                    vi.append(var_g)

        # If already has effect size columns
        elif 'Mean' in df.columns and ('CI start' in df.columns or 'CI.end' in df.columns):
            # Use existing effect sizes
            for _, row in df.iterrows():
                mean = row.get('Mean', 0)
                ci_start = row.get('CI start', mean - 1)
                ci_end = row.get('CI.end', mean + 1)

                # Estimate SE from CI
                se = (ci_end - ci_start) / 3.92  # For 95% CI

                yi.append(mean)
                vi.append(se**2)

        return np.array(yi), np.array(vi)

    @classmethod
    def get_as_meta_analysis_data(cls, dataset_name: str) -> Optional[MetaAnalysisData]:
        """
        Get a Cochrane dataset as MetaAnalysisData object.

        Args:
            dataset_name: Name of the dataset

        Returns:
            MetaAnalysisData object or None if dataset not available
        """
        df = cls.get_dataset(dataset_name)

        if df is None:
            return None

        yi, vi = cls.calculate_effect_sizes(df)

        if len(yi) == 0:
            return None

        return MetaAnalysisData(
            effect_sizes=yi,
            variances=vi,
            study_names=df.get('Study', list(range(len(yi)))).tolist()
        )

    @classmethod
    def get_sample_datasets(cls, n: int = 10) -> List[str]:
        """
        Get a sample of dataset names for testing.

        Args:
            n: Number of datasets to sample

        Returns:
            List of dataset names
        """
        all_datasets = cls.list_datasets()

        # Prioritize datasets with more studies
        all_datasets.sort(key=lambda x: x.n_studies, reverse=True)

        return [d.name for d in all_datasets[:n]]

    @classmethod
    def run_meta_analysis_on_cochrane(cls, dataset_name: str):
        """
        Run meta-analysis on a Cochrane dataset using all standard methods.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with results from different methods
        """
        from core_framework import (
            DerSimonianLaird, REML, PauleMandel, HartungKnapp,
            KnappHartungModified
        )

        data = cls.get_as_meta_analysis_data(dataset_name)

        if data is None:
            print(f"Could not process dataset: {dataset_name}")
            return None

        results = {}

        methods = [
            ("DerSimonian-Laird", DerSimonianLaird()),
            ("REML", REML()),
            ("Paule-Mandel", PauleMandel()),
            ("Hartung-Knapp", HartungKnapp()),
            ("Knapp-Hartung (no trunc)", KnappHartungModified(truncate=False)),
        ]

        print(f"\n{'='*60}")
        print(f"Cochrane Dataset: {dataset_name}")
        print(f"Studies: {data.n_studies}")
        print(f"{'='*60}")

        for name, method in methods:
            result = method.estimate(data)
            results[name] = result

            print(f"\n{name}:")
            print(f"  Effect: {result.pooled_effect:.4f}")
            print(f"  SE: {result.pooled_se:.4f}")
            print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            print(f"  tau2: {result.tau2:.4f}")
            print(f"  I2: {result.i2:.1f}%")

        return results

    @classmethod
    def generate_summary_statistics(cls) -> Dict:
        """
        Generate summary statistics across all Cochrane datasets.

        Returns:
            Dictionary with summary statistics
        """
        datasets = cls.list_datasets()

        n_studies_list = [d.n_studies for d in datasets if d.n_studies > 0]

        if not n_studies_list:
            return {}

        summary = {
            'total_datasets': len(datasets),
            'datasets_with_data': len(n_studies_list),
            'min_studies': min(n_studies_list),
            'max_studies': max(n_studies_list),
            'median_studies': np.median(n_studies_list),
            'mean_studies': np.mean(n_studies_list),
        }

        # Count by outcome type
        outcome_counts = {}
        for d in datasets:
            outcome_counts[d.outcome_type] = outcome_counts.get(d.outcome_type, 0) + 1

        summary['outcome_types'] = outcome_counts

        return summary


def demo_cochrane_datasets():
    """Demonstrate Cochrane datasets functionality"""
    print("="*60)
    print("Cochrane Datasets - Demo")
    print("="*60)

    # List first 10 datasets
    print("\n1. Listing first 10 datasets:")
    datasets = CochraneDatasets.list_datasets(limit=10)

    for i, ds in enumerate(datasets[:10], 1):
        print(f"\n  {i}. {ds.name}")
        print(f"     Cochrane ID: {ds.review_id}")
        print(f"     Studies: {ds.n_studies}")
        print(f"     Outcome: {ds.outcome_type}")

    # Show summary statistics
    print(f"\n2. Summary Statistics:")
    summary = CochraneDatasets.generate_summary_statistics()

    if summary:
        print(f"   Total datasets: {summary.get('total_datasets', 0)}")
        print(f"   With data: {summary.get('datasets_with_data', 0)}")
        print(f"   Min studies: {summary.get('min_studies', 0)}")
        print(f"   Max studies: {summary.get('max_studies', 0)}")
        print(f"   Median studies: {summary.get('median_studies', 0):.1f}")
        print(f"\n   Outcome types:")
        for outcome, count in summary.get('outcome_types', {}).items():
            print(f"     {outcome}: {count}")

    # Get sample datasets
    print(f"\n3. Sample dataset names for testing:")
    sample_names = CochraneDatasets.get_sample_datasets(n=5)
    for name in sample_names:
        print(f"   - {name}")

    print(f"\n4. To convert RDA files to CSV:")
    print(f"   Run: CochraneDatasets.convert_rda_to_csv()")
    print(f"   Or manually in R: source('{CochraneDatasets.DATA_DIR.parent / 'convert_to_csv.R'}')")


if __name__ == "__main__":
    demo_cochrane_datasets()

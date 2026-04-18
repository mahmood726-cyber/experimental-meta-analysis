"""
Efficiency Benchmarks for Experimental Meta-Analysis
====================================================
Comprehensive performance profiling of all 300+ meta-analysis methods.

Addresses Editorial Review Priority: Computational Efficiency Benchmarks
Provides users with method selection guidance based on computational constraints.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

try:
    from .core_framework import (
        MetaAnalysisData, MetaAnalysisMethod, MetaAnalysisResult,
        DerSimonianLaird, REML, PauleMandel, HartungKnapp,
        get_all_experimental_methods
    )
    from .benchmark_datasets import BENCHMARK_DATASETS
except ImportError:
    from core_framework import (
        MetaAnalysisData, MetaAnalysisMethod, MetaAnalysisResult,
        DerSimonianLaird, REML, PauleMandel, HartungKnapp
    )
    # Try to get experimental methods if available
    try:
        from methods import get_all_experimental_methods
    except ImportError:
        get_all_experimental_methods = None
    from benchmark_datasets import BENCHMARK_DATASETS


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single method"""
    method_name: str
    category: str
    sample_size_key: str  # "k=10", "k=50", "k=100"
    n_studies: int
    n_runs: int

    # Timing statistics (in milliseconds)
    avg_time_ms: float
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    percentile_50_ms: float
    percentile_95_ms: float
    percentile_99_ms: float

    # Memory (optional, if tracked)
    avg_memory_mb: float = 0.0

    # Speed categorization
    speed_category: str = ""  # "FAST", "MEDIUM", "SLOW", "VERY_SLOW"

    # Additional info
    converged: bool = True
    error_message: str = ""


@dataclass
class MethodBenchmarkSummary:
    """Summary across all sample sizes for a method"""
    method_name: str
    category: str
    overall_speed_category: str
    avg_time_all_ms: float
    fastest_sample_size: str
    slowest_sample_size: str
    scalability_ratio: float  # time_k100 / time_k10
    recommendation: str


class EfficiencyBenchmarks:
    """
    Comprehensive benchmarking of all meta-analysis methods.

    Measures:
    - Computation time across different sample sizes
    - Memory usage (if memory_profiler available)
    - Scalability patterns
    - Convergence rates

    Categories methods by speed:
    - FAST: < 1ms per analysis
    - MEDIUM: 1-100ms per analysis
    - SLOW: 100ms-5s per analysis
    - VERY_SLOW: > 5s per analysis
    """

    # Speed thresholds (in milliseconds)
    FAST_THRESHOLD = 1.0
    MEDIUM_THRESHOLD = 100.0
    SLOW_THRESHOLD = 5000.0

    # Default runs per speed category
    DEFAULT_RUNS = {
        "FAST": 1000,      # High precision for fast methods
        "MEDIUM": 100,     # Medium precision
        "SLOW": 10,        # Low precision for slow methods
        "VERY_SLOW": 3,    # Minimal runs for very slow methods
        "UNKNOWN": 50      # Default for unknown methods
    }

    # Timeout for single run (seconds)
    DEFAULT_TIMEOUT = 60.0

    def __init__(self):
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.summaries: Dict[str, MethodBenchmarkSummary] = {}

    def _categorize_speed(self, time_ms: float) -> str:
        """Categorize a method by its computation time"""
        if time_ms < self.FAST_THRESHOLD:
            return "FAST"
        elif time_ms < self.MEDIUM_THRESHOLD:
            return "MEDIUM"
        elif time_ms < self.SLOW_THRESHOLD:
            return "SLOW"
        else:
            return "VERY_SLOW"

    def _run_single_benchmark(
        self,
        method: MetaAnalysisMethod,
        data: MetaAnalysisData,
        timeout: float = DEFAULT_TIMEOUT
    ) -> Tuple[float, bool, str]:
        """
        Run a single benchmark iteration.

        Returns:
            (time_ms, success, error_message)
        """
        start_time = time.time()

        try:
            result = method.estimate(data)

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return elapsed * 1000, False, f"Timeout ({timeout}s exceeded)"

            # Check result validity
            if result is None or not result.converged:
                return elapsed * 1000, False, "Did not converge"

            return elapsed * 1000, True, ""

        except Exception as e:
            elapsed = time.time() - start_time
            return elapsed * 1000, False, str(e)

    def _determine_n_runs(
        self,
        method: MetaAnalysisMethod,
        data: MetaAnalysisData,
        initial_runs: int = 10
    ) -> int:
        """
        Determine appropriate number of runs based on method speed.
        Faster methods get more runs for better precision.
        """
        # Run a few initial iterations to gauge speed
        times = []
        for _ in range(initial_runs):
            time_ms, success, _ = self._run_single_benchmark(method, data, timeout=5.0)
            if success:
                times.append(time_ms)

        if not times:
            return self.DEFAULT_RUNS["UNKNOWN"]

        avg_time = np.mean(times)
        category = self._categorize_speed(avg_time)

        return self.DEFAULT_RUNS.get(category, self.DEFAULT_RUNS["UNKNOWN"])

    def benchmark_method(
        self,
        method: MetaAnalysisMethod,
        data: MetaAnalysisData,
        n_runs: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        sample_size_key: str = "unknown"
    ) -> BenchmarkResult:
        """
        Benchmark a single method on a single dataset.

        Args:
            method: Method to benchmark
            data: Test dataset
            n_runs: Number of runs (auto-determined if None)
            timeout: Maximum time per run
            sample_size_key: Description of sample size (e.g., "k=10")

        Returns:
            BenchmarkResult with timing statistics
        """
        # Determine number of runs if not specified
        if n_runs is None:
            n_runs = self._determine_n_runs(method, data)

        # Run benchmarks
        times = []
        successful_runs = 0
        error_messages = []

        for i in range(n_runs):
            time_ms, success, error = self._run_single_benchmark(method, data, timeout)

            if success:
                times.append(time_ms)
                successful_runs += 1
            else:
                error_messages.append(error)

            # Stop if too many failures
            if successful_runs == 0 and i >= 10:
                break

        # Compute statistics
        if times:
            times_array = np.array(times)
            avg_time = np.mean(times_array)
            median_time = np.median(times_array)
            std_time = np.std(times_array)
            min_time = np.min(times_array)
            max_time = np.max(times_array)
            p50 = np.percentile(times_array, 50)
            p95 = np.percentile(times_array, 95)
            p99 = np.percentile(times_array, 99)
            converged = True
            error_msg = ""
        else:
            avg_time = median_time = std_time = min_time = max_time = 0.0
            p50 = p95 = p99 = 0.0
            converged = False
            error_msg = "; ".join(set(error_messages)) if error_messages else "Unknown error"

        # Categorize speed
        speed_category = self._categorize_speed(avg_time) if avg_time > 0 else "UNKNOWN"

        return BenchmarkResult(
            method_name=method.name,
            category=method.category,
            sample_size_key=sample_size_key,
            n_studies=data.n_studies,
            n_runs=n_runs,
            avg_time_ms=avg_time,
            median_time_ms=median_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            percentile_50_ms=p50,
            percentile_95_ms=p95,
            percentile_99_ms=p99,
            speed_category=speed_category,
            converged=converged,
            error_message=error_msg
        )

    def benchmark_all_methods(
        self,
        methods: Optional[List[MetaAnalysisMethod]] = None,
        sample_sizes: List[str] = None,
        n_runs_per_method: Dict[str, int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        verbose: bool = True
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark all methods across multiple sample sizes.

        Args:
            methods: List of methods to benchmark (default: all available)
            sample_sizes: List of dataset keys from BENCHMARK_DATASETS
            n_runs_per_method: Override runs per method
            timeout: Maximum time per run
            verbose: Print progress

        Returns:
            Dictionary mapping method names to list of BenchmarkResults
        """
        if methods is None:
            methods = self._get_all_available_methods()

        if sample_sizes is None:
            sample_sizes = ["small_k_10", "medium_k_50", "large_k_100"]

        results = {}

        total_methods = len(methods)
        for idx, method in enumerate(methods):
            if verbose:
                print(f"\n[{idx+1}/{total_methods}] Benchmarking: {method.name} ({method.category})")

            method_results = []

            for dataset_key in sample_sizes:
                if dataset_key not in BENCHMARK_DATASETS:
                    print(f"  Warning: Dataset '{dataset_key}' not found, skipping")
                    continue

                data = BENCHMARK_DATASETS[dataset_key]

                # Determine runs
                n_runs = n_runs_per_method.get(method.name) if n_runs_per_method else None

                if verbose:
                    print(f"  Dataset: {dataset_key}...", end="", flush=True)

                result = self.benchmark_method(
                    method, data, n_runs=n_runs,
                    timeout=timeout, sample_size_key=dataset_key
                )

                if verbose:
                    status = "OK" if result.converged else "FAIL"
                    print(f" {result.avg_time_ms:.4f}ms [{result.speed_category}] ({status})")

                method_results.append(result)

            results[method.name] = method_results

        self.results = results
        return results

    def _get_all_available_methods(self) -> List[MetaAnalysisMethod]:
        """Get all available methods from the framework"""
        methods = []

        # Standard methods
        methods.extend([
            DerSimonianLaird(),
            REML(),
            PauleMandel(),
            HartungKnapp(),
        ])

        # Experimental methods (if available)
        if get_all_experimental_methods is not None:
            try:
                experimental = get_all_experimental_methods()
                methods.extend(experimental)
            except Exception as e:
                print(f"Warning: Could not load experimental methods: {e}")

        return methods

    def compute_summaries(self) -> Dict[str, MethodBenchmarkSummary]:
        """
        Compute summary statistics for each method across all sample sizes.
        """
        summaries = {}

        for method_name, results_list in self.results.items():
            if not results_list:
                continue

            # Filter successful results
            valid_results = [r for r in results_list if r.converged]

            if not valid_results:
                continue

            # Overall average time
            avg_all = np.mean([r.avg_time_ms for r in valid_results])

            # Find fastest and slowest sample sizes
            fastest = min(valid_results, key=lambda r: r.avg_time_ms)
            slowest = max(valid_results, key=lambda r: r.avg_time_ms)

            # Scalability ratio (how much slower at large k vs small k)
            small_k = [r for r in valid_results if "k=10" in r.sample_size_key]
            large_k = [r for r in valid_results if "k=100" in r.sample_size_key]

            scalability = 1.0
            if small_k and large_k:
                scalability = large_k[0].avg_time_ms / small_k[0].avg_time_ms

            # Overall speed category
            overall_category = self._categorize_speed(avg_all)

            # Generate recommendation
            recommendation = self._generate_recommendation(valid_results, overall_category)

            summaries[method_name] = MethodBenchmarkSummary(
                method_name=method_name,
                category=valid_results[0].category,
                overall_speed_category=overall_category,
                avg_time_all_ms=avg_all,
                fastest_sample_size=fastest.sample_size_key,
                slowest_sample_size=slowest.sample_size_key,
                scalability_ratio=scalability,
                recommendation=recommendation
            )

        self.summaries = summaries
        return summaries

    def _generate_recommendation(
        self,
        results: List[BenchmarkResult],
        category: str
    ) -> str:
        """Generate usage recommendation based on benchmark results"""
        if category == "FAST":
            return "Suitable for real-time or interactive use"
        elif category == "MEDIUM":
            return "Suitable for batch processing and typical analyses"
        elif category == "SLOW":
            return "Use for final analysis or when time is not critical"
        else:  # VERY_SLOW
            return "Use sparingly; consider alternatives for routine analysis"

    def generate_report(
        self,
        output_format: str = "markdown"
    ) -> str:
        """
        Generate comprehensive efficiency report.

        Args:
            output_format: "markdown", "html", or "json"

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available. Run benchmark_all_methods() first."

        summaries = self.summaries if self.summaries else self.compute_summaries()

        if output_format == "json":
            return self._generate_json_report(summaries)
        elif output_format == "html":
            return self._generate_html_report(summaries)
        else:
            return self._generate_markdown_report(summaries)

    def _generate_markdown_report(self, summaries: Dict[str, MethodBenchmarkSummary]) -> str:
        """Generate Markdown format report"""
        md = """# Computational Efficiency Benchmarks

**Generated:** {timestamp}
**Framework Version:** 1.1.0
**Methods Tested:** {n_methods}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Fastest Method | {fastest_method} ({fastest_time:.4f} ms) |
| Slowest Method | {slowest_method} ({slowest_time:.2f} ms) |
| Median Time | {median_time:.4f} ms |
| Total Methods | {n_methods} |
| Fast Methods (< 1ms) | {n_fast} |
| Medium Methods (1-100ms) | {n_medium} |
| Slow Methods (100ms-5s) | {n_slow} |
| Very Slow Methods (> 5s) | {n_very_slow} |

---

## Speed Category Distribution

{speed_distribution}

---

## By Category

{category_table}

---

## Top 20 Fastest Methods

{fastest_table}

---

## Top 20 Slowest Methods

{slowest_table}

---

## Recommendations by Time Constraint

### "I need results in < 1 second"
{fast_recommendations}

### "I can wait up to 1 minute"
{medium_recommendations}

### "Time is not critical"
{all_recommendations}

---

## Method Details

{method_details}

---

*Report generated by EfficiencyBenchmarks module*
"""

        # Compute statistics
        method_names = list(summaries.keys())
        times = [s.avg_time_all_ms for s in summaries.values()]

        fastest_method = min(summaries.items(), key=lambda x: x[1].avg_time_all_ms)
        slowest_method = max(summaries.items(), key=lambda x: x[1].avg_time_all_ms)
        median_time = np.median(times)

        # Count by speed category
        categories = [s.overall_speed_category for s in summaries.values()]
        n_fast = categories.count("FAST")
        n_medium = categories.count("MEDIUM")
        n_slow = categories.count("SLOW")
        n_very_slow = categories.count("VERY_SLOW")

        # Speed distribution
        speed_dist = f"""
| Speed Category | Count | Percentage |
|---------------|-------|------------|
| FAST (< 1ms) | {n_fast} | {100*n_fast/len(summaries):.1f}% |
| MEDIUM (1-100ms) | {n_medium} | {100*n_medium/len(summaries):.1f}% |
| SLOW (100ms-5s) | {n_slow} | {100*n_slow/len(summaries):.1f}% |
| VERY_SLOW (> 5s) | {n_very_slow} | {100*n_very_slow/len(summaries):.1f}% |
"""

        # Category table
        category_stats = {}
        for s in summaries.values():
            if s.category not in category_stats:
                category_stats[s.category] = []
            category_stats[s.category].append(s.avg_time_all_ms)

        category_table = "| Category | n | Mean Time (ms) | Median Time (ms) |\n"
        category_table += "|---------|---|----------------|------------------|\n"

        for cat, cat_times in sorted(category_stats.items()):
            category_table += f"| {cat} | {len(cat_times)} | {np.mean(cat_times):.4f} | {np.median(cat_times):.4f} |\n"

        # Fastest methods
        sorted_fastest = sorted(summaries.items(), key=lambda x: x[1].avg_time_all_ms)[:20]
        fastest_table = "| Rank | Method | Category | Time (ms) | Scalability |\n"
        fastest_table += "|------|--------|----------|-----------|-------------|\n"

        for rank, (name, s) in enumerate(sorted_fastest, 1):
            fastest_table += f"| {rank} | {name} | {s.category} | {s.avg_time_all_ms:.4f} | {s.scalability_ratio:.2f}x |\n"

        # Slowest methods
        sorted_slowest = sorted(summaries.items(), key=lambda x: x[1].avg_time_all_ms, reverse=True)[:20]
        slowest_table = "| Rank | Method | Category | Time (ms) | Scalability |\n"
        slowest_table += "|------|--------|----------|-----------|-------------|\n"

        for rank, (name, s) in enumerate(sorted_slowest, 1):
            slowest_table += f"| {rank} | {name} | {s.category} | {s.avg_time_all_ms:.2f} | {s.scalability_ratio:.2f}x |\n"

        # Recommendations
        fast_methods = [name for name, s in summaries.items() if s.overall_speed_category == "FAST"]
        fast_rec = ", ".join(fast_methods[:10]) if fast_methods else "None"

        medium_methods = [name for name, s in summaries.items()
                         if s.overall_speed_category in ["FAST", "MEDIUM"]]
        medium_rec = ", ".join(medium_methods[:15]) if medium_methods else "None"

        all_methods = ", ".join(list(summaries.keys())[:20])

        # Method details
        details = ""
        for name, s in sorted(summaries.items(), key=lambda x: x[1].avg_time_all_ms):
            details += f"""
### {name}

| Property | Value |
|----------|-------|
| Category | {s.category} |
| Speed | {s.overall_speed_category} |
| Avg Time | {s.avg_time_all_ms:.4f} ms |
| Fastest on | {s.fastest_sample_size} |
| Slowest on | {s.slowest_sample_size} |
| Scalability | {s.scalability_ratio:.2f}x |
| Recommendation | {s.recommendation} |

"""

        return md.format(
            timestamp=datetime.now().isoformat(),
            n_methods=len(summaries),
            fastest_method=fastest_method[0],
            fastest_time=fastest_method[1].avg_time_all_ms,
            slowest_method=slowest_method[0],
            slowest_time=slowest_method[1].avg_time_all_ms,
            median_time=median_time,
            n_fast=n_fast,
            n_medium=n_medium,
            n_slow=n_slow,
            n_very_slow=n_very_slow,
            speed_distribution=speed_dist,
            category_table=category_table,
            fastest_table=fastest_table,
            slowest_table=slowest_table,
            fast_recommendations=fast_rec,
            medium_recommendations=medium_rec,
            all_recommendations=all_methods,
            method_details=details
        )

    def _generate_html_report(self, summaries: Dict[str, MethodBenchmarkSummary]) -> str:
        """Generate HTML format report"""
        # Simplified HTML version
        md = self._generate_markdown_report(summaries)
        # In production, use a proper markdown to HTML converter
        return f"<html><body><pre>{md}</pre></body></html>"

    def _generate_json_report(self, summaries: Dict[str, MethodBenchmarkSummary]) -> str:
        """Generate JSON format report"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "n_methods": len(summaries),
            "summaries": {
                name: {
                    "category": s.category,
                    "speed_category": s.overall_speed_category,
                    "avg_time_ms": s.avg_time_all_ms,
                    "scalability": s.scalability_ratio,
                    "recommendation": s.recommendation
                }
                for name, s in summaries.items()
            }
        }
        return json.dumps(data, indent=2)

    def save_results(
        self,
        output_dir: str = "efficiency_results",
        formats: List[str] = None
    ):
        """
        Save benchmark results to files.

        Args:
            output_dir: Directory to save results
            formats: List of formats ("csv", "json", "markdown")
        """
        if formats is None:
            formats = ["csv", "json", "markdown"]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save raw results as CSV
        if "csv" in formats:
            rows = []
            for method_name, results_list in self.results.items():
                for r in results_list:
                    rows.append({
                        "method": r.method_name,
                        "category": r.category,
                        "sample_size": r.sample_size_key,
                        "n_studies": r.n_studies,
                        "n_runs": r.n_runs,
                        "avg_time_ms": r.avg_time_ms,
                        "median_time_ms": r.median_time_ms,
                        "std_time_ms": r.std_time_ms,
                        "min_time_ms": r.min_time_ms,
                        "max_time_ms": r.max_time_ms,
                        "p50_ms": r.percentile_50_ms,
                        "p95_ms": r.percentile_95_ms,
                        "p99_ms": r.percentile_99_ms,
                        "speed_category": r.speed_category,
                        "converged": r.converged,
                        "error": r.error_message
                    })

            df = pd.DataFrame(rows)
            csv_path = output_path / f"efficiency_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

        # Save JSON report
        if "json" in formats:
            json_report = self.generate_report(output_format="json")
            json_path = output_path / f"efficiency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                f.write(json_report)
            print(f"Saved JSON: {json_path}")

        # Save Markdown report
        if "markdown" in formats:
            md_report = self.generate_report(output_format="markdown")
            md_path = output_path / f"EFFICIENCY_BENCHMARKS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(md_path, 'w') as f:
                f.write(md_report)
            print(f"Saved Markdown: {md_path}")

    def create_recommendation_engine(self) -> Callable[[str, float], List[str]]:
        """
        Create a function that recommends methods based on time constraints.

        Returns:
            Function that takes (max_time_ms, sample_size) and returns list of suitable methods
        """
        if not self.summaries:
            self.compute_summaries()

        def recommend(max_time_ms: float, sample_size: str = "medium_k_50") -> List[str]:
            """
            Recommend methods that fit within time constraint.

            Args:
                max_time_ms: Maximum acceptable time in milliseconds
                sample_size: Expected sample size ("small_k_10", "medium_k_50", "large_k_100")

            Returns:
                List of method names that fit the constraint
            """
            suitable = []

            for method_name, results_list in self.results.items():
                # Find result for this sample size
                matching = [r for r in results_list if r.sample_size_key == sample_size]

                if not matching:
                    # Use overall average if no specific match
                    summary = self.summaries.get(method_name)
                    if summary and summary.avg_time_all_ms <= max_time_ms:
                        suitable.append((method_name, summary.avg_time_all_ms))
                else:
                    result = matching[0]
                    if result.avg_time_ms <= max_time_ms:
                        suitable.append((method_name, result.avg_time_ms))

            # Sort by time
            suitable.sort(key=lambda x: x[1])
            return [name for name, _ in suitable]

        return recommend


def demo_benchmarks():
    """Demonstrate benchmarking functionality"""
    print("="*60)
    print("Efficiency Benchmarks Demo")
    print("="*60)

    # Create benchmarker
    benchmarker = EfficiencyBenchmarks()

    # Get a small sample of methods for demo
    print("\nTesting standard methods...")
    demo_methods = [
        DerSimonianLaird(),
        REML(),
        PauleMandel(),
        HartungKnapp(),
    ]

    # Run on small dataset only for demo
    results = benchmarker.benchmark_all_methods(
        methods=demo_methods,
        sample_sizes=["small_k_10"],
        verbose=True
    )

    # Generate report
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)

    summaries = benchmarker.compute_summaries()

    print("\nMethod Performance:")
    for name, s in summaries.items():
        print(f"  {name}:")
        print(f"    Category: {s.category}")
        print(f"    Speed: {s.overall_speed_category}")
        print(f"    Avg Time: {s.avg_time_all_ms:.4f} ms")
        print(f"    Scalability: {s.scalability_ratio:.2f}x")

    # Generate recommendation engine
    print("\n" + "="*60)
    print("Recommendation Engine Demo")
    print("="*60)

    recommend = benchmarker.create_recommendation_engine()

    for max_time in [0.1, 1.0, 10.0]:
        methods = recommend(max_time, "small_k_10")
        print(f"\nMethods under {max_time} ms: {', '.join(methods) if methods else 'None'}")

    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    benchmarker.save_results(output_dir="efficiency_results", formats=["markdown"])

    print("\nDemo complete!")


if __name__ == "__main__":
    demo_benchmarks()

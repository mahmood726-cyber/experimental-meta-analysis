"""
Comparison Report Generator for Experimental Meta-Analysis
===========================================================
Generates comprehensive comparison reports for meta-analysis methods.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

try:
    from .core_framework import MetaAnalysisMethod, MetaAnalysisData, MetaAnalysisResult
except ImportError:
    from core_framework import MetaAnalysisMethod, MetaAnalysisData, MetaAnalysisResult


@dataclass
class MethodComparison:
    """Comparison result for a single method"""
    method_name: str
    pooled_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    tau2: float
    i2: float
    computation_time: float


class ComparisonReportGenerator:
    """
    Generates comprehensive comparison reports for meta-analysis methods.

    Creates HTML, Markdown, and JSON reports comparing multiple methods
    on the same dataset.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def compare_methods(
        self,
        data: MetaAnalysisData,
        methods: List[MetaAnalysisMethod],
        dataset_name: str = "Unknown"
    ) -> List[MethodComparison]:
        """
        Compare multiple methods on the same dataset.

        Args:
            data: Meta-analysis data
            methods: List of methods to compare
            dataset_name: Name of the dataset

        Returns:
            List of MethodComparison objects
        """
        comparisons = []

        for method in methods:
            result = method.estimate(data)

            comparisons.append(MethodComparison(
                method_name=method.name,
                pooled_effect=result.pooled_effect,
                se=result.pooled_se,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                tau2=result.tau2,
                i2=result.i2,
                computation_time=result.computation_time
            ))

        return comparisons

    def generate_html_report(
        self,
        comparisons: List[MethodComparison],
        dataset_name: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate an HTML comparison report.

        Args:
            comparisons: List of method comparisons
            dataset_name: Name of the dataset
            output_file: Output file path (optional)

        Returns:
            Path to generated HTML file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"comparison_{timestamp}.html"
        else:
            output_file = Path(output_file)

        # Calculate statistics for comparison
        effects = [c.pooled_effect for c in comparisons]
        tau2s = [c.tau2 for c in comparisons]

        min_effect = min(effects)
        max_effect = max(effects)
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Analysis Comparison Report - {dataset_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .numeric {{
            font-family: 'Courier New', monospace;
        }}
        .ci {{
            color: #666;
            font-size: 0.9em;
        }}
        .effect-bar {{
            height: 20px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            margin-top: 5px;
        }}
        .footer {{
            margin-top: 30px;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .rank {{
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        .rank-1 {{ background: #ffd700; color: #333; }}
        .rank-2 {{ background: #c0c0c0; color: #333; }}
        .rank-3 {{ background: #cd7f32; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Meta-Analysis Comparison Report</h1>
        <p>Dataset: {dataset_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Methods Compared</h3>
            <div class="value">{len(comparisons)}</div>
        </div>
        <div class="summary-card">
            <h3>Effect Range</h3>
            <div class="value">[{min_effect:.3f}, {max_effect:.3f}]</div>
        </div>
        <div class="summary-card">
            <h3>Mean Effect</h3>
            <div class="value">{mean_effect:.3f}</div>
        </div>
        <div class="summary-card">
            <h3>Effect SD</h3>
            <div class="value">{std_effect:.3f}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Method</th>
                <th>Effect Size</th>
                <th>95% CI</th>
                <th>SE</th>
                <th>Heterogeneity (I2)</th>
                <th>tau2</th>
                <th>Time (ms)</th>
            </tr>
        </thead>
        <tbody>
"""

        # Sort by effect size
        sorted_comparisons = sorted(comparisons, key=lambda x: x.pooled_effect)

        for i, comp in enumerate(sorted_comparisons, 1):
            rank_class = f"rank-{i}" if i <= 3 else ""
            ci_width = comp.ci_upper - comp.ci_lower

            # Normalize effect for bar (relative to range)
            if max_effect != min_effect:
                bar_width = ((comp.pooled_effect - min_effect) / (max_effect - min_effect)) * 100
            else:
                bar_width = 50

            html += f"""
            <tr>
                <td><span class="rank {rank_class}">{i}</span></td>
                <td><strong>{comp.method_name}</strong></td>
                <td class="numeric">
                    {comp.pooled_effect:.4f}
                    <div class="effect-bar" style="width: {bar_width}%; margin-left: {(50-bar_width/2) if bar_width < 100 else 0}%"></div>
                </td>
                <td class="numeric ci">[{comp.ci_lower:.4f}, {comp.ci_upper:.4f}]</td>
                <td class="numeric">{comp.se:.4f}</td>
                <td class="numeric">{comp.i2:.1f}%</td>
                <td class="numeric">{comp.tau2:.4f}</td>
                <td class="numeric">{comp.computation_time*1000:.2f}</td>
            </tr>
"""

        html += f"""
        </tbody>
    </table>

    <div class="footer">
        <p>Generated by Experimental Meta-Analysis Framework v1.1.0</p>
        <p>Results should be interpreted in conjunction with methodological expertise and domain knowledge.</p>
    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html)

        return str(output_file)

    def generate_markdown_report(
        self,
        comparisons: List[MethodComparison],
        dataset_name: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a Markdown comparison report.

        Args:
            comparisons: List of method comparisons
            dataset_name: Name of the dataset
            output_file: Output file path (optional)

        Returns:
            Path to generated Markdown file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"comparison_{timestamp}.md"
        else:
            output_file = Path(output_file)

        md = f"""# Meta-Analysis Comparison Report

**Dataset:** {dataset_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Methods Compared:** {len(comparisons)}

---

## Summary

| Metric | Value |
|--------|-------|
| Min Effect | {min(c.pooled_effect for c in comparisons):.4f} |
| Max Effect | {max(c.pooled_effect for c in comparisons):.4f} |
| Mean Effect | {np.mean([c.pooled_effect for c in comparisons]):.4f} |
| Effect SD | {np.std([c.pooled_effect for c in comparisons]):.4f} |

---

## Method Comparison

| Rank | Method | Effect | 95% CI | SE | I2 | tau2 |
|------|--------|--------|--------|----|----|----|
"""

        sorted_comparisons = sorted(comparisons, key=lambda x: x.pooled_effect)

        for i, comp in enumerate(sorted_comparisons, 1):
            md += f"| {i} | {comp.method_name} | {comp.pooled_effect:.4f} | [{comp.ci_lower:.4f}, {comp.ci_upper:.4f}] | {comp.se:.4f} | {comp.i2:.1f}% | {comp.tau2:.4f} |\n"

        md += f"""

---

## Detailed Results

"""

        for comp in sorted_comparisons:
            md += f"""
### {comp.method_name}

- **Pooled Effect:** {comp.pooled_effect:.4f}
- **95% Confidence Interval:** [{comp.ci_lower:.4f}, {comp.ci_upper:.4f}]
- **Standard Error:** {comp.se:.4f}
- **Between-Study Variance (tau2):** {comp.tau2:.4f}
- **Heterogeneity (I2):** {comp.i2:.1f}%
- **Computation Time:** {comp.computation_time*1000:.2f} ms

"""

        md += """
---

*Generated by Experimental Meta-Analysis Framework v1.1.0*
"""

        with open(output_file, 'w') as f:
            f.write(md)

        return str(output_file)

    def generate_json_report(
        self,
        comparisons: List[MethodComparison],
        dataset_name: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a JSON comparison report.

        Args:
            comparisons: List of method comparisons
            dataset_name: Name of the dataset
            output_file: Output file path (optional)

        Returns:
            Path to generated JSON file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"comparison_{timestamp}.json"
        else:
            output_file = Path(output_file)

        data = {
            "dataset_name": dataset_name,
            "generated_at": datetime.now().isoformat(),
            "n_methods": len(comparisons),
            "summary": {
                "min_effect": float(min(c.pooled_effect for c in comparisons)),
                "max_effect": float(max(c.pooled_effect for c in comparisons)),
                "mean_effect": float(np.mean([c.pooled_effect for c in comparisons])),
                "std_effect": float(np.std([c.pooled_effect for c in comparisons])),
            },
            "methods": [
                {
                    "rank": i + 1,
                    "method_name": c.method_name,
                    "pooled_effect": float(c.pooled_effect),
                    "se": float(c.se),
                    "ci_lower": float(c.ci_lower),
                    "ci_upper": float(c.ci_upper),
                    "tau2": float(c.tau2),
                    "i2": float(c.i2),
                    "computation_time": float(c.computation_time),
                }
                for i, c in enumerate(sorted(comparisons, key=lambda x: x.pooled_effect))
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def generate_all_reports(
        self,
        data: MetaAnalysisData,
        methods: List[MetaAnalysisMethod],
        dataset_name: str = "Unknown",
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate all comparison report formats.

        Args:
            data: Meta-analysis data
            methods: List of methods to compare
            dataset_name: Name of the dataset
            formats: List of formats ('html', 'markdown', 'json')

        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = ['html', 'markdown', 'json']

        comparisons = self.compare_methods(data, methods, dataset_name)

        output_files = {}

        if 'html' in formats:
            output_files['html'] = self.generate_html_report(
                comparisons, dataset_name
            )

        if 'markdown' in formats:
            output_files['markdown'] = self.generate_markdown_report(
                comparisons, dataset_name
            )

        if 'json' in formats:
            output_files['json'] = self.generate_json_report(
                comparisons, dataset_name
            )

        return output_files


def demo_comparison_report():
    """Demonstrate the comparison report generator"""
    from core_framework import (
        DerSimonianLaird, REML, PauleMandel, HartungKnapp,
        KnappHartungModified, MetaAnalysisData
    )
    import numpy as np

    print("Comparison Report Generator Demo")
    print("="*60)

    # Create sample data
    np.random.seed(42)
    data = MetaAnalysisData(
        effect_sizes=np.array([0.5, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]),
        variances=np.array([0.1, 0.15, 0.08, 0.12, 0.09, 0.11, 0.13])
    )

    # Methods to compare
    methods = [
        DerSimonianLaird(),
        REML(),
        PauleMandel(),
        HartungKnapp(),
        KnappHartungModified(truncate=False),
    ]

    # Generate reports
    generator = ComparisonReportGenerator()

    print("\nGenerating comparison reports...")
    output_files = generator.generate_all_reports(
        data, methods, dataset_name="Sample Dataset"
    )

    for format_type, file_path in output_files.items():
        print(f"  {format_type.upper()}: {file_path}")

    print("\nDone! Open the HTML file in your browser to view the report.")


if __name__ == "__main__":
    demo_comparison_report()

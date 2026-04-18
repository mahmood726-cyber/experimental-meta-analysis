"""
Visualization Module for Experimental Meta-Analysis
====================================================
Provides plotting capabilities for simulation results and method comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


@dataclass
class MethodSummary:
    """Summary statistics for a method"""
    name: str
    avg_score: float
    win_rate: float
    avg_rmse: float
    avg_coverage: float
    scenarios_tested: int


class MetaAnalysisVisualizer:
    """Visualization tools for meta-analysis simulation results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results = None
        self.method_summaries = []

    def load_results(self, filename: str = None) -> Dict:
        """Load simulation results from JSON file"""
        if filename is None:
            # Find latest summary file
            files = list(self.results_dir.glob("summary_*.json"))
            if not files:
                raise FileNotFoundError(f"No results found in {self.results_dir}")
            filename = max(files, key=lambda f: f.stat().st_mtime).name

        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        return self.results

    def plot_top_methods(self, top_n: int = 20, save_path: str = None) -> None:
        """Plot bar chart of top performing methods"""
        if self.results is None:
            self.load_results()

        methods = self.results.get('superior_methods', [])[:top_n]

        if not methods:
            print("No superior methods found in results")
            return

        names = [m['name'] for m in methods]
        scores = [m['score'] for m in methods]
        win_rates = [m['win_rate'] * 100 for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Score plot
        colors = sns.color_palette("RdYlGn", len(scores))
        ax1.barh(range(len(names)), scores, color=colors)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('Overall Score')
        ax1.set_title(f'Top {top_n} Methods by Score')
        ax1.invert_yaxis()

        # Win rate plot
        ax2.barh(range(len(names)), win_rates, color='steelblue')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel('Win Rate vs DerSimonian-Laird (%)')
        ax2.set_title(f'Top {top_n} Methods by Win Rate')
        ax2.invert_yaxis()
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_rmse_distribution(self, save_path: str = None) -> None:
        """Plot distribution of RMSE across methods"""
        if self.results is None:
            self.load_results()

        # Load detailed results
        detail_files = list(self.results_dir.glob("detailed_*.json"))
        if not detail_files:
            print("No detailed results found")
            return

        filepath = max(detail_files, key=lambda f: f.stat().st_mtime)
        with open(filepath, 'r') as f:
            detailed = json.load(f)

        # Extract RMSE values
        rmse_values = []
        method_names = []
        for method_name, scenarios in detailed.items():
            if scenarios:
                rmses = [s.get('rmse', np.nan) for s in scenarios.values() if s]
                if rmses:
                    rmse_values.append(np.nanmean(rmses))
                    method_names.append(method_name)

        # Sort by RMSE
        sorted_idx = np.argsort(rmse_values)
        rmse_values = np.array(rmse_values)[sorted_idx][:30]
        method_names = np.array(method_names)[sorted_idx][:30]

        plt.figure(figsize=(10, 8))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rmse_values)))
        plt.barh(range(len(method_names)), rmse_values, color=colors)
        plt.yticks(range(len(method_names)), method_names, fontsize=9)
        plt.xlabel('Average RMSE')
        plt.title('RMSE Comparison (Lower is Better)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_scenario_comparison(self, save_path: str = None) -> None:
        """Compare method performance across scenarios"""
        if self.results is None:
            self.load_results()

        # Load detailed results
        detail_files = list(self.results_dir.glob("detailed_*.json"))
        if not detail_files:
            print("No detailed results found")
            return

        filepath = max(detail_files, key=lambda f: f.stat().st_mtime)
        with open(filepath, 'r') as f:
            detailed = json.load(f)

        # Get top 10 methods
        top_methods = self.results.get('superior_methods', [])[:10]
        method_names = [m['name'] for m in top_methods]

        # Get scenario names from first method
        if method_names and method_names[0] in detailed:
            scenarios = list(detailed[method_names[0]].keys())
        else:
            scenarios = []

        # Create heatmap data
        scores = np.zeros((len(method_names), len(scenarios)))
        for i, method in enumerate(method_names):
            if method in detailed:
                for j, scenario in enumerate(scenarios):
                    if scenario in detailed[method]:
                        scores[i, j] = detailed[method][scenario].get('overall_score', 0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(scores, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(method_names)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(method_names, fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Overall Score')

        # Add text annotations
        for i in range(len(method_names)):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{scores[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=7)

        ax.set_title('Method Performance Across Scenarios')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_coverage_comparison(self, save_path: str = None) -> None:
        """Plot coverage rates compared to nominal 95%"""
        if self.results is None:
            self.load_results()

        # Load detailed results
        detail_files = list(self.results_dir.glob("detailed_*.json"))
        if not detail_files:
            print("No detailed results found")
            return

        filepath = max(detail_files, key=lambda f: f.stat().st_mtime)
        with open(filepath, 'r') as f:
            detailed = json.load(f)

        # Extract coverage rates
        coverage_data = []
        method_labels = []
        top_methods = self.results.get('superior_methods', [])[:20]

        for method_info in top_methods:
            method_name = method_info['name']
            if method_name in detailed:
                coverages = []
                for scenario_data in detailed[method_name].values():
                    cov = scenario_data.get('coverage_rate', np.nan)
                    if not np.isnan(cov):
                        coverages.append(cov * 100)
                if coverages:
                    coverage_data.append(coverages)
                    method_labels.append(method_name.replace('_', ' '))

        if not coverage_data:
            print("No coverage data found")
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bp = ax.boxplot(coverage_data, labels=method_labels, patch_artist=True,
                       vert=False, showmeans=True)

        # Color boxes based on median coverage
        for patch, median in zip(bp['boxes'], bp['medians']):
            median_val = median.get_xdata()[0]
            if median_val >= 94 and median_val <= 96:
                patch.set_facecolor('lightgreen')
            elif median_val >= 92:
                patch.set_facecolor('yellow')
            else:
                patch.set_facecolor('lightcoral')

        ax.axvline(x=95, color='red', linestyle='--', linewidth=2, label='Nominal 95%')
        ax.set_xlabel('Coverage Rate (%)')
        ax.set_title('Confidence Interval Coverage Rates')
        ax.legend()
        ax.set_ylim(ax.get_ylim()[::-1])  # Invert y-axis
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive visual report"""
        if output_file is None:
            output_file = self.results_dir / "visualization_report.html"

        # Generate plots
        self.load_results()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experimental Meta-Analysis Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Experimental Meta-Analysis Results</h1>

                <div class="summary">
                    <div class="metric">
                        <div class="metric-value">{self.results.get('total_methods_tested', 0)}</div>
                        <div class="metric-label">Methods Tested</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.results.get('superior_methods_count', 0)}</div>
                        <div class="metric-label">Superior Methods</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.results.get('total_scenarios', 0)}</div>
                        <div class="metric-label">Scenarios</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.results.get('simulations_per_method', 0)}</div>
                        <div class="metric-label">Sims per Method</div>
                    </div>
                </div>

                <h2>Top Performing Methods</h2>
                <p>Methods ranked by overall performance score, comparing against standard DerSimonian-Laird approach.</p>
                <!-- Images would be embedded here -->
            </div>
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        return str(output_file)


def create_summary_plots(results_dir: str = "results"):
    """Create all summary plots from simulation results"""
    viz = MetaAnalysisVisualizer(results_dir)

    print("Creating visualization plots...")

    try:
        viz.load_results()
        viz.plot_top_methods(top_n=20, save_path=f"{results_dir}/top_methods.png")
        print("  - Top methods plot created")
    except Exception as e:
        print(f"  - Error creating top methods plot: {e}")

    try:
        viz.plot_rmse_distribution(save_path=f"{results_dir}/rmse_comparison.png")
        print("  - RMSE comparison plot created")
    except Exception as e:
        print(f"  - Error creating RMSE plot: {e}")

    try:
        viz.plot_scenario_comparison(save_path=f"{results_dir}/scenario_heatmap.png")
        print("  - Scenario heatmap created")
    except Exception as e:
        print(f"  - Error creating scenario heatmap: {e}")

    try:
        viz.plot_coverage_comparison(save_path=f"{results_dir}/coverage_comparison.png")
        print("  - Coverage comparison plot created")
    except Exception as e:
        print(f"  - Error creating coverage plot: {e}")

    print(f"\nPlots saved to {results_dir}/")


if __name__ == "__main__":
    create_summary_plots()

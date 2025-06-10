#!/usr/bin/env python3
"""
Comprehensive Analysis of Persistence Diagrams

Compares persistence diagrams:
1. Epoch 0 vs Epoch 30 for each optimizer (training evolution)
2. Epoch 30 between optimizers (optimizer comparison)
3. Validation vs validation_corrupt (robustness analysis)

Creates detailed visualizations and statistical analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from typing import Dict, List, Tuple
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TDA libraries for distance computation
try:
    import gudhi as gd

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

try:
    from persim import bottleneck, sliced_wasserstein

    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    print("persim not available - using approximate distance metrics")


class PersistenceDiagramAnalyzer:
    """Comprehensive analysis of persistence diagrams."""

    def __init__(
        self, results_file: str = "./results/all_vietoris_rips_persistence.pt"
    ):
        """Initialize analyzer with persistence data."""
        self.results_file = results_file
        self.data = None
        self.load_data()

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def load_data(self):
        """Load persistence diagram data."""
        print(f"Loading persistence data from {self.results_file}")
        try:
            self.data = torch.load(
                self.results_file, map_location="cpu", weights_only=False
            )
            print(f"Loaded data for optimizers: {list(self.data.keys())}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def extract_diagram_statistics(self, diagrams: Dict) -> Dict:
        """Extract statistical features from persistence diagrams."""
        stats = {}

        for dim in ["H0", "H1", "H2"]:
            if dim in diagrams and len(diagrams[dim]) > 0:
                diagram = diagrams[dim]

                # Handle infinite values (persistent features)
                births = diagram[:, 0]
                deaths = diagram[:, 1]

                finite_mask = np.isfinite(deaths)
                finite_deaths = deaths[finite_mask]
                finite_births = births[finite_mask]

                # Count features
                total_features = len(diagram)
                infinite_features = np.sum(~finite_mask)
                finite_features = np.sum(finite_mask)

                # Compute lifetimes for finite features
                if finite_features > 0:
                    lifetimes = finite_deaths - finite_births
                    stats[dim] = {
                        "total_features": total_features,
                        "infinite_features": infinite_features,
                        "finite_features": finite_features,
                        "mean_lifetime": np.mean(lifetimes),
                        "std_lifetime": np.std(lifetimes),
                        "max_lifetime": np.max(lifetimes),
                        "median_lifetime": np.median(lifetimes),
                        "lifetime_entropy": self._compute_lifetime_entropy(lifetimes),
                        "max_birth": np.max(births),
                        "max_death": (
                            np.max(finite_deaths) if len(finite_deaths) > 0 else 0.0
                        ),
                    }
                else:
                    stats[dim] = {
                        "total_features": total_features,
                        "infinite_features": infinite_features,
                        "finite_features": 0,
                        "mean_lifetime": 0.0,
                        "std_lifetime": 0.0,
                        "max_lifetime": 0.0,
                        "median_lifetime": 0.0,
                        "lifetime_entropy": 0.0,
                        "max_birth": np.max(births) if len(births) > 0 else 0.0,
                        "max_death": 0.0,
                    }
            else:
                stats[dim] = {
                    "total_features": 0,
                    "infinite_features": 0,
                    "finite_features": 0,
                    "mean_lifetime": 0.0,
                    "std_lifetime": 0.0,
                    "max_lifetime": 0.0,
                    "median_lifetime": 0.0,
                    "lifetime_entropy": 0.0,
                    "max_birth": 0.0,
                    "max_death": 0.0,
                }

        return stats

    def _compute_lifetime_entropy(self, lifetimes: np.ndarray) -> float:
        """Compute entropy of lifetime distribution."""
        if len(lifetimes) == 0:
            return 0.0

        # Normalize to get probability distribution
        prob_dist = lifetimes / np.sum(lifetimes)

        # Compute entropy
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        return entropy

    def compute_bottleneck_distance(
        self, diag1: np.ndarray, diag2: np.ndarray
    ) -> float:
        """Compute bottleneck distance between two persistence diagrams."""
        if PERSIM_AVAILABLE:
            try:
                return bottleneck(diag1, diag2)
            except:
                pass

        # Fallback: simple approximation
        if len(diag1) == 0 and len(diag2) == 0:
            return 0.0
        elif len(diag1) == 0:
            return np.max(diag2[:, 1] - diag2[:, 0]) / 2.0 if len(diag2) > 0 else 0.0
        elif len(diag2) == 0:
            return np.max(diag1[:, 1] - diag1[:, 0]) / 2.0 if len(diag1) > 0 else 0.0

        # Simple approximation based on feature count difference
        return abs(len(diag1) - len(diag2)) * 0.1

    def aggregate_particle_statistics(
        self, optimizer: str, epoch: int, dataset: str
    ) -> Dict:
        """Aggregate statistics across all particles for a given condition."""
        particles_data = self.data[optimizer]["epochs"][epoch]["particles"]

        all_stats = {"H0": [], "H1": [], "H2": []}

        for particle_id, particle_data in particles_data.items():
            diagrams = particle_data[dataset]["diagrams"]
            stats = self.extract_diagram_statistics(diagrams)

            for dim in ["H0", "H1", "H2"]:
                all_stats[dim].append(stats[dim])

        # Compute aggregated statistics
        aggregated = {}
        for dim in ["H0", "H1", "H2"]:
            if all_stats[dim]:
                # Extract values for each metric
                metrics = [
                    "total_features",
                    "finite_features",
                    "mean_lifetime",
                    "max_lifetime",
                    "lifetime_entropy",
                ]

                dim_stats = {}
                for metric in metrics:
                    values = [s[metric] for s in all_stats[dim]]
                    dim_stats[f"{metric}_mean"] = np.mean(values)
                    dim_stats[f"{metric}_std"] = np.std(values)
                    dim_stats[f"{metric}_median"] = np.median(values)
                    dim_stats[f"{metric}_min"] = np.min(values)
                    dim_stats[f"{metric}_max"] = np.max(values)

                aggregated[dim] = dim_stats

        return aggregated

    def compare_epochs(self) -> Dict:
        """Compare epoch 0 vs epoch 30 for each optimizer."""
        print("Analyzing epoch 0 vs epoch 30 comparison...")

        results = {}

        for optimizer in self.data.keys():
            print(f"  Processing {optimizer}...")

            optimizer_results = {}

            for dataset in ["validation", "validation_corrupt"]:
                epoch_0_stats = self.aggregate_particle_statistics(
                    optimizer, 0, dataset
                )
                epoch_30_stats = self.aggregate_particle_statistics(
                    optimizer, 30, dataset
                )

                # Compute changes
                changes = {}
                for dim in ["H0", "H1", "H2"]:
                    dim_changes = {}
                    for metric in [
                        "total_features",
                        "finite_features",
                        "mean_lifetime",
                        "max_lifetime",
                    ]:
                        epoch_0_val = epoch_0_stats[dim][f"{metric}_mean"]
                        epoch_30_val = epoch_30_stats[dim][f"{metric}_mean"]

                        # Relative change
                        if epoch_0_val != 0:
                            relative_change = (epoch_30_val - epoch_0_val) / epoch_0_val
                        else:
                            relative_change = epoch_30_val

                        dim_changes[metric] = {
                            "epoch_0": epoch_0_val,
                            "epoch_30": epoch_30_val,
                            "absolute_change": epoch_30_val - epoch_0_val,
                            "relative_change": relative_change,
                        }

                    changes[dim] = dim_changes

                optimizer_results[dataset] = {
                    "epoch_0": epoch_0_stats,
                    "epoch_30": epoch_30_stats,
                    "changes": changes,
                }

            results[optimizer] = optimizer_results

        return results

    def compare_optimizers_at_epoch(self, epoch: int) -> Dict:
        """Compare all optimizers at a specific epoch."""
        print(f"Analyzing optimizer comparison at epoch {epoch}...")

        results = {}

        for dataset in ["validation", "validation_corrupt"]:
            optimizer_stats = {}

            for optimizer in self.data.keys():
                stats = self.aggregate_particle_statistics(optimizer, epoch, dataset)
                optimizer_stats[optimizer] = stats

            results[dataset] = optimizer_stats

        return results

    def create_epoch_comparison_plots(self, epoch_results: Dict, output_dir: str):
        """Create plots comparing epochs."""
        print("Creating epoch comparison plots...")

        # Feature count evolution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for dataset_idx, dataset in enumerate(["validation", "validation_corrupt"]):
            for dim_idx, dim in enumerate(["H0", "H1", "H2"]):
                ax = axes[dataset_idx, dim_idx]

                optimizers = list(epoch_results.keys())
                epoch_0_vals = []
                epoch_30_vals = []

                for opt in optimizers:
                    e0_val = epoch_results[opt][dataset]["changes"][dim][
                        "total_features"
                    ]["epoch_0"]
                    e30_val = epoch_results[opt][dataset]["changes"][dim][
                        "total_features"
                    ]["epoch_30"]
                    epoch_0_vals.append(e0_val)
                    epoch_30_vals.append(e30_val)

                x = np.arange(len(optimizers))
                width = 0.35

                ax.bar(x - width / 2, epoch_0_vals, width, label="Epoch 0", alpha=0.8)
                ax.bar(x + width / 2, epoch_30_vals, width, label="Epoch 30", alpha=0.8)

                ax.set_xlabel("Optimizer")
                ax.set_ylabel("Number of Features")
                ax.set_title(f"{dataset.replace('_', ' ').title()} - {dim} Features")
                ax.set_xticks(x)
                ax.set_xticklabels(optimizers, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/epoch_comparison_features.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Lifetime evolution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for dataset_idx, dataset in enumerate(["validation", "validation_corrupt"]):
            for dim_idx, dim in enumerate(["H0", "H1", "H2"]):
                ax = axes[dataset_idx, dim_idx]

                optimizers = list(epoch_results.keys())
                epoch_0_vals = []
                epoch_30_vals = []

                for opt in optimizers:
                    e0_val = epoch_results[opt][dataset]["changes"][dim][
                        "max_lifetime"
                    ]["epoch_0"]
                    e30_val = epoch_results[opt][dataset]["changes"][dim][
                        "max_lifetime"
                    ]["epoch_30"]
                    epoch_0_vals.append(e0_val)
                    epoch_30_vals.append(e30_val)

                x = np.arange(len(optimizers))
                width = 0.35

                ax.bar(x - width / 2, epoch_0_vals, width, label="Epoch 0", alpha=0.8)
                ax.bar(x + width / 2, epoch_30_vals, width, label="Epoch 30", alpha=0.8)

                ax.set_xlabel("Optimizer")
                ax.set_ylabel("Max Lifetime")
                ax.set_title(
                    f"{dataset.replace('_', ' ').title()} - {dim} Max Lifetime"
                )
                ax.set_xticks(x)
                ax.set_xticklabels(optimizers, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/epoch_comparison_lifetimes.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def create_optimizer_comparison_plots(
        self, optimizer_results: Dict, epoch: int, output_dir: str
    ):
        """Create plots comparing optimizers."""
        print(f"Creating optimizer comparison plots for epoch {epoch}...")

        # Feature count comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for dataset_idx, dataset in enumerate(["validation", "validation_corrupt"]):
            for dim_idx, dim in enumerate(["H0", "H1", "H2"]):
                ax = axes[dataset_idx, dim_idx]

                optimizers = list(optimizer_results[dataset].keys())
                values = []

                for opt in optimizers:
                    val = optimizer_results[dataset][opt][dim]["total_features_mean"]
                    values.append(val)

                # Create bar plot with error bars if std is available
                errors = [
                    optimizer_results[dataset][opt][dim]["total_features_std"]
                    for opt in optimizers
                ]

                bars = ax.bar(optimizers, values, yerr=errors, capsize=5, alpha=0.8)

                # Color bars by performance (higher features = different color intensity)
                normalized_vals = (
                    np.array(values) / max(values)
                    if max(values) > 0
                    else np.zeros_like(values)
                )
                for bar, norm_val in zip(bars, normalized_vals):
                    bar.set_color(plt.cm.viridis(norm_val))

                ax.set_xlabel("Optimizer")
                ax.set_ylabel("Mean Number of Features")
                ax.set_title(
                    f"{dataset.replace('_', ' ').title()} - {dim} (Epoch {epoch})"
                )
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/optimizer_comparison_epoch{epoch}_features.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Lifetime comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for dataset_idx, dataset in enumerate(["validation", "validation_corrupt"]):
            for dim_idx, dim in enumerate(["H0", "H1", "H2"]):
                ax = axes[dataset_idx, dim_idx]

                optimizers = list(optimizer_results[dataset].keys())
                values = []

                for opt in optimizers:
                    val = optimizer_results[dataset][opt][dim]["max_lifetime_mean"]
                    values.append(val)

                errors = [
                    optimizer_results[dataset][opt][dim]["max_lifetime_std"]
                    for opt in optimizers
                ]

                bars = ax.bar(optimizers, values, yerr=errors, capsize=5, alpha=0.8)

                # Color bars
                normalized_vals = (
                    np.array(values) / max(values)
                    if max(values) > 0
                    else np.zeros_like(values)
                )
                for bar, norm_val in zip(bars, normalized_vals):
                    bar.set_color(plt.cm.plasma(norm_val))

                ax.set_xlabel("Optimizer")
                ax.set_ylabel("Mean Max Lifetime")
                ax.set_title(
                    f"{dataset.replace('_', ' ').title()} - {dim} (Epoch {epoch})"
                )
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/optimizer_comparison_epoch{epoch}_lifetimes.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_heatmap_analysis(
        self, optimizer_results: Dict, epoch: int, output_dir: str
    ):
        """Create heatmap analysis of optimizer differences."""
        print(f"Creating heatmap analysis for epoch {epoch}...")

        optimizers = list(self.data.keys())

        for dataset in ["validation", "validation_corrupt"]:
            # Create feature count heatmap
            feature_data = {}
            for dim in ["H0", "H1", "H2"]:
                feature_data[dim] = [
                    optimizer_results[dataset][opt][dim]["total_features_mean"]
                    for opt in optimizers
                ]

            df_features = pd.DataFrame(feature_data, index=optimizers)

            plt.figure(figsize=(8, 6))
            sns.heatmap(df_features, annot=True, cmap="viridis", fmt=".1f")
            plt.title(
                f"Feature Counts - {dataset.replace('_', ' ').title()} (Epoch {epoch})"
            )
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/heatmap_features_{dataset}_epoch{epoch}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Create lifetime heatmap
            lifetime_data = {}
            for dim in ["H1", "H2"]:  # Skip H0 as it often has infinite lifetimes
                lifetime_data[dim] = [
                    optimizer_results[dataset][opt][dim]["max_lifetime_mean"]
                    for opt in optimizers
                ]

            df_lifetimes = pd.DataFrame(lifetime_data, index=optimizers)

            plt.figure(figsize=(6, 6))
            sns.heatmap(df_lifetimes, annot=True, cmap="plasma", fmt=".3f")
            plt.title(
                f"Max Lifetimes - {dataset.replace('_', ' ').title()} (Epoch {epoch})"
            )
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/heatmap_lifetimes_{dataset}_epoch{epoch}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def run_full_analysis(self):
        """Run complete analysis and generate all plots."""
        print("=" * 60)
        print("COMPREHENSIVE PERSISTENCE DIAGRAM ANALYSIS")
        print("=" * 60)

        # Create output directory
        output_dir = "./persistence_analysis_plots"
        os.makedirs(output_dir, exist_ok=True)

        # Run analyses
        epoch_comparison = self.compare_epochs()
        optimizer_comparison_epoch30 = self.compare_optimizers_at_epoch(30)
        optimizer_comparison_epoch0 = self.compare_optimizers_at_epoch(0)

        # Create plots
        self.create_epoch_comparison_plots(epoch_comparison, output_dir)
        self.create_optimizer_comparison_plots(
            optimizer_comparison_epoch30, 30, output_dir
        )
        self.create_optimizer_comparison_plots(
            optimizer_comparison_epoch0, 0, output_dir
        )
        self.create_heatmap_analysis(optimizer_comparison_epoch30, 30, output_dir)
        self.create_heatmap_analysis(optimizer_comparison_epoch0, 0, output_dir)

        # Save analysis results
        analysis_results = {
            "epoch_comparison": epoch_comparison,
            "optimizer_comparison_epoch0": optimizer_comparison_epoch0,
            "optimizer_comparison_epoch30": optimizer_comparison_epoch30,
        }

        torch.save(analysis_results, f"{output_dir}/analysis_results.pt")

        print(f"\nAnalysis complete! Results saved to: {output_dir}/")

        # Print summary
        self.print_analysis_summary(epoch_comparison, optimizer_comparison_epoch30)

        return analysis_results

    def print_analysis_summary(
        self, epoch_comparison: Dict, optimizer_comparison: Dict
    ):
        """Print summary of key findings."""
        print("\n" + "=" * 50)
        print("KEY FINDINGS SUMMARY")
        print("=" * 50)

        # Epoch evolution summary
        print("\n1. TRAINING EVOLUTION (Epoch 0 → 30):")
        for optimizer in epoch_comparison.keys():
            print(f"\n  {optimizer}:")
            for dataset in ["validation", "validation_corrupt"]:
                h1_change = epoch_comparison[optimizer][dataset]["changes"]["H1"][
                    "total_features"
                ]["relative_change"]
                h2_change = epoch_comparison[optimizer][dataset]["changes"]["H2"][
                    "total_features"
                ]["relative_change"]
                print(
                    f"    {dataset}: H1 change: {h1_change:+.2%}, H2 change: {h2_change:+.2%}"
                )

        # Optimizer ranking at epoch 30
        print("\n2. OPTIMIZER RANKING (Epoch 30 - Validation):")
        h1_features = [
            (opt, data["H1"]["total_features_mean"])
            for opt, data in optimizer_comparison["validation"].items()
        ]
        h1_features.sort(key=lambda x: x[1], reverse=True)

        for i, (opt, features) in enumerate(h1_features):
            print(f"    {i+1}. {opt}: {features:.1f} H1 features")


def main():
    """Run the comprehensive persistence diagram analysis."""

    # Check if persistence data exists
    results_file = "./results/all_vietoris_rips_persistence.pt"
    if not os.path.exists(results_file):
        print(f"Persistence data not found at {results_file}")
        print("Please run the Vietoris-Rips computation first.")
        return

    # Run analysis
    analyzer = PersistenceDiagramAnalyzer(results_file)
    results = analyzer.run_full_analysis()

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()

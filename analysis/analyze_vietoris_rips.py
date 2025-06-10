import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import permutation_test
import warnings
from typing import Dict, List, Tuple
import os
from collections import defaultdict

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")


class AdvancedPersistenceAnalyzer:
    def __init__(
        self, results_file: str = "../results/all_vietoris_rips_persistence.pt"
    ):
        """Initialize with persistence diagram data."""
        self.results_file = results_file
        self.data = None
        self.load_data()

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
            return

    def compute_betti_curve(
        self, diagrams: Dict, max_epsilon: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Betti curves (β₀, β₁, β₂) for given persistence diagrams."""
        if max_epsilon is None:
            # Find reasonable epsilon range from the data
            all_deaths = []
            for dim in ["H0", "H1", "H2"]:
                if dim in diagrams and len(diagrams[dim]) > 0:
                    deaths = diagrams[dim][:, 1]
                    finite_deaths = deaths[~np.isinf(deaths)]
                    if len(finite_deaths) > 0:
                        all_deaths.extend(finite_deaths)

            if all_deaths:
                max_epsilon = np.percentile(all_deaths, 95)  # Use 95th percentile
            else:
                max_epsilon = 10.0

        # Create epsilon range
        epsilon_range = np.linspace(0, max_epsilon, 1000)

        betti_0 = np.zeros_like(epsilon_range)
        betti_1 = np.zeros_like(epsilon_range)
        betti_2 = np.zeros_like(epsilon_range)

        # Compute Betti numbers for each epsilon
        for i, eps in enumerate(epsilon_range):
            # H0 (connected components)
            if "H0" in diagrams and len(diagrams["H0"]) > 0:
                h0_diag = diagrams["H0"]
                # Count features that are born before eps and die after eps (or are infinite)
                born_before = h0_diag[:, 0] <= eps
                die_after = (h0_diag[:, 1] > eps) | np.isinf(h0_diag[:, 1])
                betti_0[i] = np.sum(born_before & die_after)

            # H1 (loops)
            if "H1" in diagrams and len(diagrams["H1"]) > 0:
                h1_diag = diagrams["H1"]
                born_before = h1_diag[:, 0] <= eps
                die_after = (h1_diag[:, 1] > eps) | np.isinf(h1_diag[:, 1])
                betti_1[i] = np.sum(born_before & die_after)

            # H2 (voids)
            if "H2" in diagrams and len(diagrams["H2"]) > 0:
                h2_diag = diagrams["H2"]
                born_before = h2_diag[:, 0] <= eps
                die_after = (h2_diag[:, 1] > eps) | np.isinf(h2_diag[:, 1])
                betti_2[i] = np.sum(born_before & die_after)

        return epsilon_range, betti_0, betti_1, betti_2

    def plot_betti_curves(self, output_dir: str = "./betti_curves"):
        """Plot Betti curves for all optimizers."""
        os.makedirs(output_dir, exist_ok=True)
        print("Computing and plotting Betti curves...")

        optimizers = list(self.data.keys())
        datasets = ["validation", "validation_corrupt"]

        for dataset in datasets:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            for i, optimizer in enumerate(optimizers):
                ax = axes[i]

                # Get representative particle (particle 0, epoch 30)
                try:
                    sample_data = self.data[optimizer]["epochs"][30]["particles"][0][
                        dataset
                    ]
                    diagrams = sample_data["diagrams"]
                    epsilon_range, betti_0, betti_1, betti_2 = self.compute_betti_curve(
                        diagrams
                    )

                    # Plot Betti curves
                    ax.plot(
                        epsilon_range,
                        betti_0,
                        label="β₀ (components)",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.plot(
                        epsilon_range,
                        betti_1,
                        label="β₁ (loops)",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.plot(
                        epsilon_range,
                        betti_2,
                        label="β₂ (voids)",
                        linewidth=2,
                        alpha=0.8,
                    )

                    ax.set_xlabel("ε (filtration parameter)")
                    ax.set_ylabel("Betti number")
                    ax.set_title(f"{optimizer}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # Add annotations for key transitions
                    if len(epsilon_range) > 0:
                        # Find where β₁ peaks
                        max_b1_idx = np.argmax(betti_1)
                        if betti_1[max_b1_idx] > 0:
                            ax.axvline(
                                epsilon_range[max_b1_idx],
                                color="red",
                                linestyle="--",
                                alpha=0.5,
                            )
                            ax.text(
                                epsilon_range[max_b1_idx],
                                betti_1[max_b1_idx],
                                f"β₁ peak",
                                rotation=90,
                                va="bottom",
                            )

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error: {str(e)[:50]}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax.set_title(f"{optimizer} (Error)")

            # Remove empty subplots
            for i in range(len(optimizers), len(axes)):
                fig.delaxes(axes[i])

            plt.suptitle(
                f"Betti Curves - {dataset.replace('_', ' ').title()}", fontsize=16
            )
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/betti_curves_{dataset}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def compute_persistence_landscape(
        self, diagram: np.ndarray, num_points: int = 1000
    ) -> np.ndarray:
        """Compute persistence landscape for a single persistence diagram."""
        if len(diagram) == 0:
            return np.zeros((1, num_points))

        births = diagram[:, 0]
        deaths = diagram[:, 1]

        # Handle infinite death times
        finite_mask = ~np.isinf(deaths)
        if np.sum(finite_mask) == 0:
            return np.zeros((1, num_points))

        finite_births = births[finite_mask]
        finite_deaths = deaths[finite_mask]

        if len(finite_births) == 0:
            return np.zeros((1, num_points))

        # Create domain
        min_val = np.min(finite_births)
        max_val = np.max(finite_deaths)
        if min_val == max_val:
            return np.zeros((1, num_points))

        domain = np.linspace(min_val, max_val, num_points)

        # Compute landscape functions
        landscapes = []
        for i in range(len(finite_births)):
            b, d = finite_births[i], finite_deaths[i]
            midpoint = (b + d) / 2
            height = (d - b) / 2

            # Tent function: peaks at midpoint with height = persistence/2
            landscape = np.maximum(0, height - np.abs(domain - midpoint))
            landscapes.append(landscape)

        # Sort landscapes by maximum value and stack
        landscapes = np.array(landscapes)
        if len(landscapes) > 0:
            # Sort each column to get landscape levels
            landscapes_sorted = np.sort(landscapes, axis=0)[::-1]  # Descending order
            return landscapes_sorted
        else:
            return np.zeros((1, num_points))

    def landscape_distance(
        self, landscape1: np.ndarray, landscape2: np.ndarray, p: int = 2
    ) -> float:
        """Compute L^p distance between persistence landscapes."""
        # Make landscapes same size
        min_levels = min(landscape1.shape[0], landscape2.shape[0])
        min_points = min(landscape1.shape[1], landscape2.shape[1])

        l1 = landscape1[:min_levels, :min_points]
        l2 = landscape2[:min_levels, :min_points]

        # Compute L^p distance
        diff = l1 - l2
        if p == np.inf:
            return np.max(np.abs(diff))
        else:
            return np.power(np.sum(np.power(np.abs(diff), p)), 1 / p)

    def compute_landscape_analysis(self, output_dir: str = "./landscapes"):
        """Compute persistence landscapes and distance matrices."""
        os.makedirs(output_dir, exist_ok=True)
        print("Computing persistence landscapes...")

        optimizers = list(self.data.keys())
        datasets = ["validation", "validation_corrupt"]

        for dataset in datasets:
            print(f"Processing {dataset}...")

            # Collect H1 landscapes for all optimizers
            landscapes = {}

            for optimizer in optimizers:
                try:
                    # Average across particles
                    all_landscapes = []
                    particles_data = self.data[optimizer]["epochs"][30]["particles"]

                    for particle_id in list(particles_data.keys())[
                        :5
                    ]:  # Use first 5 particles
                        sample_data = particles_data[particle_id][dataset]
                        diagrams = sample_data["diagrams"]
                        if "H1" in diagrams and len(diagrams["H1"]) > 0:
                            landscape = self.compute_persistence_landscape(
                                diagrams["H1"]
                            )
                            all_landscapes.append(landscape)

                    if all_landscapes:
                        # Average landscapes (taking same number of levels and points)
                        min_levels = min(l.shape[0] for l in all_landscapes)
                        min_points = min(l.shape[1] for l in all_landscapes)
                        averaged = np.mean(
                            [l[:min_levels, :min_points] for l in all_landscapes],
                            axis=0,
                        )
                        landscapes[optimizer] = averaged

                except Exception as e:
                    print(f"Error processing {optimizer}: {e}")
                    continue

            # Plot representative landscapes
            if landscapes:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()

                for i, (optimizer, landscape) in enumerate(landscapes.items()):
                    if i < len(axes):
                        ax = axes[i]

                        # Plot first few landscape levels
                        num_levels = min(3, landscape.shape[0])
                        for level in range(num_levels):
                            ax.plot(
                                landscape[level], label=f"Level {level+1}", alpha=0.8
                            )

                        ax.set_xlabel("Domain")
                        ax.set_ylabel("Landscape value")
                        ax.set_title(f"{optimizer}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                # Remove empty subplots
                for i in range(len(landscapes), len(axes)):
                    fig.delaxes(axes[i])

                plt.suptitle(
                    f"Persistence Landscapes (H₁) - {dataset.replace('_', ' ').title()}",
                    fontsize=16,
                )
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/landscapes_{dataset}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # Compute distance matrix
            if len(landscapes) > 1:
                optimizer_names = list(landscapes.keys())
                n = len(optimizer_names)
                distance_matrix = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        if i != j:
                            try:
                                dist = self.landscape_distance(
                                    landscapes[optimizer_names[i]],
                                    landscapes[optimizer_names[j]],
                                )
                                distance_matrix[i, j] = dist
                            except:
                                distance_matrix[i, j] = np.nan

                # Plot distance matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    distance_matrix,
                    xticklabels=optimizer_names,
                    yticklabels=optimizer_names,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                )
                plt.title(
                    f"Landscape Distance Matrix - {dataset.replace('_', ' ').title()}"
                )
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/distance_matrix_{dataset}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def statistical_validation(self, output_dir: str = "./statistical_tests"):
        """Perform statistical tests on persistence features."""
        os.makedirs(output_dir, exist_ok=True)
        print("Performing statistical validation...")

        optimizers = list(self.data.keys())
        datasets = ["validation", "validation_corrupt"]

        results = {}

        for dataset in datasets:
            print(f"Testing {dataset}...")
            results[dataset] = {}

            # Collect H1 feature counts for all optimizers
            feature_counts = {}
            lifetimes = {}

            for optimizer in optimizers:
                counts = []
                lifetimes_list = []

                try:
                    particles_data = self.data[optimizer]["epochs"][30]["particles"]

                    for particle_id in particles_data.keys():
                        sample_data = particles_data[particle_id][dataset]
                        diagrams = sample_data["diagrams"]
                        if "H1" in diagrams and len(diagrams["H1"]) > 0:
                            # Count finite features
                            deaths = diagrams["H1"][:, 1]
                            finite_features = np.sum(~np.isinf(deaths))
                            counts.append(finite_features)

                            # Collect finite lifetimes
                            births = diagrams["H1"][:, 0]
                            finite_mask = ~np.isinf(deaths)
                            if np.sum(finite_mask) > 0:
                                finite_lifetimes = (
                                    deaths[finite_mask] - births[finite_mask]
                                )
                                lifetimes_list.extend(finite_lifetimes)

                    feature_counts[optimizer] = np.array(counts)
                    lifetimes[optimizer] = np.array(lifetimes_list)

                except Exception as e:
                    print(f"Error processing {optimizer}: {e}")
                    continue

            # Pairwise comparisons - feature counts
            print(f"  Feature count comparisons:")
            feature_comparisons = {}

            optimizer_pairs = [
                (optimizers[i], optimizers[j])
                for i in range(len(optimizers))
                for j in range(i + 1, len(optimizers))
            ]

            for opt1, opt2 in optimizer_pairs:
                if (
                    opt1 in feature_counts
                    and opt2 in feature_counts
                    and len(feature_counts[opt1]) > 0
                    and len(feature_counts[opt2]) > 0
                ):
                    try:
                        # Two-sample test
                        result = stats.mannwhitneyu(
                            feature_counts[opt1],
                            feature_counts[opt2],
                            alternative="two-sided",
                        )

                        feature_comparisons[f"{opt1}_vs_{opt2}"] = {
                            "mean_diff": np.mean(feature_counts[opt1])
                            - np.mean(feature_counts[opt2]),
                            "p_value": result.pvalue,
                            "statistic": result.statistic,
                            "significant": result.pvalue < 0.05,
                        }

                        print(
                            f"    {opt1} vs {opt2}: p={result.pvalue:.4f}, diff={np.mean(feature_counts[opt1]) - np.mean(feature_counts[opt2]):.2f}"
                        )

                    except Exception as e:
                        print(f"    Error comparing {opt1} vs {opt2}: {e}")

            results[dataset]["feature_comparisons"] = feature_comparisons

            # Bootstrap confidence intervals
            print(f"  Bootstrap confidence intervals:")
            bootstrap_results = {}

            for optimizer in feature_counts.keys():
                if len(feature_counts[optimizer]) > 0:
                    # Bootstrap mean
                    bootstrap_means = []
                    n_bootstrap = 1000

                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(
                            feature_counts[optimizer],
                            size=len(feature_counts[optimizer]),
                            replace=True,
                        )
                        bootstrap_means.append(np.mean(bootstrap_sample))

                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)

                    bootstrap_results[optimizer] = {
                        "mean": np.mean(feature_counts[optimizer]),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "ci_width": ci_upper - ci_lower,
                    }

                    print(
                        f"    {optimizer}: {np.mean(feature_counts[optimizer]):.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                    )

            results[dataset]["bootstrap_results"] = bootstrap_results

        # Save results
        torch.save(results, f"{output_dir}/statistical_results.pt")

        # Create summary plot
        self.plot_statistical_summary(results, output_dir)

        return results

    def plot_statistical_summary(self, results: Dict, output_dir: str):
        """Plot statistical test summary."""
        datasets = ["validation", "validation_corrupt"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for i, dataset in enumerate(datasets):
            ax = axes[i]

            if "bootstrap_results" in results[dataset]:
                bootstrap_data = results[dataset]["bootstrap_results"]

                optimizers = list(bootstrap_data.keys())
                means = [bootstrap_data[opt]["mean"] for opt in optimizers]
                ci_lowers = [bootstrap_data[opt]["ci_lower"] for opt in optimizers]
                ci_uppers = [bootstrap_data[opt]["ci_upper"] for opt in optimizers]

                # Error bars plot
                y_pos = np.arange(len(optimizers))
                errors = [
                    np.array(means) - np.array(ci_lowers),
                    np.array(ci_uppers) - np.array(means),
                ]

                ax.errorbar(means, y_pos, xerr=errors, fmt="o", capsize=5, capthick=2)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(optimizers)
                ax.set_xlabel("Mean H₁ Feature Count")
                ax.set_title(f'{dataset.replace("_", " ").title()}\n95% Bootstrap CI')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/statistical_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def run_complete_analysis(self, output_dir: str = "./advanced_tda_analysis"):
        """Run all advanced TDA analyses."""
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("ADVANCED TOPOLOGICAL DATA ANALYSIS")
        print("=" * 60)

        # 1. Betti curves
        self.plot_betti_curves(f"{output_dir}/betti_curves")

        # 2. Persistence landscapes
        self.compute_landscape_analysis(f"{output_dir}/landscapes")

        # 3. Statistical validation
        stats_results = self.statistical_validation(f"{output_dir}/statistical_tests")

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        return stats_results


if __name__ == "__main__":
    analyzer = AdvancedPersistenceAnalyzer()
    results = analyzer.run_complete_analysis()

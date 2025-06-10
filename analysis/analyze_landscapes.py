import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")


class LandscapeAdvancedAnalyzer:
    def __init__(
        self, results_file: str = "../results/all_vietoris_rips_persistence.pt"
    ):
        """Initialize with persistence diagram data."""
        self.results_file = results_file
        self.data = None
        self.landscapes = {}
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

    def compute_persistence_landscape(
        self, diagram: np.ndarray, num_points: int = 500
    ) -> np.ndarray:
        """Compute persistence landscape for a single persistence diagram."""
        if len(diagram) == 0:
            return np.zeros((1, num_points))

        births = diagram[:, 0]
        deaths = diagram[:, 1]

        finite_mask = ~np.isinf(deaths)
        if np.sum(finite_mask) == 0:
            return np.zeros((1, num_points))

        finite_births = births[finite_mask]
        finite_deaths = deaths[finite_mask]

        if len(finite_births) == 0:
            return np.zeros((1, num_points))

        min_val = np.min(finite_births)
        max_val = np.max(finite_deaths)
        if min_val == max_val:
            return np.zeros((1, num_points))

        domain = np.linspace(min_val, max_val, num_points)

        landscapes = []
        for i in range(len(finite_births)):
            b, d = finite_births[i], finite_deaths[i]
            midpoint = (b + d) / 2
            height = (d - b) / 2

            landscape = np.maximum(0, height - np.abs(domain - midpoint))
            landscapes.append(landscape)

        landscapes = np.array(landscapes)
        if len(landscapes) > 0:
            landscapes_sorted = np.sort(landscapes, axis=0)[::-1]
            return landscapes_sorted
        else:
            return np.zeros((1, num_points))

    def compute_all_landscapes(self, max_particles: int = 10):
        """Compute landscapes for all optimizers, epochs, and particles."""
        print("Computing comprehensive landscape database...")

        optimizers = list(self.data.keys())
        epochs = [0, 30]
        datasets = ["validation", "validation_corrupt"]

        self.landscapes = {}

        for optimizer in optimizers:
            self.landscapes[optimizer] = {}
            for epoch in epochs:
                self.landscapes[optimizer][epoch] = {}
                for dataset in datasets:
                    landscapes_list = []

                    try:
                        particles_data = self.data[optimizer]["epochs"][epoch][
                            "particles"
                        ]
                        particle_ids = list(particles_data.keys())[:max_particles]

                        for particle_id in particle_ids:
                            sample_data = particles_data[particle_id][dataset]
                            diagrams = sample_data["diagrams"]

                            if "H1" in diagrams and len(diagrams["H1"]) > 0:
                                landscape = self.compute_persistence_landscape(
                                    diagrams["H1"]
                                )
                                landscapes_list.append(landscape)

                        self.landscapes[optimizer][epoch][dataset] = landscapes_list

                    except Exception as e:
                        print(
                            f"Error processing {optimizer}, epoch {epoch}, {dataset}: {e}"
                        )
                        self.landscapes[optimizer][epoch][dataset] = []

        print(f"Computed landscapes for {len(optimizers)} optimizers")

    def landscape_time_series_analysis(
        self, output_dir: str = "./landscape_timeseries"
    ):
        """Analyze how landscapes evolve from epoch 0 to 30."""
        os.makedirs(output_dir, exist_ok=True)
        print("Performing landscape time series analysis...")

        for dataset in ["validation", "validation_corrupt"]:
            print(f"Time series for {dataset}...")

            # Compute landscape distances between epochs
            optimizer_distances = {}

            for optimizer in self.landscapes.keys():
                landscapes_0 = self.landscapes[optimizer][0][dataset]
                landscapes_30 = self.landscapes[optimizer][30][dataset]

                distances = []

                # Compare corresponding particles
                for i in range(min(len(landscapes_0), len(landscapes_30))):
                    if len(landscapes_0) > i and len(landscapes_30) > i:
                        l0 = landscapes_0[i]
                        l30 = landscapes_30[i]

                        if l0.size > 0 and l30.size > 0:
                            # Compute landscape distance
                            min_levels = min(l0.shape[0], l30.shape[0])
                            min_points = min(l0.shape[1], l30.shape[1])

                            diff = (
                                l0[:min_levels, :min_points]
                                - l30[:min_levels, :min_points]
                            )
                            distance = np.linalg.norm(diff)
                            distances.append(distance)

                optimizer_distances[optimizer] = distances

            # Plot distance evolution
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Box plot of distances
            valid_optimizers = [
                opt for opt, dist in optimizer_distances.items() if len(dist) > 0
            ]
            box_data = [optimizer_distances[opt] for opt in valid_optimizers]

            if box_data:
                bp = axes[0].boxplot(
                    box_data, labels=[opt[:6] for opt in valid_optimizers]
                )
                axes[0].set_xlabel("Optimizer")
                axes[0].set_ylabel("Landscape Distance (Epoch 0 → 30)")
                axes[0].set_title(f"Landscape Evolution Distance\n{dataset}")
                axes[0].tick_params(axis="x", rotation=45)
                axes[0].grid(True, alpha=0.3)

            # Scatter plot of distances with error bars
            x_pos = 0
            for opt, distances in optimizer_distances.items():
                if len(distances) > 0:
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    axes[1].scatter([mean_dist], [x_pos], s=100, alpha=0.8, label=opt)
                    axes[1].errorbar(
                        [mean_dist], [x_pos], xerr=[std_dist], capsize=5, alpha=0.8
                    )
                    x_pos += 1

            axes[1].set_xlabel("Mean Landscape Distance")
            axes[1].set_ylabel("Optimizer")
            axes[1].set_yticks(
                range(
                    len(
                        [
                            opt
                            for opt in optimizer_distances.keys()
                            if len(optimizer_distances[opt]) > 0
                        ]
                    )
                )
            )
            axes[1].set_yticklabels(
                [
                    opt
                    for opt in optimizer_distances.keys()
                    if len(optimizer_distances[opt]) > 0
                ]
            )
            axes[1].set_title(f"Mean Evolution Distance\n{dataset}")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/timeseries_{dataset}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def landscape_stability_analysis(self, output_dir: str = "./landscape_stability"):
        """Analyze stability of landscapes across particles."""
        os.makedirs(output_dir, exist_ok=True)
        print("Performing landscape stability analysis...")

        for epoch in [0, 30]:
            for dataset in ["validation", "validation_corrupt"]:
                print(f"Stability for epoch {epoch}, {dataset}...")

                stability_scores = {}

                for optimizer in self.landscapes.keys():
                    landscapes_list = self.landscapes[optimizer][epoch][dataset]

                    if len(landscapes_list) < 2:
                        continue

                    # Compute pairwise distances between landscapes
                    distances = []
                    for i in range(len(landscapes_list)):
                        for j in range(i + 1, len(landscapes_list)):
                            l1 = landscapes_list[i]
                            l2 = landscapes_list[j]

                            if l1.size > 0 and l2.size > 0:
                                min_levels = min(l1.shape[0], l2.shape[0])
                                min_points = min(l1.shape[1], l2.shape[1])

                                diff = (
                                    l1[:min_levels, :min_points]
                                    - l2[:min_levels, :min_points]
                                )
                                distance = np.linalg.norm(diff)
                                distances.append(distance)

                    # Stability is inverse of mean pairwise distance
                    if distances:
                        stability_scores[optimizer] = {
                            "mean_distance": np.mean(distances),
                            "std_distance": np.std(distances),
                            "stability": 1.0 / (np.mean(distances) + 1e-6),
                        }

                # Plot stability scores
                if stability_scores:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    optimizers = list(stability_scores.keys())
                    stabilities = [
                        stability_scores[opt]["stability"] for opt in optimizers
                    ]
                    mean_dists = [
                        stability_scores[opt]["mean_distance"] for opt in optimizers
                    ]
                    std_dists = [
                        stability_scores[opt]["std_distance"] for opt in optimizers
                    ]

                    # Stability scores
                    axes[0].bar(optimizers, stabilities)
                    axes[0].set_xlabel("Optimizer")
                    axes[0].set_ylabel("Stability Score")
                    axes[0].set_title(f"Landscape Stability\nEpoch {epoch}, {dataset}")
                    axes[0].tick_params(axis="x", rotation=45)
                    axes[0].grid(True, alpha=0.3)

                    # Mean distances with error bars
                    axes[1].bar(optimizers, mean_dists, yerr=std_dists, capsize=5)
                    axes[1].set_xlabel("Optimizer")
                    axes[1].set_ylabel("Mean Pairwise Distance")
                    axes[1].set_title(
                        f"Inter-particle Landscape Distance\nEpoch {epoch}, {dataset}"
                    )
                    axes[1].tick_params(axis="x", rotation=45)
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(
                        f"{output_dir}/stability_epoch{epoch}_{dataset}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

    def run_complete_landscape_analysis(
        self, output_dir: str = "../figures/activation_persistent_analysis"
    ):
        """Run stability and timeseries landscape analyses only."""
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("FOCUSED LANDSCAPE ANALYSIS - STABILITY & TIMESERIES")
        print("=" * 60)

        # Compute all landscapes first
        self.compute_all_landscapes()

        # Run analyses
        print("\n1. TIME SERIES ANALYSIS...")
        self.landscape_time_series_analysis(f"{output_dir}/timeseries")

        print("\n2. STABILITY ANALYSIS...")
        self.landscape_stability_analysis(f"{output_dir}/stability")

        print(f"\nFocused landscape analysis saved to: {output_dir}")


if __name__ == "__main__":
    analyzer = LandscapeAdvancedAnalyzer()
    analyzer.run_complete_landscape_analysis()

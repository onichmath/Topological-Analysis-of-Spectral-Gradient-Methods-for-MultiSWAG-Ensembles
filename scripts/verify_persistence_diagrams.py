#!/usr/bin/env python3
"""
Verification script for persistence diagrams.
Loads computed persistence diagrams and creates visualizations to verify correctness.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TDA visualization libraries
try:
    import persim
    from persim import plot_diagrams

    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    print("persim not available, using matplotlib only")

try:
    import gudhi as gd

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def load_persistence_data(results_dir: str = "./results"):
    """Load persistence diagram data."""

    # Try to load combined results first
    combined_file = os.path.join(results_dir, "all_vietoris_rips_persistence.pt")
    if os.path.exists(combined_file):
        print(f"Loading combined results from {combined_file}")
        return torch.load(combined_file, map_location="cpu", weights_only=False)

    # Otherwise, look for individual optimizer files
    # First check what optimizers actually have persistence data
    available_optimizers = []
    for item in os.listdir(results_dir):
        persistence_dir = os.path.join(results_dir, item, "vietoris_rips_persistence")
        if os.path.isdir(persistence_dir):
            persistence_file = os.path.join(persistence_dir, "persistence_diagrams.pt")
            if os.path.exists(persistence_file):
                available_optimizers.append(item)

    print(f"Found persistence data for optimizers: {available_optimizers}")

    if not available_optimizers:
        return {}

    # Load the first available optimizer
    for optimizer in available_optimizers:
        individual_file = os.path.join(
            results_dir,
            optimizer,
            "vietoris_rips_persistence",
            "persistence_diagrams.pt",
        )
        if os.path.exists(individual_file):
            print(f"Loading individual results from {individual_file}")
            data = torch.load(individual_file, map_location="cpu", weights_only=False)
            return {optimizer: data}

    raise FileNotFoundError("No persistence diagram files found!")


def plot_persistence_diagram_matplotlib(
    diagrams: Dict, title: str = "Persistence Diagram"
):
    """Plot persistence diagram using matplotlib."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["red", "blue", "green"]

    for dim in range(3):  # H0, H1, H2
        ax = axes[dim]
        diagram_key = f"H{dim}"

        if diagram_key in diagrams and len(diagrams[diagram_key]) > 0:
            diagram = diagrams[diagram_key]
            births = diagram[:, 0]
            deaths = diagram[:, 1]

            # Plot points
            ax.scatter(births, deaths, c=colors[dim], alpha=0.7, s=30)

            # Plot diagonal
            max_val = max(np.max(births), np.max(deaths)) if len(births) > 0 else 1
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.set_title(f"{title} - H{dim} ({len(diagram)} features)")
            ax.grid(True, alpha=0.3)

            # Add statistics
            if len(diagram) > 0:
                lifetimes = deaths - births
                ax.text(
                    0.05,
                    0.95,
                    f"Max lifetime: {np.max(lifetimes):.3f}\nMean lifetime: {np.mean(lifetimes):.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No features",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_title(f"{title} - H{dim} (0 features)")

    plt.tight_layout()
    return fig


def plot_persistence_diagram_persim(diagrams: Dict, title: str = "Persistence Diagram"):
    """Plot persistence diagram using persim if available."""

    if not PERSIM_AVAILABLE:
        return plot_persistence_diagram_matplotlib(diagrams, title)

    # Convert to persim format
    dgms = []
    for dim in range(3):
        diagram_key = f"H{dim}"
        if diagram_key in diagrams and len(diagrams[diagram_key]) > 0:
            dgms.append(diagrams[diagram_key])
        else:
            dgms.append(np.array([]).reshape(0, 2))

    # Plot using persim
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_diagrams(dgms, ax=ax, title=title, legend=True)

    return fig


def compute_basic_statistics(diagrams: Dict) -> Dict:
    """Compute basic statistics for persistence diagrams."""

    stats = {}

    for dim in range(3):
        diagram_key = f"H{dim}"
        if diagram_key in diagrams and len(diagrams[diagram_key]) > 0:
            diagram = diagrams[diagram_key]
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            lifetimes = deaths - births

            stats[f"H{dim}"] = {
                "num_features": len(diagram),
                "max_lifetime": float(np.max(lifetimes)),
                "mean_lifetime": float(np.mean(lifetimes)),
                "std_lifetime": float(np.std(lifetimes)),
                "max_birth": float(np.max(births)),
                "max_death": float(np.max(deaths)),
            }
        else:
            stats[f"H{dim}"] = {
                "num_features": 0,
                "max_lifetime": 0.0,
                "mean_lifetime": 0.0,
                "std_lifetime": 0.0,
                "max_birth": 0.0,
                "max_death": 0.0,
            }

    return stats


def verify_single_particle(
    optimizer: str, epoch: int, particle_id: int, dataset: str, data: Dict
) -> Dict:
    """Verify persistence diagram for a single particle."""

    try:
        particle_data = data[optimizer]["epochs"][epoch]["particles"][particle_id][
            dataset
        ]

        print(
            f"\n=== {optimizer} - Epoch {epoch} - Particle {particle_id} - {dataset} ==="
        )
        print(f"Method: {particle_data['method']}")
        print(f"Radius: {particle_data['radius']:.6f}")
        print(f"Original shape: {particle_data['original_shape']}")
        print(f"Processed shape: {particle_data['processed_shape']}")

        if "n_simplices" in particle_data and particle_data["n_simplices"] is not None:
            print(f"Number of simplices: {particle_data['n_simplices']}")

        diagrams = particle_data["diagrams"]
        stats = compute_basic_statistics(diagrams)

        print("\nPersistence Statistics:")
        for dim, dim_stats in stats.items():
            print(
                f"  {dim}: {dim_stats['num_features']} features, "
                f"max lifetime: {dim_stats['max_lifetime']:.6f}"
            )

        return {"diagrams": diagrams, "stats": stats, "metadata": particle_data}

    except KeyError as e:
        print(f"Error accessing data: {e}")
        return None


def main():
    """Main verification function."""

    print("=" * 60)
    print("PERSISTENCE DIAGRAM VERIFICATION")
    print("=" * 60)

    # Load data
    try:
        data = load_persistence_data()
        print(f"Loaded data for optimizers: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Pick first available optimizer and some examples
    optimizer = list(data.keys())[0]
    optimizer_data = data[optimizer]

    available_epochs = list(optimizer_data["epochs"].keys())
    print(f"Available epochs for {optimizer}: {available_epochs}")

    if not available_epochs:
        print("No epoch data found!")
        return

    # Pick first epoch
    epoch = available_epochs[0]
    epoch_data = optimizer_data["epochs"][epoch]

    available_particles = list(epoch_data["particles"].keys())
    print(f"Available particles for epoch {epoch}: {len(available_particles)}")

    if not available_particles:
        print("No particle data found!")
        return

    # Verify a few examples
    examples_to_check = [
        (optimizer, epoch, available_particles[0], "validation"),
        (optimizer, epoch, available_particles[0], "validation_corrupt"),
    ]

    if len(available_particles) > 1:
        examples_to_check.append(
            (optimizer, epoch, available_particles[1], "validation")
        )

    # Create output directory
    output_dir = "./verification_plots"
    os.makedirs(output_dir, exist_ok=True)

    for i, (opt, ep, particle, dataset) in enumerate(examples_to_check):
        result = verify_single_particle(opt, ep, particle, dataset, data)

        if result is None:
            continue

        # Create visualizations
        print(f"\nCreating visualization {i+1}...")

        # Matplotlib version
        fig1 = plot_persistence_diagram_matplotlib(
            result["diagrams"],
            title=f"{opt} - Epoch {ep} - Particle {particle} - {dataset}",
        )
        fig1.savefig(
            f"{output_dir}/persistence_{opt}_{ep}_{particle}_{dataset}_matplotlib.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig1)

        # Persim version if available
        if PERSIM_AVAILABLE:
            fig2 = plot_persistence_diagram_persim(
                result["diagrams"],
                title=f"{opt} - Epoch {ep} - Particle {particle} - {dataset}",
            )
            fig2.savefig(
                f"{output_dir}/persistence_{opt}_{ep}_{particle}_{dataset}_persim.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig2)

    # Summary statistics across all particles
    print(f"\n=== SUMMARY STATISTICS FOR {optimizer} ===")

    all_stats = {"H0": [], "H1": [], "H2": []}

    for ep in available_epochs:
        for particle_id in optimizer_data["epochs"][ep]["particles"].keys():
            for dataset in ["validation", "validation_corrupt"]:
                try:
                    diagrams = optimizer_data["epochs"][ep]["particles"][particle_id][
                        dataset
                    ]["diagrams"]
                    stats = compute_basic_statistics(diagrams)

                    for dim in ["H0", "H1", "H2"]:
                        all_stats[dim].append(stats[dim])

                except Exception as e:
                    continue

    # Aggregate statistics
    print("\nAggregate Statistics across all particles:")
    for dim in ["H0", "H1", "H2"]:
        if all_stats[dim]:
            num_features = [s["num_features"] for s in all_stats[dim]]
            max_lifetimes = [
                s["max_lifetime"] for s in all_stats[dim] if s["num_features"] > 0
            ]

            print(f"  {dim}:")
            print(f"    Mean features per particle: {np.mean(num_features):.2f}")
            print(
                f"    Features range: {np.min(num_features)} - {np.max(num_features)}"
            )
            if max_lifetimes:
                print(
                    f"    Lifetime range: {np.min(max_lifetimes):.6f} - {np.max(max_lifetimes):.6f}"
                )

    print(f"\nVisualization plots saved to: {output_dir}/")
    print(
        "\nVerification complete! Check the plots to ensure diagrams look reasonable."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to compute Vietoris-Rips filtrations for penultimate layer activations.
This script processes the activation data and computes persistence diagrams
for both validation and validation corrupt datasets.
"""

import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.vietoris_rips_computer import compute_vietoris_rips_filtrations


def main():
    """Compute Vietoris-Rips filtrations for all optimizers."""

    print("=" * 60)
    print("VIETORIS-RIPS FILTRATION COMPUTATION")
    print("=" * 60)

    # Configuration
    config = {
        "results_dir": "./results",
        "optimizers": None,  # None = all optimizers with activation data
        "max_dimension": 2,  # Compute H0, H1, H2
        "adaptive_radius": True,  # Use adaptive radius based on data
        "radius_percentile": 90,  # Use 90th percentile of distances
        "pca_components": None,  # No PCA (activations are already 256D)
        "use_gudhi": True,  # Prefer GUDHI over ripser
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    if not os.path.exists(config["results_dir"]):
        print(f"Error: Results directory '{config['results_dir']}' not found!")
        return

    activation_files = []
    for item in os.listdir(config["results_dir"]):
        activation_file = os.path.join(
            config["results_dir"], item, "penultimate_activations", "activations.pt"
        )
        if os.path.exists(activation_file):
            activation_files.append((item, activation_file))

    if not activation_files:
        print("No activation data found! Run activation extraction first.")
        return

    print(
        f"Found activation data for optimizers: {[opt for opt, _ in activation_files]}"
    )
    print()

    try:
        print("Computing Vietoris-Rips filtrations...")
        results = compute_vietoris_rips_filtrations(
            results_dir=config["results_dir"],
            optimizers=config["optimizers"],
            max_dimension=config["max_dimension"],
            adaptive_radius=config["adaptive_radius"],
            radius_percentile=config["radius_percentile"],
            pca_components=config["pca_components"],
            use_gudhi=config["use_gudhi"],
        )

        print("\n" + "=" * 60)
        print("COMPUTATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nProcessed {len(results)} optimizers:")
        for opt_name, opt_data in results.items():
            print(f"\n{opt_name}:")
            print(f"  Epochs: {len(opt_data['epochs'])}")

            total_particles = 0
            total_successful = 0
            total_failed = 0

            for epoch_data in opt_data["epochs"].values():
                summary = epoch_data["summary"]
                total_particles += summary["total_particles"]
                total_successful += summary["successful_computations"]
                total_failed += summary["failed_computations"]

            print(f"  Total particles: {total_particles}")
            print(f"  Successful computations: {total_successful}")
            print(f"  Failed computations: {total_failed}")
            success_rate = (
                total_successful / total_particles * 100 if total_particles > 0 else 0
            )
            print(f"  Success rate: {success_rate:.1f}%")

        print(f"\nOutput files:")
        print(f"  Combined: {config['results_dir']}/all_vietoris_rips_persistence.pt")
        print(f"  Summary: {config['results_dir']}/vietoris_rips_summary.json")

        for opt in results.keys():
            individual_file = f"{config['results_dir']}/{opt}/vietoris_rips_persistence/persistence_diagrams.pt"
            if os.path.exists(individual_file):
                print(f"  {opt}: {individual_file}")

        print(f"\nData structure per particle:")
        print(f"  - persistence diagrams for validation dataset (H0, H1, H2)")
        print(f"  - persistence diagrams for validation_corrupt dataset (H0, H1, H2)")
        print(f"  - computation metadata (radius, method, shapes)")

    except Exception as e:
        print(f"\nError during computation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

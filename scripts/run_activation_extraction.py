#!/usr/bin/env python3
"""
Script to extract penultimate layer activations for all optimizers and epochs.
This extracts activations from the second hidden layer (net.2) for both
validation and validation corrupt datasets.
"""

import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.activation_extractor import extract_penultimate_activations


def main():
    """Extract penultimate layer activations for TDA analysis."""

    print("=" * 60)
    print("PENULTIMATE LAYER ACTIVATION EXTRACTION")
    print("=" * 60)

    # Configuration
    config = {
        "results_dir": "./results",
        "data_dir": "./data",
        "optimizers": None,  # None = all optimizers found in results dir
        "epochs_to_extract": [0, 30],  # Only epochs 0 and 30
        "max_batches_per_dataset": 1,  # Only 1 batch per dataset (128 samples each)
        "batch_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    if not os.path.exists(config["results_dir"]):
        print(f"Error: Results directory '{config['results_dir']}' not found!")
        return

    if config["optimizers"] is None:
        available_optimizers = [
            d
            for d in os.listdir(config["results_dir"])
            if os.path.isdir(os.path.join(config["results_dir"], d))
            and os.path.exists(
                os.path.join(config["results_dir"], d, "pretrain_weights")
            )
        ]
        print(f"Auto-detected optimizers: {available_optimizers}")
        config["optimizers"] = available_optimizers

    if not config["optimizers"]:
        print("No optimizers with pretrain weights found!")
        return

    print(f"\nExtracting activations for: {config['optimizers']}")
    print(f"Using device: {config['device']}")
    print()

    # Extract activations
    try:
        results = extract_penultimate_activations(
            results_dir=config["results_dir"],
            data_dir=config["data_dir"],
            optimizers=config["optimizers"],
            epochs_to_extract=config["epochs_to_extract"],
            max_batches_per_dataset=config["max_batches_per_dataset"],
            batch_size=config["batch_size"],
            device=config["device"],
        )

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Print summary
        print(f"\nProcessed {len(results)} optimizers:")
        for opt_name, opt_data in results.items():
            print(f"\n{opt_name}:")
            print(f"  Epochs: {len(opt_data['epochs'])}")

            # Sample activation info
            if opt_data["epochs"]:
                sample_epoch = next(iter(opt_data["epochs"].values()))
                if sample_epoch["particles"]:
                    sample_particle = next(iter(sample_epoch["particles"].values()))
                    val_shape = sample_particle["validation"]["shape"]
                    corrupt_shape = sample_particle["validation_corrupt"]["shape"]
                    print(f"  Validation activations shape: {val_shape}")
                    print(f"  Validation corrupt activations shape: {corrupt_shape}")
                    print(f"  Particles per epoch: {len(sample_epoch['particles'])}")

        # Output files
        print(f"\nOutput files:")
        print(f"  Combined: {config['results_dir']}/all_penultimate_activations.pt")
        print(f"  Summary: {config['results_dir']}/activation_extraction_summary.json")

        for opt in config["optimizers"]:
            individual_file = (
                f"{config['results_dir']}/{opt}/penultimate_activations/activations.pt"
            )
            if os.path.exists(individual_file):
                print(f"  {opt}: {individual_file}")

    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

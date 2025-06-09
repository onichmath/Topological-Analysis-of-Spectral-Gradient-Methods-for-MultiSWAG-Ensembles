import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from projections import DataLoader, ProjectionEngine


def analyze_variance_explained():
    optimizers = [
        "adam",
        "adamw",
        "muon",
        "10p",
        "muon10p",
        "muonspectralnorm",
        "spectralnorm",
    ]
    data_loader = DataLoader(max_trajectories=620)

    results = {}

    for optimizer in optimizers:
        trajectories, _ = data_loader.load_optimizer_trajectories(
            optimizer=optimizer, epochs=range(0, 30), max_particles=20
        )
        print(f"Loaded trajectories: {trajectories.shape}")

        if len(trajectories) >= 50:
            variance_explained = ProjectionEngine.get_variance_explained_50d(
                trajectories
            )
            results[optimizer] = variance_explained
            print(
                f"{optimizer}: {variance_explained:.3f} ({variance_explained*100:.1f}%)"
            )
        else:
            print(f"{optimizer}: Insufficient data ({len(trajectories)} trajectories)")

    print(f"\nRanked by variance explained:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (optimizer, variance) in enumerate(sorted_results, 1):
        print(f"{i}. {optimizer}: {variance*100:.1f}%")

    with open("./results/variance_explained.csv", "w") as f:
        f.write("optimizer,variance_explained\n")
        for optimizer, variance in sorted_results:
            f.write(f"{optimizer},{variance*100:.1f}\n")

    return results


if __name__ == "__main__":
    analyze_variance_explained()

#!/usr/bin/env python3
"""
Vietoris-Rips Filtration Computer for Penultimate Layer Activations

Computes persistence diagrams for each particle's activations using Vietoris-Rips filtration.
Handles both validation and validation_corrupt datasets.
"""

import torch
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# TDA libraries
try:
    import gudhi as gd

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not available. Install with: pip install gudhi")

try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available. Install with: pip install ripser")

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


class VietorisRipsComputer:
    """Compute Vietoris-Rips filtrations from penultimate layer activations."""

    def __init__(
        self,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None,
        adaptive_radius: bool = True,
        radius_percentile: float = 90,
        pca_components: Optional[int] = None,
        use_gudhi: bool = True,
    ):
        """
        Initialize Vietoris-Rips computer.

        Args:
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex (None = adaptive)
            adaptive_radius: Whether to compute adaptive radius per activation set
            radius_percentile: Percentile of pairwise distances to use as max radius
            pca_components: Number of PCA components (None = no reduction)
            use_gudhi: Use GUDHI if available, else fallback to ripser
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.adaptive_radius = adaptive_radius
        self.radius_percentile = radius_percentile
        self.pca_components = pca_components
        self.use_gudhi = use_gudhi and GUDHI_AVAILABLE

        if not GUDHI_AVAILABLE and not RIPSER_AVAILABLE:
            raise ImportError(
                "Neither gudhi nor ripser available. Install with: pip install gudhi ripser"
            )

    def _preprocess_activations(self, activations: np.ndarray) -> np.ndarray:
        """Preprocess activations with optional PCA."""
        if self.pca_components and activations.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components, random_state=42)
            return pca.fit_transform(activations)
        return activations

    def _compute_adaptive_radius(self, activations: np.ndarray) -> float:
        """Compute adaptive radius based on pairwise distances."""
        distances = pairwise_distances(activations)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        if len(upper_triangle) > 0:
            return np.percentile(upper_triangle, self.radius_percentile)
        return 1.0

    def _compute_persistence_gudhi(self, activations: np.ndarray) -> Dict:
        """Compute persistence using GUDHI."""
        try:
            if self.adaptive_radius or self.max_edge_length is None:
                radius = self._compute_adaptive_radius(activations)
            else:
                radius = self.max_edge_length

            rips_complex = gd.RipsComplex(points=activations, max_edge_length=radius)
            simplex_tree = rips_complex.create_simplex_tree(
                max_dimension=self.max_dimension + 1
            )

            persistence = simplex_tree.persistence()

            diagrams = {}
            for dim in range(self.max_dimension + 1):
                diagrams[f"H{dim}"] = []

            for dimension, (birth, death) in persistence:
                if dimension <= self.max_dimension:
                    diagrams[f"H{dimension}"].append([birth, death])

            for dim in diagrams:
                diagrams[dim] = (
                    np.array(diagrams[dim])
                    if diagrams[dim]
                    else np.array([]).reshape(0, 2)
                )

            return {
                "diagrams": diagrams,
                "radius": radius,
                "n_simplices": simplex_tree.num_simplices(),
                "method": "gudhi",
            }

        except Exception as e:
            print(f"GUDHI computation failed: {e}")
            return self._compute_persistence_ripser(activations)

    def _compute_persistence_ripser(self, activations: np.ndarray) -> Dict:
        """Compute persistence using ripser."""
        try:
            if self.adaptive_radius or self.max_edge_length is None:
                radius = self._compute_adaptive_radius(activations)
                thresh = radius
            else:
                thresh = self.max_edge_length
                radius = thresh

            result = ripser(activations, maxdim=self.max_dimension, thresh=thresh)
            diagrams_raw = result["dgms"]

            diagrams = {}
            for dim in range(self.max_dimension + 1):
                if dim < len(diagrams_raw):
                    diagrams[f"H{dim}"] = diagrams_raw[dim]
                else:
                    diagrams[f"H{dim}"] = np.array([]).reshape(0, 2)

            return {
                "diagrams": diagrams,
                "radius": radius,
                "n_simplices": None,  # Ripser doesn't provide this easily
                "method": "ripser",
            }

        except Exception as e:
            print(f"Ripser computation failed: {e}")
            return {
                "diagrams": {
                    f"H{dim}": np.array([]).reshape(0, 2)
                    for dim in range(self.max_dimension + 1)
                },
                "radius": 0.0,
                "n_simplices": 0,
                "method": "failed",
            }

    def compute_persistence(self, activations: np.ndarray) -> Dict:
        """Compute persistence diagram for activation matrix."""

        processed_activations = self._preprocess_activations(activations)

        if len(processed_activations) < 3:
            return {
                "diagrams": {
                    f"H{dim}": np.array([]).reshape(0, 2)
                    for dim in range(self.max_dimension + 1)
                },
                "radius": 0.0,
                "n_simplices": 0,
                "method": "insufficient_data",
                "original_shape": activations.shape,
                "processed_shape": processed_activations.shape,
            }

        if self.use_gudhi:
            result = self._compute_persistence_gudhi(processed_activations)
        else:
            result = self._compute_persistence_ripser(processed_activations)

        result["original_shape"] = activations.shape
        result["processed_shape"] = processed_activations.shape

        return result

    def process_activation_data(self, activation_file: str) -> Dict:
        """Process activation data file and compute all persistence diagrams."""

        print(f"Loading activation data from {activation_file}")
        try:
            data = torch.load(activation_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading {activation_file}: {e}")
            return {}

        optimizer_name = data.get("optimizer", "unknown")
        print(f"Processing optimizer: {optimizer_name}")

        results = {
            "optimizer": optimizer_name,
            "epochs": {},
            "metadata": {
                "max_dimension": self.max_dimension,
                "adaptive_radius": self.adaptive_radius,
                "radius_percentile": self.radius_percentile,
                "pca_components": self.pca_components,
                "method": "gudhi" if self.use_gudhi else "ripser",
            },
        }

        epochs_data = data.get("epochs", {})

        for epoch, epoch_data in epochs_data.items():
            print(f"\n  Processing epoch {epoch}")

            epoch_results = {
                "particles": {},
                "summary": {
                    "total_particles": 0,
                    "successful_computations": 0,
                    "failed_computations": 0,
                },
            }

            particles_data = epoch_data.get("particles", {})

            for particle_id, particle_data in tqdm(
                particles_data.items(), desc=f"Epoch {epoch}"
            ):
                particle_results = {}

                val_activations = particle_data["validation"]["activations"]
                val_persistence = self.compute_persistence(val_activations)
                particle_results["validation"] = val_persistence

                val_corrupt_activations = particle_data["validation_corrupt"][
                    "activations"
                ]
                val_corrupt_persistence = self.compute_persistence(
                    val_corrupt_activations
                )
                particle_results["validation_corrupt"] = val_corrupt_persistence

                epoch_results["particles"][particle_id] = particle_results
                epoch_results["summary"]["total_particles"] += 1

                if val_persistence["method"] not in [
                    "failed",
                    "insufficient_data",
                ] and val_corrupt_persistence["method"] not in [
                    "failed",
                    "insufficient_data",
                ]:
                    epoch_results["summary"]["successful_computations"] += 1
                else:
                    epoch_results["summary"]["failed_computations"] += 1

            results["epochs"][epoch] = epoch_results

        return results

    def process_all_optimizers(
        self,
        results_dir: str = "./results",
        optimizers: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict:
        """Process all optimizer activation files."""

        if optimizers is None:
            optimizers = []
            for item in os.listdir(results_dir):
                activation_file = os.path.join(
                    results_dir, item, "penultimate_activations", "activations.pt"
                )
                if os.path.exists(activation_file):
                    optimizers.append(item)

        print(f"Processing optimizers: {optimizers}")

        all_results = {}

        for optimizer in optimizers:
            activation_file = os.path.join(
                results_dir, optimizer, "penultimate_activations", "activations.pt"
            )

            if not os.path.exists(activation_file):
                print(f"Activation file not found for {optimizer}: {activation_file}")
                continue

            try:
                results = self.process_activation_data(activation_file)
                if results:
                    all_results[optimizer] = results

                    if save_results:
                        output_dir = os.path.join(
                            results_dir, optimizer, "vietoris_rips_persistence"
                        )
                        os.makedirs(output_dir, exist_ok=True)

                        output_file = os.path.join(
                            output_dir, "persistence_diagrams.pt"
                        )
                        torch.save(results, output_file)
                        print(
                            f"Saved persistence diagrams for {optimizer} to {output_file}"
                        )

            except Exception as e:
                print(f"Error processing {optimizer}: {e}")

        if save_results and all_results:
            combined_output = os.path.join(
                results_dir, "all_vietoris_rips_persistence.pt"
            )
            torch.save(all_results, combined_output)
            print(f"\nSaved combined persistence diagrams to {combined_output}")

            summary = self._create_summary(all_results)
            summary_file = os.path.join(results_dir, "vietoris_rips_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {summary_file}")

        return all_results

    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics of persistence computations."""
        summary = {
            "optimizers": list(results.keys()),
            "computation_stats": {},
            "topology_stats": {},
        }

        for opt, data in results.items():
            opt_stats = {
                "epochs": list(data["epochs"].keys()),
                "total_particles": 0,
                "successful_computations": 0,
                "failed_computations": 0,
                "betti_numbers": {"H0": [], "H1": [], "H2": []},
            }

            for epoch, epoch_data in data["epochs"].items():
                epoch_summary = epoch_data["summary"]
                opt_stats["total_particles"] += epoch_summary["total_particles"]
                opt_stats["successful_computations"] += epoch_summary[
                    "successful_computations"
                ]
                opt_stats["failed_computations"] += epoch_summary["failed_computations"]

                for particle_data in epoch_data["particles"].values():
                    for dataset in ["validation", "validation_corrupt"]:
                        diagrams = particle_data[dataset]["diagrams"]
                        for dim in ["H0", "H1", "H2"]:
                            if dim in diagrams:
                                betti = len(diagrams[dim])
                                opt_stats["betti_numbers"][dim].append(betti)

            summary["computation_stats"][opt] = opt_stats

        return summary


def compute_vietoris_rips_filtrations(
    results_dir: str = "./results",
    optimizers: Optional[List[str]] = None,
    max_dimension: int = 2,
    adaptive_radius: bool = True,
    radius_percentile: float = 90,
    pca_components: Optional[int] = None,
    use_gudhi: bool = True,
) -> Dict:
    """
    Convenience function to compute Vietoris-Rips filtrations.

    Args:
        results_dir: Directory containing activation results
        optimizers: List of optimizers to process (None = all)
        max_dimension: Maximum homology dimension
        adaptive_radius: Use adaptive radius computation
        radius_percentile: Percentile for adaptive radius
        pca_components: PCA components for dimensionality reduction
        use_gudhi: Prefer GUDHI over ripser

    Returns:
        Dictionary of all persistence results
    """

    computer = VietorisRipsComputer(
        max_dimension=max_dimension,
        adaptive_radius=adaptive_radius,
        radius_percentile=radius_percentile,
        pca_components=pca_components,
        use_gudhi=use_gudhi,
    )

    return computer.process_all_optimizers(
        results_dir=results_dir, optimizers=optimizers, save_results=True
    )


if __name__ == "__main__":
    print("Computing Vietoris-Rips filtrations for penultimate layer activations...")

    results = compute_vietoris_rips_filtrations(
        max_dimension=2,  # Compute H0, H1, H2
        adaptive_radius=True,
        radius_percentile=90,  # Use 90th percentile of distances
        pca_components=None,  # No PCA reduction (activations are already 256D)
        use_gudhi=True,
    )

    print(f"\nProcessed {len(results)} optimizers")
    for opt, data in results.items():
        print(f"  {opt}: {len(data['epochs'])} epochs")

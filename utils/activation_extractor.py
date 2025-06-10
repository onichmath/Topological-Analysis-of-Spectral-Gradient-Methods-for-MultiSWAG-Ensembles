import torch
import torch.nn as nn
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import json

from utils.dataloaders import build_train_dataloaders
from utils.transforms import get_transform, get_corrupt_transform
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.MLP import MLP


class PenultimateActivationExtractor:
    """Extract activations from penultimate layer of MLP for TDA analysis."""

    def __init__(
        self,
        results_dir: str = "./results",
        data_dir: str = "./data",
        batch_size: int = 128,
        val_size: int = 10000,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = torch.device(device)
        self.seed = seed

        self.model_config = {
            "input_dim": 28 * 28,
            "hidden_dim": 256,
            "output_dim": 10,
            "num_hidden_layers": 2,
        }

        self._setup_dataloaders()

    def _setup_dataloaders(self):
        """Setup validation and validation corrupt dataloaders."""
        print("Setting up dataloaders...")
        _, self.val_dataloader, self.val_corrupt_dataloader = build_train_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            val_size=self.val_size,
            transform=get_transform(),
            corrupt_transform=get_corrupt_transform(),
            seed=self.seed,
        )
        print(f"Validation dataset size: {len(self.val_dataloader.dataset)}")
        print(
            f"Validation corrupt dataset size: {len(self.val_corrupt_dataloader.dataset)}"
        )
        print("Using first batch only for consistent analysis across all models")

    def _create_model(self) -> nn.Module:
        """Create model instance with correct architecture."""
        model = MLP(self.model_config)
        return model

    def _get_penultimate_hook(self, activations_dict: Dict, key: str):
        """Create hook to capture penultimate layer activations."""

        def hook(module, input, output):
            activations_dict[key] = output.detach().cpu().numpy()

        return hook

    def _extract_activations_for_weights(
        self,
        weights_path: str,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> np.ndarray:
        """Extract penultimate layer activations for given model weights and dataloader."""

        model = self._create_model()
        try:
            weights = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(weights)
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")
            return np.array([])

        model.to(self.device)
        model.eval()

        activations_dict = {}
        hook_handle = model.net[2].register_forward_hook(
            self._get_penultimate_hook(activations_dict, "penultimate")
        )

        all_activations = []
        all_labels = []

        try:
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(dataloader):
                    if max_batches and batch_idx >= max_batches:
                        break

                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    _ = model(data)

                    if "penultimate" in activations_dict:
                        all_activations.append(activations_dict["penultimate"])
                        all_labels.append(targets.cpu().numpy())
                        del activations_dict["penultimate"]

        finally:
            hook_handle.remove()

        if not all_activations:
            return np.array([]), np.array([])

        return np.vstack(all_activations), np.concatenate(all_labels)

    def extract_activations_for_optimizer(
        self,
        optimizer_name: str,
        max_batches_per_dataset: Optional[int] = None,
        epochs_to_extract: Optional[List[int]] = None,
    ) -> Dict:
        """Extract activations for all epochs and particles of a given optimizer."""

        optimizer_dir = os.path.join(
            self.results_dir, optimizer_name, "pretrain_weights"
        )

        if not os.path.exists(optimizer_dir):
            print(f"No pretrain weights found for optimizer: {optimizer_name}")
            return {}

        weight_files = glob.glob(
            os.path.join(optimizer_dir, "particle*_epoch*_weights.pt")
        )

        if not weight_files:
            print(f"No weight files found in {optimizer_dir}")
            return {}

        file_info = []
        for file_path in weight_files:
            filename = os.path.basename(file_path)
            parts = filename.replace("_weights.pt", "").split("_")

            if len(parts) >= 2:
                try:
                    particle_id = int(parts[0].replace("particle", ""))
                    epoch = int(parts[1].replace("epoch", ""))

                    if epochs_to_extract is None or epoch in epochs_to_extract:
                        file_info.append(
                            {"path": file_path, "particle": particle_id, "epoch": epoch}
                        )
                except ValueError as e:
                    print(f"Error parsing {filename}: {e}")

        if not file_info:
            print(f"No valid weight files found for {optimizer_name}")
            return {}

        file_info.sort(key=lambda x: (x["epoch"], x["particle"]))

        print(f"\nExtracting activations for {optimizer_name}")
        print(f"Found {len(file_info)} weight files")

        epochs = {}
        for info in file_info:
            epoch = info["epoch"]
            if epoch not in epochs:
                epochs[epoch] = []
            epochs[epoch].append(info)

        results = {
            "optimizer": optimizer_name,
            "epochs": {},
            "metadata": {
                "model_config": self.model_config,
                "batch_size": self.batch_size,
                "val_size": self.val_size,
                "max_batches_per_dataset": max_batches_per_dataset,
            },
        }

        for epoch in sorted(epochs.keys()):
            print(f"\n  Processing epoch {epoch} ({len(epochs[epoch])} particles)")

            epoch_results = {
                "particles": {},
                "validation": {},
                "validation_corrupt": {},
            }

            for particle_info in tqdm(epochs[epoch], desc=f"Epoch {epoch}"):
                particle_id = particle_info["particle"]
                weights_path = particle_info["path"]

                val_activations, val_labels = self._extract_activations_for_weights(
                    weights_path, self.val_dataloader, max_batches_per_dataset
                )

                val_corrupt_activations, val_corrupt_labels = (
                    self._extract_activations_for_weights(
                        weights_path,
                        self.val_corrupt_dataloader,
                        max_batches_per_dataset,
                    )
                )

                if val_activations.size > 0 and val_corrupt_activations.size > 0:
                    epoch_results["particles"][particle_id] = {
                        "validation": {
                            "activations": val_activations,
                            "labels": val_labels,
                            "shape": val_activations.shape,
                        },
                        "validation_corrupt": {
                            "activations": val_corrupt_activations,
                            "labels": val_corrupt_labels,
                            "shape": val_corrupt_activations.shape,
                        },
                    }

            results["epochs"][epoch] = epoch_results

        return results

    def extract_all_optimizers(
        self,
        optimizers: Optional[List[str]] = None,
        max_batches_per_dataset: Optional[int] = None,
        epochs_to_extract: Optional[List[int]] = None,
        save_results: bool = True,
    ) -> Dict:
        """Extract activations for all optimizers."""

        if optimizers is None:
            optimizers = [
                d
                for d in os.listdir(self.results_dir)
                if os.path.isdir(os.path.join(self.results_dir, d))
            ]

        print(f"Processing optimizers: {optimizers}")

        all_results = {}

        for optimizer in optimizers:
            try:
                results = self.extract_activations_for_optimizer(
                    optimizer, max_batches_per_dataset, epochs_to_extract
                )
                if results:
                    all_results[optimizer] = results

                    if save_results:
                        output_dir = os.path.join(
                            self.results_dir, optimizer, "penultimate_activations"
                        )
                        os.makedirs(output_dir, exist_ok=True)

                        output_file = os.path.join(output_dir, "activations.pt")
                        torch.save(results, output_file)
                        print(f"Saved activations for {optimizer} to {output_file}")

            except Exception as e:
                print(f"Error processing optimizer {optimizer}: {e}")

        if save_results and all_results:
            combined_output = os.path.join(
                self.results_dir, "all_penultimate_activations.pt"
            )
            torch.save(all_results, combined_output)
            print(f"\nSaved combined activations to {combined_output}")

            summary = {
                "optimizers": list(all_results.keys()),
                "epochs_processed": {},
                "particles_per_optimizer": {},
                "activation_shapes": {},
            }

            for opt, data in all_results.items():
                summary["epochs_processed"][opt] = list(data["epochs"].keys())
                summary["particles_per_optimizer"][opt] = {}
                summary["activation_shapes"][opt] = {}

                for epoch, epoch_data in data["epochs"].items():
                    summary["particles_per_optimizer"][opt][epoch] = len(
                        epoch_data["particles"]
                    )
                    if epoch_data["particles"]:
                        first_particle = next(iter(epoch_data["particles"].values()))
                        summary["activation_shapes"][opt][epoch] = {
                            "validation": first_particle["validation"]["shape"],
                            "validation_corrupt": first_particle["validation_corrupt"][
                                "shape"
                            ],
                        }

            summary_file = os.path.join(
                self.results_dir, "activation_extraction_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved extraction summary to {summary_file}")

        return all_results


def extract_penultimate_activations(
    results_dir: str = "./results",
    data_dir: str = "./data",
    optimizers: Optional[List[str]] = None,
    epochs_to_extract: Optional[List[int]] = None,
    max_batches_per_dataset: Optional[int] = 50,  # Limit for memory management
    batch_size: int = 128,
    device: str = "cpu",
) -> Dict:
    """
    Convenience function to extract penultimate layer activations.

    Args:
        results_dir: Directory containing optimizer results
        data_dir: Directory containing MNIST data
        optimizers: List of optimizer names to process (None = all)
        epochs_to_extract: List of epochs to extract (None = all)
        max_batches_per_dataset: Max batches per dataset to limit memory usage
        batch_size: Batch size for data loading
        device: Device to use for computation

    Returns:
        Dictionary containing all extracted activations
    """

    extractor = PenultimateActivationExtractor(
        results_dir=results_dir, data_dir=data_dir, batch_size=batch_size, device=device
    )

    return extractor.extract_all_optimizers(
        optimizers=optimizers,
        max_batches_per_dataset=max_batches_per_dataset,
        epochs_to_extract=epochs_to_extract,
        save_results=True,
    )


if __name__ == "__main__":
    # Example usage
    results = extract_penultimate_activations(
        optimizers=[
            "adam",
            "adamw",
            "muon",
            "10p",
            "muon10p",
            "muonspectralnorm",
            "spectralnorm",
        ],  # Specify optimizers or None for all
        epochs_to_extract=[0, 30],  # Only epochs 0 and 30
        max_batches_per_dataset=1,  # Only 1 batch per dataset
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nExtraction completed!")
    print(f"Processed {len(results)} optimizers")
    for opt, data in results.items():
        print(f"  {opt}: {len(data['epochs'])} epochs")

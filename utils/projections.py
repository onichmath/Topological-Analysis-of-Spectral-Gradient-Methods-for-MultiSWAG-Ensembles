import torch
import numpy as np
from pathlib import Path
import gc
import os
import time
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import numpy.core.multiarray

torch.serialization.add_safe_globals(
    [
        numpy.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        getattr(np.dtypes, "Float32DType", type(None)),
        getattr(np.dtypes, "Float64DType", type(None)),
        getattr(np.dtypes, "Int32DType", type(None)),
        getattr(np.dtypes, "Int64DType", type(None)),
    ]
)

os.environ.update(
    {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
)

warnings.filterwarnings("ignore")


class DataLoader:
    def __init__(self, max_trajectories=620, max_total=30000):
        self.max_trajectories = max_trajectories
        self.max_total = max_total
        print(
            f"DataLoader initialized with max_trajectories={max_trajectories} and max_total={max_total}"
        )

    def load_data(self, optimizers, epochs=range(0, 31), max_particles=20):
        all_metadata = []

        for optimizer in optimizers:
            weights_dir = Path("results") / optimizer / "pretrain_weights"
            eval_dir = Path("results") / optimizer / "evaluation_results"

            if not weights_dir.exists():
                print(f"No weights directory for {optimizer}")
                continue

            optimizer_files = []
            for epoch in epochs:
                for particle_id in range(max_particles):
                    weight_file = (
                        weights_dir / f"particle{particle_id}_epoch{epoch}_weights.pt"
                    )
                    if weight_file.exists():
                        optimizer_files.append(
                            {
                                "file": weight_file,
                                "optimizer": optimizer,
                                "epoch": epoch,
                                "particle_id": particle_id,
                            }
                        )

            if len(optimizer_files) > self.max_trajectories:
                print(
                    f"Subsampling {optimizer_files} to {self.max_trajectories} trajectories"
                )
                step = len(optimizer_files) // self.max_trajectories
                optimizer_files = optimizer_files[::step][: self.max_trajectories]

            all_metadata.extend(optimizer_files)
            print(f"{optimizer}: {len(optimizer_files)} files")

        if len(all_metadata) > self.max_total:
            print(f"Subsampling {all_metadata} to {self.max_total} trajectories")
            step = len(all_metadata) // self.max_total
            all_metadata = all_metadata[::step][: self.max_total]

        return all_metadata

    def load_trajectories(self, metadata_list):
        trajectories, valid_metadata = [], []

        numpy_safe_globals = [
            numpy.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            getattr(np.dtypes, "Float32DType", type(None)),
            getattr(np.dtypes, "Float64DType", type(None)),
            getattr(np.dtypes, "Int32DType", type(None)),
            getattr(np.dtypes, "Int64DType", type(None)),
        ]

        for i, meta in enumerate(metadata_list):
            try:
                with torch.serialization.safe_globals(numpy_safe_globals):
                    weights = torch.load(meta["file"], map_location="cpu")
                weight_vector = []
                for param in weights.values():
                    weight_vector.extend(param.flatten().detach().cpu().numpy())

                trajectories.append(np.array(weight_vector, dtype=np.float32))
                valid_metadata.append(meta)
                del weights

                if (i + 1) % 50 == 0:
                    print(f"Loaded {i + 1}/{len(metadata_list)} trajectories")
                    gc.collect()
            except Exception as e:
                print(f"Error loading {meta['file']}: {e}")
                continue

        print(f"Loaded {len(trajectories)} trajectories")
        return np.array(trajectories), valid_metadata

    def load_optimizer_trajectories(
        self, optimizer, epochs=range(0, 31), max_particles=20
    ):
        """Load trajectories for a single optimizer."""
        print(f"Loading trajectories for {optimizer}...")

        weights_dir = Path("results") / optimizer / "pretrain_weights"
        eval_dir = Path("results") / optimizer / "evaluation_results"

        if not weights_dir.exists():
            print(f"No weights directory for {optimizer}")
            return np.array([]), []
        trajectories, metadata = [], []
        for epoch in epochs:
            for particle_id in range(max_particles):
                weight_file = (
                    weights_dir / f"particle{particle_id}_epoch{epoch}_weights.pt"
                )
                if weight_file.exists():
                    try:
                        numpy_safe_globals = [
                            numpy.core.multiarray._reconstruct,
                            np.ndarray,
                            np.dtype,
                            np.float32,
                            np.float64,
                            np.int32,
                            np.int64,
                            getattr(np.dtypes, "Float32DType", type(None)),
                            getattr(np.dtypes, "Float64DType", type(None)),
                            getattr(np.dtypes, "Int32DType", type(None)),
                            getattr(np.dtypes, "Int64DType", type(None)),
                        ]

                        with torch.serialization.safe_globals(numpy_safe_globals):
                            weights = torch.load(weight_file, map_location="cpu")
                        weight_vector = []
                        for param in weights.values():
                            weight_vector.extend(param.flatten().detach().cpu().numpy())

                        trajectories.append(np.array(weight_vector, dtype=np.float32))
                        metadata.append(
                            {
                                "optimizer": optimizer,
                                "epoch": epoch,
                                "particle_id": particle_id,
                            }
                        )
                        del weights
                    except Exception as e:
                        continue

        if len(trajectories) > self.max_trajectories:
            print(f"Subsampling {trajectories} to {self.max_trajectories} trajectories")
            step = len(trajectories) // self.max_trajectories
            trajectories = trajectories[::step][: self.max_trajectories]
            metadata = metadata[::step][: self.max_trajectories]

        print(f"Loaded {len(trajectories)} trajectories for {optimizer}")
        return np.array(trajectories) if trajectories else np.array([]), metadata


class ProjectionEngine:
    @staticmethod
    def create_projections(trajectories, include_tsne=True, n_components=50):
        """Create standardized projections from trajectories."""
        print("Creating projections...")

        if len(trajectories) == 0:
            return {}, None

        scaler = StandardScaler()
        trajectories_scaled = scaler.fit_transform(trajectories)

        pca = PCA(n_components=n_components, random_state=42)
        pca_result = pca.fit_transform(trajectories_scaled)

        projections = {"pca_2d": pca_result[:, :2], "pca_extended": pca_result}

        if include_tsne and len(trajectories) <= 1000:
            print("Computing t-SNE projection...")
            perplexity = min(30, len(trajectories) // 4)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_input = pca_result[:, :50] if pca_result.shape[1] > 50 else pca_result
            projections["tsne_2d"] = tsne.fit_transform(tsne_input)

        print(f"Created projections: {list(projections.keys())}")
        return projections, pca

    @staticmethod
    def get_variance_explained_50d(trajectories, N=50):
        """
        Calculate percentage of variance explained by N PCA dimensions.
        """
        if len(trajectories) == 0:
            return 0.0

        scaler = StandardScaler()
        trajectories_scaled = scaler.fit_transform(trajectories)

        pca = PCA(n_components=N, random_state=42)
        pca.fit(trajectories_scaled)

        return pca.explained_variance_ratio_.sum()

    @staticmethod
    def create_lens_functions(projections, metadata):
        """Create lens functions from projections and metadata."""
        print("Creating lens functions...")
        lens_functions = {}

        if "pca_2d" in projections:
            lens_functions["pc1_pc2"] = projections["pca_2d"]

        if "pca_2d" in projections and metadata:
            pc1 = projections["pca_2d"][:, 0]

            val_losses = [m.get("val_loss") for m in metadata]
            if any(v is not None for v in val_losses):
                val_losses = np.array(
                    [
                        (
                            v
                            if v is not None
                            else np.mean([x for x in val_losses if x is not None])
                        )
                        for v in val_losses
                    ]
                )
                val_loss_norm = (val_losses - np.mean(val_losses)) / (
                    np.std(val_losses) + 1e-8
                )
                lens_functions["pc1_valloss"] = np.column_stack([pc1, val_loss_norm])

            epochs = np.array([m["epoch"] for m in metadata])
            epoch_norm = (epochs - np.mean(epochs)) / (np.std(epochs) + 1e-8)
            lens_functions["pc1_epoch"] = np.column_stack([pc1, epoch_norm])

        if "pca_extended" in projections and projections["pca_extended"].shape[1] >= 3:
            lens_functions["pc1_pc3"] = projections["pca_extended"][:, [0, 2]]

        if "tsne_2d" in projections:
            lens_functions["tsne"] = projections["tsne_2d"]

        print(f"Created lens functions: {list(lens_functions.keys())}")
        return lens_functions

    @staticmethod
    def save_projections(
        projections, lens_functions, metadata, optimizer, results_dir="results"
    ):
        """Save projections and lens functions for an optimizer."""
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)

        projection_data = {
            "optimizer": optimizer,
            "projections": projections,
            "lens_functions": lens_functions,
            "metadata": metadata,
            "n_trajectories": len(metadata),
        }
        print(projection_data.keys())
        filename = results_path / f"{optimizer}/{optimizer}_projections.pt"
        torch.save(projection_data, filename)
        print(f"Saved projections to {filename}")
        return filename

    @staticmethod
    def load_projections(optimizer, results_dir="results"):
        """Load saved projections for an optimizer."""
        filename = Path(results_dir) / f"{optimizer}/{optimizer}_projections.pt"
        print(filename)
        if filename.exists():
            numpy_safe_globals = [
                numpy.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                np.float32,
                np.float64,
                np.int32,
                np.int64,
                getattr(np.dtypes, "Float32DType", type(None)),
                getattr(np.dtypes, "Float64DType", type(None)),
                getattr(np.dtypes, "Int32DType", type(None)),
                getattr(np.dtypes, "Int64DType", type(None)),
            ]

            with torch.serialization.safe_globals(numpy_safe_globals):
                return torch.load(filename, map_location="cpu")
        else:
            print(f"No saved projections found for {optimizer}")
            return None


def generate_projections_for_optimizer(
    optimizer, data_loader, projection_engine, include_tsne=True
):
    """Generate and save projections for a single optimizer."""
    print(f"\nGENERATING PROJECTIONS FOR {optimizer.upper()}")
    print("-" * 50)

    trajectories, metadata = data_loader.load_optimizer_trajectories(optimizer)

    if len(trajectories) == 0:
        print(f"No trajectories found for {optimizer}")
        return None

    print(f" Loaded {len(trajectories)} trajectories")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Num epochs: {len(set([m['epoch'] for m in metadata]))}")
    print(f"Num particles: {len(set([m['particle_id'] for m in metadata]))}")

    projections, pca = projection_engine.create_projections(
        trajectories, include_tsne=include_tsne
    )

    if not projections:
        print(f"Failed to create projections for {optimizer}")
        return None

    lens_functions = projection_engine.create_lens_functions(projections, metadata)

    filename = projection_engine.save_projections(
        projections, lens_functions, metadata, optimizer
    )

    print(f"Summary:")
    print(f"    - Trajectories: {len(trajectories)}")
    print(f"    - Projections: {list(projections.keys())}")
    print(f"    - Lens functions: {list(lens_functions.keys())}")
    print(f"    - Saved to: {filename}")

    return {
        "optimizer": optimizer,
        "n_trajectories": len(trajectories),
        "projections": list(projections.keys()),
        "lens_functions": list(lens_functions.keys()),
        "filename": filename,
    }


def generate_optimizer_projections():
    """Generate projections for all optimizers."""

    start_time = time.time()

    optimizers = [
        "adam",
        "adamw",
        "muon",
        "10p",
        "muon10p",
        "muonspectralnorm",
        "spectralnorm",
    ]
    include_tsne = True

    # Initialize engines
    data_loader = DataLoader(max_trajectories=30000, max_total=30000)
    projection_engine = ProjectionEngine()

    all_results = {}

    for optimizer in optimizers:
        try:
            result = generate_projections_for_optimizer(
                optimizer, data_loader, projection_engine, include_tsne=include_tsne
            )
            if result:
                all_results[optimizer] = result
        except Exception as e:
            print(f"Failed to process {optimizer}: {e}")
            continue

    summary = {
        "optimizers_processed": list(all_results.keys()),
        "total_optimizers": len(all_results),
        "include_tsne": include_tsne,
        "generation_time": time.time() - start_time,
        "results": all_results,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_file = results_dir / "projection_summary.pt"
    torch.save(summary, summary_file)

    total_time = time.time() - start_time
    print(f"\nPROJECTION GENERATION COMPLETE!")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Processed {len(all_results)} optimizers")
    print(f"Summary saved to: {summary_file}")
    print(f"📁 Individual projections: results/*_projections.pt")

    # Print summary table
    print(f"\nSUMMARY TABLE:")
    print("-" * 80)
    print(
        f"{'Optimizer':<15} {'Trajectories':<12} {'Projections':<15} {'Lens Functions':<15}"
    )
    print("-" * 80)

    for optimizer, result in all_results.items():
        projections_str = f"{len(result['projections'])}"
        lens_str = f"{len(result['lens_functions'])}"
        print(
            f"{optimizer:<15} {result['n_trajectories']:<12} {projections_str:<15} {lens_str:<15}"
        )

    print("-" * 80)

    return all_results


def test_projections():
    """Test loading saved projections."""
    print("\nTESTING PROJECTION LOADING")
    print("-" * 40)

    projection_engine = ProjectionEngine()
    optimizers = ["adam", "adamw", "muon"]

    for optimizer in optimizers:
        print(f"Testing {optimizer}...")
        data = projection_engine.load_projections(optimizer)
        if data:
            print(f"Loaded {data['n_trajectories']} trajectories")
            print(f"    Projections: {list(data['projections'].keys())}")
            print(f"    Lens functions: {list(data['lens_functions'].keys())}")
        else:
            print(f"Failed to load projections for {optimizer}")


if __name__ == "__main__":
    results = generate_optimizer_projections()

    if results:
        test_projections()

import torch
import numpy as np
from typing import Dict, List
import os


class PosteriorAnalyzer:
    def __init__(self, results_base_dir: str = "results"):
        self.results_dir = results_base_dir

    def load_swag_posterior(
        self, optimizer_name: str, epoch: int, num_particles: int = 20
    ):
        """Load SWAG posterior for a specific optimizer and epoch."""
        posterior_data = {}

        for particle_id in range(num_particles):
            mom1_file = f"{self.results_dir}/{optimizer_name}/swag_moments/particle{particle_id}_epoch{epoch}_mom1.pt"
            mom2_file = f"{self.results_dir}/{optimizer_name}/swag_moments/particle{particle_id}_epoch{epoch}_mom2.pt"
            cov_file = f"{self.results_dir}/{optimizer_name}/swag_moments/particle{particle_id}_epoch{epoch}_cov_mat_sqrt.pt"

            if all(os.path.exists(f) for f in [mom1_file, mom2_file, cov_file]):
                posterior_data[particle_id] = {
                    "mean": torch.load(mom1_file, map_location="cpu"),
                    "second_moment": torch.load(mom2_file, map_location="cpu"),
                    "cov_sqrt": torch.load(cov_file, map_location="cpu"),
                }

        return posterior_data

    def compute_posterior_variance(self, posterior_data: Dict):
        """Compute variance from SWAG moments."""
        variances = {}

        for particle_id, data in posterior_data.items():
            mean = data["mean"]
            second_moment = data["second_moment"]

            particle_vars = []
            for m, sm in zip(mean, second_moment):
                var = torch.clamp(sm - m**2, 1e-8)
                particle_vars.append(var)

            variances[particle_id] = particle_vars

        return variances

    def compare_posterior_diversity(self, opt1_posterior: Dict, opt2_posterior: Dict):
        """Compare diversity between two optimizer posteriors."""

        def compute_ensemble_diversity(posterior_data):
            """Compute diversity within an ensemble."""
            means = [data["mean"] for data in posterior_data.values()]

            pairwise_distances = []
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    total_dist = 0
                    for param1, param2 in zip(means[i], means[j]):
                        total_dist += torch.norm(param1 - param2).item()
                    pairwise_distances.append(total_dist)

            return np.mean(pairwise_distances), np.std(pairwise_distances)

        div1_mean, div1_std = compute_ensemble_diversity(opt1_posterior)
        div2_mean, div2_std = compute_ensemble_diversity(opt2_posterior)

        return {
            "optimizer1_diversity": {"mean": div1_mean, "std": div1_std},
            "optimizer2_diversity": {"mean": div2_mean, "std": div2_std},
            "diversity_ratio": div1_mean / div2_mean if div2_mean > 0 else float("inf"),
        }

    def compute_uncertainty_calibration_metrics(self, optimizer_name: str, epoch: int):
        """Compute metrics for uncertainty calibration quality."""

        pred_files = []
        eval_dir = f"{self.results_dir}/{optimizer_name}/evaluation_results"

        for phase in ["test", "test_corrupt"]:
            pred_file = f"{eval_dir}/swag_epoch{epoch}_{phase}_predictions.pt"
            if os.path.exists(pred_file):
                pred_files.append((phase, torch.load(pred_file, map_location="cpu")))

        calibration_metrics = {}
        for phase, predictions in pred_files:
            if "prob" in predictions:
                probs = predictions["prob"]
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                calibration_metrics[f"{phase}_entropy"] = {
                    "mean": entropy.mean().item(),
                    "std": entropy.std().item(),
                }

        return calibration_metrics


class TrajectoryAnalyzer:
    def __init__(self, results_base_dir: str = "results"):
        self.results_dir = results_base_dir

    def extract_weight_trajectories(
        self, optimizer_name: str, particle_id: int, max_epoch: int
    ):
        """Extract weight trajectory for a specific particle."""
        trajectory = []

        for epoch in range(0, max_epoch + 1):
            weight_file = f"{self.results_dir}/{optimizer_name}/pretrain_weights/particle{particle_id}_epoch{epoch}_weights.pt"
            if os.path.exists(weight_file):
                weights = torch.load(weight_file, map_location="cpu")
                flattened = torch.cat([param.flatten() for param in weights.values()])
                trajectory.append(flattened.numpy())

        return np.array(trajectory)

    def extract_gradient_trajectories(
        self, optimizer_name: str, particle_id: int, max_epoch: int
    ):
        """Extract gradient trajectory for a specific particle."""
        trajectory = []

        for epoch in range(1, max_epoch + 1):
            grad_file = f"{self.results_dir}/{optimizer_name}/gradients/particle{particle_id}_epoch{epoch}_gradients.pt"
            if os.path.exists(grad_file):
                gradients = torch.load(grad_file, map_location="cpu")
                flattened = torch.cat([grad.flatten() for grad in gradients.values()])
                trajectory.append(flattened.numpy())

        return np.array(trajectory)

    def compute_trajectory_persistence(self, trajectory: np.ndarray):
        """Compute topological persistence of a trajectory."""
        path_lengths = []
        for i in range(1, len(trajectory)):
            path_lengths.append(np.linalg.norm(trajectory[i] - trajectory[i - 1]))

        return {
            "total_path_length": np.sum(path_lengths),
            "mean_step_size": np.mean(path_lengths),
            "step_size_variance": np.var(path_lengths),
            "trajectory_endpoints_distance": np.linalg.norm(
                trajectory[-1] - trajectory[0]
            ),
        }


def setup_posterior_analysis():
    """Set up complete posterior analysis framework."""
    return {
        "posterior_analyzer": PosteriorAnalyzer(),
        "trajectory_analyzer": TrajectoryAnalyzer(),
    }

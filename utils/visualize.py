import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional


def plot_training_losses(optimizer: str, 
                        results_dir: str = "results",
                        save_path: Optional[str] = None,
                        show_plot: bool = True):
    """
    Visualize training losses for both pretrain and SWAG phases.
    
    Args:
        optimizer: Name of the optimizer (e.g., 'adam', 'muon')
        results_dir: Base directory containing results
        save_path: Path to save the plot (if None, uses default naming)
        show_plot: Whether to display the plot
    """
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load loss data
    loss_data = _load_loss_data(optimizer, results_dir)
    
    if not loss_data:
        print(f"No loss data found for optimizer: {optimizer}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Loss Analysis - {optimizer.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Pretrain losses over epochs
    _plot_pretrain_losses(axes[0, 0], loss_data['pretrain'])
    
    # Plot 2: SWAG losses over epochs  
    _plot_swag_losses(axes[0, 1], loss_data['swag'])
    
    # Plot 3: Combined comparison
    _plot_combined_losses(axes[1, 0], loss_data)
    
    # Plot 4: Loss statistics
    _plot_loss_statistics(axes[1, 1], loss_data)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = f"{results_dir}/{optimizer}/evaluation_results/training_losses_visualization.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_optimizers_comparison(optimizers: List[str],
                                 results_dir: str = "results", 
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True):
    """
    Compare training losses across different optimizers.
    
    Args:
        optimizers: List of optimizer names to compare
        results_dir: Base directory containing results
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    plt.style.use('default')
    sns.set_palette("tab10")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Optimizer Comparison - Training Losses', fontsize=16, fontweight='bold')
    
    # Load data for all optimizers
    all_data = {}
    for opt in optimizers:
        loss_data = _load_loss_data(opt, results_dir)
        if loss_data:
            all_data[opt] = loss_data
    
    if not all_data:
        print("No loss data found for any optimizer")
        return
    
    # Plot pretrain comparison
    _plot_optimizer_comparison(axes[0], all_data, 'pretrain', 'Pretrain Phase')
    
    # Plot SWAG comparison
    _plot_optimizer_comparison(axes[1], all_data, 'swag', 'SWAG Phase')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{results_dir}/optimizer_comparison_losses.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimizer comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def _load_loss_data(optimizer: str, results_dir: str) -> Dict:
    """Load loss data from saved files."""
    eval_dir = Path(results_dir) / optimizer / "evaluation_results"
    
    if not eval_dir.exists():
        return {}
    
    data = {'pretrain': {}, 'swag': {}}
    
    # Load pretrain losses
    for loss_file in eval_dir.glob("pretrain_epoch*_loss.pt"):
        try:
            loss_data = torch.load(loss_file, map_location='cpu')
            epoch = loss_data['epoch']
            loss = loss_data['loss']
            data['pretrain'][epoch] = loss
        except Exception as e:
            print(f"Error loading {loss_file}: {e}")
    
    # Load SWAG losses
    for loss_file in eval_dir.glob("swag_epoch*_loss.pt"):
        try:
            loss_data = torch.load(loss_file, map_location='cpu')
            epoch = loss_data['epoch']
            loss = loss_data['loss']
            data['swag'][epoch] = loss
        except Exception as e:
            print(f"Error loading {loss_file}: {e}")
    
    return data


def _plot_pretrain_losses(ax, pretrain_data: Dict):
    """Plot pretrain losses over epochs."""
    if not pretrain_data:
        ax.text(0.5, 0.5, 'No Pretrain Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pretrain Losses')
        return
    
    epochs = sorted(pretrain_data.keys())
    losses = [pretrain_data[ep] for ep in epochs]
    
    ax.plot(epochs, losses, 'o-', linewidth=2, markersize=4, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Pretrain Phase Losses')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(epochs) > 1:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), '--', alpha=0.6, color='red', label='Trend')
        ax.legend()


def _plot_swag_losses(ax, swag_data: Dict):
    """Plot SWAG losses over epochs."""
    if not swag_data:
        ax.text(0.5, 0.5, 'No SWAG Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SWAG Losses')
        return
    
    epochs = sorted(swag_data.keys())
    losses = [swag_data[ep] for ep in epochs]
    
    ax.plot(epochs, losses, 's-', linewidth=2, markersize=4, alpha=0.8, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('SWAG Phase Losses')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(epochs) > 1:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), '--', alpha=0.6, color='red', label='Trend')
        ax.legend()


def _plot_combined_losses(ax, loss_data: Dict):
    """Plot combined pretrain and SWAG losses."""
    pretrain_data = loss_data['pretrain']
    swag_data = loss_data['swag']
    
    if pretrain_data:
        epochs = sorted(pretrain_data.keys())
        losses = [pretrain_data[ep] for ep in epochs]
        ax.plot(epochs, losses, 'o-', label='Pretrain', linewidth=2, markersize=4, alpha=0.8)
    
    if swag_data:
        epochs = sorted(swag_data.keys())
        losses = [swag_data[ep] for ep in epochs]
        ax.plot(epochs, losses, 's-', label='SWAG', linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Combined Phase Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_loss_statistics(ax, loss_data: Dict):
    """Plot loss distribution statistics."""
    pretrain_losses = list(loss_data['pretrain'].values()) if loss_data['pretrain'] else []
    swag_losses = list(loss_data['swag'].values()) if loss_data['swag'] else []
    
    data_to_plot = []
    labels = []
    
    if pretrain_losses:
        data_to_plot.append(pretrain_losses)
        labels.append('Pretrain')
    
    if swag_losses:
        data_to_plot.append(swag_losses)
        labels.append('SWAG')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Loss Distribution')
    ax.grid(True, alpha=0.3)


def _plot_optimizer_comparison(ax, all_data: Dict, phase: str, title: str):
    """Plot comparison across optimizers for a specific phase."""
    for opt_name, data in all_data.items():
        phase_data = data[phase]
        if phase_data:
            epochs = sorted(phase_data.keys())
            losses = [phase_data[ep] for ep in epochs]
            ax.plot(epochs, losses, 'o-', label=opt_name, linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def generate_loss_summary_table(optimizer: str, results_dir: str = "results") -> pd.DataFrame:
    """
    Generate a summary table of loss statistics.
    
    Args:
        optimizer: Name of the optimizer
        results_dir: Base directory containing results
        
    Returns:
        DataFrame with loss statistics
    """
    loss_data = _load_loss_data(optimizer, results_dir)
    
    summary_data = []
    
    for phase in ['pretrain', 'swag']:
        phase_data = loss_data[phase]
        if phase_data:
            losses = list(phase_data.values())
            summary_data.append({
                'Phase': phase.capitalize(),
                'Min Loss': np.min(losses),
                'Max Loss': np.max(losses),
                'Mean Loss': np.mean(losses),
                'Std Loss': np.std(losses),
                'Final Loss': losses[-1] if losses else None,
                'Num Epochs': len(losses)
            })
    
    return pd.DataFrame(summary_data)


def plot_test_accuracies(optimizer: str,
                        results_dir: str = "results",
                        save_path: Optional[str] = None,
                        show_plot: bool = True):
    """
    Visualize test accuracies (mean and mode) for both test and test_corrupt datasets.
    
    Args:
        optimizer: Name of the optimizer (e.g., 'adam', 'muon')
        results_dir: Base directory containing results
        save_path: Path to save the plot (if None, uses default naming)
        show_plot: Whether to display the plot
    """
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load accuracy data
    accuracy_data = _load_accuracy_data(optimizer, results_dir)
    
    if not accuracy_data:
        print(f"No accuracy data found for optimizer: {optimizer}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Test Accuracy Analysis - {optimizer.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Test dataset accuracies
    _plot_test_dataset_accuracies(axes[0, 0], accuracy_data, 'test', 'Clean Test Dataset')
    
    # Plot 2: Test corrupt dataset accuracies
    _plot_test_dataset_accuracies(axes[0, 1], accuracy_data, 'test_corrupt', 'Corrupted Test Dataset')
    
    # Plot 3: Mean accuracy comparison
    _plot_accuracy_comparison(axes[1, 0], accuracy_data, 'mean', 'Mean Accuracy Comparison')
    
    # Plot 4: Mode accuracy comparison
    _plot_accuracy_comparison(axes[1, 1], accuracy_data, 'mode', 'Mode Accuracy Comparison')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = f"{results_dir}/{optimizer}/evaluation_results/test_accuracies_visualization.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_optimizers_accuracy_comparison(optimizers: List[str],
                                          results_dir: str = "results",
                                          save_path: Optional[str] = None,
                                          show_plot: bool = True):
    """
    Compare test accuracies across different optimizers.
    
    Args:
        optimizers: List of optimizer names to compare
        results_dir: Base directory containing results
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    plt.style.use('default')
    sns.set_palette("tab10")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Comparison - Test Accuracies', fontsize=16, fontweight='bold')
    
    # Load data for all optimizers
    all_accuracy_data = {}
    for opt in optimizers:
        accuracy_data = _load_accuracy_data(opt, results_dir)
        if accuracy_data:
            all_accuracy_data[opt] = accuracy_data
    
    if not all_accuracy_data:
        print("No accuracy data found for any optimizer")
        return
    
    # Plot comparisons
    _plot_multi_optimizer_accuracy(axes[0, 0], all_accuracy_data, 'test', 'mean', 'Test Mean Accuracy')
    _plot_multi_optimizer_accuracy(axes[0, 1], all_accuracy_data, 'test', 'mode', 'Test Mode Accuracy')
    _plot_multi_optimizer_accuracy(axes[1, 0], all_accuracy_data, 'test_corrupt', 'mean', 'Test Corrupt Mean Accuracy')
    _plot_multi_optimizer_accuracy(axes[1, 1], all_accuracy_data, 'test_corrupt', 'mode', 'Test Corrupt Mode Accuracy')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{results_dir}/optimizer_comparison_accuracies.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimizer accuracy comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def _load_accuracy_data(optimizer: str, results_dir: str) -> Dict:
    """Load accuracy data by computing from saved prediction files."""
    eval_dir = Path(results_dir) / optimizer / "evaluation_results"
    
    if not eval_dir.exists():
        return {}
    
    data = {
        'pretrain': {'test': {}, 'test_corrupt': {}},
        'swag': {'test': {}, 'test_corrupt': {}}
    }
    
    # Load test dataset targets once
    test_targets = _load_test_targets()
    test_corrupt_targets = _load_test_corrupt_targets()
    
    # Load pretrain predictions and compute accuracies
    for pred_file in eval_dir.glob("pretrain_epoch*_test_predictions.pt"):
        epoch = _extract_epoch_from_filename(pred_file.name)
        if epoch is not None:
            try:
                preds = torch.load(pred_file, map_location='cpu')
                if 'mean' in preds and 'mode' in preds:
                    mean_acc = _compute_accuracy(preds['mean'], test_targets)
                    mode_acc = _compute_accuracy(preds['mode'], test_targets)
                    data['pretrain']['test'][epoch] = {'mean': mean_acc, 'mode': mode_acc}
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
    
    for pred_file in eval_dir.glob("pretrain_epoch*_test_corrupt_predictions.pt"):
        epoch = _extract_epoch_from_filename(pred_file.name)
        if epoch is not None:
            try:
                preds = torch.load(pred_file, map_location='cpu')
                if 'mean' in preds and 'mode' in preds:
                    mean_acc = _compute_accuracy(preds['mean'], test_corrupt_targets)
                    mode_acc = _compute_accuracy(preds['mode'], test_corrupt_targets)
                    data['pretrain']['test_corrupt'][epoch] = {'mean': mean_acc, 'mode': mode_acc}
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
    
    # Load SWAG predictions and compute accuracies
    for pred_file in eval_dir.glob("swag_epoch*_test_predictions.pt"):
        epoch = _extract_epoch_from_filename(pred_file.name)
        if epoch is not None:
            try:
                preds = torch.load(pred_file, map_location='cpu')
                if 'mean' in preds and 'mode' in preds:
                    mean_acc = _compute_accuracy(preds['mean'], test_targets)
                    mode_acc = _compute_accuracy(preds['mode'], test_targets)
                    data['swag']['test'][epoch] = {'mean': mean_acc, 'mode': mode_acc}
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
    
    for pred_file in eval_dir.glob("swag_epoch*_test_corrupt_predictions.pt"):
        epoch = _extract_epoch_from_filename(pred_file.name)
        if epoch is not None:
            try:
                preds = torch.load(pred_file, map_location='cpu')
                if 'mean' in preds and 'mode' in preds:
                    mean_acc = _compute_accuracy(preds['mean'], test_corrupt_targets)
                    mode_acc = _compute_accuracy(preds['mode'], test_corrupt_targets)
                    data['swag']['test_corrupt'][epoch] = {'mean': mean_acc, 'mode': mode_acc}
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
    
    return data


def _load_test_targets():
    """Load test dataset targets - this is a placeholder, you may need to adjust based on your data loading."""
    # This should load the actual test targets from your dataset
    # For now, returning None - you'll need to implement based on your dataloader structure
    from utils.dataloaders import build_test_dataloaders
    from utils.transforms import get_transform, get_corrupt_transform
    
    try:
        test_dataloader, _ = build_test_dataloaders(
            "data/MNIST",  # Adjust path as needed
            batch_size=256,
            transform=get_transform(),
            corrupt_transform=get_corrupt_transform(),
        )
        
        targets = []
        for _, target in test_dataloader:
            targets.append(target)
        return torch.cat(targets, dim=0)
    except:
        print("Warning: Could not load test targets automatically")
        return None


def _load_test_corrupt_targets():
    """Load test corrupt dataset targets."""
    from utils.dataloaders import build_test_dataloaders
    from utils.transforms import get_transform, get_corrupt_transform
    
    try:
        _, test_corrupt_dataloader = build_test_dataloaders(
            "data/MNIST",  # Adjust path as needed
            batch_size=256,
            transform=get_transform(),
            corrupt_transform=get_corrupt_transform(),
        )
        
        targets = []
        for _, target in test_corrupt_dataloader:
            targets.append(target)
        return torch.cat(targets, dim=0)
    except:
        print("Warning: Could not load test corrupt targets automatically")
        return None


def _extract_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch number from filename."""
    import re
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else None


def _compute_accuracy(predictions, targets):
    """Compute accuracy given predictions and targets."""
    if targets is None:
        return 0.0
    
    if len(predictions) != len(targets):
        print(f"Warning: Prediction length {len(predictions)} != target length {len(targets)}")
        return 0.0
    
    # Convert targets to class indices if needed
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)
    
    # Ensure predictions are class indices
    if predictions.dim() == 2:
        predictions = predictions.argmax(dim=1)
    
    return (predictions == targets).float().mean().item()


def _plot_test_dataset_accuracies(ax, accuracy_data: Dict, dataset: str, title: str):
    """Plot accuracies for a specific test dataset."""
    pretrain_data = accuracy_data['pretrain'][dataset]
    swag_data = accuracy_data['swag'][dataset]
    
    # Plot pretrain accuracies
    if pretrain_data:
        epochs = sorted(pretrain_data.keys())
        mean_accs = [pretrain_data[ep]['mean'] for ep in epochs]
        mode_accs = [pretrain_data[ep]['mode'] for ep in epochs]
        
        ax.plot(epochs, mean_accs, 'o-', label='Pretrain Mean', linewidth=2, markersize=4, alpha=0.8)
        ax.plot(epochs, mode_accs, 's-', label='Pretrain Mode', linewidth=2, markersize=4, alpha=0.8)
    
    # Plot SWAG accuracies
    if swag_data:
        epochs = sorted(swag_data.keys())
        mean_accs = [swag_data[ep]['mean'] for ep in epochs]
        mode_accs = [swag_data[ep]['mode'] for ep in epochs]
        
        ax.plot(epochs, mean_accs, '^-', label='SWAG Mean', linewidth=2, markersize=4, alpha=0.8)
        ax.plot(epochs, mode_accs, 'v-', label='SWAG Mode', linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def _plot_accuracy_comparison(ax, accuracy_data: Dict, acc_type: str, title: str):
    """Plot comparison of specific accuracy type across datasets."""
    datasets = ['test', 'test_corrupt']
    colors = ['blue', 'red']
    
    for dataset, color in zip(datasets, colors):
        # Pretrain data
        pretrain_data = accuracy_data['pretrain'][dataset]
        if pretrain_data:
            epochs = sorted(pretrain_data.keys())
            accs = [pretrain_data[ep][acc_type] for ep in epochs]
            ax.plot(epochs, accs, 'o-', color=color, alpha=0.7, 
                   label=f'Pretrain {dataset.replace("_", " ").title()}', linewidth=2, markersize=4)
        
        # SWAG data
        swag_data = accuracy_data['swag'][dataset]
        if swag_data:
            epochs = sorted(swag_data.keys())
            accs = [swag_data[ep][acc_type] for ep in epochs]
            ax.plot(epochs, accs, 's-', color=color, alpha=0.9, 
                   label=f'SWAG {dataset.replace("_", " ").title()}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def _plot_multi_optimizer_accuracy(ax, all_accuracy_data: Dict, dataset: str, acc_type: str, title: str):
    """Plot accuracy comparison across optimizers."""
    for opt_name, accuracy_data in all_accuracy_data.items():
        # Plot SWAG data (more interesting than pretrain)
        swag_data = accuracy_data['swag'][dataset]
        if swag_data:
            epochs = sorted(swag_data.keys())
            accs = [swag_data[ep][acc_type] for ep in epochs]
            ax.plot(epochs, accs, 'o-', label=opt_name, linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


if __name__ == "__main__":
    # Example usage
    optimizer = "adam"
    
    # # Plot single optimizer
    # plot_training_losses(optimizer, show_plot=False)
    
    # # Generate summary table
    # summary = generate_loss_summary_table(optimizer)
    # print(summary)
    
    # Compare multiple optimizers losses
    optimizers = ["adam", "muon", "adamw", "10p", "muon10p", "muonspectralnorm", "spectralnorm"]
    plot_all_optimizers_comparison(optimizers, show_plot=False)
    
    # Plot single optimizer test accuracies
    plot_test_accuracies(optimizer, show_plot=False)
    
    # Compare multiple optimizers accuracies
    plot_all_optimizers_accuracy_comparison(optimizers, show_plot=False)

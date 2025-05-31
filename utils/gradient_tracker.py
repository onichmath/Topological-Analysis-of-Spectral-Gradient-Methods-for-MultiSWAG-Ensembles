import torch
import os
from typing import Dict, List

class GradientTracker:
    def __init__(self, save_dir: str, optimizer_name: str, 
                 max_batches_tracked: int = 50):
        self.save_dir = save_dir
        self.optimizer_name = optimizer_name
        self.gradients_dir = os.path.join(save_dir, optimizer_name, "gradients")
        os.makedirs(self.gradients_dir, exist_ok=True)
        
        # Configuration for memory management
        self.max_batches_tracked = max_batches_tracked    # Limit batches tracked per epoch
        
        # For per-epoch norm statistics only
        self.epoch_norms = {}      # Store gradient norms for current epoch
        self.current_epoch = None
        self.batch_count = 0
    
    def compute_fast_spectral_norm(self, grad_2d, max_iter=10):
        """Power iteration method for fast spectral norm approximation.
        
        Uses 10 iterations which provides good accuracy while being 10-100x faster than SVD.
        For most neural network gradients, 10 iterations gives spectral norm accuracy within ~1% of SVD.
        """
        if grad_2d.numel() == 0:
            return 0.0
        
        m, n = grad_2d.shape
        if min(m, n) == 0:
            return 0.0
        
        # Power iteration (much faster than SVD)
        u = torch.randn(m, 1, device=grad_2d.device, dtype=grad_2d.dtype)
        
        for _ in range(max_iter):
            # v = A^T u / ||A^T u||
            v = grad_2d.t() @ u
            v_norm = torch.norm(v)
            if v_norm > 1e-8:
                v = v / v_norm
            else:
                break
                
            # u = A v / ||A v||
            u = grad_2d @ v
            u_norm = torch.norm(u)
            if u_norm > 1e-8:
                u = u / u_norm
            else:
                break
        
        # Compute spectral norm: u^T A v
        spectral_norm = (u.t() @ grad_2d @ v).item()
        return abs(spectral_norm)
    
    def compute_gradient_norms(self, model):
        """Compute L2 and spectral norms for gradients using fast methods."""
        norms = {}
        total_l2_norm_sq = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # L2 norm of this parameter's gradient
                l2_norm = torch.norm(grad, p=2).item()
                norms[f"{name}_l2"] = l2_norm
                total_l2_norm_sq += l2_norm ** 2
                
                # Fast spectral norm for matrix/tensor gradients
                if grad.dim() >= 2:
                    # Reshape to 2D for spectral norm computation
                    grad_2d = grad.view(grad.size(0), -1)
                    spectral_norm = self.compute_fast_spectral_norm(grad_2d)
                    norms[f"{name}_spectral"] = spectral_norm
                else:
                    # For 1D tensors (biases), spectral norm is just L2 norm
                    norms[f"{name}_spectral"] = l2_norm
        
        # Overall L2 norm across all parameters
        norms['total_l2'] = total_l2_norm_sq ** 0.5
        
        return norms
    
    def start_epoch_tracking(self, epoch: int):
        """Start tracking gradients for a new epoch."""
        self.current_epoch = epoch
        self.epoch_norms = {}
        self.batch_count = 0
    
    def accumulate_batch_gradients(self, model, particle_id: int, batch_id: int):
        """Accumulate gradient norms from a batch for epoch-level statistics."""
        if self.current_epoch is None:
            return
        
        # Skip if we've reached the batch limit
        if self.batch_count >= self.max_batches_tracked:
            return
            
        particle_key = f"particle_{particle_id}"
        if particle_key not in self.epoch_norms:
            self.epoch_norms[particle_key] = []
        
        # Only compute and store norms (much smaller than full gradients)
        batch_norms = self.compute_gradient_norms(model)
        self.epoch_norms[particle_key].append(batch_norms)
        
        self.batch_count += 1
    
    def compute_epoch_statistics(self):
        """Compute mean and variance of gradient norms for the current epoch."""
        if self.current_epoch is None:
            return
        
        epoch_stats = {}
        
        for particle_key in self.epoch_norms.keys():
            batch_norms_list = self.epoch_norms[particle_key]
            
            if not batch_norms_list:
                continue
                
            particle_stats = {}
            
            # === NORM STATISTICS ===
            # Compute statistics for L2 and spectral norms
            norm_stats = {}
            
            # Get all norm names from first batch
            if batch_norms_list:
                norm_names = list(batch_norms_list[0].keys())
                
                for norm_name in norm_names:
                    # Extract norm values across all batches
                    norm_values = [batch_norms[norm_name] for batch_norms in batch_norms_list if norm_name in batch_norms]
                    
                    if norm_values:
                        norm_tensor = torch.tensor(norm_values)
                        
                        norm_stats[norm_name] = {
                            'mean': torch.mean(norm_tensor).item(),
                            'var': torch.var(norm_tensor).item(),
                            'std': torch.std(norm_tensor).item(),
                            'min': torch.min(norm_tensor).item(),
                            'max': torch.max(norm_tensor).item(),
                            'num_batches': len(norm_values)
                        }
            
            particle_stats['norm_statistics'] = norm_stats
            epoch_stats[particle_key] = particle_stats
        
        # Save epoch statistics
        self.save_epoch_statistics(epoch_stats)
        
        # Clear accumulated data to free memory
        self.epoch_norms.clear()
        self.current_epoch = None
        self.batch_count = 0
        
        return epoch_stats
    
    def save_epoch_statistics(self, epoch_stats: Dict):
        """Save the computed epoch statistics."""
        if self.current_epoch is None:
            return
        
        # Save compact summary statistics (only norms)
        summary_stats = {}
        for particle_key, particle_stats in epoch_stats.items():
            if 'norm_statistics' in particle_stats:
                summary_stats[particle_key] = particle_stats['norm_statistics']
        
        # Save norm statistics (this is now the main file)
        norms_filepath = os.path.join(self.gradients_dir, f"epoch{self.current_epoch}_norm_stats.pt")
        torch.save(summary_stats, norms_filepath)
        
        print(f"Saved gradient norm statistics for epoch {self.current_epoch} ({self.batch_count} batches tracked)")

class LossLandscapeTracker:
    def __init__(self, save_dir: str, optimizer_name: str):
        self.save_dir = save_dir
        self.optimizer_name = optimizer_name
        self.landscape_dir = os.path.join(save_dir, optimizer_name, "loss_landscape")
        os.makedirs(self.landscape_dir, exist_ok=True)
    
    def save_loss_components(self, train_loss: float, val_loss: float, 
                           particle_id: int, epoch: int, additional_metrics: Dict = None):
        """Save detailed loss information."""
        loss_data = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'particle_id': particle_id
        }
        
        if additional_metrics:
            loss_data.update(additional_metrics)
        
        filepath = os.path.join(self.landscape_dir, f"particle{particle_id}_epoch{epoch}_losses.pt")
        torch.save(loss_data, filepath)

def setup_comprehensive_tracking(optimizer_name: str):
    """Set up tracking utilities for an optimizer (optimized for speed)."""
    base_dir = "results"
    
    return {
        'gradient_tracker': GradientTracker(base_dir, optimizer_name, max_batches_tracked=50),
        'landscape_tracker': LossLandscapeTracker(base_dir, optimizer_name)
        # Note: Weight distance tracking removed for performance
    } 
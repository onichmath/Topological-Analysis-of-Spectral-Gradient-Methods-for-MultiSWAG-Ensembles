import torch
from torch.utils.data import DataLoader
from push.bayes.swag import MultiSWAG
from push.particle import Particle
import os
import gc
from tqdm import tqdm
from models.MLP import MLP

from utils.dataloaders import build_test_dataloaders, build_train_dataloaders
from utils.transforms import get_corrupt_transform, get_transform


def evaluate_predictions(preds: dict, dataloader: DataLoader, label=""):
    if isinstance(dataloader.dataset, torch.utils.data.Subset):
        base_dataset = dataloader.dataset.dataset
        indices = dataloader.dataset.indices
        targets = torch.tensor(base_dataset.targets)[indices]
    else:
        targets = dataloader.dataset.targets

    mean_pred = preds["mean"]
    mode_pred = preds["mode"]

    mean_acc = (mean_pred == targets).float().mean()
    mode_acc = (mode_pred == targets).float().mean()

    print(f"[{label}] Mean Accuracy: {mean_acc:.4f}")
    print(f"[{label}] Mode Accuracy: {mode_acc:.4f}")

def run_posterior_eval(
    mswag: MultiSWAG, num_samples: int, dataloader: DataLoader, label: str
):
    preds = mswag.posterior_pred(
        dataloader,
        num_samples=num_samples,
        f_reg=False,
        loss_fn=torch.nn.CrossEntropyLoss(),
        mode=["mean", "mode", "std", "logits", "prob"],
    )
    #evaluate_predictions(preds, dataloader, label)
    return preds

def compute_loss(mswag, dataloader, num_samples):
    total_loss = 0
    num_batches = 0
    
    for data, target in dataloader:
        preds = mswag.posterior_pred(
            data,
            num_samples=num_samples,
            mode=["mean"],
            f_reg=False
        )
        
        loss = torch.nn.CrossEntropyLoss()(preds["mean"], target)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def _clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _move_to_cpu_and_cleanup(mswag):
    """Move model to CPU and cleanup GPU memory."""
    try:
        # For MultiSWAG, use the inherited cleanup method from Infer
        # The particles are managed by push_dist, not directly accessible
        if hasattr(mswag, 'push_dist') and hasattr(mswag.push_dist, '_cleanup'):
            mswag.push_dist._cleanup()
        
        # Delete the mswag object
        del mswag
        
        # Clear GPU memory
        _clear_gpu_memory()
        
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
        _clear_gpu_memory()

def evaluate_all_epochs(args):
    """
    Evaluate all trained epochs by loading weights/moments and computing predictions on test sets.
    
    Args:
        args: Arguments containing model and evaluation parameters
    """
    # Clear GPU memory at start
    _clear_gpu_memory()
    
    # Build test dataloaders
    test_dataloader, test_corrupt_dataloader = build_test_dataloaders(
        args.data_dir,
        args.batch_size,
        transform=get_transform(),
        corrupt_transform=get_corrupt_transform(),
    )
    
    # Build train dataloader for training loss computation
    train_dataloader, _, _ = build_train_dataloaders(
        args.data_dir,
        args.batch_size,
        val_size=args.val_size,
        transform=get_transform(),
        corrupt_transform=get_corrupt_transform(),
        seed=args.seed,
    )
    
    model_args = (
        {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "num_hidden_layers": args.num_hidden_layers,
        },
    )
    
    # Evaluation parameters
    eval_params = {
        'scale': getattr(args, 'scale', 1.0),
        'num_samples': getattr(args, 'num_samples', 10),
        'mode': ['mean', 'mode', 'std', 'logits', 'prob'],
        'f_reg': False,
    }
    
    # Create results directories
    os.makedirs(f"results/{args.optimizer}/evaluation_results", exist_ok=True)
    
    # Evaluate pretrain epochs
    print("Evaluating pretrain epochs...")
    for epoch in tqdm(range(args.pretrain_epochs + 1)):  # +1 to include epoch 0
        _evaluate_pretrain_epoch(
            epoch, args.optimizer, model_args, test_dataloader, test_corrupt_dataloader,
            train_dataloader, eval_params, args.num_models
        )
        # Clear memory after each epoch
        _clear_gpu_memory()
    
    # Evaluate SWAG epochs  
    print("Evaluating SWAG epochs...")
    for epoch in tqdm(range(args.swag_epochs + 1)):  # +1 to include epoch 0
        _evaluate_swag_epoch(
            epoch, args.optimizer, model_args, test_dataloader, test_corrupt_dataloader,
            train_dataloader, eval_params, args.num_models
        )
        # Clear memory after each epoch
        _clear_gpu_memory()
    
    print(f"Evaluation complete! Results saved in results/{args.optimizer}/evaluation_results/")


def _evaluate_pretrain_epoch(epoch, optimizer, model_args, test_dataloader, test_corrupt_dataloader, 
                           train_dataloader, eval_params, num_models):
    """Evaluate a specific pretrain epoch."""
    mswag = None
    try:
        # Load pretrained weights
        mswag = MultiSWAG(MLP, *model_args)
        mswag.load_from_pretrained_epoch(
            opt=optimizer,
            epoch=epoch,
            num_models=num_models,
        )
        
        # Compute training loss (using pretrained ensemble, not SWAG sampling)
        train_loss = _compute_pretrain_loss(mswag, train_dataloader, eval_params)
        
        # Save training loss
        _save_training_loss(train_loss, optimizer, epoch, "pretrain")
        
        # Evaluate on test sets using simple ensemble (no SWAG sampling for pretrain)
        test_preds = _run_pretrain_eval(mswag, test_dataloader, eval_params)
        test_corrupt_preds = _run_pretrain_eval(mswag, test_corrupt_dataloader, eval_params)
        
        # Save predictions
        _save_predictions(test_preds, optimizer, epoch, "pretrain", "test")
        _save_predictions(test_corrupt_preds, optimizer, epoch, "pretrain", "test_corrupt")
        
    except FileNotFoundError:
        print(f"Warning: Pretrain epoch {epoch} weights not found, skipping...")
    except Exception as e:
        print(f"Error evaluating pretrain epoch {epoch}: {e}")
    finally:
        # Always cleanup, even if there was an error
        if mswag is not None:
            _move_to_cpu_and_cleanup(mswag)


def _evaluate_swag_epoch(epoch, optimizer, model_args, test_dataloader, test_corrupt_dataloader,
                        train_dataloader, eval_params, num_models):
    """Evaluate a specific SWAG epoch."""
    mswag = None
    try:
        # Load SWAG moments
        mswag = MultiSWAG(MLP, *model_args)
        mswag.load_from_swag_epoch(
            opt=optimizer,
            epoch=epoch,
            num_models=num_models,
        )
        
        # Compute training loss (using SWAG mean)
        train_loss = _compute_swag_loss(mswag, train_dataloader, eval_params)
        
        # Save training loss
        _save_training_loss(train_loss, optimizer, epoch, "swag")
        
        # Evaluate on test sets using SWAG sampling
        if eval_params['f_reg']:
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        test_preds = mswag.posterior_pred(
            test_dataloader,
            loss_fn=loss_fn,
            num_samples=eval_params['num_samples'],
            scale=eval_params['scale'],
            var_clamp=1e-30,
            mode=eval_params['mode'],
            f_reg=eval_params['f_reg']
        )
        
        test_corrupt_preds = mswag.posterior_pred(
            test_corrupt_dataloader,
            loss_fn=loss_fn,
            num_samples=eval_params['num_samples'],
            scale=eval_params['scale'],
            var_clamp=1e-30,
            mode=eval_params['mode'],
            f_reg=eval_params['f_reg']
        )
        
        # Save predictions
        _save_predictions(test_preds, optimizer, epoch, "swag", "test")
        _save_predictions(test_corrupt_preds, optimizer, epoch, "swag", "test_corrupt")
        
    except FileNotFoundError:
        print(f"Warning: SWAG epoch {epoch} moments not found, skipping...")
    except Exception as e:
        print(f"Error evaluating SWAG epoch {epoch}: {e}")
    finally:
        # Always cleanup, even if there was an error
        if mswag is not None:
            _move_to_cpu_and_cleanup(mswag)


def _compute_pretrain_loss(mswag, dataloader, eval_params):
    """Compute loss for pretrained model (simple ensemble, no SWAG)."""
    # Use the full dataloader to get predictions, then compute loss separately
    if eval_params['f_reg']:
        loss_fn = torch.nn.MSELoss()
        modes = ["mean"]
    else:
        # Use a loss function that won't cause shape issues during internal computation
        loss_fn = torch.nn.CrossEntropyLoss()
        # Only request logits to avoid internal loss computation issues
        modes = ["logits"]

    
    preds = mswag.posterior_pred(
        dataloader,
        loss_fn=loss_fn,
        num_samples=1,
        var_clamp=1e-30,
        mode=modes,
        f_reg=eval_params['f_reg']
    )
    
    all_targets = []
    for _, target in dataloader:
        all_targets.append(target)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute loss based on task type
    if eval_params['f_reg']:
        # Regression: use mean predictions
        loss = torch.nn.MSELoss()(preds["mean"], all_targets)
    else:
        # Classification: use logits for CrossEntropyLoss
        logits = preds["logits"]
        
        # Handle the extra samples dimension: logits might be [batch_size, num_samples, num_classes]
        # We need [batch_size, num_classes] for CrossEntropyLoss
        if logits.dim() == 3:
            # Average across the samples dimension
            logits = logits.mean(dim=1)
        
        # Ensure target is the right shape and type for CrossEntropyLoss
        if all_targets.dim() == 2:
            # Targets are one-hot encoded, convert to class indices
            all_targets = all_targets.argmax(dim=1)
        elif all_targets.dim() == 1 and all_targets.dtype == torch.float:
            # Targets might be float indices, convert to long
            all_targets = all_targets.long()
        
        # Final check: ensure target is long type for CrossEntropyLoss
        if all_targets.dtype != torch.long:
            all_targets = all_targets.long()
        
        loss = torch.nn.CrossEntropyLoss()(logits, all_targets)
    
    return loss.item()


def _compute_swag_loss(mswag, dataloader, eval_params):
    """Compute loss for SWAG model using mean weights."""
    # Use the full dataloader to get predictions, then compute loss separately
    if eval_params['f_reg']:
        loss_fn = torch.nn.MSELoss()
        modes = ["mean"]
    else:
        # Use a loss function that won't cause shape issues during internal computation
        loss_fn = torch.nn.CrossEntropyLoss()
        # Only request logits to avoid internal loss computation issues
        modes = ["logits"]
    
    preds = mswag.posterior_pred(
        dataloader,
        loss_fn=loss_fn,
        num_samples=1,  # Just use mean
        # scale=0.0,      # No sampling, just mean
        var_clamp=1e-30,
        mode=modes,
        f_reg=eval_params['f_reg']
    )
    
    # Get targets from the dataloader
    all_targets = []
    for _, target in dataloader:
        all_targets.append(target)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute loss based on task type
    if eval_params['f_reg']:
        # Regression: use mean predictions
        loss = torch.nn.MSELoss()(preds["mean"], all_targets)
    else:
        # Classification: use logits for CrossEntropyLoss
        logits = preds["logits"]
        
        # Handle the extra samples dimension: logits might be [batch_size, num_samples, num_classes]
        # We need [batch_size, num_classes] for CrossEntropyLoss
        if logits.dim() == 3:
            # Average across the samples dimension
            logits = logits.mean(dim=1)
        
        # Ensure target is the right shape and type for CrossEntropyLoss
        if all_targets.dim() == 2:
            # Targets are one-hot encoded, convert to class indices
            all_targets = all_targets.argmax(dim=1)
        elif all_targets.dim() == 1 and all_targets.dtype == torch.float:
            # Targets might be float indices, convert to long
            all_targets = all_targets.long()
        
        # Final check: ensure target is long type for CrossEntropyLoss
        if all_targets.dtype != torch.long:
            all_targets = all_targets.long()
        
        loss = torch.nn.CrossEntropyLoss()(logits, all_targets)
    
    return loss.item()


def _run_pretrain_eval(mswag, dataloader, eval_params):
    """Run evaluation for pretrained model (simple ensemble)."""
    # Use the full dataloader directly instead of processing batch by batch
    # This ensures consistent handling with the SWAG evaluation
    if eval_params['f_reg']:
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    preds = mswag.posterior_pred(
        dataloader,
        loss_fn=loss_fn,
        num_samples=1,
        # scale=0.0,  # No sampling for pretrain
        var_clamp=1e-30,
        mode=eval_params['mode'],
        f_reg=eval_params['f_reg']
    )
    
    return preds


def _run_pretrain_eval_batch(mswag, data, eval_params):
    """Run evaluation for a single batch using pretrained ensemble."""
    # Create a temporary single-batch dataloader to ensure proper handling
    batch_dataset = torch.utils.data.TensorDataset(data, torch.zeros(len(data)))  # dummy targets
    batch_dataloader = torch.utils.data.DataLoader(batch_dataset, batch_size=len(data), shuffle=False)
    
    if eval_params['f_reg']:
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # Get predictions from all particles (simple ensemble, no SWAG sampling)
    preds = mswag.posterior_pred(
        batch_dataloader,
        loss_fn=loss_fn,
        num_samples=1,
        # scale=0.0,  # No sampling for pretrain
        var_clamp=1e-30,
        mode=eval_params['mode'],
        f_reg=eval_params['f_reg']
    )
    return preds


def _save_training_loss(loss, optimizer, epoch, phase):
    """Save training loss to disk."""
    loss_data = {
        'epoch': epoch,
        'phase': phase,
        'loss': loss
    }
    
    save_dir = f"results/{optimizer}/evaluation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(
        loss_data,
        os.path.join(save_dir, f"{phase}_epoch{epoch}_loss.pt")
    )


def _save_predictions(pred_dict, optimizer, epoch, phase, dataset_name):
    """Save prediction dictionary to disk."""
    save_dir = f"results/{optimizer}/evaluation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(
        pred_dict,
        os.path.join(save_dir, f"{phase}_epoch{epoch}_{dataset_name}_predictions.pt")
    )
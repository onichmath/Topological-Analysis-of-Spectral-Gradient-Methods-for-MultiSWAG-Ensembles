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
    # print("Evaluating pretrain epochs...")
    # for epoch in tqdm(range(args.pretrain_epochs + 1)):  # +1 to include epoch 0
    #     _evaluate_pretrain_epoch(
    #         epoch, args.optimizer, model_args, test_dataloader, test_corrupt_dataloader,
    #         train_dataloader, eval_params, args.num_models
    #     )
    #     # Clear memory after each epoch
    #     _clear_gpu_memory()
    
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


def _compute_pretrain_loss_simple_ensemble(optimizer, epoch, model_args, dataloader, eval_params, num_models):
    """Compute pretrain loss using simple ensemble of individual models (no MultiSWAG)."""
    from models.MLP import MLP
    import os
    
    # Create individual model instances
    models = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_num in range(num_models):
        # Load weights for this model
        weights_dir = os.path.join("results", optimizer, "pretrain_weights")
        weights_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_weights.pt")
        
        if not os.path.exists(weights_path):
            print(f"Warning: Weights not found for model {model_num} epoch {epoch}")
            continue
            
        # Create model and load weights
        model = MLP(model_args[0]).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    
    if len(models) == 0:
        print(f"No valid models found for epoch {epoch}")
        return 0.0
    
    print(f"Debug: Loaded {len(models)} models for ensemble evaluation")
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            # Get predictions from all models
            all_preds = []
            for model in models:
                pred = model(data)
                all_preds.append(pred)
            
            # Average predictions across ensemble
            ensemble_pred = torch.stack(all_preds).mean(dim=0)
            
            # Check for NaN
            if torch.isnan(ensemble_pred).any():
                print(f"Warning: NaN in ensemble predictions, skipping batch")
                continue
            
            # Compute loss
            if eval_params['f_reg']:
                loss = torch.nn.MSELoss()(ensemble_pred, target)
            else:
                if target.dim() == 2:
                    target = target.argmax(dim=1)
                elif target.dtype != torch.long:
                    target = target.long()
                loss = torch.nn.CrossEntropyLoss()(ensemble_pred, target)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
    
    # Cleanup models
    del models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def _run_pretrain_eval_simple_ensemble(optimizer, epoch, model_args, dataloader, eval_params, num_models):
    """Run pretrain evaluation using simple ensemble of individual models (no MultiSWAG)."""
    from models.MLP import MLP
    import os
    
    # Create individual model instances
    models = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_num in range(num_models):
        # Load weights for this model
        weights_dir = os.path.join("results", optimizer, "pretrain_weights")
        weights_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_weights.pt")
        
        if not os.path.exists(weights_path):
            print(f"Warning: Weights not found for model {model_num} epoch {epoch}")
            continue
            
        # Create model and load weights
        model = MLP(model_args[0]).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    
    if len(models) == 0:
        print(f"No valid models found for epoch {epoch}")
        # Return dummy predictions
        dummy_preds = {}
        for mode in eval_params['mode']:
            if mode == "logits":
                dummy_preds[mode] = torch.zeros(50000, 10)
            elif mode in ["mean", "mode"]:
                dummy_preds[mode] = torch.zeros(50000, dtype=torch.long)
            else:
                dummy_preds[mode] = torch.zeros(50000, 10)
        return dummy_preds
    
    print(f"Debug: Loaded {len(models)} models for ensemble evaluation")
    
    all_preds = {mode: [] for mode in eval_params['mode']}
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # Get predictions from all models
            batch_preds = []
            for model in models:
                pred = model(data)
                batch_preds.append(pred)
            
            # Stack predictions: [num_models, batch_size, num_classes]
            stacked_preds = torch.stack(batch_preds)
            
            # Check for NaN
            if torch.isnan(stacked_preds).any():
                print(f"Warning: NaN in batch predictions, skipping batch")
                continue
            
            # Compute ensemble statistics
            batch_results = {}
            
            if eval_params['f_reg']:
                # Regression modes
                if "mean" in eval_params['mode']:
                    batch_results["mean"] = stacked_preds.mean(dim=0)
                if "std" in eval_params['mode']:
                    batch_results["std"] = stacked_preds.std(dim=0)
                if "mode" in eval_params['mode']:
                    batch_results["mode"] = stacked_preds.mean(dim=0)
                if "logits" in eval_params['mode']:
                    batch_results["logits"] = stacked_preds.mean(dim=0)
                if "prob" in eval_params['mode']:
                    batch_results["prob"] = stacked_preds.mean(dim=0)
            else:
                # Classification modes
                if "logits" in eval_params['mode']:
                    batch_results["logits"] = stacked_preds.mean(dim=0)
                if "mean" in eval_params['mode']:
                    batch_results["mean"] = stacked_preds.mean(dim=0).argmax(dim=1)
                if "mode" in eval_params['mode']:
                    # Get mode across ensemble predictions
                    pred_classes = stacked_preds.argmax(dim=2)  # [num_models, batch_size]
                    batch_results["mode"] = torch.mode(pred_classes, dim=0).values
                if "prob" in eval_params['mode']:
                    # Average softmax probabilities
                    softmax_preds = torch.softmax(stacked_preds, dim=2)
                    batch_results["prob"] = softmax_preds.mean(dim=0)
                if "std" in eval_params['mode']:
                    softmax_preds = torch.softmax(stacked_preds, dim=2)
                    batch_results["std"] = softmax_preds.std(dim=0)
            
            # Accumulate results
            for mode in eval_params['mode']:
                if mode in batch_results:
                    all_preds[mode].append(batch_results[mode].cpu())
    
    # Concatenate all batch results
    final_preds = {}
    for mode in eval_params['mode']:
        if mode in all_preds and len(all_preds[mode]) > 0:
            final_preds[mode] = torch.cat(all_preds[mode], dim=0)
        else:
            # Create dummy predictions if no valid predictions were obtained
            if mode == "logits":
                final_preds[mode] = torch.zeros(50000, 10)
            elif mode in ["mean", "mode"]:
                final_preds[mode] = torch.zeros(50000, dtype=torch.long)
            else:
                final_preds[mode] = torch.zeros(50000, 10)
    
    # Cleanup models
    del models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return final_preds


def _evaluate_pretrain_epoch(epoch, optimizer, model_args, test_dataloader, test_corrupt_dataloader, 
                           train_dataloader, eval_params, num_models):
    """Evaluate a specific pretrain epoch using simple ensemble (no MultiSWAG)."""
    try:
        print(f"Debug: Evaluating pretrain epoch {epoch} using simple ensemble")
        
        # Compute training loss using simple ensemble
        train_loss = _compute_pretrain_loss_simple_ensemble(
            optimizer, epoch, model_args, train_dataloader, eval_params, num_models
        )
        
        # Save training loss
        _save_training_loss(train_loss, optimizer, epoch, "pretrain")
        
        # Evaluate on test sets using simple ensemble
        test_preds = _run_pretrain_eval_simple_ensemble(
            optimizer, epoch, model_args, test_dataloader, eval_params, num_models
        )
        test_corrupt_preds = _run_pretrain_eval_simple_ensemble(
            optimizer, epoch, model_args, test_corrupt_dataloader, eval_params, num_models
        )
        
        # Save predictions
        _save_predictions(test_preds, optimizer, epoch, "pretrain", "test")
        _save_predictions(test_corrupt_preds, optimizer, epoch, "pretrain", "test_corrupt")
        
        print(f"Debug: Successfully evaluated pretrain epoch {epoch}, loss: {train_loss:.6f}")
        
    except FileNotFoundError:
        print(f"Warning: Pretrain epoch {epoch} weights not found, skipping...")
    except Exception as e:
        print(f"Error evaluating pretrain epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()


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

        # TODO: Do this later
        
        # # Compute training loss (using SWAG mean)
        # train_loss = _compute_swag_loss(mswag, train_dataloader, eval_params)
        
        # # Save training loss
        # _save_training_loss(train_loss, optimizer, epoch, "swag")
        
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
        
        # Print test accuracies
        if not eval_params['f_reg'] and 'mean' in test_preds and 'mode' in test_preds:
            # Get test dataset targets
            test_targets = []
            for _, target in test_dataloader:
                test_targets.append(target)
            test_targets = torch.cat(test_targets, dim=0)
            
            # Convert targets to class indices if needed
            if test_targets.dim() == 2:
                test_targets = test_targets.argmax(dim=1)
            
            # Compute accuracies
            test_mean_acc = (test_preds["mean"] == test_targets).float().mean()
            test_mode_acc = (test_preds["mode"] == test_targets).float().mean()
            
            print(f"Debug: Test Mean Accuracy: {test_mean_acc:.4f}")
            print(f"Debug: Test Mode Accuracy: {test_mode_acc:.4f}")
            
            # Get test_corrupt dataset targets
            test_corrupt_targets = []
            for _, target in test_corrupt_dataloader:
                test_corrupt_targets.append(target)
            test_corrupt_targets = torch.cat(test_corrupt_targets, dim=0)
            
            # Convert targets to class indices if needed
            if test_corrupt_targets.dim() == 2:
                test_corrupt_targets = test_corrupt_targets.argmax(dim=1)
            
            # Compute accuracies for corrupted test set
            test_corrupt_mean_acc = (test_corrupt_preds["mean"] == test_corrupt_targets).float().mean()
            test_corrupt_mode_acc = (test_corrupt_preds["mode"] == test_corrupt_targets).float().mean()
            
            print(f"Debug: Test Corrupt Mean Accuracy: {test_corrupt_mean_acc:.4f}")
            print(f"Debug: Test Corrupt Mode Accuracy: {test_corrupt_mode_acc:.4f}")
        
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


def _compute_swag_loss_simple(optimizer, epoch, model_args, dataloader, eval_params, num_models):
    """Compute SWAG loss using mean weights only (no sampling)."""
    try:
        # Load SWAG moments and extract mean weights
        models = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for model_num in range(num_models):
            # Load SWAG first moment (mean) for this model
            swag_dir = os.path.join("results", optimizer, "swag_moments")
            mom1_path = os.path.join(swag_dir, f"particle{model_num}_epoch{epoch}_mom1.pt")
            
            if not os.path.exists(mom1_path):
                print(f"Warning: SWAG mom1 not found for model {model_num} epoch {epoch}")
                continue
            
            # Load first moment (mean weights)
            mom1_data = torch.load(mom1_path, map_location=device)
            print(f"Debug: Loaded mom1 for model {model_num}, type: {type(mom1_data)}")
            
            # Create model to get the parameter structure
            model = MLP(model_args[0]).to(device)
            
            # Convert mom1 data to state dict format
            if isinstance(mom1_data, list):
                print(f"Debug: mom1 is list with {len(mom1_data)} tensors")
                
                # Check for NaN in loaded tensors
                for i, tensor in enumerate(mom1_data):
                    if torch.isnan(tensor).any():
                        print(f"Debug: NaN found in mom1 tensor {i}")
                    print(f"Debug: Tensor {i} shape: {tensor.shape}, min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}")
                
                # mom1 is a list of tensors, need to map back to parameter names
                state_dict = {}
                param_names = list(model.state_dict().keys())
                print(f"Debug: Model has {len(param_names)} parameters: {param_names}")
                
                if len(mom1_data) == len(param_names):
                    for i, param_name in enumerate(param_names):
                        state_dict[param_name] = mom1_data[i].to(device)
                        if torch.isnan(state_dict[param_name]).any():
                            print(f"Debug: NaN in reconstructed parameter {param_name}")
                else:
                    print(f"Warning: mom1 list length {len(mom1_data)} doesn't match model params {len(param_names)}")
                    continue
                    
            elif isinstance(mom1_data, dict):
                # mom1 is already a state dict
                state_dict = mom1_data
                print(f"Debug: mom1 is dict with keys: {list(state_dict.keys())}")
                
                # Check for NaN in dict values
                for k, v in state_dict.items():
                    if torch.isnan(v).any():
                        print(f"Debug: NaN found in dict parameter {k}")
            else:
                print(f"Warning: Unexpected mom1 data type: {type(mom1_data)}")
                continue
            
            # Load the reconstructed state dict
            model.load_state_dict(state_dict)
            model.eval()
            
            # Test the model with a small sample
            with torch.no_grad():
                test_input = torch.randn(2, model_args[0]['input_dim']).to(device)
                test_output = model(test_input)
                print(f"Debug: Model {model_num} test output shape: {test_output.shape}")
                print(f"Debug: Model {model_num} test output has NaN: {torch.isnan(test_output).any()}")
                if torch.isnan(test_output).any():
                    print(f"Debug: Test output min/max: {test_output.min().item():.6f}/{test_output.max().item():.6f}")
            
            models.append(model)
            print(f"Debug: Successfully loaded SWAG mean weights for model {model_num}")
        
        if len(models) == 0:
            print(f"No valid SWAG models found for epoch {epoch}")
            return 0.0
        
        print(f"Debug: Loaded {len(models)} SWAG mean weight models")
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 3:  # Only process first few batches for debugging
                    break
                    
                data = data.to(device)
                target = target.to(device)
                print(f"Debug: Batch {batch_idx} - data shape: {data.shape}, target shape: {target.shape}")
                
                # Get predictions from all models using mean weights
                all_preds = []
                for model_idx, model in enumerate(models):
                    pred = model(data)
                    print(f"Debug: Model {model_idx} pred shape: {pred.shape}, has NaN: {torch.isnan(pred).any()}")
                    if torch.isnan(pred).any():
                        print(f"Debug: Model {model_idx} pred min/max: {pred.min().item():.6f}/{pred.max().item():.6f}")
                    all_preds.append(pred)
                
                # Average predictions across ensemble
                ensemble_pred = torch.stack(all_preds).mean(dim=0)
                print(f"Debug: Ensemble pred shape: {ensemble_pred.shape}, has NaN: {torch.isnan(ensemble_pred).any()}")
                
                # Check for NaN
                if torch.isnan(ensemble_pred).any():
                    print(f"Warning: NaN in SWAG ensemble predictions, skipping batch")
                    continue
                
                # Compute loss
                if eval_params['f_reg']:
                    loss = torch.nn.MSELoss()(ensemble_pred, target)
                else:
                    if target.dim() == 2:
                        target = target.argmax(dim=1)
                    elif target.dtype != torch.long:
                        target = target.long()
                    print(f"Debug: Final target shape: {target.shape}, dtype: {target.dtype}")
                    print(f"Debug: ensemble_pred shape: {ensemble_pred.shape}")
                    loss = torch.nn.CrossEntropyLoss()(ensemble_pred, target)
                
                print(f"Debug: Batch {batch_idx} loss: {loss.item()}")
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    print(f"Debug: NaN loss in batch {batch_idx}")
        
        # Cleanup models
        del models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        result_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Debug: SWAG simple loss computed: {result_loss}")
        return result_loss
        
    except Exception as e:
        print(f"Error in simple SWAG loss computation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def _run_swag_eval_simple(optimizer, epoch, model_args, dataloader, eval_params, num_models):
    """Run SWAG evaluation using mean weights only (no sampling)."""
    try:
        # Load SWAG moments and extract mean weights
        models = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for model_num in range(num_models):
            # Load SWAG first moment (mean) for this model
            swag_dir = os.path.join("results", optimizer, "swag_moments")
            mom1_path = os.path.join(swag_dir, f"particle{model_num}_epoch{epoch}_mom1.pt")
            
            if not os.path.exists(mom1_path):
                print(f"Warning: SWAG mom1 not found for model {model_num} epoch {epoch}")
                continue
            
            # Load first moment (mean weights)
            mom1_data = torch.load(mom1_path, map_location=device)
            
            # Create model to get the parameter structure
            model = MLP(model_args[0]).to(device)
            
            # Convert mom1 data to state dict format
            if isinstance(mom1_data, list):
                # mom1 is a list of tensors, need to map back to parameter names
                state_dict = {}
                param_names = list(model.state_dict().keys())
                
                if len(mom1_data) == len(param_names):
                    for i, param_name in enumerate(param_names):
                        state_dict[param_name] = mom1_data[i].to(device)
                else:
                    print(f"Warning: mom1 list length {len(mom1_data)} doesn't match model params {len(param_names)}")
                    continue
                    
            elif isinstance(mom1_data, dict):
                # mom1 is already a state dict
                state_dict = mom1_data
            else:
                print(f"Warning: Unexpected mom1 data type: {type(mom1_data)}")
                continue
            
            # Load the reconstructed state dict
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        
        if len(models) == 0:
            print(f"No valid SWAG models found for epoch {epoch}")
            # Return dummy predictions
            dummy_preds = {}
            for mode in eval_params['mode']:
                if mode == "logits":
                    dummy_preds[mode] = torch.zeros(50000, 10)
                elif mode in ["mean", "mode"]:
                    dummy_preds[mode] = torch.zeros(50000, dtype=torch.long)
                else:
                    dummy_preds[mode] = torch.zeros(50000, 10)
            return dummy_preds
        
        print(f"Debug: Loaded {len(models)} SWAG mean weight models for evaluation")
        
        all_preds = {mode: [] for mode in eval_params['mode']}
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                
                # Get predictions from all models using mean weights
                batch_preds = []
                for model in models:
                    pred = model(data)
                    batch_preds.append(pred)
                
                # Stack predictions: [num_models, batch_size, num_classes]
                stacked_preds = torch.stack(batch_preds)
                
                # Check for NaN
                if torch.isnan(stacked_preds).any():
                    print(f"Warning: NaN in batch predictions, skipping batch")
                    continue
                
                # Compute ensemble statistics (same as pretrain)
                batch_results = {}
                
                if eval_params['f_reg']:
                    # Regression modes
                    if "mean" in eval_params['mode']:
                        batch_results["mean"] = stacked_preds.mean(dim=0)
                    if "std" in eval_params['mode']:
                        batch_results["std"] = stacked_preds.std(dim=0)
                    if "mode" in eval_params['mode']:
                        batch_results["mode"] = stacked_preds.mean(dim=0)
                    if "logits" in eval_params['mode']:
                        batch_results["logits"] = stacked_preds.mean(dim=0)
                    if "prob" in eval_params['mode']:
                        batch_results["prob"] = stacked_preds.mean(dim=0)
                else:
                    # Classification modes
                    if "logits" in eval_params['mode']:
                        batch_results["logits"] = stacked_preds.mean(dim=0)
                    if "mean" in eval_params['mode']:
                        batch_results["mean"] = stacked_preds.mean(dim=0).argmax(dim=1)
                    if "mode" in eval_params['mode']:
                        # Get mode across ensemble predictions
                        pred_classes = stacked_preds.argmax(dim=2)  # [num_models, batch_size]
                        batch_results["mode"] = torch.mode(pred_classes, dim=0).values
                    if "prob" in eval_params['mode']:
                        # Average softmax probabilities
                        softmax_preds = torch.softmax(stacked_preds, dim=2)
                        batch_results["prob"] = softmax_preds.mean(dim=0)
                    if "std" in eval_params['mode']:
                        softmax_preds = torch.softmax(stacked_preds, dim=2)
                        batch_results["std"] = softmax_preds.std(dim=0)
                
                # Accumulate results
                for mode in eval_params['mode']:
                    if mode in batch_results:
                        all_preds[mode].append(batch_results[mode].cpu())
        
        # Concatenate all batch results
        final_preds = {}
        for mode in eval_params['mode']:
            if mode in all_preds and len(all_preds[mode]) > 0:
                final_preds[mode] = torch.cat(all_preds[mode], dim=0)
            else:
                # Create dummy predictions if no valid predictions were obtained
                if mode == "logits":
                    final_preds[mode] = torch.zeros(50000, 10)
                elif mode in ["mean", "mode"]:
                    final_preds[mode] = torch.zeros(50000, dtype=torch.long)
                else:
                    final_preds[mode] = torch.zeros(50000, 10)
        
        # Cleanup models
        del models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return final_preds
        
    except Exception as e:
        print(f"Error in simple SWAG evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy predictions
        dummy_preds = {}
        for mode in eval_params['mode']:
            if mode == "logits":
                dummy_preds[mode] = torch.zeros(50000, 10)
            elif mode in ["mean", "mode"]:
                dummy_preds[mode] = torch.zeros(50000, dtype=torch.long)
            else:
                dummy_preds[mode] = torch.zeros(50000, 10)
        return dummy_preds
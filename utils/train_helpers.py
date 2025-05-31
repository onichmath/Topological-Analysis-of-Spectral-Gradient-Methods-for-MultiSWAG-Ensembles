import torch
import os
from utils.dataloaders import build_train_dataloaders
from utils.transforms import get_transform, get_corrupt_transform
from utils.optimizers import map_create_optimizer
from models.MLP import MLP
from push.bayes.swag import train_mswag
from utils.eval_helpers import run_posterior_eval
from .gradient_tracker import setup_comprehensive_tracking


def train(args):
    """
    Train MultiSWAG model with comprehensive tracking for TDA analysis.
    
    This function implements Option 2 ensemble design:
    - random_seed=False: Same initialization across particles
    - bootstrap=True: Different data samples per particle  
    - save_metrics=True: Comprehensive metric saving for TDA analysis
    - Enhanced with gradient norm tracking (L2/spectral norm statistics only, memory efficient)
    """
    
    print(f"Training {args.optimizer} with bootstrap ensemble configuration")
    print(f"Comprehensive tracking enabled for TDA analysis (norm-only gradient tracking)")
    
    # Set up tracking utilities (gradient norm tracking, weight distances, loss landscapes)
    tracking_utils = setup_comprehensive_tracking(args.optimizer)
    
    train_dataloader, val_dataloader, val_corrupt_dataloader = build_train_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        transform=get_transform(),
        corrupt_transform=get_corrupt_transform(),
        seed=args.seed,
    )

    create_optimizer = map_create_optimizer(args.optimizer)
    
    results_dir = os.path.join(args.results_dir, args.optimizer)
    os.makedirs(results_dir, exist_ok=True)

    model_args = (
        {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "num_hidden_layers": args.num_hidden_layers,
        },
    )

    # Enhanced training with tracking
    mswag = train_mswag(
        train_dataloader,
        torch.nn.CrossEntropyLoss(),
        create_optimizer,
        args.pretrain_epochs,
        args.swag_epochs,
        MLP,
        model_args[0],
        cov_mat_rank=args.cov_mat_rank,
        num_models=args.num_models,
        f_save=False,
        random_seed=False,
        bootstrap=True,
        val_dataloader=val_dataloader,
        val_corrupt_dataloader=val_corrupt_dataloader,
        save_metrics=True,
        optimizer_name=args.optimizer,
        lr=args.lr,
        mswag_state={},
        tracking_utils=tracking_utils  # Pass tracking utilities
    )
    
    print(f"Training completed for {args.optimizer}")

    # Post-training analysis setup
    evaluation_dir = os.path.join(results_dir, "evaluation_results")
    if os.path.exists(evaluation_dir):
        import glob
        metric_files = glob.glob(os.path.join(evaluation_dir, "*_metrics.pt"))
        print(f"Created {len(metric_files)} metric files")
    
    # Compute final trajectory summaries
    print("Computing trajectory summaries for TDA analysis...")
    compute_trajectory_summaries(args.optimizer, args.pretrain_epochs, args.swag_epochs, args.num_models)

    print("Running validation evaluation...")
    run_posterior_eval(mswag, args.num_samples, val_dataloader, label="ID")
    run_posterior_eval(mswag, args.num_samples, val_corrupt_dataloader, label="OOD")

def compute_trajectory_summaries(optimizer_name: str, pretrain_epochs: int, swag_epochs: int, num_models: int):
    """Compute and save trajectory summaries for TDA analysis."""
    from .posterior_analyzer import setup_posterior_analysis
    
    analyzers = setup_posterior_analysis()
    trajectory_analyzer = analyzers['trajectory_analyzer']
    
    # Analyze weight trajectories for each particle
    trajectory_summaries = {}
    
    for particle_id in range(num_models):
        # Pretrain trajectories
        pretrain_traj = trajectory_analyzer.extract_weight_trajectories(
            optimizer_name, particle_id, pretrain_epochs
        )
        
        if len(pretrain_traj) > 0:
            pretrain_summary = trajectory_analyzer.compute_trajectory_persistence(pretrain_traj)
            trajectory_summaries[f'particle_{particle_id}_pretrain'] = pretrain_summary
    
    # Save trajectory summaries
    summary_dir = os.path.join("results", optimizer_name, "trajectory_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    torch.save(trajectory_summaries, os.path.join(summary_dir, "trajectory_summaries.pt"))
    print(f"Saved trajectory summaries to {summary_dir}/trajectory_summaries.pt")

def prepare_tda_analysis_data(optimizers: list):
    """Prepare comprehensive data for TDA analysis across all optimizers."""
    from .posterior_analyzer import setup_posterior_analysis
    
    analyzers = setup_posterior_analysis()
    posterior_analyzer = analyzers['posterior_analyzer']
    
    print("Preparing TDA analysis data...")
    
    tda_data = {
        'posterior_comparisons': {},
        'trajectory_data': {},
        'uncertainty_metrics': {}
    }
    
    # Compare posteriors between optimizers
    for i, opt1 in enumerate(optimizers):
        for j, opt2 in enumerate(optimizers[i+1:], i+1):
            try:
                opt1_posterior = posterior_analyzer.load_swag_posterior(opt1, epoch=20)  # Final SWAG epoch
                opt2_posterior = posterior_analyzer.load_swag_posterior(opt2, epoch=20)
                
                if opt1_posterior and opt2_posterior:
                    comparison = posterior_analyzer.compare_posterior_diversity(opt1_posterior, opt2_posterior)
                    tda_data['posterior_comparisons'][f'{opt1}_vs_{opt2}'] = comparison
                    print(f"Compared posteriors: {opt1} vs {opt2}")
            except Exception as e:
                print(f"Error comparing {opt1} vs {opt2}: {e}")
    
    # Save TDA preparation data
    torch.save(tda_data, "results/tda_analysis_data.pt")
    print("Saved TDA analysis preparation data to results/tda_analysis_data.pt")
    
    return tda_data

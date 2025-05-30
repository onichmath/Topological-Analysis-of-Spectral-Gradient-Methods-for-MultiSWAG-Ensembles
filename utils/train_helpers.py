import torch
import os
from utils.dataloaders import build_train_dataloaders
from utils.transforms import get_transform, get_corrupt_transform
from utils.optimizers import map_create_optimizer
from models.MLP import MLP
from push.bayes.swag import train_mswag
from utils.eval_helpers import run_posterior_eval


def train(args):
    """
    Train MultiSWAG model.
    
    This function implements Option 2 ensemble design:
    - random_seed=False: Same initialization across particles
    - bootstrap=True: Different data samples per particle  
    - save_metrics=True: Comprehensive metric saving for TDA analysis
    """
    
    print(f"Training {args.optimizer} with bootstrap ensemble configuration")
    
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

    mswag = train_mswag(
        dataloader=train_dataloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        create_optimizer=create_optimizer,
        pretrain_epochs=args.pretrain_epochs,
        swag_epochs=args.swag_epochs,
        nn=MLP,
        *model_args,
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
        mswag_state={}
    )
    
    print(f"Training completed for {args.optimizer}")

    evaluation_dir = os.path.join(results_dir, "evaluation_results")
    if os.path.exists(evaluation_dir):
        import glob
        metric_files = glob.glob(os.path.join(evaluation_dir, "*_metrics.pt"))
        print(f"Created {len(metric_files)} metric files")

    print("Running validation evaluation...")
    run_posterior_eval(mswag, args.num_samples, val_dataloader, label="ID")
    run_posterior_eval(mswag, args.num_samples, val_corrupt_dataloader, label="OOD")

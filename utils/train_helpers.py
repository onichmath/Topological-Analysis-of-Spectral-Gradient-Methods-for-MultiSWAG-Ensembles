import torch
from utils.dataloaders import build_train_dataloaders
from utils.transforms import get_transform, get_corrupt_transform
from utils.optimizers import map_create_optimizer
from models.MLP import MLP
from push.bayes.swag import train_mswag
from utils.eval_helpers import run_posterior_eval


def train(args):
    train_dataloader, val_dataloader, val_corrupt_dataloader = build_train_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        transform=get_transform(),
        corrupt_transform=get_corrupt_transform(),
        seed=args.seed,
    )

    create_optimizer = map_create_optimizer(args.optimizer)

    model_args = (
        {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "num_hidden_layers": args.num_hidden_layers,
        },
    )

    mswag = train_mswag(
        train_dataloader,
        torch.nn.CrossEntropyLoss(),
        create_optimizer,
        args.pretrain_epochs,
        args.swag_epochs,
        MLP,
        *model_args,
        cov_mat_rank=args.cov_mat_rank,
        num_models=args.num_models,
        f_save=False,
    )

    print("\nEvaluating on in-distribution validation set:")
    run_posterior_eval(mswag, args.num_samples, val_dataloader, label="ID")

    print("\nEvaluating on out-of-distribution corrupted validation set:")
    run_posterior_eval(mswag, args.num_samples, val_corrupt_dataloader, label="OOD")

from torch.utils.data import DataLoader
import torch
from models.MLP import MLP
from utils.parser import build_parser
from utils.transforms import get_transform, get_corrupt_transform
from utils.dataloaders import build_train_dataloaders
from utils.optimizers import create_adam_optimizer, create_muon_optimizer
from push.bayes.swag import MultiSWAG, train_mswag


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
    evaluate_predictions(preds, dataloader, label)


def main():
    """
    Main function for training MultiSWAG model and evaluating on ID and OOD validation datasets.
    """
    # TODO: evaluate on train set for epistemic uncertainty
    # TODO: evaluate on test set for aleatoric uncertainty
    # Note: p_params is list of [tensor(num_models, model_params, layer)]
    args = build_parser().parse_args()

    train_dataloader, val_dataloader, val_corrupt_dataloader = build_train_dataloaders(
        data_dir="./data",
        batch_size=args.batch_size,
        val_size=args.val_size,
        transform=get_transform(),
        corrupt_transform=get_corrupt_transform(),
        seed=args.seed,
    )

    create_optimizer = (
        create_muon_optimizer if args.optimizer == "muon" else create_adam_optimizer
    )

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
    # Pac Bayes vs Bayesian, is this paper testing agaisnt multiswag or just MCMC?
    # https://arxiv.org/html/2406.05469v1#S3


if __name__ == "__main__":
    main()

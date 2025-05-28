import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
from models.MLP import MLP
from utils.parser import build_parser
from experiments.baseline import baseline
from utils.transforms import get_transform, get_corrupt_transform
from utils.dataloaders import build_train_dataloaders, build_test_dataloaders
from utils.optimizers import create_adam_optimizer, create_muon_optimizer
from push.bayes.swag import MultiSWAG, train_mswag
from torchvision.transforms import Compose


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


def run_posterior_eval(mswag: MultiSWAG, dataloader: DataLoader, label: str):
    preds = mswag.posterior_pred(
        dataloader,
        num_samples=2,
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
        create_adam_optimizer,
        args.pretrain_epochs,
        args.swag_epochs,
        MLP,
        *model_args,
        cov_mat_rank=args.cov_mat_rank,
        num_models=args.num_models,
        f_save=False,
    )

    print("\nEvaluating on in-distribution validation set:")
    run_posterior_eval(mswag, val_dataloader, label="ID")

    print("\nEvaluating on out-of-distribution corrupted validation set:")
    run_posterior_eval(mswag, val_corrupt_dataloader, label="OOD")
    # Pac Bayes vs Bayesian, is this paper testing agaisnt multiswag or just MCMC?
    # https://arxiv.org/html/2406.05469v1#S3


if __name__ == "__main__":
    main()

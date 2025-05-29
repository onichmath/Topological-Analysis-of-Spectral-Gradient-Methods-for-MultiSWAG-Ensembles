import torch
from models.MLP import MLP
from utils.parser import build_parser
from utils.transforms import get_transform, get_corrupt_transform
from utils.dataloaders import build_train_dataloaders
from utils.optimizers import map_create_optimizer
from push.bayes.swag import train_mswag
from utils.eval import run_posterior_eval

# TODO: function space filtration on test + test corrupt
# TODO: train set evaluation for epistemic uncertainty + test set for aleatoric uncertainty
# TODO: weight space filtration over time (delta?)
# TODO: cov mat as distance matrix or PCD?
# TODO: eNTK?


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
    # Pac Bayes vs Bayesian, is this paper testing agaisnt multiswag or just MCMC?
    # https://arxiv.org/html/2406.05469v1#S3


if __name__ == "__main__":
    main()

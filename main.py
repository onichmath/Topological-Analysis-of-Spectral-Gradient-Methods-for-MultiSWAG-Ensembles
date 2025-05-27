import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
from models.MLP import MLP
from utils.parser import build_parser
from experiments.baseline import baseline
from utils.transforms import get_transform, get_corrupt_transform
from push.bayes.swag import MultiSWAG, train_mswag


def create_optimizer(lr):
    """
    Create a function that returns Adam optimizer with a specific learning rate.

    Args:
        lr (float): Learning rate for the optimizer.

    Returns:
        function: Function that generates Adam optimizer with the specified learning rate.
    """

    def mk_optim(params):
        """
        Returns Adam optimizer with the specified learning rate.

        Args:
            params: Model parameters.

        Returns:
            torch.optim.Adam: Adam optimizer.
        """
        return torch.optim.Adam(params, lr=lr, weight_decay=lr / 1e3)

    return mk_optim


def main():
    parser = build_parser()
    args = parser.parse_args()

    transform = get_transform()
    corrupt_transform = get_corrupt_transform()

    train_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root="./data", train=False, download=True, transform=corrupt_transform
    )
    # test_dataset_corrupted: datasets.mnist.MNIST = datasets.MNIST(
    #     root="./data", train=False, download=True, transform=corrupt_transform
    # )
    print(train_dataset.data.shape)
    print(test_dataset.data.shape)

    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    # test_dataloader_corrupted: DataLoader = DataLoader(
    #     test_dataset_corrupted, batch_size=args.batch_size, shuffle=False
    # )

    # mswag = pus
    model_args: tuple = (
        {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
        },
    )
    mswag: MultiSWAG = train_mswag(
        train_dataloader,
        torch.nn.CrossEntropyLoss(),
        # create_optimizer,
        args.pretrain_epochs,
        args.swag_epochs,
        MLP,
        *model_args,
        cov_mat_rank=args.cov_mat_rank,
        num_models=args.num_models,
    )
    # TODO: evaluate on train set for epistemic uncertainty
    # TODO: evaluate on test set for aleatoric uncertainty
    # Note: p_params is list of [tensor(num_models, model_params, layer)]

    posterior_preds = mswag.posterior_pred(
        test_dataloader,
        num_samples=4,  # Number of models used for preds: num particles x num samples
        f_reg=False,
        # loss_fn=torch.nn.CrossEntropyLoss(),
        mode=[
            "mean",
            "mode",
            "std",
            "logits",
        ],  # Mode := standard NN since it only uses most likely weights
    )

    print(posterior_preds.keys())
    mean_pred = posterior_preds["mean"]
    print(type(mean_pred))
    print(mean_pred.shape)
    mean_acc = (mean_pred == test_dataset.targets).float().mean()
    print(f"Mean Accuracy: {mean_acc}")

    mode_pred = posterior_preds["mode"]
    print(mode_pred.shape)
    mode_acc = (mode_pred == test_dataset.targets).float().mean()
    print(f"Mode Accuracy: {mode_acc}")

    # Pac Bayes vs Bayesian, is this paper testing agaisnt multiswag or just MCMC?
    # https://arxiv.org/html/2406.05469v1#S3

    # TODO: Compare mean vs mode vs baseline of regularly trained model
    # 1 layer ensemble vs 2 layer non ensemble vs convnet non ensemble

    # TRAin the MLP

    # NOTE: Instead of a hyperparameter search, we use best params from other experiments


if __name__ == "__main__":
    main()

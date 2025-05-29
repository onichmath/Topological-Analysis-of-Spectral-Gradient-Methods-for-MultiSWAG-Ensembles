import argparse


def add_train_subparser(subparsers):
    train_parser = subparsers.add_parser(
        "train", help="Train the MultiSWAG model on the MNIST dataset"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for DataLoader"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for the optimizer"
    )
    train_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Number of epochs for training"
    )
    train_parser.add_argument(
        "--swag_epochs",
        type=int,
        default=20,
        help="Number of epochs for SWAG training",
    )
    train_parser.add_argument(
        "--cov_mat_rank",
        type=int,
        default=20,
        help="Rank of covariance matrix (used during posterior predictive inference)",
    )
    train_parser.add_argument(
        "--num_models",
        type=int,
        default=20,
        help="Number of models in the SWAG ensemble",
    )
    train_parser.add_argument(
        "--input_dim",
        type=int,
        default=28 * 28,
        help="Input dimension for the model",
    )
    train_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the model",
    )
    train_parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the model",
    )
    train_parser.add_argument(
        "--output_dim",
        type=int,
        default=10,
        help="Output dimension for the model",
    )
    train_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples (model params) from each particle",
    )
    train_parser.add_argument(
        "--val_size",
        type=int,
        default=10000,
        help="Size of the validation set (default: 10000)",
    )


def add_eval_subparser(subparsers):
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate the MultiSWAG model on the MNIST dataset"
    )
    eval_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for DataLoader"
    )
    eval_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples (model params) from each particle",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="MNIST DataLoader Example")

    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Mode to run: train, eval, or analysis"
    )

    add_train_subparser(subparsers)
    add_eval_subparser(subparsers)

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the MNIST dataset is stored",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=[
            "adam",
            "adamw",
            "muon",
            "10p",
            "muon10p",
            "muonspectralnorm",
            "spectralnorm",
        ],
        help="Optimizer to use for training (default: adam)",
    )
    return parser

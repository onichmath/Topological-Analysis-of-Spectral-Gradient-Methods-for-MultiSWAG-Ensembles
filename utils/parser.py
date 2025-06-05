import argparse
import torch


def add_train_subparser(subparsers):
    train_parser = subparsers.add_parser(
        "train", help="Train the MultiSWAG model on the MNIST dataset"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for DataLoader"
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
        "--batch_size", type=int, default=128, help="Batch size for DataLoader"
    )
    eval_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples (model params) from each particle",
    )
    eval_parser.add_argument(
        "--cov_mat_rank",
        type=int,
        default=20,
        help="Rank of covariance matrix (used during posterior predictive inference)",
    )
    eval_parser.add_argument(
        "--num_models",
        type=int,
        default=20,
        help="Number of models in the SWAG ensemble",
    )
    eval_parser.add_argument(
        "--input_dim",
        type=int,
        default=28 * 28,
        help="Input dimension for the model",
    )
    eval_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the model",
    )
    eval_parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the model",
    )
    eval_parser.add_argument(
        "--output_dim",
        type=int,
        default=10,
        help="Output dimension for the model",
    )
    eval_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Number of epochs for training"
    )
    eval_parser.add_argument(
        "--swag_epochs",
        type=int,
        default=20,
        help="Number of epochs for SWAG training",
    )
    eval_parser.add_argument(
        "--eval_all_epochs",
        type=bool,
        default=True,
        help="Evaluate all epochs",
    )
    eval_parser.add_argument(
        "--val_size",
        type=int,
        default=10000,
        help="Size of the validation set (default: 10000)",
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
        "--results_dir",
        type=str,
        default="./results",
        help="Directory where model weights are saved, and where results will be stored",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training and evaluation",
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
            "muon10p_fast",
            "muonspectralnorm",
            "muonspectralnorm_fast",
            "spectralnorm",
            "spectralnorm_fast",
            "10p_fast",
        ],
        help="Optimizer to use for training (default: adam)",
    )
    return parser

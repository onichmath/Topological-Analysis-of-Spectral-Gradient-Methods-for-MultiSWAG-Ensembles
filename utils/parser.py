import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="MNIST DataLoader Example")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=3, help="Number of epochs for training"
    )
    parser.add_argument(
        "--swag_epochs",
        type=int,
        default=2,
        help="Number of epochs for SWAG training",
    )
    parser.add_argument(
        "--cov_mat_rank",
        type=int,
        default=20,
        help="Rank of covariance matrix (used during posterior predictive inference)",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=2,
        help="Number of models in the SWAG ensemble",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=28 * 28,
        help="Input dimension for the model",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for the model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the model",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=10,
        help="Output dimension for the model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples (model params) from each particle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=10000,
        help="Size of the validation set (default: 10000)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "muon"],
        help="Optimizer to use for training (default: adam)",
    )
    return parser

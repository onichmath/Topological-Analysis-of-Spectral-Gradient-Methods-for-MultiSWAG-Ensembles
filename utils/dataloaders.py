from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch
from torchvision.transforms import Compose


def build_train_dataloaders(
    data_dir: str,
    batch_size: int,
    val_size: int,
    transform: Compose,
    corrupt_transform: Compose,
    seed: int,
):
    in_distribution = datasets.mnist.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    out_of_distribution = datasets.mnist.MNIST(
        root=data_dir, train=True, download=True, transform=corrupt_transform
    )

    train_size = len(in_distribution) - val_size

    assert train_size + val_size == len(
        in_distribution
    ), "Train and validation sizes do not match the in-distribution dataset size."
    assert len(in_distribution) == len(
        out_of_distribution
    ), "In-distribution and out-of-distribution datasets must have the same length."

    train_subset, val_subset = torch.utils.data.random_split(
        in_distribution,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    _, val_subset_corrupt = torch.utils.data.random_split(
        out_of_distribution,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    val_corrupt_dataloader = DataLoader(
        val_subset_corrupt, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader, val_corrupt_dataloader


def build_test_dataloaders(
    data_dir: str,
    batch_size: int,
    transform: Compose,
    corrupt_transform: Compose,
):
    test_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    test_corrupt_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=corrupt_transform
    )
    test_dataloader_corrupt: DataLoader = DataLoader(
        test_corrupt_dataset, batch_size=batch_size, shuffle=False
    )
    return test_dataloader, test_dataloader_corrupt

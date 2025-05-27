from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch
from torchvision.transforms import Compose


def build_train_dataloader(
    data_dir: str, batch_size: int, val_size: int, transform: Compose, seed: int
):
    full_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_size: int = len(full_dataset) - val_size
    assert train_size + val_size == len(
        full_dataset
    ), "Train and validation sizes do not match the full dataset size."

    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataloader: DataLoader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    val_dataloader: DataLoader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, val_dataloader


def build_test_dataloader(data_dir: str, batch_size: int, transform: Compose):
    test_dataset: datasets.mnist.MNIST = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return test_dataloader

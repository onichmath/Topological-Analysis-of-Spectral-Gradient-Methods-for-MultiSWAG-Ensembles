import torch
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(
            tensor + torch.randn_like(tensor) * self.std + self.mean, 0.0, 1.0
        )


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )


def get_corrupt_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            AddGaussianNoise(0.0, 0.2),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

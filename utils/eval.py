import torch
from torch.utils.data import DataLoader
from push.bayes.swag import MultiSWAG


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

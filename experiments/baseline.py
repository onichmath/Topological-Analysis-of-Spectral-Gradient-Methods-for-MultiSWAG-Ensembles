import torch
from models.MLP import MLP
import inspect


def baseline(train_dataloader, test_dataloader, test_dataloader_corrupted, model_args):
    # NOTE: We are not performing a hyperparameter search for either the MLP or the SWAG so we are not using a validation set
    # Instead, we are assuming the final model is given by the 10th epoch parameters

    if inspect.currentframe().f_back.f_code.co_name != "main":
        raise ValueError("This function should only be called from main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_MLP = MLP(*model_args).to(device)
    optimizer = torch.optim.Adam(single_MLP.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    results = {
        "train_loss": [],
        "test_acc": [],
        "test_acc_corrupted": [],
        "model_checkpoints": [],
    }

    for epoch in range(num_epochs):
        single_MLP.train()
        running_loss = 0.0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = single_MLP(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_dataloader.dataset)
        results["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        torch.save(
            single_MLP.state_dict(),
            f"./results/baseline/baseline_model_epoch_{epoch+1}.pth",
        )
        results["model_checkpoints"].append(f"baseline_model_epoch_{epoch+1}.pth")

    single_MLP.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            outputs = single_MLP(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total * 100
    results["test_acc"].append(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_dataloader_corrupted:
            x, y = x.to(device), y.to(device)
            outputs = single_MLP(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total * 100
    results["test_acc_corrupted"].append(accuracy)
    print(f"Test Accuracy Corrupted: {accuracy:.2f}%")

    with open("./results/baseline/baseline_training_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

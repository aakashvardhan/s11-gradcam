import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from config import get_config
from models.model_utils import model_summary
from models.resnet import ResNet18
from utils.transform import CIFAR10Dataset


def test_model_sanity(model_):
    """
    Function to perform a sanity check on a model by training it on a small dataset and checking if the loss decreases.

    Args:
        model_ (torch.nn.Module): The model to be tested.

    Returns:
        None
    """
    from tqdm import tqdm

    # Sanity check for model
    # Check if the model is capable of overfitting on a small dataset
    # Load CIFAR10 dataset
    cifar_train = CIFAR10Dataset(
        root="./data", train=True, transform="test", download=True
    )
    cifar_test = CIFAR10Dataset(
        root="./data", train=False, transform="test", download=True
    )
    cifar_subset = Subset(cifar_train, range(100))

    # Set the seed
    torch.manual_seed(1)

    # Create model
    model = model_
    criterion = F.cross_entropy
    # Using Adam as the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Create data loader
    train_loader = DataLoader(cifar_subset, batch_size=10, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size=1, shuffle=False)

    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (train_data, train_targets) in enumerate(pbar):
        # Train on each small batch
        for epoch in range(epochs):
            train_data, train_targets = train_data.to(device), train_targets.to(device)
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_loss, correct, total = 0, 0, 0
            for test_data, test_targets in test_loader:
                test_data, test_targets = test_data.to(device), test_targets.to(device)
                outputs = model(test_data)
                test_loss += criterion(outputs, test_targets).item()
                _, predicted = torch.max(outputs.data, 1)
                total += test_targets.size(0)
                correct += (predicted == test_targets).sum().item()

        train_loss = loss.item()
        train_accuracy = (
            100.0
            * np.sum(
                np.argmax(outputs.cpu().numpy(), axis=1) == train_targets.cpu().numpy()
            )
            / train_targets.size(0)
        )
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / total

        print(
            f"Batch {batch_idx + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Batch {batch_idx + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )

        # Check for signs of overfitting
        if train_loss < test_loss and train_accuracy > test_accuracy:
            print(f"Overfitting detected in batch {batch_idx + 1}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--summary", action="store_true", help="Print model summary")
    parser.add_argument(
        "--sanity", action="store_true", help="Perform model sanity check"
    )
    parser.add_argument(
        "--shapes", action="store_true", help="Print shapes of all layers"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create
    config = get_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet18().to(device)

    if args.summary:
        config["debug"] = True
        model_summary(model, input_size=(3, 32, 32))

    if args.sanity:
        test_model_sanity(model)

    if args.shapes:
        # print shapes of all layers (excluding bias)
        for name, param in model.named_parameters():
            print(name, param.shape)

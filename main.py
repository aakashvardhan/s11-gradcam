import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from models.model_utils import save_model
from utils.transform import CIFAR10Dataset


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def setup_cifar10(config):
    """
    Sets up the CIFAR-10 dataset and data loaders.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the CIFAR-10 train dataset, CIFAR-10 test dataset,
               CIFAR-10 train data loader, and CIFAR-10 test data loader.
    """
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(1)

    if cuda:
        torch.cuda.manual_seed(1)
    train_data = CIFAR10Dataset(
        root="./data", train=True, download=True, transform="train"
    )
    test_data = CIFAR10Dataset(
        root="./data", train=False, download=True, transform="test"
    )

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = (
        dict(
            shuffle=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        if cuda
        else dict(shuffle=True, batch_size=64)
    )
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_data, test_data, train_loader, test_loader

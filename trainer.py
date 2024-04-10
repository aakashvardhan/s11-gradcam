import torch
from tqdm import tqdm
import torch.nn.functional as F


class ModelTrainer:
    """
    A class to train and evaluate a model.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss function for training.
        ocp_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss function for training.
        ocp_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        train_losses (list): List to store training losses.
        test_losses (list): List to store test losses.
        train_acc (list): List to store training accuracies.
        test_acc (list): List to store test accuracies.
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        ocp_scheduler,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.ocp_scheduler = ocp_scheduler
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, epoch):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.criterion(y_pred, target)
            self.train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )
            self.ocp_scheduler.step()
        self.train_acc.append(100 * correct / processed)
        lrs = self.ocp_scheduler.get_last_lr()
        print(f"Max Learning Rate: {max(lrs)}")

    def test(self):
        """
        Evaluates the model on the test dataset.

        Returns:
            float: The average test loss.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({100.0 * correct / len(self.test_loader.dataset):.2f}%)\n"
        )
        self.test_acc.append(100.0 * correct / len(self.test_loader.dataset))
        return test_loss

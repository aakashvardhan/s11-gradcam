import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch
import cv2


class CIFAR10Dataset(datasets.CIFAR10):
    """
    Custom dataset class for CIFAR-10 dataset.

    Args:
        root (str): Root directory where the dataset exists or will be saved.
        train (bool): If True, creates a dataset from the training set, otherwise from the test set.
        download (bool): If True, downloads the dataset from the internet and puts it in the root directory.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            Default: None.

    Attributes:
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version.

    """

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        if transform == "train":
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=40, min_width=40,border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                    A.RandomCrop(32, 32, always_apply=True),
                    A.CoarseDropout(
                        max_holes=1,
                        max_height=16,
                        max_width=16,
                        min_holes=1,
                        min_height=16,
                        min_width=16,
                        fill_value=(0.4914, 0.4822, 0.4465),
                        mask_fill_value=None,
                    ),
                    A.RandomBrightnessContrast(p=0.2),
                    A.CenterCrop(32, 32, always_apply=True),
                    A.Normalize(
                        mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
                    ),
                    ToTensorV2(),
                ]
            )
        elif transform == "test":
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves the image and label at the given index.

        Args:
            index (int): Index of the image.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label.

        """
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

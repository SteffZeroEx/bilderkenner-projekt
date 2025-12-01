"""
Dataset Loader für CIFAR-10
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10Dataset:
    """
    CIFAR-10 Dataset Handler

    Klassen:
    0: Flugzeug
    1: Auto
    2: Vogel
    3: Katze
    4: Hirsch
    5: Hund
    6: Frosch
    7: Pferd
    8: Schiff
    9: LKW
    """

    def __init__(self, data_dir="./data/cifar10"):
        self.data_dir = data_dir
        self.classes = [
            "Flugzeug",
            "Auto",
            "Vogel",
            "Katze",
            "Hirsch",
            "Hund",
            "Frosch",
            "Pferd",
            "Schiff",
            "LKW",
        ]

    def get_train_dataset(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        return datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )

    def get_test_dataset(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        return datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )

    def get_class_name(self, class_idx):
        """Gibt Klassennamen für Index zurück"""
        return self.classes[class_idx]


# BEISPIEL-NUTZUNG (zum Testen):
if __name__ == "__main__":
    dataset_handler = CIFAR10Dataset()
    train_data = dataset_handler.get_train_dataset()
    print(f"Training Samples: {len(train_data)}")
    print(f"Klassen: {dataset_handler.classes}")

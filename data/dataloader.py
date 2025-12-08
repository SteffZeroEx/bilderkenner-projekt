# -*- coding: utf-8 -*-
"""
DataLoader Wrapper fuer einfache Verwendung
"""

from torch.utils.data import DataLoader
from data.dataset import CIFAR10Dataset
from data.transforms import get_train_transforms, get_test_transforms


def get_train_loader(batch_size=64, num_workers=2):
    """
    Erstellt DataLoader fuer Trainingsdaten

    Args:
        batch_size: Groesse der Batches
        num_workers: Anzahl der Worker-Prozesse

    Returns:
        DataLoader fuer Training
    """
    dataset = CIFAR10Dataset()
    train_data = dataset.get_train_dataset(transform=get_train_transforms())
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(batch_size=64, num_workers=2):
    """
    Erstellt DataLoader fuer Testdaten

    Args:
        batch_size: Groesse der Batches
        num_workers: Anzahl der Worker-Prozesse

    Returns:
        DataLoader fuer Testing
    """
    dataset = CIFAR10Dataset()
    test_data = dataset.get_test_dataset(transform=get_test_transforms())
    return DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    print(f"Train Batches: {len(train_loader)}")
    print(f"Test Batches: {len(test_loader)}")

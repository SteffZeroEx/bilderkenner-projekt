# -*- coding: utf-8 -*-
"""
Training-Pipeline fuer das CNN-Modell
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.cnn import get_model
from data.dataset import CIFAR10Dataset
from data.transforms import get_train_transforms, get_test_transforms


class Trainer:
    """
    Trainer-Klasse fuer das CNN-Modell
    """

    def __init__(self, model, device=None, learning_rate=0.001):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, train_loader):
        """Trainiert eine Epoche"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        return avg_loss, accuracy

    def evaluate(self, test_loader):
        """Evaluiert das Modell"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(test_loader)
        return avg_loss, accuracy

    def train(self, train_loader, test_loader, epochs=20, save_path="saved_models/model.pth"):
        """Vollstaendiges Training"""
        best_accuracy = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            self.scheduler.step()

            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_model(save_path)
                print(f"  -> Neues bestes Modell gespeichert!")

        return best_accuracy

    def save_model(self, path):
        """Speichert das Modell"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Laedt ein gespeichertes Modell"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def main():
    """Hauptfunktion zum Starten des Trainings"""
    # Dataset laden
    dataset = CIFAR10Dataset()
    train_data = dataset.get_train_dataset(transform=get_train_transforms())
    test_data = dataset.get_test_dataset(transform=get_test_transforms())

    # DataLoader erstellen
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    # Modell und Trainer
    model = get_model()
    trainer = Trainer(model)

    print(f"Training auf: {trainer.device}")
    print(f"Training Samples: {len(train_data)}")
    print(f"Test Samples: {len(test_data)}")

    # Training starten
    best_acc = trainer.train(train_loader, test_loader, epochs=20)
    print(f"\nBeste Genauigkeit: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

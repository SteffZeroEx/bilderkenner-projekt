# -*- coding: utf-8 -*-
"""
Prediction/Inferenz fuer trainierte Modelle
"""

import torch
from PIL import Image
from torchvision import transforms

from models.cnn import get_model
from data.dataset import CIFAR10Dataset


class Predictor:
    """
    Klasse fuer Vorhersagen mit trainiertem Modell
    """

    def __init__(self, model_path="saved_models/model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.dataset = CIFAR10Dataset()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def predict_image(self, image):
        """
        Vorhersage fuer ein einzelnes Bild

        Args:
            image: PIL Image oder Tensor

        Returns:
            dict mit Klasse und Wahrscheinlichkeiten
        """
        if isinstance(image, Image.Image):
            image = self.transform(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        return {
            "class_id": class_idx,
            "class_name": self.dataset.get_class_name(class_idx),
            "confidence": confidence.item(),
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }

    def predict_from_path(self, image_path):
        """Vorhersage fuer Bild aus Dateipfad"""
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)


if __name__ == "__main__":
    print("Predictor-Modul geladen")
    print("Verwendung:")
    print("  predictor = Predictor('saved_models/model.pth')")
    print("  result = predictor.predict_from_path('bild.jpg')")

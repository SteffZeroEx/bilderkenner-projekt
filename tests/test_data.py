# -*- coding: utf-8 -*-
"""
Tests fuer Data-Module
"""

import pytest
import torch
from torchvision import transforms

from data.dataset import CIFAR10Dataset
from data.transforms import get_train_transforms, get_test_transforms


class TestCIFAR10Dataset:
    """Tests fuer CIFAR10Dataset"""

    def test_init(self):
        """Dataset kann initialisiert werden"""
        dataset = CIFAR10Dataset()
        assert dataset is not None
        assert len(dataset.classes) == 10

    def test_classes(self):
        """Alle 10 Klassen sind definiert"""
        dataset = CIFAR10Dataset()
        expected = ["Flugzeug", "Auto", "Vogel", "Katze", "Hirsch",
                   "Hund", "Frosch", "Pferd", "Schiff", "LKW"]
        assert dataset.classes == expected

    def test_get_class_name(self):
        """Klassennamen werden korrekt zurueckgegeben"""
        dataset = CIFAR10Dataset()
        assert dataset.get_class_name(0) == "Flugzeug"
        assert dataset.get_class_name(9) == "LKW"


class TestTransforms:
    """Tests fuer Transformations"""

    def test_train_transforms(self):
        """Train-Transforms werden erstellt"""
        t = get_train_transforms()
        assert t is not None
        assert isinstance(t, transforms.Compose)

    def test_test_transforms(self):
        """Test-Transforms werden erstellt"""
        t = get_test_transforms()
        assert t is not None
        assert isinstance(t, transforms.Compose)

    def test_transform_output_shape(self):
        """Transforms erzeugen korrekte Output-Shape"""
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        t = get_test_transforms()
        result = t(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

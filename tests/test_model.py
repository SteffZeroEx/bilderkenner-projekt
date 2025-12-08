# -*- coding: utf-8 -*-
"""
Tests fuer Model-Module
"""

import pytest
import torch

from models.cnn import CIFAR10CNN, get_model


class TestCIFAR10CNN:
    """Tests fuer das CNN-Modell"""

    def test_model_creation(self):
        """Modell kann erstellt werden"""
        model = get_model()
        assert model is not None
        assert isinstance(model, CIFAR10CNN)

    def test_forward_pass(self):
        """Forward-Pass funktioniert"""
        model = get_model()
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)

    def test_batch_forward(self):
        """Forward-Pass mit Batch funktioniert"""
        model = get_model()
        x = torch.randn(16, 3, 32, 32)
        output = model(x)
        assert output.shape == (16, 10)

    def test_output_logits(self):
        """Output sind Logits (keine Softmax)"""
        model = get_model()
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.sum().item() != pytest.approx(1.0, abs=0.1)

    def test_model_parameters(self):
        """Modell hat trainierbare Parameter"""
        model = get_model()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 0

    def test_eval_mode(self):
        """Modell kann in Eval-Modus gesetzt werden"""
        model = get_model()
        model.eval()
        assert not model.training

    def test_train_mode(self):
        """Modell kann in Train-Modus gesetzt werden"""
        model = get_model()
        model.train()
        assert model.training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

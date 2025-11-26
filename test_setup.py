"""
Test ob alles funktioniert
"""
import sys
import torch
import torchvision
import fastapi
from PIL import Image
import numpy as np


def test_python_version():
    """Test Python Version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3
    assert version.minor >= 9


def test_pytorch():
    """Test PyTorch"""
    print(f"✓ PyTorch {torch.__version__}")
    # Einfacher Tensor Test
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0
    print("✓ PyTorch funktioniert!")


def test_torchvision():
    """Test torchvision"""
    print(f"✓ torchvision {torchvision.__version__}")
    # Test Transform
    from torchvision import transforms
    transform = transforms.ToTensor()
    print("✓ torchvision funktioniert!")


def test_pillow():
    """Test PIL/Pillow"""
    from PIL import __version__
    print(f"✓ Pillow {__version__}")
    # Erstelle dummy image
    img = Image.new('RGB', (100, 100), color='red')
    assert img.size == (100, 100)
    print("✓ Pillow funktioniert!")


def test_numpy():
    """Test NumPy"""
    print(f"✓ NumPy {np.__version__}")
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    print("✓ NumPy funktioniert!")


def test_fastapi():
    """Test FastAPI"""
    print(f"✓ FastAPI {fastapi.__version__}")
    print("✓ FastAPI importiert!")


if __name__ == "__main__":
    print("=" * 50)
    print("SYSTEM CHECK - Teste Installation")
    print("=" * 50)

    try:
        test_python_version()
        test_pytorch()
        test_torchvision()
        test_pillow()
        test_numpy()
        test_fastapi()

        print("=" * 50)
        print("✅ ALLE TESTS BESTANDEN!")
        print("✅ Dein System ist bereit für den Unterricht!")
        print("=" * 50)
    except Exception as e:
        print("=" * 50)
        print(f"❌ FEHLER: {e}")
        print("=" * 50)
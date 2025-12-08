"""
Data Transformations und Augmentation
"""

from torchvision import transforms


def get_train_transforms():
    """
    Transformations für Training mit Augmentation
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


def get_test_transforms():
    """
    Transformations für Testing (ohne Augmentation)
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


if __name__ == "__main__":
    train_t = get_train_transforms()
    test_t = get_test_transforms()
    print("Train Transforms:", train_t)
    print("Test Transforms:", test_t)

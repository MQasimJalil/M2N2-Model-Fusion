import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size, data_dir='cifar-10-batches-py'):
    """
    Returns CIFAR-10 training and testing data loaders with appropriate transformations.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): Directory where CIFAR-10 dataset is located or will be downloaded.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # CIFAR-10 statistics (mean and std for normalization)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Testing transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load CIFAR-10 datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2, # Using 2 workers for data loading
        pin_memory=True # Pinning memory for faster GPU transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == '__main__':
    # Example usage
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Get one batch
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}") # Expected: torch.Size([64, 3, 32, 32])
        print(f"Label batch shape: {labels.shape}") # Expected: torch.Size([64])
        break

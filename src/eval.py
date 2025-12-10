import torch
import torch.nn as nn
import yaml
import os
import argparse
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.cifar10 import get_cifar10_loaders
from src.models.efficientnet_lite import EfficientNetLite0
from src.models.hybrid_model import EfficientNetBaseline, HybridEfficientNetTRM
from src.utils.metrics import calculate_accuracy

def evaluate_model(model_name, checkpoint_path, num_classes, batch_size, data_dir, trm_hidden_dim=128, trm_num_recursions=2, test_loader=None):
    """
    Evaluates a trained model checkpoint on the CIFAR-10 test set.

    Args:
        model_name (str): The name of the model to evaluate ('EfficientNetBaseline' or 'HybridEfficientNetTRM').
        checkpoint_path (str): Path to the saved model state_dict.
        num_classes (int): Number of output classes (e.g., 10 for CIFAR-10).
        batch_size (int): Batch size for evaluation.
        data_dir (str): Directory where CIFAR-10 dataset is located.
        trm_hidden_dim (int): Hidden dimension for TRM (only relevant for HybridEfficientNetTRM).
        trm_num_recursions (int): Number of recursions for TRM (only relevant for HybridEfficientNetTRM).
        test_loader (DataLoader, optional): Pre-loaded test data loader. If None, loads standard CIFAR-10.

    Returns:
        float: Final test accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data if not provided
    if test_loader is None:
        _, test_loader = get_cifar10_loaders(batch_size=batch_size, data_dir=data_dir)

    # Model instantiation
    if model_name == 'EfficientNetBaseline':
        model = EfficientNetBaseline(num_classes=num_classes).to(device)
    elif model_name == 'HybridEfficientNetTRM':
        model = HybridEfficientNetTRM(
            num_classes=num_classes,
            trm_hidden_dim=trm_hidden_dim,
            trm_num_recursions=trm_num_recursions
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}")

    # Evaluation
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_accuracy_sum += calculate_accuracy(outputs, labels) * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy_sum / total_samples

    print(f"\n--- Evaluation Results ---")
    print(f"Model: {model_name}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_accuracy:.4f}")
    
    return avg_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on CIFAR-10.')
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['EfficientNetBaseline', 'HybridEfficientNetTRM'],
                        help='Name of the model to evaluate.')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='Number of output classes.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for evaluation.')
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py', 
                        help='Directory where CIFAR-10 dataset is located.')
    parser.add_argument('--trm_hidden_dim', type=int, default=128, 
                        help='TRM hidden dimension (if evaluating HybridEfficientNetTRM).')
    parser.add_argument('--trm_num_recursions', type=int, default=2, 
                        help='TRM number of recursions (if evaluating HybridEfficientNetTRM).')
    args = parser.parse_args()

    evaluate_model(args.model_name, args.checkpoint, args.num_classes, args.batch_size, 
                   args.data_dir, args.trm_hidden_dim, args.trm_num_recursions)

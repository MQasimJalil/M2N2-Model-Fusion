import torch

def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculates the top-1 accuracy.

    Args:
        outputs (torch.Tensor): Predicted logits or scores from the model.
                                Shape: (batch_size, num_classes).
        labels (torch.Tensor): True labels. Shape: (batch_size).

    Returns:
        float: Top-1 accuracy as a percentage.
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    # Example usage
    num_classes = 10
    batch_size = 64

    # Simulate model outputs (logits) and true labels
    dummy_outputs = torch.randn(batch_size, num_classes)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    acc = calculate_accuracy(dummy_outputs, dummy_labels)
    print(f"Example Accuracy: {acc:.4f}")

    # Example with perfect prediction
    perfect_outputs = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        perfect_outputs[i, dummy_labels[i]] = 10.0 # Set high score for true class
    perfect_acc = calculate_accuracy(perfect_outputs, dummy_labels)
    print(f"Perfect Accuracy: {perfect_acc:.4f}")

    # Example with zero prediction
    zero_outputs = torch.zeros(batch_size, num_classes)
    zero_labels = torch.randint(0, num_classes, (batch_size,))
    zero_acc = calculate_accuracy(zero_outputs, zero_labels)
    print(f"Zero Accuracy (random chance): {zero_acc:.4f}")

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import copy
from torch.optim.swa_utils import update_bn

from src.models.hybrid_model import HybridEfficientNetTRM
from src.m2n2_fusion import M2N2WeightMerger
from src.data.cifar10 import get_cifar10_loaders
from src.utils.metrics import calculate_accuracy
from src.utils.seed import set_seed

def main():
    # Configuration
    model_name = 'HybridEfficientNetTRM'
    checkpoint_path_a = 'checkpoints/hybrid/HybridEfficientNetTRM_best_seed_42.pth'
    checkpoint_path_b = 'checkpoints/hybrid/HybridEfficientNetTRM_best_seed_43.pth'
    num_classes = 10
    batch_size = 128
    data_dir = 'cifar-10-batches-py'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, data_dir=data_dir)

    # Helper to limit BN update to N batches to save time during search
    class LimitedLoader:
        def __init__(self, loader, limit=50):
            self.loader = loader
            self.limit = limit
        def __iter__(self):
            count = 0
            for batch in self.loader:
                if count >= self.limit:
                    break
                yield batch
                count += 1
        def __len__(self):
            return self.limit

    limited_train_loader = LimitedLoader(train_loader, limit=50) # Use 50 batches for BN stats

    # Define Evaluation Function
    def eval_func(model):
        model.to(device)
        
        # CRITICAL: Update BatchNorm statistics before evaluation
        # We use a limited subset of training data for speed during search
        model.train() # update_bn requires train mode
        # We manually implement a simple update_bn or use the util with the limited loader
        # update_bn(limited_train_loader, model, device=device) -> This expects a standard loader.
        # Let's just do a manual pass for robustness
        with torch.no_grad():
            for i, (inputs, _) in enumerate(train_loader):
                if i >= 50: break
                inputs = inputs.to(device)
                model(inputs) # Forward pass updates running stats
        
        model.eval()
        total_accuracy_sum = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_accuracy_sum += calculate_accuracy(outputs, labels) * labels.size(0)
                total_samples += labels.size(0)
        
        return total_accuracy_sum / total_samples

    # Load Models
    print(f"Loading checkpoints for {model_name}...")
    model_a = HybridEfficientNetTRM(num_classes=num_classes)
    model_a.load_state_dict(torch.load(checkpoint_path_a, map_location='cpu'))
    
    model_b = HybridEfficientNetTRM(num_classes=num_classes)
    model_b.load_state_dict(torch.load(checkpoint_path_b, map_location='cpu'))

    checkpoint_path_c = 'checkpoints/hybrid/HybridEfficientNetTRM_best_seed_44.pth'
    model_c = HybridEfficientNetTRM(num_classes=num_classes)
    if os.path.exists(checkpoint_path_c):
        print(f"Loading checkpoint C: {checkpoint_path_c}")
        model_c.load_state_dict(torch.load(checkpoint_path_c, map_location='cpu'))
        models_to_merge = [model_a, model_b, model_c]
    else:
        print(f"Checkpoint C not found at {checkpoint_path_c}, skipping.")
        models_to_merge = [model_a, model_b]

    # Initialize M2N2 Merger
    merger = M2N2WeightMerger(num_classes=num_classes)

    # Perform Fusion
    print("\nStarting M2N2 Fusion Search (Iterative)...")
    # Reduced grid for speed
    best_sd, best_acc, _, _ = merger.find_best_merge(
        HybridEfficientNetTRM, 
        models_to_merge, 
        eval_func, 
        num_mix_ratios=6,  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        num_split_points=3  # 0, 50%, 100%
    )

    print("\n" + "="*30)
    print("FUSION RESULTS")
    print("="*30)
    print(f"Fused Model Accuracy:     {best_acc:.4f}")
    
    # Save
    save_path = 'checkpoints/hybrid/HybridEfficientNetTRM_fused.pth'
    torch.save(best_sd, save_path)
    print(f"\nFused model saved to: {save_path}")

if __name__ == '__main__':
    main()
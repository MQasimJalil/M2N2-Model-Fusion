import torch
import torch.nn as nn
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from imagecorruptions import corrupt
import torchvision
import matplotlib.pyplot as plt
import warnings # Import the warnings module

from src.eval import evaluate_model
from src.models.hybrid_model import HybridEfficientNetTRM

# Suppress the specific pkg_resources deprecation warning from imagecorruptions
warnings.filterwarnings('ignore', category=UserWarning, module='imagecorruptions.corruptions')

# Config
CORRUPTION_TYPES = ['gaussian_noise', 'defocus_blur', 'frost']
SEVERITY_LEVELS = [1, 2, 3] # Test range of severities

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def denormalize(tensor):
    """Reverses the CIFAR-10 normalization for visualization."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def visualize_predictions(model, data_dir, output_dir):
    print("Generating visualization plots...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for c_type in CORRUPTION_TYPES:
        fig, axs = plt.subplots(len(SEVERITY_LEVELS), 5, figsize=(15, 9))
        fig.suptitle(f"Predictions on {c_type.replace('_', ' ').title()}", fontsize=16)
        
        for i, sev in enumerate(SEVERITY_LEVELS):
            loader = get_corrupted_loader(c_type, sev, data_dir, batch_size=5)
            # Get one batch
            images, labels = next(iter(loader))
            images, labels = images[:5].to(device), labels[:5].to(device)
            
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            
            for j in range(5):
                ax = axs[i, j]
                img_np = denormalize(images[j])
                ax.imshow(img_np)
                ax.axis('off')
                
                true_label = CIFAR10_CLASSES[labels[j].item()]
                pred_label = CIFAR10_CLASSES[preds[j].item()]
                
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f"Sev {sev}\nT: {true_label}\nP: {pred_label}", color=color, fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'viz_{c_type}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

def get_corrupted_loader(corruption_type, severity, data_dir, batch_size):
    """
    Returns a DataLoader for the specified corruption. 
    Generates and saves the dataset to disk if it doesn't exist.
    """
    corrupted_dir = os.path.join(data_dir, 'corrupted')
    os.makedirs(corrupted_dir, exist_ok=True)
    
    file_name = f"test_{corruption_type}_{severity}.pt"
    file_path = os.path.join(corrupted_dir, file_name)
    
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]
    
    if os.path.exists(file_path):
        print(f"Loading cached corrupted data from {file_path}")
        data, targets = torch.load(file_path)
    else:
        print(f"Generating corrupted data for {corruption_type} level {severity}...")
        # Load raw test set
        test_dataset_raw = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=None
        )
        
        corrupted_images = []
        targets = []
        
        # Transform for final tensor
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        
        for img_pil, label in test_dataset_raw:
            img_np = np.array(img_pil, dtype=np.uint8)
            img_corrupted_np = corrupt(img_np, corruption_name=corruption_type, severity=severity)
            img_tensor = post_transform(Image.fromarray(img_corrupted_np))
            corrupted_images.append(img_tensor)
            targets.append(label)
        
        data = torch.stack(corrupted_images)
        targets = torch.tensor(targets)
        
        print(f"Saving to {file_path}")
        torch.save((data, targets), file_path)
        
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader

def save_plots(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Dashboard Layout:
    # Row 1: Clean Data (Left), Text Summary (Right)
    # Row 2: 3 Subplots for each corruption type (Line charts: Severity vs Accuracy)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # --- 1. Clean Data (Top Left) ---
    ax_clean = fig.add_subplot(gs[0, 0])
    methods = ['Baseline Avg', 'Fused Hybrid']
    accuracies = [results['clean']['baseline'], results['clean']['fused']]
    
    ax_clean.bar(methods, accuracies, color=['gray', 'cornflowerblue'], width=0.6)
    ax_clean.set_ylim(0, 1.0)
    ax_clean.set_ylabel('Accuracy')
    ax_clean.set_title('Clean Data Performance')
    ax_clean.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(accuracies):
        ax_clean.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')

    # --- 2. Corruption Plots (Bottom Row) ---
    if results['corrupted']:
        corruptions = list(results['corrupted'].keys())
        # Severity levels found in the results
        # We assume all corruptions have the same severities tested
        first_corr = corruptions[0]
        # Structure is results['corrupted'][type][severity]['baseline']
        # Wait, current structure in main() loop is:
        # results['corrupted'][c_type] = {'baseline': val, 'fused': val} 
        # BUT we are now iterating severities. We need to update the data structure in main() first!
        # The plotting logic below assumes the data structure IS updated.
        # Let's write the plotting logic assuming:
        # results['corrupted'][c_type] = { 1: {'baseline': x, 'fused': y}, 2: ... }
        
        # NOTE: I need to update main() first to support this structure. 
        # But this tool call is for save_plots. I will write generic logic here that expects the new structure.
        
        for idx, c_type in enumerate(corruptions):
            if idx >= 3: break # Limit to 3 columns for now
            ax = fig.add_subplot(gs[1, idx])
            
            severities = sorted(results['corrupted'][c_type].keys())
            baseline_vals = [results['corrupted'][c_type][s]['baseline'] for s in severities]
            fused_vals = [results['corrupted'][c_type][s]['fused'] for s in severities]
            
            ax.plot(severities, baseline_vals, 'o-', label='Baseline Avg', color='gray')
            ax.plot(severities, fused_vals, 'o-', label='Fused Hybrid', color='cornflowerblue')
            
            ax.set_title(c_type.replace('_', ' ').title())
            ax.set_xlabel('Severity')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1.0)
            ax.set_xticks(severities)
            ax.grid(True, linestyle='--', alpha=0.7)
            if idx == 0: ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'dashboard.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nDashboard saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='cifar-10-batches-py')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    # Results container
    results = {
        'clean': {'baseline': 0.0, 'fused': 0.0},
        'corrupted': {}
    }
    
    # 1. Define Models to Evaluate
    baseline_checkpoints = [
        'checkpoints/baseline/EfficientNetBaseline_best_seed_42.pth',
        'checkpoints/baseline/EfficientNetBaseline_best_seed_43.pth',
        'checkpoints/baseline/EfficientNetBaseline_best_seed_44.pth'
    ]
    # Check if they exist
    baseline_checkpoints = [cp for cp in baseline_checkpoints if os.path.exists(cp)]
    
    fused_checkpoint = 'checkpoints/hybrid/HybridEfficientNetTRM_fused.pth'
    has_fused = os.path.exists(fused_checkpoint)
    
    if not baseline_checkpoints and not has_fused:
        print("No checkpoints found to evaluate.")
        return

    # Logging File
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'evaluation_results.txt')
    
    with open(log_file_path, 'w') as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + '\n')

        log("="*40)
        log("FULL EVALUATION REPORT")
        log("="*40 + "\n")

        # 2. Evaluate Clean Accuracy (Baseline Avg vs Fused)
        log("--- CLEAN DATA (Original CIFAR-10) ---")
        
        # Baseline
        if baseline_checkpoints:
            accs = []
            for cp in baseline_checkpoints:
                acc = evaluate_model('EfficientNetBaseline', cp, 10, args.batch_size, args.data_dir)
                accs.append(acc)
            avg_baseline = sum(accs)/len(accs)
            results['clean']['baseline'] = avg_baseline
            log(f"-> Average Baseline Accuracy: {avg_baseline:.4f}")
        
        # Fused
        if has_fused:
            acc_fused = evaluate_model('HybridEfficientNetTRM', fused_checkpoint, 10, args.batch_size, args.data_dir)
            results['clean']['fused'] = acc_fused
            log(f"-> Fused Hybrid Accuracy:     {acc_fused:.4f}")

        # 3. Evaluate Corrupted Accuracy
        for c_type in CORRUPTION_TYPES:
            results['corrupted'][c_type] = {} # Initialize dict for this corruption type
            
            for sev in SEVERITY_LEVELS:
                log(f"\n--- {c_type.upper()} (Severity {sev}) ---")
                
                results['corrupted'][c_type][sev] = {'baseline': 0.0, 'fused': 0.0}
                
                # Get Loader (Generates if missing)
                loader = get_corrupted_loader(c_type, sev, args.data_dir, args.batch_size)
                
                # Baseline
                if baseline_checkpoints:
                    accs = []
                    for cp in baseline_checkpoints:
                        acc = evaluate_model('EfficientNetBaseline', cp, 10, args.batch_size, args.data_dir, test_loader=loader)
                        accs.append(acc)
                    avg_baseline = sum(accs)/len(accs)
                    results['corrupted'][c_type][sev]['baseline'] = avg_baseline
                    log(f"-> Average Baseline Accuracy: {avg_baseline:.4f}")

                # Fused
                if has_fused:
                    acc_fused = evaluate_model('HybridEfficientNetTRM', fused_checkpoint, 10, args.batch_size, args.data_dir, test_loader=loader)
                    results['corrupted'][c_type][sev]['fused'] = acc_fused
                    log(f"-> Fused Hybrid Accuracy:     {acc_fused:.4f}")
    
    # Generate Plots
    save_plots(results, output_dir)
    print(f"\nResults saved to {log_file_path}")

    # Generate Visualizations
    if has_fused:
        print("\nLoading Fused Model for Visualization...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridEfficientNetTRM(num_classes=10)
        model.load_state_dict(torch.load(fused_checkpoint, map_location=device))
        visualize_predictions(model, args.data_dir, output_dir)

if __name__ == '__main__':
    main()

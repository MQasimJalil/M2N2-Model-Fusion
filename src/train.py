import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os
import argparse
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.cifar10 import get_cifar10_loaders
from src.models.efficientnet_lite import EfficientNetLite0
from src.models.hybrid_model import EfficientNetBaseline, HybridEfficientNetTRM
from src.utils.metrics import calculate_accuracy

def train_model(config_path, override_seed=None):
    """
    Trains a model based on the provided configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override seed if provided via command line
    if override_seed is not None:
        config['seed'] = override_seed
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed for reproducibility
    set_seed(config['seed'])

    # Checkpoint directory
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dynamic batch size adjustment logic
    current_batch_size = config['batch_size']
    
    # Store the run-specific seed
    run_seed = config['seed'] if override_seed is None else override_seed

    while current_batch_size >= 1:
        try:
            print(f"Attempting training with batch size: {current_batch_size}")

            # 1. FIXED INITIALIZATION
            # We set a constant seed (0) for model initialization so that all runs 
            # start from the EXACT SAME random weights. This is crucial for Model Merging (M2N2).
            print("Initializing model with fixed seed 0 (for weight alignment)...")
            set_seed(0)
            
            # Model (re-initialized)
            if config['model_name'] == 'EfficientNetBaseline':
                model = EfficientNetBaseline(num_classes=config['num_classes']).to(device)
            elif config['model_name'] == 'HybridEfficientNetTRM':
                model = HybridEfficientNetTRM(
                    num_classes=config['num_classes'],
                    trm_hidden_dim=config.get('trm_hidden_dim', 128),
                    trm_num_recursions=config.get('trm_num_recursions', 2)
                ).to(device)
            else:
                raise ValueError(f"Unknown model name: {config['model_name']}")

            # 2. RUN-SPECIFIC DYNAMICS
            # Now we switch back to the specific seed for this run (e.g. 42)
            # This ensures that data shuffling, dropout, and augmentation are unique.
            print(f"Switching to run-specific seed {run_seed} for training dynamics...")
            set_seed(run_seed)

            # Data loaders (re-initialized with new batch size)
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=current_batch_size,
                data_dir=config['data_dir']
            )

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

            best_val_accuracy = -1.0
            best_epoch = -1

            print(f"Starting training for {config['model_name']}...")
            for epoch in range(config['epochs']):
                # Training loop
                model.train()
                running_loss = 0.0
                train_accuracy_sum = 0.0
                train_total_samples = 0

                for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Training")):
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    train_accuracy_sum += calculate_accuracy(outputs, labels) * labels.size(0)
                    train_total_samples += labels.size(0)

                train_loss = running_loss / len(train_loader)
                train_accuracy = train_accuracy_sum / train_total_samples
                scheduler.step()

                # Validation loop
                model.eval()
                val_loss = 0.0
                val_accuracy_sum = 0.0
                val_total_samples = 0
                with torch.no_grad():
                    for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Validation"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        val_accuracy_sum += calculate_accuracy(outputs, labels) * labels.size(0)
                        val_total_samples += labels.size(0)
                
                val_loss /= len(test_loader)
                val_accuracy = val_accuracy_sum / val_total_samples

                print(f"Epoch {epoch+1}/{config['epochs']} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    checkpoint_path = os.path.join(checkpoint_dir, f"{config['model_name']}_best_seed_{config['seed']}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"New best model saved at {checkpoint_path} with Val Acc: {best_val_accuracy:.4f}")
            
            print(f"\nTraining finished for seed {config['seed']}. Best Validation Accuracy: {best_val_accuracy:.4f} at Epoch {best_epoch}")
            return best_val_accuracy, best_epoch

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nCUDA Out of Memory with batch size {current_batch_size}.")
                torch.cuda.empty_cache()
                current_batch_size //= 2
                if current_batch_size < 1:
                    print("Batch size dropped below 1. Cannot train.")
                    raise e
                print(f"Retrying with batch size: {current_batch_size}")
            else:
                raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--seed', type=int, help='Override the random seed specified in the config file.')
    args = parser.parse_args()

    train_model(args.config, args.seed)

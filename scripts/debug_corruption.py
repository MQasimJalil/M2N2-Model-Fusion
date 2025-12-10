import torch
import torchvision
import torchvision.transforms as transforms
from imagecorruptions import corrupt
import numpy as np
from PIL import Image
import os

def debug_corruption_pipeline():
    print("Debugging Corruption Pipeline...")
    
    # Load ONE image from CIFAR-10
    dataset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py', train=False, download=True)
    img_pil, label = dataset[0]
    
    print(f"\n1. Original Image: {img_pil.size} (PIL)")
    
    # Convert to Numpy
    img_np = np.array(img_pil)
    print(f"2. Numpy Array: shape={img_np.shape}, dtype={img_np.dtype}, min={img_np.min()}, max={img_np.max()}")
    
    # Corrupt
    print("\nApplying Gaussian Noise (Severity 3)...")
    try:
        corrupted_np = corrupt(img_np, corruption_name='gaussian_noise', severity=3)
        print(f"3. Corrupted Numpy: shape={corrupted_np.shape}, dtype={corrupted_np.dtype}, min={corrupted_np.min()}, max={corrupted_np.max()}")
        
        # Check if it looks like an image (should be uint8 0-255)
        if corrupted_np.max() <= 1.0:
            print("WARNING: Corrupted output seems to be float 0-1, but expected uint8 0-255!")
            
    except Exception as e:
        print(f"ERROR during corruption: {e}")
        return

    # Transform
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]
    
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # PIL Re-conversion check
    img_re_pil = Image.fromarray(corrupted_np)
    print(f"4. Re-PIL Image: {img_re_pil.size}")
    
    # Tensor Conversion
    tensor = post_transform(img_re_pil)
    print(f"5. Final Tensor: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.4f}, max={tensor.max():.4f}")
    
    # Check if tensor values are reasonable (should be roughly -2 to +2)
    if tensor.min() < -5 or tensor.max() > 5:
        print("WARNING: Tensor values are very large! Normalization might be wrong.")
    else:
        print("Tensor values look reasonable (Standard Normal Distribution).")

if __name__ == "__main__":
    debug_corruption_pipeline()

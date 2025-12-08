import torch
import torch.nn as nn

from src.models.efficientnet_lite import EfficientNetLite0
from src.models.trm import TRMBlock

class EfficientNetBaseline(nn.Module):
    """
    EfficientNet-Lite0 model with its own classification head.
    Used as the baseline model.
    """
    def __init__(self, num_classes=10):
        super(EfficientNetBaseline, self).__init__()
        # EfficientNetLite0 already includes its own classifier
        self.efficientnet = EfficientNetLite0(num_classes=num_classes)

    def forward(self, x):
        return self.efficientnet(x)

class HybridEfficientNetTRM(nn.Module):
    """
    Hybrid model combining EfficientNet-Lite0 as a feature extractor
    and TRMBlock for recursive feature refinement, followed by a classifier.
    """
    def __init__(self, num_classes=10, trm_hidden_dim=128, trm_num_recursions=2):
        super(HybridEfficientNetTRM, self).__init__()
        
        # 1. EfficientNet-Lite0 as feature extractor
        # We need to remove its original classifier to attach TRM and a new classifier.
        efficientnet_feature_extractor = EfficientNetLite0(num_classes=1000) # num_classes doesn't matter, we remove head
        self.efficientnet_features = nn.Sequential(
            efficientnet_feature_extractor.stem,
            efficientnet_feature_extractor.blocks,
            efficientnet_feature_extractor.head_conv,
            efficientnet_feature_extractor.avgpool,
            nn.Flatten(1)
        )
        
        # Get the output dimension of EfficientNet's feature extractor
        # This is hardcoded to 1280 for EfficientNet-Lite0 after GAP
        efficientnet_output_dim = 1280 

        # 2. TRM Block for recursive feature refinement
        # TRM will output a refined feature vector
        trm_final_feature_dim = 256 # Define a specific dimension for TRM's output features
        self.trm_block = TRMBlock(
            input_dim=efficientnet_output_dim,
            final_feature_dim=trm_final_feature_dim,
            hidden_dim=trm_hidden_dim,
            num_recursions=trm_num_recursions
        )

        # 3. Final Classifier Head
        self.classifier = nn.Linear(trm_final_feature_dim, num_classes)

    def forward(self, x):
        # Extract features from EfficientNet
        efficientnet_output = self.efficientnet_features(x)
        
        # Refine features using TRM
        trm_output_features = self.trm_block(efficientnet_output)
        
        # Classify the refined features
        logits = self.classifier(trm_output_features)
        return logits

if __name__ == '__main__':
    num_classes = 10
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32) # CIFAR-10 size

    # Test EfficientNetBaseline
    print("--- Testing EfficientNetBaseline ---")
    baseline_model = EfficientNetBaseline(num_classes=num_classes)
    print(baseline_model)
    baseline_output = baseline_model(input_tensor)
    print(f"Baseline output shape: {baseline_output.shape}")

    total_params_baseline = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters for EfficientNetBaseline: {total_params_baseline / 1e6:.2f} M")

    # Test HybridEfficientNetTRM
    print("\n--- Testing HybridEfficientNetTRM ---")
    hybrid_model = HybridEfficientNetTRM(num_classes=num_classes)
    print(hybrid_model)
    hybrid_output = hybrid_model(input_tensor)
    print(f"Hybrid output shape: {hybrid_output.shape}")

    total_params_hybrid = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters for HybridEfficientNetTRM: {total_params_hybrid / 1e6:.2f} M")

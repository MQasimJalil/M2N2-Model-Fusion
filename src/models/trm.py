import torch
import torch.nn as nn

class TRMNet(nn.Module):
    """
    The 'net' within the Tiny Recursive Model, which performs the latent reasoning.
    Simplified as a small MLP. This is the core recursive unit.
    """
    def __init__(self, input_and_latent_dim, latent_dim):
        super(TRMNet, self).__init__()
        # As per TRM.pdf, the 'net' is a small network.
        # We model it as a 2-layer MLP with a residual connection for stability.
        self.fc1 = nn.Linear(input_and_latent_dim, latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, combined_input_latent):
        """
        Processes the combined input features and latent state to produce a new latent state.
        """
        # A simple residual-like connection in the recursive update
        # z_new = z_old + MLP(combined)
        return self.relu(self.fc1(combined_input_latent)) + self.fc2(self.relu(self.fc1(combined_input_latent)))


class TRMBlock(nn.Module):
    """
    Nano-TRM block for feature refinement based on the Tiny Recursive Model principles.
    Adapted for low VRAM by reducing hidden_dim and recursion steps.
    
    References:
        - TRM.pdf, Section 4. (Tiny Recursion Models)
        - TRM.pdf, Figure 1 & Figure 3 (Pseudocode).
    """
    def __init__(self, input_dim: int, final_feature_dim: int, hidden_dim: int = 128, num_recursions: int = 2):
        """
        Args:
            input_dim (int): Dimension of the input feature vector from the backbone.
            final_feature_dim (int): Dimension of the output feature vector from TRM (to be used by classifier).
            hidden_dim (int): Latent dimension for recursive processing (D in TRM paper, here 'z' latent dim).
            num_recursions (int): Number of recursive steps (n in TRM paper). Default to 2 for efficiency.
                                  The paper often uses n=2 or n=6.
        """
        super(TRMBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_recursions = num_recursions

        # Project input x to an initial latent z
        # In TRM.pdf, initially z is an embedding of x.
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # The core recursive network: it takes (x_input, z_latent) and outputs a new z_latent
        # The input to TRMNet will be the concatenation of x_input and z_latent
        self.trm_net = TRMNet(input_dim + hidden_dim, hidden_dim)

        # Output head to project the final latent state to the desired final_feature_dim
        # These features will then be passed to a final classification layer.
        self.output_projection = nn.Linear(hidden_dim, final_feature_dim)

    def forward(self, x_input):
        """
        Args:
            x_input (Tensor): Input features from the backbone (e.g., from EfficientNet's GAP).
                              Shape: (batch_size, input_dim).
        """
        # Initialize latent state z with a projection of the input features
        z_latent = self.input_projection(x_input)

        # Recursive processing
        # In TRM.pdf, Figure 1, the arrow "Update z given x, y, z"
        # and "recursive reasoning" block
        for _ in range(self.num_recursions):
            # Concatenate x_input with current z_latent to form the input to TRMNet
            combined_input_latent = torch.cat((x_input, z_latent), dim=-1)
            z_latent = self.trm_net(combined_input_latent)
        
        # Project the final latent state to output features
        output_features = self.output_projection(z_latent)
        return output_features

if __name__ == '__main__':
    # Test with dummy data
    batch_size = 4
    efficientnet_output_dim = 1280 # Assuming EfficientNet-Lite0 output features after GAP
    num_classes = 10

    # Initialize TRM Block (Nano-TRM parameters)
    trm_output_features_dim = 256 # TRM will output 256 features to be classified later
    trm_block = TRMBlock(input_dim=efficientnet_output_dim,
                         final_feature_dim=trm_output_features_dim,
                         hidden_dim=128, # Reduced hidden_dim for VRAM constraints
                         num_recursions=2) # Reduced recursions for efficiency
    print(trm_block)

    # Simulate EfficientNet output (flattened features after GAP)
    dummy_input_features = torch.randn(batch_size, efficientnet_output_dim)
    
    # Forward pass through TRM
    output = trm_block(dummy_input_features)
    print(f"TRM output shape: {output.shape}") # Expected: torch.Size([4, 256])
    
    # Check parameter count for the Nano-TRM
    total_params = sum(p.numel() for p in trm_block.parameters() if p.requires_grad)
    print(f"Total trainable parameters for TRMBlock (Nano-TRM): {total_params / 1e6:.2f} M") 
    # This should be significantly less than the 7M mentioned in the paper for their config,
    # making it suitable for the MX450 GPU.

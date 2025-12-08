import torch
import torch.nn as nn
from copy import deepcopy

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation for model weights.
    Adapted from: https://github.com/soumith/dcgan.torch/issues/9
    
    Args:
        t (float): Interpolation factor between 0 and 1.
        v0 (torch.Tensor): First tensor.
        v1 (torch.Tensor): Second tensor.
        DOT_THRESHOLD (float): Threshold to switch to linear interpolation if tensors are too close.
                               Prevents numerical instability for very small angles.
    Returns:
        torch.Tensor: Interpolated tensor.
    """
    if not isinstance(v0, torch.Tensor):
        v0 = torch.from_numpy(v0)
    if not isinstance(v1, torch.Tensor):
        v1 = torch.from_numpy(v1)

    dot = torch.sum(v0 * v1)
    
    if dot.item() > DOT_THRESHOLD: # Linear interpolation if too close
        return (1 - t) * v0 + t * v1

    theta_0 = torch.acos(dot.clamp(-1, 1)) # Clamp to avoid NaN from floating point errors
    sin_theta_0 = torch.sin(theta_0) 
    
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

class M2N2WeightMerger:
    """
    A simplified M2N2 (Model Merging of Natural Niches) weight merging utility.
    This utility focuses on merging two model state_dicts using SLERP
    and applying a split point strategy as described in M2N2.pdf (Eq. 2).
    
    This is a simplified implementation of the M2N2 *merging mechanism*, not the full
    evolutionary search with "attraction" and "competition" which would require
    complex population management and training on diverse tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def merge_state_dicts(self, state_dict_a, state_dict_b, mix_ratio, split_idx=None):
        """
        Merges two model state dictionaries using SLERP and a split point.
        
        Args:
            state_dict_a (dict): State dictionary of the first model.
            state_dict_b (dict): State dictionary of the second model.
            mix_ratio (float): The mixing coefficient (wm in M2N2.pdf).
                               Weights for the first part: (1-mix_ratio)*A + mix_ratio*B.
                               Weights for the second part: mix_ratio*A + (1-mix_ratio)*B.
            split_idx (int, optional): Index where to split the flattened parameter list (ws in M2N2.pdf).
                                     If None, applies mix_ratio globally to all parameters (no split).
        Returns:
            dict: The merged state dictionary.
        """
        merged_state_dict = deepcopy(state_dict_a) # Start with a copy of A's weights

        # Collect parameters as a flattened list of tensors to apply split_idx
        params_a_list = [param for param in state_dict_a.values()]
        params_b_list = [param for param in state_dict_b.values()]

        if len(params_a_list) != len(params_b_list):
            raise ValueError("State dictionaries must have the same number of parameters (keys).")

        if split_idx is None:
            # If no split_idx, apply mix_ratio globally
            split_idx = len(params_a_list) + 1 # Effectively makes all params fall into the first part
        
        # Merge by iterating through the state_dict keys to maintain structure
        param_counter = 0
        for name in merged_state_dict.keys():
            param_a = state_dict_a[name].to('cpu')
            param_b = state_dict_b[name].to('cpu')
            
            # This logic applies split_idx on a per-tensor basis.
            # A more faithful implementation of M2N2's `w_s` would flatten all params,
            # split, then re-pack. For simplicity and managing tensor shapes, we split
            # based on the *number of tensors* encountered.
            
            if param_counter < split_idx:
                # First part of the model (f_wm in Eq. 2)
                merged_state_dict[name] = slerp(mix_ratio, param_a, param_b)
            else:
                # Second part of the model (f_1-wm in Eq. 2)
                # Note: M2N2 paper has f_1-wm(theta_A^>=ws, theta_B^>=ws)
                # This implies mixing with (1-t) on A and t on B for first part,
                # then (t) on A and (1-t) on B for second part.
                merged_state_dict[name] = slerp(1.0 - mix_ratio, param_a, param_b)
            
            param_counter += 1

        return merged_state_dict

    def find_best_merge(self, model_class, models_to_merge, eval_func, num_mix_ratios=5, num_split_points=5):
        """
        A simplified grid search to find the best merge parameters (mix_ratio, split_idx).
        This approximates the evolutionary search aspect of M2N2 for finding optimal merge.
        
        Args:
            model_class (nn.Module): The class of the models (e.g., HybridEfficientNetTRM or EfficientNetBaseline).
                                     Used to instantiate a temporary model for evaluation.
            models_to_merge (list of nn.Module): A list of 2 trained models (instances) to merge.
            eval_func (callable): A function that takes a model (nn.Module) and returns its validation accuracy.
                                  Signature: eval_func(model) -> accuracy (float).
            num_mix_ratios (int): Number of mix_ratio values to test between 0 and 1 (inclusive).
            num_split_points (int): Number of split_point_index values to test.
                                     These are indices into the list of parameter tensors.
        Returns:
            tuple: (best_merged_model_state_dict, best_accuracy, best_mix_ratio, best_split_idx)
        """
        if len(models_to_merge) != 2:
            raise ValueError("M2N2 merging currently supports merging exactly two models.")
        
        model_a = models_to_merge[0]
        model_b = models_to_merge[1]

        best_accuracy = -1.0
        best_merged_state_dict = None
        best_mix_ratio = 0.0
        best_split_idx = 0

        # Evaluate individual models first and set as initial best
        print("Evaluating individual models before merging:")
        acc_a = eval_func(model_a)
        print(f"  Model A accuracy: {acc_a:.4f}")
        acc_b = eval_func(model_b)
        print(f"  Model B accuracy: {acc_b:.4f}")

        # Determine the initial best
        if acc_a >= acc_b: # Prefer A if equal
            best_accuracy = acc_a
            best_merged_state_dict = deepcopy(model_a.state_dict())
            best_mix_ratio = 0.0 # Convention: 0.0 means purely model A
            best_split_idx = None
        else:
            best_accuracy = acc_b
            best_merged_state_dict = deepcopy(model_b.state_dict())
            best_mix_ratio = 1.0 # Convention: 1.0 means purely model B
            best_split_idx = None
            
        param_tensor_count = len(list(model_a.state_dict().keys()))

        mix_ratios = [i / (num_mix_ratios - 1) for i in range(num_mix_ratios)] if num_mix_ratios > 1 else [0.5]
        
        # Define strategic split points based on the number of parameter tensors
        # Example split points: 0%, 25%, 50%, 75%, 100% of the parameter tensors
        split_indices = []
        if num_split_points > 1:
            for i in range(num_split_points):
                idx = int(param_tensor_count * i / (num_split_points - 1))
                split_indices.append(idx)
            split_indices = sorted(list(set(split_indices))) # Ensure uniqueness
        else: # If only 1 split point requested, use None (global merge)
            split_indices = [None] 

        print(f"Testing {len(mix_ratios)} mix ratios and {len(split_indices)} split points...")
        print(f"Mix Ratios: {['{:.2f}'.format(r) for r in mix_ratios]}")
        print(f"Split Indices (approx. tensor index): {split_indices}")

        for mix_ratio in mix_ratios:
            for split_idx in split_indices:
                if (mix_ratio == 0.0 and split_idx is None) or \
                   (mix_ratio == 1.0 and split_idx is None):
                    # Skip re-evaluating pure A or pure B if already considered as initial best
                    continue

                print(f"  Merging with mix_ratio={mix_ratio:.2f}, split_idx={split_idx}")
                merged_sd = self.merge_state_dicts(model_a.state_dict(), model_b.state_dict(), mix_ratio, split_idx)
                
                # Create a temporary model to evaluate the merged state_dict
                temp_model = model_class(num_classes=self.num_classes)
                temp_model.load_state_dict(merged_sd)
                
                current_accuracy = eval_func(temp_model)
                print(f"    -> Accuracy: {current_accuracy:.4f}")

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_merged_state_dict = deepcopy(merged_sd) # Deepcopy to prevent modification
                    best_mix_ratio = mix_ratio
                    best_split_idx = split_idx
        
        print(f"\nBest merge found: Accuracy={best_accuracy:.4f}, mix_ratio={best_mix_ratio:.2f}, split_idx={best_split_idx}")
        return best_merged_state_dict, best_accuracy, best_mix_ratio, best_split_idx


if __name__ == '__main__':
    # Dummy Model for testing
    class DummyModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, num_classes)
        
        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    # Create two dummy models with different initial weights
    model1 = DummyModel(num_classes=2)
    model2 = DummyModel(num_classes=2)
    for p in model2.parameters():
        p.data.fill_(0.5) # Initialize differently

    # Dummy evaluation function (e.g., on a validation set)
    def dummy_eval(model):
        # In a real scenario, this would run model.eval() and compute accuracy
        # For testing, we just return a simulated accuracy based on some parameter
        # For simplicity, let's say models with higher average weight magnitude are better
        avg_weight_mag = sum(p.data.abs().mean().item() for p in model.parameters())
        # Simulate some accuracy improvement for merging
        return 0.5 + avg_weight_mag * 0.1 # Placeholder: replace with real validation logic

    # Test M2N2WeightMerger
    merger = M2N2WeightMerger(num_classes=2) # num_classes is needed for model_class instantiation
    
    # Example: merging model1 and model2
    models = [model1, model2]
    best_sd, best_acc, best_mix_r, best_split_i = merger.find_best_merge(
        DummyModel, models, dummy_eval, num_mix_ratios=3, num_split_points=3
    )

    print(f"\nResult of merging (main test):")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Mix Ratio: {best_mix_r:.2f}")
    print(f"Best Split Index: {best_split_i}")

    # Load the best merged state dict into a new model
    final_model = DummyModel(num_classes=2)
    final_model.load_state_dict(best_sd)

    # Verify a few parameters (optional)
    print("\nVerifying merged model parameters (example):")
    print(f"Final model linear1.weight mean: {final_model.linear1.weight.mean().item():.4f}")
    print(f"Model1 linear1.weight mean: {model1.linear1.weight.mean().item():.4f}")
    print(f"Model2 linear1.weight mean: {model2.linear1.weight.mean().item():.4f}")

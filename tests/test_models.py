import unittest
import torch
from src.models.efficientnet_lite import EfficientNetLite0
from src.models.trm import TRMBlock
from src.models.hybrid_model import EfficientNetBaseline, HybridEfficientNetTRM

class TestModelShapes(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 4
        self.input_size = (3, 32, 32) # CIFAR-10 image size
        self.dummy_input = torch.randn(self.batch_size, *self.input_size)

    def test_efficientnet_lite0_output_shape(self):
        model = EfficientNetLite0(num_classes=self.num_classes)
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes), 
                         f"EfficientNetLite0 output shape mismatch: {output.shape}")

    def test_efficientnet_baseline_output_shape(self):
        model = EfficientNetBaseline(num_classes=self.num_classes)
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes), 
                         f"EfficientNetBaseline output shape mismatch: {output.shape}")

    def test_trm_block_output_shape(self):
        efficientnet_output_dim = 1280 # Expected from EfficientNetLite0 after GAP
        trm_final_feature_dim = 256 # As defined in hybrid_model.py
        trm_block = TRMBlock(input_dim=efficientnet_output_dim, 
                             final_feature_dim=trm_final_feature_dim)
        
        dummy_efficientnet_features = torch.randn(self.batch_size, efficientnet_output_dim)
        output = trm_block(dummy_efficientnet_features)
        self.assertEqual(output.shape, (self.batch_size, trm_final_feature_dim),
                         f"TRMBlock output shape mismatch: {output.shape}")

    def test_hybrid_efficientnet_trm_output_shape(self):
        model = HybridEfficientNetTRM(num_classes=self.num_classes)
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes),
                         f"HybridEfficientNetTRM output shape mismatch: {output.shape}")

class TestForwardPass(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 4
        self.input_size = (3, 32, 32) # CIFAR-10 image size
        self.dummy_input = torch.randn(self.batch_size, *self.input_size)

    def test_efficientnet_lite0_forward_pass(self):
        model = EfficientNetLite0(num_classes=self.num_classes)
        try:
            output = model(self.dummy_input)
            self.assertIsInstance(output, torch.Tensor)
            self.assertFalse(torch.isnan(output).any())
        except Exception as e:
            self.fail(f"EfficientNetLite0 forward pass failed: {e}")

    def test_hybrid_efficientnet_trm_forward_pass(self):
        model = HybridEfficientNetTRM(num_classes=self.num_classes)
        try:
            output = model(self.dummy_input)
            self.assertIsInstance(output, torch.Tensor)
            self.assertFalse(torch.isnan(output).any())
        except Exception as e:
            self.fail(f"HybridEfficientNetTRM forward pass failed: {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
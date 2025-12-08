import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor=8, min_value=None):
    """
    Ensure that all layers have a channel number divisible by 8.
    This is to optimize for certain hardware.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU6 activation.
    Used in EfficientNetLite stem and head.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu6(self.bn(self.conv(x)))

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block.
    EfficientNet-Lite uses ReLU6 and no Squeeze-and-Excitation (SE).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1, 1, 0))
        
        # Depthwise convolution
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetLite0(nn.Module):
    """
    EfficientNet-Lite0 model adapted for CIFAR-10 (32x32 input).
    Changes: initial stride of 2 replaced with 1 to preserve resolution.
    """
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0):
        super(EfficientNetLite0, self).__init__()
        
        # Inverted residual blocks configs: (in_channels, kernel_size, stride, expand_ratio, num_blocks)
        # Note: input_channel will be set dynamically based on previous layer's out_channels.
        # The number of blocks for each stage is scaled by depth_mult.
        inverted_residual_setting = [
            # k, s, e, c, n
            [3, 1, 1, 16, 1], # MBConv1_3x3, 16, 1
            [3, 2, 6, 24, 2], # MBConv6_3x3, 24, 2
            [5, 2, 6, 40, 2], # MBConv6_5x5, 40, 2
            [3, 2, 6, 80, 3], # MBConv6_3x3, 80, 3
            [5, 1, 6, 112, 3], # MBConv6_5x5, 112, 3 (stride 1 here for CIFAR, original is 2 for this block in B0)
            [5, 2, 6, 192, 4], # MBConv6_5x5, 192, 4
            [3, 1, 6, 320, 1], # MBConv6_3x3, 320, 1
        ]
        
        # Stem
        in_channels = 3 # RGB input
        out_channels = _make_divisible(32 * width_mult)
        # CIFAR Adaptation: Initial stride 1 to keep 32x32 resolution
        self.stem = ConvBNReLU(in_channels, out_channels, 3, 1, 1) 
        in_channels = out_channels

        self.blocks = nn.Sequential()
        # Build blocks
        for k, s, e, c, n in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.add_module(f"mbconv_{in_channels}_{out_channels}_k{k}_s{stride}_e{e}_{i}",
                                       MBConvBlock(in_channels, out_channels, k, stride, e))
                in_channels = out_channels
        
        # Head
        self.head_conv = ConvBNReLU(in_channels, _make_divisible(1280 * width_mult), 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(_make_divisible(1280 * width_mult), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Test with dummy data
    model = EfficientNetLite0(num_classes=10)
    print(model)
    
    input_tensor = torch.randn(1, 3, 32, 32) # CIFAR-10 size
    output = model(input_tensor)
    print(f"Output shape: {output.shape}") # Expected: torch.Size([1, 10])

    # Check parameter count (roughly)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M") # Should be around 4.x M

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

class FeatureReturner(nn.Module):
    def __init__(self):
        super(FeatureReturner, self).__init__()

    def forward(self, x):
        return x
    
class Modified_CNN(nn.Module):
    def __init__(self, cnn_model, layer_idx_ls):
        super(Modified_CNN, self).__init__()
        self.features = None
        layer_ls = [cnn_model.features[:layer_idx_ls[0] + 1], FeatureReturner()]
        for i in range(1, len(layer_idx_ls)):
            layer_ls.append(cnn_model.features[layer_idx_ls[i-1] + 1:layer_idx_ls[i] + 1])
            layer_ls.append(FeatureReturner())
        self.model = nn.Sequential(*layer_ls)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, FeatureReturner):
                features.append(x)
        return features
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        return self.conv(self.reflection_pad(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, 3, 1)
        self.relu = nn.ReLU()
        self.conv2 = ConvBlock(channels, channels, 3, 1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        return self.conv(self.reflection_pad(self.upsample(x)))
    
class ImageTranformNet(nn.Module):
    def __init__(self):
        super(ImageTranformNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ConvBlock(32, 64, 3, 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ConvBlock(64, 128, 3, 2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleBlock(128, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            UpsampleBlock(64, 32, 3, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
            ConvBlock(32, 3, 9, 1),
            nn.InstanceNorm2d(3),
            nn.Tanh()
        )
    
    def forward(self, x):
        return (self.model(x) + 1) / 2
    
    
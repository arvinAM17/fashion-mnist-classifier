import torch.nn as nn
from .blocks import Conv2DBlock, DoubleLinearBlock

class FashionMNISTCNN(nn.Module):
    def __init__(self, output_dimension: int = 10, hidden_layer_dimension: int = 128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            Conv2DBlock(padding=3), # 1 * 28 * 28 -> 64 * 16 * 16
            Conv2DBlock(), # 64 * 16 * 16 -> 64 * 8 * 8 
            Conv2DBlock(out_channels=128), # 64 * 8 * 8 -> 128 * 4 * 4
            DoubleLinearBlock(hidden_dim=hidden_layer_dimension, output_dim=output_dimension) # 128 * 4 * 4 -> 128 -> 10
        )

    def forward(self, x):
        return self.conv_layers(x)

import torch.nn as nn
from .blocks import DoubleLinearBlock

class FashionMNISTBaseline(nn.Module):
    def __init__(self, hidden_layers: int, output_dimension: int):
        super().__init__()

        self.stacked_layers = DoubleLinearBlock(hidden_dim=hidden_layers, output_dim=output_dimension)

    def forward(self, x):
        return self.stacked_layers(x)

import torch.nn as nn
from .module_name_map import pooling_layer_map

class Conv2DBlock(nn.Module):
    def __init__(
        self,
        out_channels: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        activation: nn.Module = nn.ReLU, 
        batch_norm: bool = True,
        dropout: float = 0.1,
        pooling: str = "MaxPool2d", 
        pooling_params: dict = None,
    ):
        super().__init__()

        self.pooling_layer = pooling_layer_map[pooling]

        if pooling_params is None:
            pooling_params = {}

        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LazyBatchNorm2d() if batch_norm else nn.Identity(),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            pooling(**pooling_params) if pooling else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)
    
class DoubleLinearBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 10,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.3
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.block(x)

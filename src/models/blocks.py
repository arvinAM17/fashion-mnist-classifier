import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(
        self,
        out_channels: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            activation(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )

    def forward(self, x):
        return self.block(x)
    
class DoubleLinearBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 10,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.block(x)

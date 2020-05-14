import torch.nn as nn
from .marching_squares import MarchingSquares


class SnakeNet(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels=16):
        super().__init__()
        bc = base_channels

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 2, stride=2),
            nn.Conv2d(1 * bc, 2 * bc, 2, stride=2), nn.Tanh(),
            nn.Conv2d(2 * bc, 4 * bc, 2, stride=2), nn.Tanh(),
            nn.Conv2d(4 * bc, 8 * bc, 2, stride=2), nn.Tanh(),
        )

        self.make_coarse = nn.Conv2d(8 * bc, 1, 1)

        self.marching_squares = MarchingSquares([16, 16])

    def forward(self, x):
        features = self.backbone(x)
        coarse = self.make_coarse(features)

        # Do Marching squares
        x = features

        return x, coarse

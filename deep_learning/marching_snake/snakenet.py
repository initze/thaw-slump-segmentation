import torch.nn as nn


class SnakeNet(nn.Module):
    """
    A straight-forward UNet implementation
    """

    def __init__(self, input_channels, output_channels, base_channels=16):
        super().__init__()
        bc = base_channels

        # Backbone
        self.backbone = nn.ModuleList([
            nn.Conv2d(input_channels, base_channels, 2, stride=2),
            nn.Conv2d(1 * bc, 2 * bc, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(2 * bc, 4 * bc, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(4 * bc, 8 * bc, 2, stride=2), nn.ReLU(inplace=True),
        ])

    def forward(self, x):
        coarse = self.backbone(x)

        # Do Marching squares
 
        x = coarse

        return x, coarse

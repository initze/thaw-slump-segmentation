import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock, UpBlock, WithSE


class UNet(nn.Module):
    """
    A straight-forward UNet implementation
    """
    tasks = ['seg']

    def __init__(self, input_channels, output_channels=1, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', norm=None, stack_height=4):
        super().__init__()
        bc = base_channels
        self.init = conv_block(input_channels, bc, norm, padding_mode=padding_mode)

        conv_args = dict(
            conv_block=conv_block,
            norm=norm,
            padding_mode=padding_mode
        )

        self.down_blocks = nn.ModuleList([
            DownBlock((1<<i)*bc, (2<<i)*bc, **conv_args)
            for i in range(stack_height)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock((2<<i)*bc, (1<<i)*bc, **conv_args)
            for i in reversed(range(stack_height))
        ])

        self.final = conv_block(bc, bc, norm, padding_mode=padding_mode)
        self.segment = nn.Conv2d(bc, output_channels, 1)

    def encode(self, x):
        x = self.init(x)
        for block in self.down_blocks:
            x = block(x)
        return x


    def forward(self, x):
        x = self.init(x)

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)

        x = self.final(x)
        x = self.segment(x)

        return x

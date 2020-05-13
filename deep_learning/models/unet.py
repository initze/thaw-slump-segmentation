import torch.nn as nn
from .layers import Convx2, DownBlock, UpBlock


class UNet(nn.Module):
    """
    A straight-forward UNet implementation
    """

    def __init__(self, input_channels, output_channels, base_channels=16, conv_block=Convx2, 
            padding_mode='zeros', batch_norm=True):
        super().__init__()
        bc = base_channels
        self.init = conv_block(input_channels, bc, batch_norm, padding_mode)

        self.down1 = DownBlock(1 * bc,  2 * bc, conv_block, batch_norm, padding_mode)
        self.down2 = DownBlock(2 * bc,  4 * bc, conv_block, batch_norm, padding_mode)
        self.down3 = DownBlock(4 * bc,  8 * bc, conv_block, batch_norm, padding_mode)
        self.down4 = DownBlock(8 * bc, 16 * bc, conv_block, batch_norm, padding_mode)

        self.up1 = UpBlock(16 * bc, 8 * bc, conv_block, batch_norm, padding_mode)
        self.up2 = UpBlock( 8 * bc, 4 * bc, conv_block, batch_norm, padding_mode)
        self.up3 = UpBlock( 4 * bc, 2 * bc, conv_block, batch_norm, padding_mode)
        self.up4 = UpBlock( 2 * bc, 1 * bc, conv_block, batch_norm, padding_mode)

        self.final = conv_block(bc, bc, batch_norm, padding_mode)
        self.segment = nn.Conv2d(bc, output_channels, 1)

    def forward(self, x):
        x = self.init(x)

        skip1 = x
        x = self.down1(x)
        skip2 = x
        x = self.down2(x)
        skip3 = x
        x = self.down3(x)
        skip4 = x
        x = self.down4(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.final(x)
        x = self.segment(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock


class BoringBackbone(nn.Module):
    def __init__(self, input_channels, num_features=64, conv_block=Convx2,
                 batch_norm=True):
        super().__init__()
        bc = num_features // 4
        self.pre = Convx2(2, bc),

        self.pyramid = nn.ModuleList([
            DownBlock(bc,  bc, conv_block, batch_norm),
            DownBlock(bc,  bc, conv_block, batch_norm),
            DownBlock(bc,  bc, conv_block, batch_norm),
        ])

    def forward(self, x):
        x = self.pre(x)
        scale = 1
        features = [x]
        for block in self.pyramid:
            x = block(x)
            scale *= 2
            features.append(F.interpolate(x, scale_factor=scale, mode='bilinear'))
        return torch.cat(features, dim=1)

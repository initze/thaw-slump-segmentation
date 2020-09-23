import torch
import math
import torch.nn as nn
import torch.nn.functional as F



def get_norm(name, channels):
    if name is None:
        return nn.Identity()
    if name in ('BN', 'BatchNorm'):
        return nn.BatchNorm2d(channels)
    elif name in ('IN', 'InstanceNorm'):
        return nn.InstanceNorm2d(channels)
    elif name in ('SE', 'SqueezeExcitation'):
        return SqueezeExcitation(c_out, reduction=8)
    else:
        raise ValueError(f'No norm named "{name}" known.')


class Convx2(nn.Module):
    def __init__(self, c_in, c_out, norm, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=1, padding_mode=padding_mode, bias=(norm is None))
        self.conv1 = nn.Conv2d(c_in, c_out, 3, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, **conv_args)
        self.norm1 = get_norm(norm, c_out)
        self.norm2 = get_norm(norm, c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, conv_block=Convx2, batch_norm=True):
        super().__init__()
        if c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1)
        else:
            self.skip = nn.Identity()

        self.convblock = conv_block(c_in, c_out, batch_norm)

    def forward(self, x):
        skipped = self.skip(x)
        residual = self.convblock(x)
        return skipped + residual


class DenseBlock(nn.Module):
    def __init__(self, c_in, c_out, bn, dense_size=8):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, bias=not bn)
        self.dense_convs = nn.ModuleList([
            nn.Conv2d(c_in + i * dense_size, dense_size, **conv_args)
            for i in range(4)
        ])
        self.final = nn.Conv2d(c_in + 4 * dense_size, c_out, **conv_args)

        if bn:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(dense_size)
                for i in range(4)
            ])
            self.bn_final = nn.BatchNorm2d(c_out)
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(4)])
            self.bn_final = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv, bn in zip(self.dense_convs, self.bns):
            x = torch.cat([x, self.relu(bn(conv(x)))], dim=1)
        x = self.relu(self.bn_final(self.final(x)))
        return x


class SqueezeExcitation(nn.Module):
    """
    adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    See: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = int(math.ceil(channels / reduction))
        self.squeeze = nn.Conv2d(channels, reduced, 1)
        self.excite = nn.Conv2d(reduced, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.avg_pool2d(x, x.shape[2:])
        y = self.relu(self.squeeze(y))
        y = torch.sigmoid(self.excite(y))
        return x * y


def WithSE(conv_block, reduction=8):
    def make_block(c_in, c_out, **kwargs):
        return nn.Sequential(
            conv_block(c_in, c_out, **kwargs),
            SqueezeExcitation(c_out, reduction=reduction)
        )
    make_block.__name__ = f"WithSE({conv_block.__name__})"
    return make_block



class DownBlock(nn.Module):
    """
    UNet Downsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2, norm=None, padding_mode='zeros'):
        super().__init__()
        bias = (norm is None)
        self.convdown = nn.Conv2d(c_in, c_in, 2, stride=2, bias=bias)
        self.norm = get_norm(norm, c_in)
        self.relu = nn.ReLU(inplace=True)

        self.conv_block = conv_block(c_in, c_out, norm=norm, padding_mode=padding_mode)

    def forward(self, x):
        x = self.relu(self.norm(self.convdown(x)))
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    UNet Upsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2, norm=None, padding_mode='zeros'):
        super().__init__()
        bias = (norm is None)
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2, bias=bias)
        self.norm = get_norm(norm, c_in // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = conv_block(c_in, c_out, norm=norm, padding_mode=padding_mode)

    def forward(self, x, skip):
        x = self.relu(self.norm(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

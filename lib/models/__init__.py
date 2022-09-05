from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .unet3p.unet3p import UNet3Plus
from .manet import MAnet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN
import torch.nn as nn

from . import encoders
from . import utils
from . import losses

from .__version__ import __version__

from typing import Optional
import torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "random",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parameters
    """

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN, UNet3Plus]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Avalibale options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )


def create_loss(
    name: str,
) -> torch.nn.Module:
    """LossFn wrapper. Allows to create any loss_fn just with parameters"""

    losses_list = [
        losses.JaccardLoss,
        losses.DiceLoss,
        losses.FocalLoss,
        losses.LovaszLoss,
        losses.SoftBCEWithLogitsLoss,
        losses.SoftCrossEntropyLoss
    ]
    losses_dict = {l.__name__.lower(): l for l in losses_list}

    builtin_losses = [nn.BCELoss]
    builtin_losses_dict = {l.__name__.lower(): l for l in builtin_losses}
    name = name.lower()
    if name in losses_dict:
        return losses_dict[name](mode=losses.BINARY_MODE)
    elif name in builtin_losses_dict:
        return builtin_losses_dict[name]()
    raise KeyError("Wrong loss type `{}`. Available options are: {}".format(
        name, list(builtin_losses_dict.keys()) + list(losses_dict.keys()),
    ))

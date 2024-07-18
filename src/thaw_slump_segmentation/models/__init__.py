from thaw_slump_segmentation.models.unet import Unet
from thaw_slump_segmentation.models.unetplusplus import UnetPlusPlus
from thaw_slump_segmentation.models.unet3p.unet3p import Unet3Plus
from thaw_slump_segmentation.models.manet import MAnet
from thaw_slump_segmentation.models.linknet import Linknet
from thaw_slump_segmentation.models.fpn import FPN
from thaw_slump_segmentation.models.pspnet import PSPNet
from thaw_slump_segmentation.models.deeplabv3 import DeepLabV3, DeepLabV3Plus
from thaw_slump_segmentation.models.pan import PAN
import torch.nn as nn

from thaw_slump_segmentation.models import encoders
from thaw_slump_segmentation.models import utils
from thaw_slump_segmentation.models import losses

from thaw_slump_segmentation.models.__version__ import __version__

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

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN, Unet3Plus]
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

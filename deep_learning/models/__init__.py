# flake8: noqa
from .unet import UNet
from .boring_backbone import BoringBackbone
from .ocr import OCRNet
from .logistic_regression import LogisticRegression
from .enet import ENet
from .merger import Merger

_MODELS = {
    'UNet': UNet,
    'OCR': OCRNet,
    'ENet': ENet
}

def get_model(model_name):
    try:
        return _MODELS[model_name]
    except KeyError:
        print(f'Can\'t provide Model called "{model_name}"')

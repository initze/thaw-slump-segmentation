from math import sqrt
from typing import Literal

import torch
import torchvision

MatchingMetric = Literal['iou', 'boundary']


@torch.no_grad()
def erode_pytorch(mask: torch.Tensor, iterations: int = 1, validate_args: bool = False) -> torch.Tensor:
    """Erodes a binary mask using a square kernel in PyTorch.

    Args:
        mask (torch.Tensor): The binary mask. Shape: (batch_size, height, width), dtype: torch.uint8
        iterations (int, optional): The size of the erosion. Defaults to 1.

    Returns:
        torch.Tensor: The eroded mask. Shape: (batch_size, height, width), dtype: torch.uint8
    """
    if validate_args:
        assert mask.dim() == 3, f'Expected 3 dimensions, got {mask.dim()}'
        assert mask.dtype == torch.uint8, f'Expected torch.uint8, got {mask.dtype}'
        assert mask.min() >= 0 and mask.max() <= 1, f'Expected binary mask, got {mask.min()} and {mask.max()}'

    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    erode = torch.nn.functional.conv2d(mask.float().unsqueeze(1), kernel, padding=1, stride=1)

    for _ in range(iterations - 1):
        erode = torch.nn.functional.conv2d(erode, kernel, padding=1, stride=1)

    return (erode == erode.max()).to(torch.uint8).squeeze(1)


@torch.no_grad()
def get_boundaries(
    instances_target_onehot: torch.Tensor,
    instances_preds_onehot: torch.Tensor,
    dilation: float | int = 0.02,
    validate_args: bool = False,
):
    if validate_args:
        assert instances_target_onehot.dim() == 3, f'Expected 3 dimensions, got {instances_target_onehot.dim()}'
        assert instances_preds_onehot.dim() == 3, f'Expected 3 dimensions, got {instances_preds_onehot.dim()}'
        assert (
            instances_target_onehot.dtype == torch.uint8
        ), f'Expected torch.uint8, got {instances_target_onehot.dtype}'
        assert instances_preds_onehot.dtype == torch.uint8, f'Expected torch.uint8, got {instances_preds_onehot.dtype}'
        assert (
            instances_target_onehot.shape[1:] == instances_preds_onehot.shape[1:]
        ), f'Shapes (..., H, W) do not match: {instances_target_onehot.shape} and {instances_preds_onehot.shape}'
        assert (
            instances_target_onehot.min() >= 0 and instances_target_onehot.max() <= 1
        ), f'Expected binary mask, got {instances_target_onehot.min()} and {instances_target_onehot.max()}'
        assert (
            instances_preds_onehot.min() >= 0 and instances_preds_onehot.max() <= 1
        ), f'Expected binary mask, got {instances_preds_onehot.min()} and {instances_preds_onehot.max()}'

    n, h, w = instances_target_onehot.shape
    if isinstance(dilation, float):
        img_diag = sqrt(h**2 + w**2)
        dilation = int(round(dilation * img_diag))
        if dilation < 1:
            dilation = 1

    # Pad the instances to avoid boundary issues
    pad = torchvision.transforms.Pad(1)
    instances_target_onehot_padded = pad(instances_target_onehot)
    instances_preds_onehot_padded = pad(instances_preds_onehot)

    # Erode the instances to get the boundaries
    eroded_target = erode_pytorch(instances_target_onehot_padded, iterations=dilation, validate_args=validate_args)
    eroded_preds = erode_pytorch(instances_preds_onehot_padded, iterations=dilation, validate_args=validate_args)

    # Remove the padding
    eroded_target = eroded_target[:, 1:-1, 1:-1]
    eroded_preds = eroded_preds[:, 1:-1, 1:-1]

    # Calculate the boundary of the instances
    boundaries_target = instances_target_onehot - eroded_target
    boundaries_preds = instances_preds_onehot - eroded_preds
    return boundaries_target, boundaries_preds


@torch.no_grad()
def instance_boundary_iou(
    instances_target_onehot: torch.Tensor,
    instances_preds_onehot: torch.Tensor,
    dilation: float | int = 0.02,
    validate_args: bool = False,
) -> torch.Tensor:
    """Calculates the IoU of the boundaries of instances.
    Expects non-batched, one-hot encoded input from skimage.measure.label

    Args:
        instances_target (torch.Tensor): The instance mask of the target.
            Shape: (num_instances, height, width), dtype: torch.uint8
        instances_preds (torch.Tensor): The instance mask of the prediction.
            Shape: (num_instances, height, width), dtype: torch.uint8
        dilation (float | int): The dilation factor for the boundary.
            Dilation in pixels if int, else ratio to calculate `dilation = dilation_ratio * image_diagonal`.
            Default: 0.02

    Returns:
        torch.Tensor: The IoU of the boundaries. Shape: (num_instances,)
    """

    # Calculate the boundary of the instances
    boundaries_target, boundaries_preds = get_boundaries(
        instances_target_onehot, instances_preds_onehot, dilation, validate_args
    )

    # Calculate the IoU of the boundaries (broadcast because of the different number of instances)
    intersection = (boundaries_target.unsqueeze(1) & boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    union = (boundaries_target.unsqueeze(1) | boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    iou = intersection / union  # Shape: (num_instances_target, num_instances_preds)

    return iou

from typing import Literal

import torch

from thaw_slump_segmentation.metrics.boundary_helpers import instance_boundary_iou

try:
    import cupy as cp
    from cucim.skimage.measure import label as label_cucim

    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    from skimage.measure import label as label_skimage


MatchingMetric = Literal['iou', 'boundary']


@torch.no_grad()
def mask_to_instances(x: torch.Tensor, validate_args: bool = False) -> list[torch.Tensor]:
    """Converts a binary segmentation mask into multiple instance masks. Expects a batched version of the input.

    Args:
        x (torch.Tensor): The binary segmentation mask. Shape: (batch_size, height, width), dtype: torch.uint8

    Returns:
        list[torch.Tensor]: The instance masks. Length of list: batch_size.
            Shape of a tensor: (height, width), dtype: torch.uint8
    """
    if validate_args:
        assert x.dim() == 3, f'Expected 3 dimensions, got {x.dim()}'
        assert x.dtype == torch.uint8, f'Expected torch.uint8, got {x.dtype}'
        assert x.min() >= 0 and x.max() <= 1, f'Expected binary mask, got {x.min()} and {x.max()}'

    if CUCIM_AVAILABLE:
        # Check if device is cuda
        assert x.device.type == 'cuda', f'Expected device to be cuda, got {x.device.type}'
        x = cp.asarray(x).astype(cp.uint8)

        instances = []
        for x_i in x:
            instances_i = label_cucim(x_i)
            instances_i = torch.tensor(instances_i, dtype=torch.uint8)
            instances.append(instances_i)
        return instances

    else:
        instances = []
        for x_i in x:
            x_i = x_i.cpu().numpy()
            instances_i = label_skimage(x_i)
            instances_i = torch.tensor(instances_i, dtype=torch.uint8)
            instances.append(instances_i)
        return instances

@torch.no_grad()
def match_instances(
    instances_target: torch.Tensor,
    instances_preds: torch.Tensor,
    match_metric: MatchingMetric = 'iou',
    match_threshold: float = 0.5,
    boundary_dilation: float | int = 0.02,
    validate_args: bool = False,
) -> tuple[int, int, int]:
    """Matches instances between target and prediction masks. Expects non-batched input from skimage.measure.label.

    Args:
        instances_target (torch.Tensor): The instance mask of the target. Shape: (height, width), dtype: torch.uint8
        instances_preds (torch.Tensor): The instance mask of the prediction. Shape: (height, width), dtype: torch.uint8

    Returns:
        tuple[int, int, int]: True positives, false positives, false negatives
    """
    if validate_args:
        assert instances_target.dim() == 2, f'Expected 2 dimensions, got {instances_target.dim()}'
        assert instances_preds.dim() == 2, f'Expected 2 dimensions, got {instances_preds.dim()}'
        assert instances_target.dtype == torch.uint8, f'Expected torch.uint8, got {instances_target.dtype}'
        assert instances_preds.dtype == torch.uint8, f'Expected torch.uint8, got {instances_preds.dtype}'
        assert (
            instances_target.shape == instances_preds.shape
        ), f'Shapes do not match: {instances_target.shape} and {instances_preds.shape}'

    # Create one-hot encoding of instances, so that each instance is a channel
    target_labels = list(range(1, instances_target.max() + 1))
    pred_labels = list(range(1, instances_preds.max() + 1))

    if len(target_labels) == 0:
        return 0, len(pred_labels), 0

    if len(pred_labels) == 0:
        return 0, 0, len(target_labels)

    instances_target_onehot = torch.stack([instances_target == i for i in target_labels], dim=0).to(torch.uint8)
    instances_preds_onehot = torch.stack([instances_preds == i for i in pred_labels], dim=0).to(torch.uint8)

    # Now the instances are channels, hence tensors of shape (num_instances, height, width)

    # Calculate IoU (we need to do a n-m intersection and union, therefore we need to broadcast)
    intersection = (instances_target_onehot.unsqueeze(1) & instances_preds_onehot.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    union = (instances_target_onehot.unsqueeze(1) | instances_preds_onehot.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    iou = intersection / union  # Shape: (num_instances_target, num_instances_preds)

    if match_metric == 'boundary':
        boundary_iou = instance_boundary_iou(
            instances_target_onehot, instances_preds_onehot, boundary_dilation, validate_args
        )
        # Use the minimum IoU of the instance and its boundary, like described in the BoundaryIoU paper
        iou = torch.min(iou, boundary_iou)

    # Match instances based on IoU
    tp = (iou >= match_threshold).sum().item()
    fp = len(pred_labels) - tp
    fn = len(target_labels) - tp

    return tp, fp, fn

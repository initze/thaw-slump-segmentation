from typing import Any, Literal, Optional, Sequence

import torch
import torchvision
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification.stat_scores import _AbstractStatScores
from torchmetrics.functional.classification.f_beta import _binary_fbeta_score_arg_validation, _fbeta_reduce
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_tensor_validation,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

try:
    import cupy as cp
    from cucim.skimage.measure import label as label_cucim

    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    from skimage.measure import label as label_skimage


MatchingMetric = Literal['iou', 'boundary']


@torch.no_grad()
def mask_to_instances(x: torch.Tensor) -> list[torch.Tensor]:
    """Converts a binary segmentation mask into multiple instance masks. Expects a batched version of the input.

    Args:
        x (torch.Tensor): The binary segmentation mask. Shape: (batch_size, height, width), dtype: torch.uint8

    Returns:
        list[torch.Tensor]: The instance masks. Length of list: batch_size. Shape of a tensor: (height, width), dtype: torch.uint8
    """
    assert x.dim() == 3, f'Expected 3 dimensions, got {x.dim()}'
    assert x.dtype == torch.uint8, f'Expected torch.uint8, got {x.dtype}'
    assert x.min() >= 0 and x.max() <= 1, f'Expected binary mask, got {x.min()} and {x.max()}'

    if CUCIM_AVAILABLE:
        assert x.device == torch.device('cuda'), f'Expected CUDA device, got {x.device}'
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
) -> tuple[int, int, int]:
    """Matches instances between target and prediction masks. Expects non-batched input from skimage.measure.label.

    Args:
        instances_target (torch.Tensor): The instance mask of the target. Shape: (height, width), dtype: torch.uint8
        instances_preds (torch.Tensor): The instance mask of the prediction. Shape: (height, width), dtype: torch.uint8

    Returns:
        tuple[int, int, int]: True positives, false positives, false negatives
    """
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
        boundary_iou = instance_boundary_iou(instances_target_onehot, instances_preds_onehot)
        # Use the minimum IoU of the instance and its boundary, like described in the BoundaryIoU paper
        iou = torch.min(iou, boundary_iou)

    # Match instances based on IoU
    tp = (iou >= match_threshold).sum().item()
    fp = len(pred_labels) - tp
    fn = len(target_labels) - tp

    return tp, fp, fn


@torch.no_grad()
def erode_pytorch(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """Erodes a binary mask using a square kernel in PyTorch.

    Args:
        mask (torch.Tensor): The binary mask. Shape: (batch_size, height, width), dtype: torch.uint8
        iterations (int, optional): The size of the erosion. Defaults to 1.

    Returns:
        torch.Tensor: The eroded mask. Shape: (batch_size, height, width), dtype: torch.uint8
    """
    assert mask.dim() == 3, f'Expected 3 dimensions, got {mask.dim()}'
    assert mask.dtype == torch.uint8, f'Expected torch.uint8, got {mask.dtype}'
    assert mask.min() >= 0 and mask.max() <= 1, f'Expected binary mask, got {mask.min()} and {mask.max()}'

    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    erode = torch.nn.functional.conv2d(mask.float().unsqueeze(1), kernel, padding=1, stride=1)

    for _ in range(iterations - 1):
        erode = torch.nn.functional.conv2d(erode, kernel, padding=1, stride=1)

    return (erode == erode.max()).to(torch.uint8).squeeze(1)


@torch.no_grad()
def instance_boundary_iou(
    instances_target_onehot: torch.Tensor, instances_preds_onehot: torch.Tensor, dilation: float | int = 0.02
) -> torch.Tensor:
    """Calculates the IoU of the boundaries of instances. Expects non-batched, one-hot encoded input from skimage.measure.label

    Args:
        instances_target (torch.Tensor): The instance mask of the target. Shape: (num_instances, height, width), dtype: torch.uint8
        instances_preds (torch.Tensor): The instance mask of the prediction. Shape: (num_instances, height, width), dtype: torch.uint8
        dilation (float | int): The dilation factor for the boundary. Dilation in pixels if int, else ratio to calculate `dilation = dilation_ratio * image_diagonal`. Default: 0.02

    Returns:
        torch.Tensor: The IoU of the boundaries. Shape: (num_instances,)
    """

    assert instances_target_onehot.dim() == 3, f'Expected 3 dimensions, got {instances_target_onehot.dim()}'
    assert instances_preds_onehot.dim() == 3, f'Expected 3 dimensions, got {instances_preds_onehot.dim()}'
    assert instances_target_onehot.dtype == torch.uint8, f'Expected torch.uint8, got {instances_target_onehot.dtype}'
    assert instances_preds_onehot.dtype == torch.uint8, f'Expected torch.uint8, got {instances_preds_onehot.dtype}'
    assert (
        instances_target_onehot.shape == instances_preds_onehot.shape
    ), f'Shapes do not match: {instances_target_onehot.shape} and {instances_preds_onehot.shape}'
    assert (
        instances_target_onehot.min() >= 0 and instances_target_onehot.max() <= 1
    ), f'Expected binary mask, got {instances_target_onehot.min()} and {instances_target_onehot.max()}'
    assert (
        instances_preds_onehot.min() >= 0 and instances_preds_onehot.max() <= 1
    ), f'Expected binary mask, got {instances_preds_onehot.min()} and {instances_preds_onehot.max()}'

    n, h, w = instances_target_onehot.shape
    if isinstance(dilation, float):
        img_diag = torch.sqrt(h**2 + w**2)
        dilation = int(round(dilation * img_diag))
        if dilation < 1:
            dilation = 1

    # Pad the instances to avoid boundary issues
    pad = torchvision.transforms.Pad(1)
    instances_target_onehot_padded = pad(instances_target_onehot)
    instances_preds_onehot_padded = pad(instances_preds_onehot)

    # Erode the instances to get the boundaries
    eroded_target = erode_pytorch(instances_target_onehot_padded, iterations=dilation)
    eroded_preds = erode_pytorch(instances_preds_onehot_padded, iterations=dilation)

    print(eroded_target.shape, eroded_preds.shape)

    # Remove the padding
    eroded_target = eroded_target[:, 1:-1, 1:-1]
    eroded_preds = eroded_preds[:, 1:-1, 1:-1]

    print(eroded_target.shape, eroded_preds.shape)

    # Calculate the boundary of the instances
    boundaries_target = instances_target_onehot - eroded_target
    boundaries_preds = instances_preds_onehot - eroded_preds

    # Calculate the IoU of the boundaries (broadcast because of the different number of instances)
    intersection = (boundaries_target.unsqueeze(1) & boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    union = (boundaries_target.unsqueeze(1) | boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    iou = intersection / union  # Shape: (num_instances_target, num_instances_preds)

    return iou


### Implementation of torchmetric classes, following the implementation of classification metrics of torchmetrics ###
### The inheritance order is: Metric -> _AbstractStatScores -> BinaryInstanceStatScores -> [BinaryInstanceRecall, BinaryInstancePrecision, BinaryInstanceF1] ###
class BinaryInstanceStatScores(_AbstractStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        matching_threshold: float = 0.5,
        multidim_average: Literal['global', 'samplewise'] = 'global',
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        zero_division = kwargs.pop('zero_division', 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
            if not (isinstance(matching_threshold, float) and (0 <= matching_threshold <= 1)):
                raise ValueError(
                    f'Expected argument `matching_threshold` to be a float in the [0,1] range, but got {matching_threshold}.'
                )

        self.threshold = threshold
        self.matching_threshold = matching_threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        if ignore_index is not None:
            # TODO: implement ignore_index
            raise ValueError('Argument `ignore_index` is not supported for this metric yet.')

        self._create_state(size=1, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
            if not preds.shape == target.shape:
                raise ValueError(
                    f'Expected `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}.'
                )
            if not preds.dim() == 3:
                raise ValueError(f'Expected `preds` and `target` to have 3 dimensions, but got {preds.dim()}.')

        # Format
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                # preds is logits, convert with sigmoid
                preds = preds.sigmoid()
            preds = preds > self.threshold

        if self.ignore_index is not None:
            idx = target == self.ignore_index
            target = target.clone()
            target[idx] = -1

        # Update state
        instance_list_target = mask_to_instances(target.to(torch.uint8))
        instance_list_preds = mask_to_instances(preds.to(torch.uint8))

        for target_i, preds_i in zip(instance_list_target, instance_list_preds):
            tp, fp, fn = match_instances(target_i, preds_i, match_threshold=self.matching_threshold)
            self._update_state(tp, fp, 0, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _binary_stat_scores_compute(tp, fp, tn, fn, self.multidim_average)


class BinaryInstanceRecall(BinaryInstanceStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            'recall',
            tp,
            fp,
            tn,
            fn,
            average='binary',
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(
        self,
        val: Optional[Tensor | Sequence[Tensor]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstancePrecision(BinaryInstanceStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            'precision',
            tp,
            fp,
            tn,
            fn,
            average='binary',
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(
        self,
        val: Optional[Tensor | Sequence[Tensor]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceFBetaScore(BinaryInstanceStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        beta: float,
        threshold: float = 0.5,
        multidim_average: Literal['global', 'samplewise'] = 'global',
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _binary_fbeta_score_arg_validation(beta, threshold, multidim_average, ignore_index, zero_division)
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.beta = beta

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            self.beta,
            average='binary',
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(
        self,
        val: Optional[Tensor | Sequence[Tensor]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceF1Score(BinaryInstanceFBetaScore):
    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal['global', 'samplewise'] = 'global',
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            beta=1.0,
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
        )

    def plot(
        self,
        val: Optional[Tensor | Sequence[Tensor]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryBoundaryIoU(Metric):
    intersection: Tensor
    union: Tensor

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('intersection', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('union', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        # TODO: Implement
        pass

    def compute(self) -> Tensor:
        return self.intersection / self.union

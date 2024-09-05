from typing import Literal, Optional

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_tensor_validation,
)

from thaw_slump_segmentation.metrics.boundary_helpers import get_boundaries

MatchingMetric = Literal['iou', 'boundary']



class BinaryBoundaryIoU(Metric):
    intersection: Tensor | list[Tensor]
    union: Tensor | list[Tensor]

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        dilation: float | int = 0.02,
        threshold: float = 0.5,
        multidim_average: Literal['global', 'samplewise'] = 'global',
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs,
    ):
        zero_division = kwargs.pop('zero_division', 0)
        super().__init__(**kwargs)

        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
            if not isinstance(dilation, (float, int)):
                raise ValueError(f'Expected argument `dilation` to be a float or int, but got {dilation}.')

        self.dilation = dilation
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        if ignore_index is not None:
            # TODO: implement ignore_index
            raise ValueError('Argument `ignore_index` is not supported for this metric yet.')

        if multidim_average == 'samplewise':
            self.add_state('intersection', default=[], dist_reduce_fx='cat')
            self.add_state('union', default=[], dist_reduce_fx='cat')
        else:
            self.add_state('intersection', default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state('union', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
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

        target = target.to(torch.uint8)
        preds = preds.to(torch.uint8)

        target, preds = get_boundaries(target, preds, self.dilation, self.validate_args)

        intersection = (target & preds).sum().item()
        union = (target | preds).sum().item()

        if self.multidim_average == 'global':
            self.intersection += intersection
            self.union += union
        else:
            self.intersection.append(intersection)
            self.union.append(union)

    def compute(self) -> Tensor:
        if self.multidim_average == 'global':
            return self.intersection / self.union
        else:
            self.intersection = torch.tensor(self.intersection)
            self.union = torch.tensor(self.union)
            return self.intersection / self.union

from typing import Any, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.average_precision import _binary_average_precision_compute
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_tensor_validation,
)
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

from thaw_slump_segmentation.metrics.boundary_helpers import _boundary_arg_validation
from thaw_slump_segmentation.metrics.instance_helpers import mask_to_instances, match_instances

MatchingMetric = Literal['iou', 'boundary']


### Implementation of torchmetric classes, following the implementation of classification metrics of torchmetrics ###
### The inheritance order is:
# Metric ->
# BinaryInstancePrecisionRecallCurve ->
# [BinaryInstanceAUROC, BinaryROC]
###


class BinaryInstancePrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for binary instance segmentation.

    This metric works similar to `torchmetrics.classification.PrecisionRecallCurve`, with two key differences:
    1. It calculates the tp, fp, fn values for each instance (blob) in the batch, and then aggregates them.
        Instead of calculating the values for each pixel.
    2. The "thresholds" argument is required.
        Calculating the thresholds at the compute stage would cost to much memory for this usecase.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    confmat: Tensor
    thesholds: Tensor

    def __init__(
        self,
        thresholds: int | List[float] | Tensor = None,
        matching_threshold: float = 0.5,
        matching_metric: MatchingMetric = 'iou',
        boundary_dilation: float | int = 0.02,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
            _boundary_arg_validation(matching_threshold, matching_metric, boundary_dilation)
            if thresholds is None:
                raise ValueError('Argument `thresholds` must be provided for this metric.')

        self.matching_threshold = matching_threshold
        self.matching_metric = matching_metric
        self.boundary_dilation = boundary_dilation
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        if ignore_index is not None:
            # TODO: implement ignore_index
            raise ValueError('Argument `ignore_index` is not supported for this metric yet.')

        thresholds = _adjust_threshold_arg(thresholds)
        self.register_buffer('thresholds', thresholds, persistent=False)
        self.add_state('confmat', default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
            if not preds.dim() == 3:
                raise ValueError(f'Expected `preds` and `target` to have 3 dimensions (BHW), but got {preds.dim()}.')

        # Format
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()

        instance_list_target = mask_to_instances(target.to(torch.uint8), self.validate_args)

        len_t = len(self.thresholds)
        confmat = self.thresholds.new_zeros((len_t, 2, 2), dtype=torch.int64)
        for i in range(len_t):
            preds_i = preds >= self.thresholds[i]
            instance_list_preds_i = mask_to_instances(preds_i.to(torch.uint8), self.validate_args)
            for target_i, preds_i in zip(instance_list_target, instance_list_preds_i):
                tp, fp, fn = match_instances(
                    target_i,
                    preds_i,
                    match_threshold=self.matching_threshold,
                    match_metric=self.matching_metric,
                    boundary_dilation=self.boundary_dilation,
                    validate_args=self.validate_args,
                )
                confmat[i, 1, 1] += tp
                confmat[i, 0, 1] += fp
                confmat[i, 1, 0] += fn
        self.confmat += confmat

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute metric."""
        return _binary_precision_recall_curve_compute(self.confmat, self.thresholds)

    def plot(
        self,
        curve: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        score: Optional[Tensor | bool] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        """Plot a single curve from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryPrecisionRecallCurve
            >>> preds = rand(20)
            >>> target = randint(2, (20,))
            >>> metric = BinaryPrecisionRecallCurve()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        # switch order as the standard way is recall along x-axis and precision along y-axis
        curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])

        score = (
            _auc_compute_without_check(curve_computed[0], curve_computed[1], direction=-1.0)
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed, score=score, ax=ax, label_names=('Recall', 'Precision'), name=self.__class__.__name__
        )

class BinaryInstanceAveragePrecision(BinaryInstancePrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        return _binary_average_precision_compute(self.confmat, self.thresholds)

    def plot(  # type: ignore[override]
        self, val: Optional[Tensor | Sequence[Tensor]] = None, ax: Optional[_AX_TYPE] = None  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore

        return self._plot(val, ax)

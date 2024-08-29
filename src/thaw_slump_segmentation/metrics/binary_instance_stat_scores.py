from typing import Any, Literal, Optional, Sequence

import torch
from torch import Tensor
from torchmetrics.classification.stat_scores import _AbstractStatScores
from torchmetrics.functional.classification.accuracy import _accuracy_reduce
from torchmetrics.functional.classification.f_beta import _binary_fbeta_score_arg_validation, _fbeta_reduce
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_tensor_validation,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from thaw_slump_segmentation.metrics.instance_helpers import mask_to_instances, match_instances

MatchingMetric = Literal['iou', 'boundary']



### Implementation of torchmetric classes, following the implementation of classification metrics of torchmetrics ###
### The inheritance order is:
# Metric ->
# _AbstractStatScores ->
# BinaryInstanceStatScores ->
# [BinaryInstanceRecall, BinaryInstancePrecision, BinaryInstanceF1]
###
class BinaryInstanceStatScores(_AbstractStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        matching_threshold: float = 0.5,
        matching_metric: MatchingMetric = 'iou',
        boundary_dilation: float | int = 0.02,
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
                    f'Expected arg `matching_threshold` to be a float in the [0,1] range, but got {matching_threshold}.'
                )
            if matching_metric not in ['iou', 'boundary']:
                raise ValueError(
                    f'Expected argument `matching_metric` to be either "iou" or "boundary", but got {matching_metric}.'
                )
            if not isinstance(boundary_dilation, (float, int)) and matching_metric == 'boundary':
                raise ValueError(
                    f'Expected argument `boundary_dilation` to be a float or int, but got {boundary_dilation}.'
                )

        self.threshold = threshold
        self.matching_threshold = matching_threshold
        self.matching_metric = matching_metric
        self.boundary_dilation = boundary_dilation
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
        instance_list_target = mask_to_instances(target.to(torch.uint8), self.validate_args)
        instance_list_preds = mask_to_instances(preds.to(torch.uint8), self.validate_args)

        for target_i, preds_i in zip(instance_list_target, instance_list_preds):
            tp, fp, fn = match_instances(
                target_i,
                preds_i,
                match_threshold=self.matching_threshold,
                match_metric=self.matching_metric,
                boundary_dilation=self.boundary_dilation,
                validate_args=self.validate_args,
            )
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


class BinaryInstanceAccuracy(BinaryInstanceStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp,
            fp,
            tn,
            fn,
            average='binary',
            multidim_average=self.multidim_average,
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

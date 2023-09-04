from typing import Union, Optional, Sequence, Any

from bcblib.tools.nifti_utils import load_nifti
from monai.metrics import HausdorffDistanceMetric, CumulativeIterationMetric, is_binary_tensor, \
    compute_hausdorff_distance, do_metric_reduction
from monai.utils import MetricReduction
from monai.data import MetaTensor
from torch.nn.modules.loss import _Loss
import numpy as np
import torch


class HausdorffDistanceRatioMetric(CumulativeIterationMetric):
    # TODO transform it into a ratio
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: Optional[float] = None,
        directed: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
        )
    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class BinaryEmptyLabelLoss(_Loss):
    def __init__(self, sigmoid: bool = True, threshold: float = 0.5, reduction: str = "mean") -> None:
        """
            sigmoid: if True, apply a sigmoid function to the prediction.

        """
        super().__init__(reduction=reduction)
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        if self.threshold is not None:
            input[input >= self.threshold] = torch.tensor(1.0)

        f: torch.Tensor = torch.count_nonzero(input, dim=[-3, -2, -1])
        # threshold f >= 1 becomes 1
        f[f >= 1] = torch.tensor(1.0, dtype=torch.float)
        if self.reduction.lower() == 'mean':
            f = torch.mean(f, dtype=torch.float)  # the batch and channel average
        elif self.reduction.lower() == 'sum':
            f = torch.sum(f, dtype=torch.float)  # sum over the batch and channel dims
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        # if self.reduction.lower() == 'none' we only return the loss value per channel and batch
        return f


class BinaryEmptyLabelLoss(_Loss):
    def __init__(self, sigmoid: bool = True, threshold: float = 0.5, reduction: str = "mean") -> None:
        """
            sigmoid: if True, apply a sigmoid function to the prediction.

        """
        super().__init__(reduction=reduction)
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        if self.threshold is not None:
            input[input >= self.threshold] = torch.tensor(1.0)

        f: torch.Tensor = torch.count_nonzero(input, dim=[-3, -2, -1])
        # threshold f >= 1 becomes 1
        f[f >= 1] = torch.tensor(1.0, dtype=torch.float)
        if self.reduction.lower() == 'mean':
            f = torch.mean(f, dtype=torch.float)  # the batch and channel average
        elif self.reduction.lower() == 'sum':
            f = torch.sum(f, dtype=torch.float)  # sum over the batch and channel dims
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        # if self.reduction.lower() == 'none' we only return the loss value per channel and batch
        return f


class ThresholdedAverageLoss(_Loss):
    def __init__(self, sigmoid: bool = True, threshold: float = 0.5, reduction: str = "mean") -> None:
        """
            sigmoid: if True, apply a sigmoid function to the prediction.

        """
        super().__init__(reduction=reduction)
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
        """
        # input = input.clone().detach()
        if isinstance(input, MetaTensor):
            input = input.as_tensor()
        if self.sigmoid:
            input = torch.sigmoid(input)
        # thr_data = torch.where(input >= self.threshold, input, torch.nan)
        # data = input * mask
        # thr_data = torch.sum(mask) / torch.sum(mask)
        thr_data = input[input >= self.threshold]
        if self.reduction.lower() == 'mean':
            out_data = torch.mean(thr_data)  # the batch and channel average
        elif self.reduction.lower() == 'sum':
            out_data = torch.sum(thr_data)  # sum over the batch and channel dims
        elif self.reduction.lower() == 'max':
            out_data = torch.max(thr_data)
        elif self.reduction.lower() == 'min':
            out_data = torch.min(thr_data)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return torch.nan_to_num(out_data, nan=0.0)


def distance_ratio(label, prediction):
    """
    Compute the distance ratio between the prediction and the label.
    The distance ratio is defined as 1 - (Hausdorff distance / max distance).
    The max distance is the distance between the two opposite corners of the label.
    The Hausdorff distance is computed using the HausdorffDistanceMetric from MONAI.
    Parameters
    ----------
    label
    prediction

    Returns
    -------
    distance_ratio: float
        The distance ratio between the prediction and the label.

    """
    label_hdr = load_nifti(label)
    label_data = label_hdr.get_fdata()
    pred_hdr = load_nifti(prediction)
    pred_data = pred_hdr.get_fdata()

    if np.count_nonzero(label_data) == 0 ^ np.count_nonzero(pred_data) == 0:
        return 0
    max_coord = np.sum([axis - 1 for axis in label_hdr.shape])
    max_distance = np.sqrt(np.sum((np.array([0, 0, 0]) - max_coord) ** 2))

    hausdorff_distance = HausdorffDistanceMetric(include_background=True, reduction="mean")
    label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_data).unsqueeze(0).unsqueeze(0)
    distance = hausdorff_distance(y_pred=pred_tensor, y=label_tensor).item()

    return 1 - (distance / max_distance)



# Use the same structure as for HausdorffDistanceMetric to create a DistanceRatioMetric using a function similar to
# distance_ratio (without taking images as input but tensors)


def compute_distance_ratio(
        y_pred: torch.Tensor,
        y: torch.Tensor,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: float | None = None,
        directed: bool = False,
        spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
        ) -> torch.Tensor:
    """
    Compute the distance ratio between the prediction and the label.
    Parameters
    ----------
    y_pred: torch.Tensor
        The prediction tensor.
    y: torch.Tensor
        The label tensor.
    include_background: bool
        Whether to include the background in the computation.
    distance_metric: str
        The metric used to compute surface distance.
    percentile: float
        The percentile of the Hausdorff distance.
    directed: bool
        Whether to calculate directed Hausdorff distance.
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None
        The pixel spacing in physical world units.

    Returns
    -------

    """
    if not isinstance(y_pred, torch.Tensor):
        raise ValueError("y_pred must be a torch.Tensor")
    if not isinstance(y, torch.Tensor):
        raise ValueError("y must be a torch.Tensor")
    if y_pred.shape != y.shape:
        raise ValueError("y_pred and y must have the same shape")
    if y_pred.ndim < 3:
        raise ValueError("y_pred must have at least three dimensions")
    if y.ndim < 3:
        raise ValueError("y must have at least three dimensions")
    if y_pred.ndim != y.ndim:
        raise ValueError("y_pred and y must have the same number of dimensions")
    if not isinstance(include_background, bool):
        raise ValueError("include_background must be a bool")
    if not isinstance(distance_metric, str):
        raise ValueError("distance_metric must be a str")
    if percentile is not None and not isinstance(percentile, float):
        raise ValueError("percentile must be a float")
    if not isinstance(directed, bool):
        raise ValueError("directed must be a bool")
    if spacing is not None and not isinstance(spacing, (int, float, np.ndarray, Sequence)):
        raise ValueError("spacing must be an int, float, np.ndarray or Sequence")
    if isinstance(spacing, (int, float)):
        spacing = [spacing] * (y.ndim - 2)
    if isinstance(spacing, np.ndarray):
        spacing = spacing.tolist()
    if isinstance(spacing, Sequence):
        if len(spacing) != y.ndim - 2:
            raise ValueError("spacing must have the same length as y.ndim - 2")
        for i in spacing:
            if not isinstance(i, (int, float, np.ndarray, Sequence)):
                raise ValueError("spacing must be an int, float, np.ndarray or Sequence")
    if isinstance(spacing, Sequence):
        for i in spacing:
            if isinstance(i, Sequence):
                for j in i:
                    if not isinstance(j, (int, float)):
                        raise ValueError("spacing must be an int, float, np.ndarray or Sequence")
            else:
                if not isinstance(i, (int, float)):
                    raise ValueError("spacing must be an int, float, np.ndarray or Sequence")

    # Compute the distance ratio
    # If either y_pred or y is empty, the distance ratio is 0 (lowest possible value)
    if torch.count_nonzero(y_pred) == 0 ^ torch.count_nonzero(y) == 0:
        return torch.tensor(0.0)
    # Compute the max distance
    max_coord = np.sum([axis - 1 for axis in y.shape])
    max_distance = np.sqrt(np.sum((np.array([0, 0, 0]) - max_coord) ** 2))
    # Compute the Hausdorff distance
    hausdorff_distance = HausdorffDistanceMetric(include_background=include_background, reduction="mean")
    distance = hausdorff_distance(y_pred=y_pred, y=y).item()
    # Compute the distance ratio
    dist_ratio = 1 - (distance / max_distance)
    return torch.tensor(dist_ratio)


class DistanceRatioMetric(CumulativeIterationMetric):
    """
    Compute the distance ratio between two tensors (ratio of the Hausdorff distance over the maximum distance possible in the image).
    It can support both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(self,
                 include_background: bool = False,
                 distance_metric: str = "euclidean",
                 percentile: Optional[float] = None,
                 directed: bool = False,
                 reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
                 get_not_nans: bool = False,
                 ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute (BxC) for each channel for each batch
        return compute_distance_ratio(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


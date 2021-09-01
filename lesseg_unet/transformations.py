from math import radians
import logging
from copy import deepcopy
from typing import Mapping, Dict, Hashable, Any, Optional, Callable, Union, List, Tuple, Sequence
import time
from enum import Enum

import numpy as np
from monai.utils import Method, fall_back_tuple
from monai.transforms.compose import Randomizable
from monai.transforms.inverse import InvertibleTransform
from monai.config import KeysCollection, DtypeLike
import torch
from torch.nn.functional import pad
from monai.transforms import (
    ToNumpyd,
    GaussianSmoothd,
    MapTransform,
    Transform,
    AddChanneld,
    LoadImaged,
    Compose,
    RandRotate90d,
    RandFlipd,
    RandSpatialCropd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ToTensord,
    Resize,
    Resized,
    RandAffined,
    Rand3DElasticd,
    RandDeformGrid,
    RandBiasFieldd,
    Spacingd,
    RandHistogramShiftd,
    NormalizeIntensityd,
    ThresholdIntensityd,
    SplitChanneld,
    SqueezeDimd,
    CropForegroundd,
    AsChannelFirstd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd, RandomizableTransform
)
import torchio
from torchio import Subject, ScalarImage
from torchio.transforms import (
    RandomBlur,
    RandomNoise,
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomBiasField,
)
from torchvision.transforms import RandomOrder

"""
Custom Transformation Classes
"""
timeziz = time.time()
timean = 0


class TorchPadMode(Enum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """
    CONSTANT = "constant"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
    REFLECT = "reflect"


def get_data_and_pkg(data, detach=True):
    if isinstance(data, np.ndarray):
        return data, np
    if isinstance(data, torch.Tensor):
        if detach:
            data = data.detach()
        return data, torch
    raise TypeError('The data does not have the right type. It should either be a numpy array or a torch Tensor')


def create_gradient(img_spatial_size, low=-1, high=1):
    x = np.linspace(low, high, img_spatial_size[0])
    y = np.linspace(low, high, img_spatial_size[1])
    z = np.linspace(low, high, img_spatial_size[2])
    x_grad = np.ones(img_spatial_size)
    for ind in range(x_grad.shape[0]):
        x_grad[ind, :, :] = x[ind]
    y_grad = np.ones(img_spatial_size)
    for ind in range(y_grad.shape[0]):
        y_grad[:, ind, :] = y[ind]
    z_grad = np.ones(img_spatial_size)
    for ind in range(z_grad.shape[0]):
        z_grad[:, :, ind] = z[ind]
    x_grad = np.expand_dims(x_grad, 0)
    y_grad = np.expand_dims(y_grad, 0)
    z_grad = np.expand_dims(z_grad, 0)
    return np.concatenate((x_grad, y_grad, z_grad), 0)


class Binarize(Transform):
    """
    Set every above threshold voxel to 1.0
    """

    def __init__(self, lower_threshold: float = 0) -> None:
        self.lower_threshold = lower_threshold

    def __call__(self, img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply the transform to `img`.
        """
        # s = str(img.shape)
        # tensor_shape = img.shape
        # if isinstance(img, torch.Tensor):
        # img = img[0, :, :, :].detach()
        # img = np.asarray(img[0, :, :, :].detach().numpy())
        # else:
        #     # img = np.asarray(img[0, :, :, :])
        #     img = img[0, :, :, :]
        if isinstance(img, torch.Tensor):
            img[0, :, :, :] = torch.where(img[0, :, :, :] > self.lower_threshold, 1, 0)
        else:
            img[0, :, :, :] = np.where(img[0, :, :, :] > self.lower_threshold, 1, 0)
        # output = torch.tensor(torch.where(img > self.lower_threshold, 1, 0), dtype=img.dtype)
        # if len(tensor_shape) == 4:
        #     output = output.unsqueeze(0)
        # if len(tensor_shape) == 5:
        #     output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
        # s += '\n {}'.format(output.shape)
        # print('Binarized ######\n{}\n#####'.format(s))
        return img


class Binarized(MapTransform):
    """
    Set every above threshold voxel to 1.0

    """

    def __init__(self, keys: KeysCollection, lower_threshold: float = 0) -> None:
        super().__init__(keys)
        self.lower_threshold = lower_threshold
        self.binarize = Binarize(lower_threshold)

    def __call__(self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
                 ) -> Dict[Hashable, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.binarize(d[key])
        return d


class CoordConvAlt(Transform):
    """
    Implement CordConv
    """
    def __init__(
        self,
        spatial_channels: Tuple[int] = (1, 2, 3),
    ) -> None:
        self.spatial_channels = spatial_channels

    def __call__(self,  img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`.
        """

        spatial_dims = img.shape[1:]
        # pre-allocate memory
        if isinstance(img, torch.Tensor):
            img = np.asarray(img.detach().numpy())
        coord_channels = np.ones((len(self.spatial_channels), *spatial_dims)).astype(img.dtype)

        for i, dim in enumerate(self.spatial_channels):
            ones = np.ones((1, *spatial_dims))
            channel_size = img.shape[dim]
            range = np.arange(channel_size)
            non_channel_dims = list(set(np.arange(img.ndim)).difference([dim]))
            channel = ones * np.expand_dims(range,  non_channel_dims)
            channel = channel/channel_size - 0.5
            coord_channels[i] = channel

        return torch.Tensor(np.concatenate((img, coord_channels), axis=0))
        # return np.concatenate((img, coord_channels), axis=0)


class CoordConvAltd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CordConv`.
    """

    def __init__(self, keys: KeysCollection, spatial_channels: Tuple[int] = (1, 2, 3)) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """

        super().__init__(keys)
        self.coord_conv = CoordConvAlt(spatial_channels)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.coord_conv(d[key])
        return d


class CoordConv(Transform):
    def __init__(self, gradients: np.ndarray = None) -> None:
        self.gradients = gradients

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # if isinstance(img, torch.Tensor):
        #     img = np.asarray(img[0, :, :, :].detach().numpy())
        # else:
        #     img = np.asarray(img[0, :, :, :])
        if self.gradients is None:
            img = create_gradient(img.shape[1:])
        else:
            if isinstance(img, np.ndarray):
                img = np.concatenate((img, self.gradients), 0).astype(np.float32)
            else:
                img = torch.cat([img, self.gradients], dim=0).type(torch.float32)
        return img


class CoordConvd(MapTransform, InvertibleTransform):
    """
    """

    def __init__(self, keys: KeysCollection, gradients: Union[np.ndarray, torch.Tensor] = None) -> None:
        super().__init__(keys)
        self.coord_conv = CoordConv(gradients)

    def __call__(
            self,
            data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.coord_conv(d[key])
            self.push_transform(
                d,
                key
            )
        return d

    def inverse(
            self,
            data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # transform = self.get_most_recent_transform(d, key)
            # Apply inverse transform
            d[key] = d[key][:1, :, :, :]
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class SpatialCrop(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    def __init__(
        self,
        roi_center: Union[Sequence[int], torch.Tensor, None] = None,
        roi_size: Union[Sequence[int], torch.Tensor, None] = None,
        roi_start: Union[Sequence[int], torch.Tensor, None] = None,
        roi_end: Union[Sequence[int], torch.Tensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("Only slice steps of 1/None are currently supported")
            self.slices = list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center = torch.as_tensor(roi_center, dtype=torch.int16)
                roi_size = torch.as_tensor(roi_size, dtype=torch.int16)
                roi_start = torch.maximum(roi_center - torch.floor_divide(roi_size, 2), torch.tensor(0))
                roi_end = torch.maximum(roi_start + roi_size, roi_start)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start = torch.maximum(torch.as_tensor(roi_start, dtype=torch.int16), torch.tensor(0))
                roi_end = torch.maximum(torch.as_tensor(roi_end, dtype=torch.int16), roi_start)
            # Allow for 1D by converting back to np.array (since np.maximum will convert to int)
            roi_start = roi_start if isinstance(roi_start, torch.Tensor) else torch.as_tensor([roi_start])
            roi_end = roi_end if isinstance(roi_end, torch.Tensor) else torch.as_tensor([roi_end])
            # convert to slices
            self.slices = [slice(s, e) for s, e in zip(roi_start, roi_end)]

    def __call__(self, img: torch.Tensor):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        return img[tuple(slices)]


class MyCenterSpatialCrop(Transform):
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def __call__(self, img: torch.Tensor):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        center = [i // 2 for i in img.shape[1:]]
        cropper = SpatialCrop(roi_center=center, roi_size=roi_size)
        return cropper(img)


class MySpatialPad(Transform):
    """
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
    for additional details.

    Args:
        spatial_size: the spatial size of output data after padding, if a dimension of the input
            data size is bigger than the pad size, will not pad that dimension.
            If its components have non-positive values, the corresponding size of input image will be used
            (no padding). for example: if the spatial size of input data is [30, 30, 30] and
            `spatial_size=[32, 25, -1]`, the spatial size of output data will be [32, 30, 30].
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: Union[TorchPadMode, str] = TorchPadMode.CONSTANT,
        **torch_kwargs,
    ) -> None:
        self.spatial_size = spatial_size
        self.method: Method = Method(method)
        self.mode: TorchPadMode = TorchPadMode(mode)
        self.np_kwargs = torch_kwargs

    def _determine_data_pad_width(self, data_shape: Sequence[int]) -> List[Tuple[int, int]]:
        spatial_size = fall_back_tuple(self.spatial_size, data_shape)
        if self.method == Method.SYMMETRIC:
            pad_width = []
            for i, sp_i in enumerate(spatial_size):
                width = max(sp_i - data_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
            return pad_width
        return [(0, max(sp_i - data_shape[i], 0)) for i, sp_i in enumerate(spatial_size)]

    def __call__(self, img: torch.Tensor, mode: Optional[Union[TorchPadMode, str]] = None) -> torch.Tensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"circular"``, ``"replicate"``, ``"reflect"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        data_pad_width = self._determine_data_pad_width(img.shape[1:])
        all_pad_width = [(0, 0)] + data_pad_width
        # torch pad works the other way around with the pad_width
        all_pad_width.reverse()
        all_pad_width = [item for tu in all_pad_width for item in tu]
        if not torch.as_tensor(all_pad_width).any():
            # all zeros, skip padding
            return img

        mode = self.mode.value if mode is None else TorchPadMode(mode).value
        img = pad(img, all_pad_width, mode=mode, value=0)
        return img


class MyNormalizeIntensity(Transform):
    """
    Normalize input based on provided args, using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        out_min_max
        clamp_quantile
        nonzero: whether only normalize non-zero values.
        dtype: str
            output data type, defaults to float32.
        in_min_max
    """

    def __init__(
        self,
        out_min_max: Union[float, Tuple[float, float]] = None,
        clamp_quantile: Tuple[float, float] = None,
        nonzero: bool = False,
        dtype: str = 'float32',
        in_min_max: Union[float, Tuple[float, float]] = None,
    ) -> None:
        self.in_min_max = in_min_max
        if in_min_max is not None:
            if len(self.in_min_max) == 2:
                self.out_min = self.in_min_max[0]
                self.out_max = self.in_min_max[1]
            else:
                raise ValueError('in_min_max must be either a len 2 Tuple')
        self.clamp_quantile = clamp_quantile
        if clamp_quantile is not None:
            if not (0 <= clamp_quantile[0] <= 1) or not (0 <= clamp_quantile[1] <= 1):
                raise ValueError('Clamping quantile values must be between 0 and 1')
        self.out_min_max = out_min_max
        if out_min_max is not None:
            if isinstance(self.out_min_max, float):
                self.out_min = -self.out_min_max
                self.out_max = self.out_min_max
            elif len(self.out_min_max) == 2:
                # if self.out_min_max[0] > self.out_min_max[1]:
                #     raise ValueError('First element of out_min_max range must be lesser or equal than the second one')
                self.out_min = self.out_min_max[0]
                self.out_max = self.out_min_max[1]
            else:
                raise ValueError('out_min_max must be either a len 2 Tuple or a float')
        self.nonzero = nonzero
        self._dtype = dtype

    def _normalize(self, img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        img, pkg = get_data_and_pkg(img)
        slices = (img != 0) if self.nonzero else pkg.ones(img.shape, dtype=pkg.bool)
        # print(f'slices : {slices.shape}')
        if not pkg.any(slices):
            return img

        _sub = pkg.mean(img[slices])
        if pkg == np:
            _div = pkg.std(img[slices])
        else:
            # torch std is applying a correction that numpy does not. Disabling it to get the same results between the 2
            _div = pkg.std(img[slices], unbiased=False)
        if _div == 0.0:
            _div = 1.0
        img[slices] = (img[slices] - _sub) / _div
        return img

    def _clamp(self, img):
        img, pkg = get_data_and_pkg(img)
        if isinstance(img, np.ndarray):
            cutoff = pkg.quantile(img, self.clamp_quantile)
        else:
            cutoff = pkg.quantile(img, torch.tensor(self.clamp_quantile, device=img.device))
        pkg.clip(img, *cutoff, out=img)

    def _rescale(
            self,
            img: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        array, pkg = get_data_and_pkg(img)
        if self.in_min_max is None:
            in_min, in_max = pkg.min(array), pkg.max(array)
        else:
            in_min, in_max = self.in_min_max
        array -= in_min
        in_range = in_max - in_min
        if in_range == 0:
            message = (
                f'Rescaling image not possible'
                ' due to division by zero (the image contains only one value)'
            )
            logging.warning(message)
            return img
        array /= in_range
        out_range = self.out_max - self.out_min
        array *= out_range
        array += self.out_min
        return torch.as_tensor(array)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to `img`
        """
        img, pkg = get_data_and_pkg(img)
        self.dtype = getattr(pkg, self._dtype)
        if self.clamp_quantile is not None:
            self._clamp(img)
        img = self._normalize(img)
        if self.out_min_max is not None:
            self._rescale(img)
        if isinstance(img, torch.Tensor):
            img = torch.as_tensor(img, dtype=self.dtype)
        else:
            img = np.asarray(img, dtype=self.dtype)
        return img


class MyNormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        nonzero: whether only normalize non-zero values.
        dtype: output data type, defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        out_min_max: Union[float, Tuple[float, float]] = None,
        clamp_quantile: Tuple[float, float] = None,
        nonzero: bool = False,
        dtype: str = 'float32',
        in_min_max: Union[float, Tuple[float, float]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = MyNormalizeIntensity(out_min_max, clamp_quantile, nonzero, dtype, in_min_max)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d


class MyRandHistogramShiftd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandHistogramShift`.
    Apply random nonlinear transform the the image's intensity histogram.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_control_points: Union[Tuple[int, int], int] = 10,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        if isinstance(num_control_points, int):
            if num_control_points <= 2:
                raise AssertionError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (num_control_points, num_control_points)
        else:
            if len(num_control_points) != 2:
                raise AssertionError("num_control points should be a number or a pair of numbers")
            if min(num_control_points) <= 2:
                raise AssertionError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (min(num_control_points), max(num_control_points))
        self.reference_control_points = None
        self.floating_control_points = None

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        num_control_point = self.R.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        self.reference_control_points = torch.linspace(0, 1, num_control_point)
        self.floating_control_points = torch.clone(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = self.R.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            img_min, img_max = torch.min(d[key]), torch.max(d[key])
            reference_control_points_scaled = self.reference_control_points * (img_max - img_min) + img_min
            floating_control_points_scaled = self.floating_control_points * (img_max - img_min) + img_min
            dtype = d[key].dtype
            d[key] = np.interp(d[key], reference_control_points_scaled, floating_control_points_scaled).astype(dtype)
        return d


class PrintDim(MapTransform):
    """
    Set every above threshold voxel to 1.0

    """

    def __init__(self, keys: KeysCollection, msg: str = None) -> None:
        super().__init__(keys)
        self.msg = msg

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        if 'start time' in self.msg.lower():
            global timeziz
            timeziz = time.time()
        if 'end time' in self.msg.lower():
            # global timeziz
            ol_timziz = timeziz
            timeziz = time.time()
            global timean
            timean = np.mean([timeziz - ol_timziz, timean])
            print(f'Mean time between start and end time: {timean}')
        if self.msg:
            s = self.msg + '\n'
        else:
            s = ''
        d = dict(data)
        for idx, key in enumerate(self.keys):
            s += 'key: {}\n'.format(key)
            s += 'type key: {}\n'.format(type(d[key]))
            if isinstance(d[key], torch.Tensor):
                s += 'shape: {}\n'.format(d[key].shape)
            else:
                s += 'size: {}\n'.format(d[key].size())
                print(d[key].size())
            s += 'dtype: {}\n'.format(d[key].dtype)

        s += 'End printdim'
        print('#######PRINTDIM#####\n{}\n#############'.format(s))
        return d


class TorchIOWrapper(Randomizable, MapTransform):
    """
    Use torchio transformations in Monai and control which dictionary entries are transformed in synchrony!
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    p: probability that this trans will be applied (to all the keys listed in 'keys')
    """
    def __init__(self, keys: KeysCollection, trans: Callable, p: float = 1) -> None:
        super().__init__(keys)
        self.keys = keys
        self.trans = trans
        self.prob = p
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        transformed = None
        for idx, key in enumerate(self.keys):
            subject = Subject(datum=ScalarImage(tensor=d[key]))
            if transformed is None:
                transformed = self.trans
            else:
                transformed = transformed.get_composed_history()
            transformed = transformed(subject)
            d[key] = transformed['datum'].data
        return d


class RandTransformWrapper(Randomizable, MapTransform):
    """
    Use torchio transformations in Monai and control which dictionary entries are transformed in synchrony!
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    p: probability that this trans will be applied (to all the keys listed in 'keys')
    """
    def __init__(self, trans: Callable) -> None:
        prob_key = None
        keys_key = None
        trans_dict = trans.__dict__
        if 'probability' in trans_dict:
            prob_key = 'probability'
        if 'include' in trans_dict:
            keys_key = 'include'
        keys = getattr(trans, keys_key)
        prob = 1
        if prob_key is not None:
            prob = getattr(trans, prob_key)
            setattr(trans, prob_key, 1)
        super().__init__(keys)
        self.keys = keys
        self.trans = trans
        # Remove from the initial transformation
        setattr(self.trans, keys_key, None)
        self.prob = prob
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        transformed = None
        for idx, key in enumerate(self.keys):
            scalar_img = ScalarImage(tensor=d[key])
            subject = Subject(datum=scalar_img)
            if transformed is None:
                transformed = self.trans
            else:
                transformed = transformed.get_composed_history()
            transformed = transformed(subject)
            d[key] = transformed['datum'].data
        return d


class ThreeDHaircutd(Randomizable, MapTransform):
    """
    """

    def __init__(self,
                 keys: KeysCollection,
                 index_range: Union[float, List[Union[float, int]]],
                 prob: float = 0.1) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            index_range:
            prob: probability of cutting. (If a list/tuple is given, the first cell is the global probability that
                the transformation happens, the second is the local probability that it cuts on 2 axes
                (so if this transformation happens, it will have another probability to cut on 2 axes), and the third
                one is the local probability of cutting on the 3 axes.
                e.g. (0.1, 0.15, 0.05) means a 0.1 that the transformation happens at all. If
                (Default 0.1, with 10% probability it returns a rotated array.)

        """
        super().__init__(keys)
        if isinstance(index_range, float) and (index_range < 0 or index_range > 0.5):
            raise ValueError(f'If a single float given for index_range it must be between 0 and 0.5 because the'
                             f'cut is done on both sides of the axes')
        self.index_range = index_range
        self.prob = prob
        self._do_transform = False
        self.prob_x = None
        self.prob_y = None
        self.prob_z = None
        self.pick_axis = None

    def randomize(self, data: Optional[Any] = None) -> None:
        if isinstance(self.prob, tuple) or isinstance(self.prob, list):
            g_prob = self.prob[0]
            one_cut_prob = 1
            # Now we make an array with the different values to get each of the cut axes permutations
            permutations_values = [1/3, 1/3, 1/3, 0, 0, 0, 0]
            if len(self.prob) > 1:
                two_cut_prob = self.prob[1]
                one_cut_prob -= two_cut_prob
                permutations_values[3:6] = [two_cut_prob / 3] * 3
            if len(self.prob) > 2:
                three_cut_prob = self.prob[2]
                one_cut_prob -= three_cut_prob
                permutations_values[6] = 1
            else:
                permutations_values[5] = 1
            permutations_values[0] *= one_cut_prob
            permutations_values[1] *= one_cut_prob
            permutations_values[2] *= one_cut_prob
        else:
            g_prob = self.prob
            permutations_values = [1/3, 1/3, 1/3, 0, 0, 0, 0]
        self._do_transform = self.R.random() < g_prob
        if self._do_transform:
            # self.index_0 = int(self.R.uniform(low=0, high=self.index_range[0]))
            # self.index_1 = int(self.R.uniform(low=0, high=self.index_range[1]))
            # self.index_2 = int(self.R.uniform(low=0, high=self.index_range[2]))
            # different cuts: 0,1,2,01,02,12,012
            cut_axes = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
            cut_axis_prob = self.R.random()
            self.pick_axis = (0,)
            for ind, val in enumerate(permutations_values):
                if cut_axis_prob < val:
                    self.pick_axis = cut_axes[ind]
                    break
            self.prob_x = self.R.random()
            self.prob_y = self.R.random()
            self.prob_z = self.R.random()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize(data)
        if not self._do_transform:
            return d

        for key in self.keys:
            mask = np.zeros_like(d[key])
            # idk if it's useful but with that it could be used with images of different sizes
            if isinstance(self.index_range, float):
                ind_range = [self.index_range * (s - 1) for s in mask.shape]
            else:
                ind_range = self.index_range
            ind_cut_x = round(ind_range[0] * self.prob_x)
            ind_cut_y = round(ind_range[1] * self.prob_y)
            ind_cut_z = round(ind_range[2] * self.prob_z)
            for ax in self.pick_axis:
                if ax == 0:
                    mask[0, ind_cut_x:-ind_cut_x, :, :] = 1
                elif ax == 1:
                    mask[0, :, ind_cut_y:-ind_cut_y, :] = 1
                else:
                    mask[0, :, :, ind_cut_z:-ind_cut_z] = 1

            d[key] = d[key] * mask

        return d


class Anisotropiserd(Randomizable, MapTransform):
    """
    """

    def __init__(self,
                 keys: KeysCollection,
                 scale_range: Union[list, tuple, np.ndarray],
                 prob: float = 0.1) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            scale_range:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)

        """
        super().__init__(keys)

        self.scale_range = scale_range
        self.prob = prob
        self._do_transform = False
        self.scale = None
        self.pick_axis = None

    def randomize(self, data: Optional[Any] = None) -> None:
        self.scale = self.R.uniform(low=self.scale_range[0], high=self.scale_range[1])
        self._do_transform = self.R.random() < self.prob
        self.pick_axis = self.R.random()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.keys:
            shape = d[key].shape[1:]

            if 0 <= self.pick_axis < 0.333:
                self.resizer_1 = Resize(spatial_size=(int(self.scale * shape[0]), -1, -1))
                self.resizer_2 = Resize(spatial_size=(shape[0], -1, -1))
            elif 0.333 <= self.pick_axis < 0.666:
                self.resizer_1 = Resize(spatial_size=(-1, int(self.scale * shape[1]), -1))
                self.resizer_2 = Resize(spatial_size=(-1, shape[1], -1))
            else:
                self.resizer_1 = Resize(spatial_size=(-1, -1, int(self.scale * shape[2])))
                self.resizer_2 = Resize(spatial_size=(-1, -1, shape[2]))

            d[key] = self.resizer_2(self.resizer_1(d[key]))

        return d

"""
Transformation parameters
"""
high_prob = .2
low_prob = .1
tiny_prob = 0.05
# high_prob = 1
# low_prob = 1
# tiny_prob = 1
def_spatial_size = [96, 128, 96]

# def_spatial_size = [96, 96, 96]
min_small_crop_size = [int(0.95 * d) for d in def_spatial_size]

# .74 aspect ratio? maybe change to 96x128x96 or crop to 64cube and increase the epoch number by a lot

hyper_dict = {
    'first_transform': [
        {'LoadImaged': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'AsChannelFirstd': {
        #     'keys': ['image', 'label'],
        #     'channel_dim': -1}
        # },
        # {'Resized': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size,
        #     # 'mode': 'nearest'
        # }},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         },
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'First resize'}},
    ],
    'monai_transform': [

        # {'ScaleIntensityd': {'keys': "image"}},

        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Fisrt monai'}},
        # {'RandCropByPosNegLabeld': {
        #     'keys': ["image", "label"],
        #     'label_key': "label",
        #     'spatial_size': def_spatial_size,
        #     'pos': 1,
        #     'neg': 1,
        #     'num_samples': 4
        # }},
        {'RandSpatialCropd': {
            'keys': ["image", "label"],
            'roi_size': min_small_crop_size,
            'random_size': False
        }},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'After RandCrop'}},
    ],
    'labelonly_transform': [],
    'last_transform': [
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Last binarized'}},
        # {'Resized': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size,
        #     'mode': 'nearest'
        # }},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         },
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.5
        }},
        {'NormalizeIntensityd': {'keys': ['image']}},
    ]
}

# TODO Import all transforms from the dict


def check_imports(hyper_param_dict):
    # The main dict
    for k in hyper_param_dict:
        # Each dict in the sublist
        for d in hyper_param_dict[k]:
            # Each Transformation name in the sublist
            for name in d:
                try:
                    eval(name)
                except NameError:
                    raise ImportError('{} not imported'.format(name))


def check_hyper_param_dict_shape(hyper_param_dict, print_list=True):
    full_str = ''
    if print_list:
        full_str = '######\nList of transformations used:\n'
    # The main dict
    for k in hyper_param_dict:
        # each sublist (like first_transform or monai_transform)
        if print_list:
            full_str += k + '\n'
        if not isinstance(hyper_param_dict[k], list):
            raise ValueError('Hyper param dict values must be lists')
        # Each dict in the sublist
        for d in hyper_param_dict[k]:
            if not isinstance(d, dict):
                raise ValueError('Hyper param dict lists must be dicts')
            if len(d) > 1:
                raise ValueError('Transformation dicts must be singletons (1 dict 1 transformation)')
            # The transformation in d
            for name in d:
                if not isinstance(d[name], dict):
                    raise ValueError('Transformation dict parameters must be dict (e.g. {keys: ["image", "label"]')
                if print_list:
                    full_str += f'\t{name}\n'
    logging.info(full_str)


"""
Helper functions
"""


def trans_from_dict(transform_dict: dict):
    for transform_name in transform_dict:
        try:
            if transform_name in dir(torchio.transforms):
                return RandTransformWrapper(eval(transform_name)(**transform_dict[transform_name]))
            else:
                return eval(transform_name)(**transform_dict[transform_name])
        except Exception as e:
            print('Exception found with transform {}: {}'.format(transform_name, e))


def trans_list_from_list(param_list):
    trans_list = []
    for transform in param_list:
        trans_list.append(trans_from_dict(transform))
    return trans_list


def trans_list_from_dict(hyper_param_dict):
    for k in hyper_param_dict:
        trans_list_from_list(hyper_param_dict[k])


def find_param_from_hyper_dict(hyper_param_dict, param_name, transform_list_name=None, transform_name=None):
    for list_name in hyper_param_dict:
        if transform_list_name is not None and list_name == transform_list_name:
            for d in hyper_param_dict[list_name]:
                for t in d:
                    if transform_name is not None and t == transform_name:
                        if param_name in d[t]:
                            return d[t][param_name]
                    if transform_name is None:
                        if param_name in d[t]:
                            return d[t][param_name]
        if transform_list_name is None:
            for d in hyper_param_dict[list_name]:
                for t in d:
                    if transform_name is not None and t == transform_name:
                        if param_name in d[t]:
                            return d[t][param_name]
                    if transform_name is None:
                        if param_name in d[t]:
                            return d[t][param_name]
    return None


"""
Transformation compositions for the image segmentation
"""


def setup_coord_conv(hyper_param_dict):
    spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size', 'last_transform')
    if spatial_size is None:
        spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size')
    logging.info(f'Spatial resize to {spatial_size}')

    gradients = None
    for k in hyper_param_dict:
        # Each dict in the sublist
        for d in hyper_param_dict[k]:
            # Each Transformation name in the sublist
            for name in d:
                if name == 'CoordConvd' or name == 'CoordConv':
                    if gradients is None:
                        gradients = create_gradient(spatial_size)
                    d[name]['gradients'] = gradients


def segmentation_train_transformd(hyper_param_dict=None):
    if hyper_param_dict is None:
        hyper_param_dict = hyper_dict
    setup_coord_conv(hyper_param_dict)
    check_imports(hyper_param_dict)
    check_hyper_param_dict_shape(hyper_param_dict)
    compose_list = []
    for d_list_name in hyper_param_dict:
        trs = trans_list_from_list(hyper_param_dict[d_list_name])
        if trs is not None:
            compose_list += trs
    train_transd = Compose(
        compose_list
    )
    return train_transd


def segmentation_val_transformd(hyper_param_dict=None):
    if hyper_param_dict is None:
        hyper_param_dict = hyper_dict
    setup_coord_conv(hyper_param_dict)
    val_transd = Compose(
        trans_list_from_list(hyper_param_dict['first_transform']) +
        # trans_list_from_list(hyper_dict['labelonly_transform']) +
        trans_list_from_list(hyper_param_dict['last_transform'])
    )
    return val_transd


def image_only_transformd(hyper_param_dict=None, training=True, add_clamping=True):
    if hyper_param_dict is None:
        hyper_param_dict = hyper_dict
    setup_coord_conv(hyper_param_dict)
    check_imports(hyper_param_dict)
    check_hyper_param_dict_shape(hyper_param_dict)
    seg_tr_dict = {}
    for li in hyper_param_dict:
        seg_tr_dict[li] = []
        for di in hyper_param_dict[li]:
            for tr in di:
                # TODO make it cleaner!
                if tr == 'MyNormalizeIntensityd':
                    if add_clamping and 'clamp_quantile' not in di[tr]:
                        new_tr = deepcopy(di)
                        new_tr[tr]['clamp_quantile'] = (.01, .99)
                        seg_tr_dict[li].append(new_tr)
                keys_key = 'keys'
                if 'keys' not in di[tr]:
                    keys_key = 'include'
                if 'image' not in di[tr][keys_key]:
                    continue
                if 'label' in di[tr][keys_key]:
                    new_tr = deepcopy(di)
                    new_tr[tr][keys_key] = ['image']
                    seg_tr_dict[li].append(new_tr)
                else:
                    seg_tr_dict[li].append(di)
    if training:
        compose_list = []
        for d_list_name in seg_tr_dict:
            trs = trans_list_from_list(seg_tr_dict[d_list_name])
            if trs is not None:
                compose_list += trs
        train_transd = Compose(
            compose_list
        )
        return train_transd
    else:
        seg_transd = Compose(
            trans_list_from_list(seg_tr_dict['first_transform']) +
            trans_list_from_list(seg_tr_dict['last_transform'])
        )
        return seg_transd

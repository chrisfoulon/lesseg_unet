from math import radians
import logging
from copy import deepcopy
from typing import Mapping, Dict, Hashable, Any, Optional, Callable, Union, List, Tuple, Sequence
import time
from enum import Enum

import numpy as np
from monai.data import get_track_meta

from lesseg_unet.utils import logging_rank_0
from monai.utils import Method, fall_back_tuple, convert_to_tensor, ensure_tuple_rep
from monai.transforms.compose import Randomizable
from monai.transforms.inverse import InvertibleTransform
from monai.config import KeysCollection, DtypeLike
import torch
from torch.nn.functional import pad
from monai.transforms import RandRicianNoise
from monai.transforms import (
    ToNumpyd,
    GaussianSmoothd,
    MapTransform,
    Transform,
    AddChanneld,
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
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
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandRicianNoised,
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
    ResizeWithPadOrCropd,
    RandomizableTransform,
    RandShiftIntensityd,
    RandSpatialCropSamplesd
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
    raise TypeError(f'The data does not have the right type ({type(data)}. '
                    f'It should either be a numpy array or a torch Tensor')


def create_gradient(img_spatial_size, low=0, high=1):
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

    def __init__(self, keys: KeysCollection, lower_threshold: float = 0,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
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

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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
            channel = ones * np.expand_dims(range, non_channel_dims)
            channel = channel / channel_size - 0.5
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
    def __init__(self, gradients: np.ndarray = None, ) -> None:
        self.gradients = gradients

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # if isinstance(img, torch.Tensor):
        #     img = np.asarray(img[0, :, :, :].detach().numpy())
        # else:
        #     img = np.asarray(img[0, :, :, :])
        if self.gradients is None:
            self.gradients = create_gradient(img.shape[1:])
        if isinstance(img, np.ndarray):
            img = np.concatenate((img, self.gradients), 0).astype(np.float32)
        else:
            img = torch.cat([img, torch.tensor(self.gradients).to(img.device)], dim=0).type(torch.float32)
        return img


class CoordConvd(MapTransform, InvertibleTransform):
    """
    The input data must be channel first
    """

    def __init__(self, keys: KeysCollection, gradients: Union[np.ndarray, torch.Tensor] = None,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
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
        # from monai.utils.enums import InverseKeys
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # transform = self.get_most_recent_transform(d, key)
            # Apply inverse transform
            d[key] = d[key][:1, :, :, :]
            # Remove the applied transform
            self.pop_transform(d, key, check=False)
        return d


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
        no_std: bool = True,
        interpolation='linear'
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
        self.no_std = no_std
        self.interpolation = interpolation

    def _normalize(self, img: Union[torch.Tensor, np.ndarray], no_std: bool = True) -> Union[torch.Tensor, np.ndarray]:
        img, pkg = get_data_and_pkg(img)
        if pkg == np:
            bool_dtype = bool
        else:
            bool_dtype = pkg.bool
        slices = (img != 0) if self.nonzero else pkg.ones(img.shape, dtype=bool_dtype)
        if not pkg.any(slices):
            return img
        _sub = pkg.mean(img[slices])
        if no_std:
            _div = 1.0
        else:
            if pkg == np:
                _div = pkg.std(img[slices])
            else:
                # torch std is applying a correction that numpy does not.
                # Disabling it to get the same results between the 2
                _div = pkg.std(img[slices], unbiased=False)
        if _div == 1.0 or _div == 0.0:
            img[slices] = (img[slices] - _sub)
        else:
            img[slices] = (img[slices] - _sub) / _div
        return img

    def _clamp(self, img):
        img, pkg = get_data_and_pkg(img)
        if isinstance(img, np.ndarray):
            cutoff = pkg.quantile(img, self.clamp_quantile, interpolation=self.interpolation)
        else:
            cutoff = pkg.quantile(img, torch.tensor(self.clamp_quantile, device=img.device),
                                  interpolation=self.interpolation)
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
        img = self._normalize(img, self.no_std)
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
            no_std: bool = True,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = MyNormalizeIntensity(out_min_max, clamp_quantile, nonzero, dtype, in_min_max, no_std)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d



class RicianBlobRobotNoise(Transform):
    def __init__(self):
        super().__init__()
    def __call__(self, img: torch.Tensor, blob_robot_img: torch.Tensor) -> torch.Tensor:
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=self.dtype)

        if not isinstance(self.mean, (int, float)):
            raise RuntimeError("If channel_wise is False, mean must be a float or int number.")
        if not isinstance(self.std, (int, float)):
            raise RuntimeError("If channel_wise is False, std must be a float or int number.")
        std = self.std * img.std().item() if self.relative else self.std
        if not isinstance(std, (int, float)):
            raise RuntimeError("std must be a float or int number.")
        img = self._add_noise(img, mean=self.mean, std=std)
        return img



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
            # TODO https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/array.html#Resize might help
            d[key] = np.interp(d[key], reference_control_points_scaled, floating_control_points_scaled).astype(dtype)
        return d


class PrintDim(MapTransform, InvertibleTransform):
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
            s += 'shape: {}\n'.format(d[key].shape)
            # if isinstance(d[key], torch.Tensor):
            #     s += 'shape: {}\n'.format(d[key].shape)
            # else:
            #     s += 'size: {}\n'.format(d[key].size())
            s += 'dtype: {}\n'.format(d[key].dtype)
        s += 'End printdim'
        print('#######PRINTDIM#####\n{}\n#############'.format(s))
        return d

    def inverse(
            self,
            data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = self.__call__(data)
        return d


class TorchIOWrapper(Randomizable, MapTransform):
    """
    Use torchio transformations in Monai and control which dictionary entries are transformed in synchrony!
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    p: probability that this trans will be applied (to all the keys listed in 'keys')
    """

    def __init__(self, keys: KeysCollection, trans: Callable, p: float = 1,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
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
        for idx, key in enumerate(self.key_iterator(d)):
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


def check_hyper_param_dict_shape(hyper_param_dict, writing_rank, print_list=True):
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
    logging_rank_0(full_str, writing_rank)


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


def find_param_from_hyper_dict(hyper_param_dict, param_name, transform_list_name=None, transform_name=None,
                               find_last=True):
    param_val = None
    for list_name in hyper_param_dict:
        if transform_list_name is not None and list_name == transform_list_name:
            for d in hyper_param_dict[list_name]:
                for t in d:
                    if transform_name is not None and t == transform_name:
                        if param_name in d[t]:
                            param_val = d[t][param_name]
                            if not find_last:
                                return param_val
                    if transform_name is None:
                        if param_name in d[t]:
                            param_val = d[t][param_name]
                            if not find_last:
                                return param_val
        if transform_list_name is None:
            for d in hyper_param_dict[list_name]:
                for t in d:
                    if transform_name is not None and t == transform_name:
                        if param_name in d[t]:
                            param_val = d[t][param_name]
                            if not find_last:
                                return param_val
                    if transform_name is None:
                        if param_name in d[t]:
                            param_val = d[t][param_name]
                            if not find_last:
                                return param_val
    return param_val


"""
Transformation compositions for the image segmentation
"""


def setup_coord_conv(hyper_param_dict, gradient_shape=None):
    # create_gradient_before = list(hyper_param_dict.keys())[-1] == 'last_transform'
    # if create_gradient_before:
    # spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size', 'last_transform',
    #                                           find_last=True)
    # if spatial_size is None:
    #     spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size', find_last=True)
    logging.info(f'CoordConv gradient size to {gradient_shape}')

    gradients = None
    for k in hyper_param_dict:
        # Each dict in the sublist
        for d in hyper_param_dict[k]:
            # Each Transformation name in the sublist
            for name in d:
                if name == 'CoordConvd' or name == 'CoordConv':
                    if gradients is None:
                        gradients = create_gradient(gradient_shape)
                    d[name]['gradients'] = gradients


def train_transformd(hyper_param_dict=None, clamping=None, device=None, writing_rank=-1):
    if hyper_param_dict is None:
        raise ValueError('Hyper dict is None')
    # setup_coord_conv(hyper_param_dict)
    check_imports(hyper_param_dict)
    check_hyper_param_dict_shape(hyper_param_dict, writing_rank=writing_rank)
    seg_tr_dict = deepcopy(hyper_param_dict)
    normalize_count = 0
    for li in seg_tr_dict:
        for di in seg_tr_dict[li]:
            for tr in di:
                if tr == 'ToTensord':
                    if device is not None:
                        di[tr]['device'] = device
                if tr == 'MyNormalizeIntensityd':
                    normalize_count += 1
                    # if MyNormalizeIntensityd is called more than once, we do not want to clamp twice
                    if clamping is not None and not normalize_count > 1:  # and 'clamp_quantile' not in di[tr]:
                        di[tr]['clamp_quantile'] = clamping
    compose_list = []
    for d_list_name in seg_tr_dict:
        trs = trans_list_from_list(seg_tr_dict[d_list_name])
        if trs is not None:
            compose_list += trs
    train_transd = Compose(
        compose_list
    )
    return train_transd


def val_transformd(hyper_param_dict=None, clamping=None, device=None):
    if hyper_param_dict is None:
        raise ValueError('Hyper dict is None')
    # setup_coord_conv(hyper_param_dict)
    seg_tr_dict = deepcopy(hyper_param_dict)
    if clamping is not None:
        for li in seg_tr_dict:
            for di in seg_tr_dict[li]:
                for tr in di:
                    if tr == 'ToTensord':
                        if device is not None:
                            di[tr]['device'] = device
                    if tr == 'MyNormalizeIntensityd':
                        if clamping is not None and 'clamp_quantile' not in di[tr]:
                            di[tr]['clamp_quantile'] = clamping
    val_transd = Compose(
        trans_list_from_list(seg_tr_dict['first_transform']) +
        # trans_list_from_list(hyper_dict['labelonly_transform']) +
        trans_list_from_list(seg_tr_dict['last_transform'])
    )
    return val_transd


def image_only_transformd(hyper_param_dict=None, training=True, clamping=None, device=None):
    if hyper_param_dict is None:
        raise ValueError('Hyper dict is None')
    # setup_coord_conv(hyper_param_dict)
    check_imports(hyper_param_dict)
    check_hyper_param_dict_shape(hyper_param_dict, 0)
    seg_tr_dict = {}
    # Here we want to get rid of the 'label' operations (because controls datasets don't use labels)
    for li in hyper_param_dict:
        seg_tr_dict[li] = []
        for di in hyper_param_dict[li]:
            new_di = deepcopy(di)
            # There should be only one transformation per dict
            for tr in di:
                if tr == 'ToTensord':
                    if device is not None:
                        new_di[tr]['device'] = device
                if tr == 'MyNormalizeIntensityd':
                    if clamping is not None and 'clamp_quantile' not in di[tr]:
                        new_di[tr]['clamp_quantile'] = clamping
                new_di[tr]['allow_missing_keys'] = True
                keys_key = 'keys'
                if 'keys' not in new_di[tr]:
                    keys_key = 'include'
                if 'image' not in new_di[tr][keys_key]:
                    continue
                # if 'label' in new_di[tr][keys_key]:
                #     new_di[tr][keys_key] = ['image']
                #     seg_tr_dict[li].append(new_di)
                # else:
                #     seg_tr_dict[li].append(new_di)
                seg_tr_dict[li].append(new_di)
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


def add_control_key(transform_dict):
    """
    Add the control key to the transform dict in every transform that has an 'image' key
    Parameters
    ----------
    transform_dict: dict

    Returns
    -------

    """
    for tr in transform_dict:
        # 'image' can be in 'keys' or 'include'
        keys_or_include = 'keys'
        if 'keys' not in transform_dict[tr]:
            keys_or_include = 'include'
        if 'image' in transform_dict[tr][keys_or_include]:
            if 'control' not in transform_dict[tr][keys_or_include]:
                transform_dict[tr][keys_or_include].append('control')
    return transform_dict

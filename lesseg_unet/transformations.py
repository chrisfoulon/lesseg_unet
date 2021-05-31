from math import radians
import logging
from copy import deepcopy
from typing import Mapping, Dict, Hashable, Any, Optional, Callable, Union, List, Tuple

import numpy as np
from monai.transforms.compose import Randomizable
from monai.transforms.inverse import InvertibleTransform
from monai.config import KeysCollection
import torch
from monai.transforms import (
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
    ResizeWithPadOrCropd
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

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to `img`.
        """
        s = str(img.shape)
        tensor_shape = img.shape
        if isinstance(img, torch.Tensor):
            img = np.asarray(img[0, :, :, :].detach().numpy())
        else:
            img = np.asarray(img[0, :, :, :])
        output = np.asarray(np.where(img > self.lower_threshold, 1, 0), dtype=img.dtype)
        if len(tensor_shape) == 4:
            output = torch.from_numpy(output).unsqueeze(0)
        # if len(tensor_shape) == 5:
        #     output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
        s += '\n {}'.format(output.shape)
        # print('Binarized ######\n{}\n#####'.format(s))
        return output


class Binarized(MapTransform):
    """
    Set every above threshold voxel to 1.0

    """

    def __init__(self, keys: KeysCollection, lower_threshold: float = 0) -> None:
        super().__init__(keys)
        self.lower_threshold = lower_threshold
        self.binarize = Binarize(lower_threshold)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # if isinstance(img, torch.Tensor):
        #     img = np.asarray(img[0, :, :, :].detach().numpy())
        # else:
        #     img = np.asarray(img[0, :, :, :])
        if self.gradients is None:
            img = create_gradient(img.shape[1:])
        else:
            img = np.concatenate((img, self.gradients), 0)
        return torch.Tensor(img)


class CoordConvd(MapTransform, InvertibleTransform):
    """
    """

    def __init__(self, keys: KeysCollection, gradients: np.ndarray = None) -> None:
        super().__init__(keys)
        self.coord_conv = CoordConv(gradients)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.coord_conv(d[key])
            self.push_transform(
                d,
                key
            )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Apply inverse transform
            d[key] = d[key][:1, :, :, :]
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class PrintDim(MapTransform):
    """
    Set every above threshold voxel to 1.0

    """

    def __init__(self, keys: KeysCollection, msg: str = None) -> None:
        super().__init__(keys)
        self.msg = msg

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        if self.msg:
            s = self.msg + '\n'
        else:
            s = ''
        d = dict(data)
        for idx, key in enumerate(self.keys):
            s += 'key: {}\n'.format(key)
            s += 'type key: {}\n'.format(type(d[key]))
            if isinstance(d[key], np.ndarray):
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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
    if print_list:
        print('######\nList of transformations used:')
    # The main dict
    for k in hyper_param_dict:
        # each sublist (like first_transform or monai_transform)
        if print_list:
            print(k)
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
                    print('\t{}'.format(name))


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
    logging.info(f'Spatial resize to {spatial_size}')
    if spatial_size is None:
        spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size')

    gradients = create_gradient(spatial_size)
    for k in hyper_param_dict:
        # Each dict in the sublist
        for d in hyper_param_dict[k]:
            # Each Transformation name in the sublist
            for name in d:
                if name == 'CoordConvd' or name == 'CoordConv':
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


def segmentation_transformd(hyper_param_dict=None):
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
    hyper_param_dict = seg_tr_dict
    seg_transd = Compose(
        trans_list_from_list(hyper_param_dict['first_transform']) +
        trans_list_from_list(hyper_param_dict['last_transform'])
    )
    return seg_transd

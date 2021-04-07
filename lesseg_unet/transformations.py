from math import radians
import logging
from copy import deepcopy
from typing import Mapping, Dict, Hashable, Any, Optional, Callable, Union

import numpy as np
from monai.transforms.compose import Randomizable
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


class CoordConv(Transform):
    def __init__(self, gradients: np.ndarray = None) -> None:
        self.gradients = gradients

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # if isinstance(img, torch.Tensor):
        #     img = np.asarray(img[0, :, :, :].detach().numpy())
        # else:
        #     img = np.asarray(img[0, :, :, :])
        if self.gradients is None:
            x_grad = np.zeros_like(img)
            y_grad = np.zeros_like(img)
            z_grad = np.zeros_like(img)
            # print(x_grad.shape)
            for k in range(img.shape[1]):
                x_grad[0, k, :, :] = k
            for k in range(img.shape[2]):
                y_grad[0, :, k, :] = k
            for k in range(img.shape[3]):
                z_grad[0, :, :, k] = k
            img = np.concatenate((img, x_grad, y_grad, z_grad), 0)
        else:
            img = np.concatenate((img, self.gradients), 0)
        return torch.Tensor(img)


class CoordConvd(MapTransform):
    """
    """

    def __init__(self, keys: KeysCollection, gradients: np.ndarray = None) -> None:
        super().__init__(keys)
        self.coord_conv = CoordConv(gradients)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.coord_conv(d[key])
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
# TODO verify that random transformations are applied the same way on the image and the seg
hyper_dict = {
    'first_transform': [
        {'LoadImaged': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'AsChannelFirstd': {
        #     'keys': ['image', 'label'],
        #     'channel_dim': -1}
        # },
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            'mode': 'nearest'}
         },
        {'NormalizeIntensityd': {'keys': ['image']}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Fisrt resize'}},
    ],
    'monai_transform': [
        # {'ScaleIntensity': {}}
        # {'PrintDim': {'keys': ['image', 'label']}},
        {'RandSpatialCropd': {'keys': ['image', 'label'],
                              'roi_size': min_small_crop_size,
                              'random_center': True,
                              'random_size': False}
         },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
         },
        # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': high_prob,
            'rotate_range': radians(5),
            'shear_range': None,
            'translate_range': None,
            'scale_range': 0.3,
            'spatial_size': None,
            'padding_mode': 'border',
            'as_tensor_output': False}
         },
        # TODO check
        # {'RandFlipd': {
        #     'keys': ['image', 'label'],
        #     'prob': low_prob,
        #     'spatial_axis': 0}
        # },
        # {'RandDeformGrid':
        #     {'keys': ['image', 'label']}
        # },
        # {'Spacingd':
        #     {'keys': ['image', 'label']}
        # },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'spatial_size': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': False}
         },
        # {'SqueezeDimd':
        #     {'keys': ["image", "label"],
        #      'dim': 0}
        # },
        {'ToTensord': {'keys': ['image', 'label']}},
        # 'AddChanneld': {'keys': ['image', 'label']},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'After MONAI'},
    ],
    'torchio_transform': [
        # 'PrintDim': {'keys': ['image', 'label']},
        {'RandomNoise': {
            'include': ['image'],
            'mean': 0,
            'std': (0.01, 0.1),
            'p': low_prob}
         },
        # 'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (4, 10)
        # },
        {'RandomBlur': {
            'include': ['image', 'label'],
            'std': (0.1, 0.5),
            'p': low_prob}
         },
        {'RandomBiasField': {
            'include': ['image'],
            'p': tiny_prob,
            'coefficients': 0.5}
         },
        {'RandomMotion': {
            'include': ['image', 'label'],
            'p': tiny_prob,
            'num_transforms': 1}
         },
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'After TORCHIO'}},
        # {'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0}},
    ],
    'labelonly_transform': [
        # {'ToTensord': {'keys': ['label']}},
        # {'AddChanneld': {'keys': ['label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize'}},
    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.5}
         },
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            'mode': 'nearest'}
         },
        # 'ToTensord': {'keys': ['image', 'label']},

        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize and resize'},
        # 'AddChanneld': {'keys': ['image']},
        # 'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0},
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
    if spatial_size is None:
        spatial_size = find_param_from_hyper_dict(hyper_param_dict, 'spatial_size')
    x_grad = np.expand_dims(np.zeros(spatial_size), 0)
    y_grad = np.expand_dims(np.zeros(spatial_size), 0)
    z_grad = np.expand_dims(np.zeros(spatial_size), 0)
    for k in range(spatial_size[0]):
        x_grad[0, k, :, :] = k
    for k in range(spatial_size[1]):
        y_grad[0, :, k, :] = k
    for k in range(spatial_size[2]):
        z_grad[0, :, :, k] = k
    gradients = np.concatenate((x_grad, y_grad, z_grad), 0)
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

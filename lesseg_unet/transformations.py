from math import radians
import logging
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
    AsChannelFirstd
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
        if len(tensor_shape) == 5:
            output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
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
high_prob = 0.5
low_prob = 0.1
def_spatial_size = [96, 96, 96]
# .74 aspect ratio? maybe change to 96x128x96 or crop to 64cube and increase the epoch number by a lot
# TODO verify that random transformations are applied the same way on the image and the seg
hyper_dict = {
    'first_transform': {
        'LoadImaged': {'keys': ['image', 'label']
                       },
        'AddChanneld': {'keys': ['image', 'label']},
        'Binarized': {'keys': ['label']},
        # 'AsChannelFirstd': {
        #     'keys': ['image', 'label'],
        #     'channel_dim': -1
        # },
        'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'Fisrt resize'},
    },
    'monai_transform': {
        # 'ScaleIntensity': {}
        # 'PrintDim': {'keys': ['image', 'label']},
        'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob
        },
        # TODO maybe 'Orientation': {} but it would interact with the flip,
        'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(15),
            'shear_range': None,
            'translate_range': None,
            'scale_range': 0.3,
            'spatial_size': None,
            'padding_mode': 'border',
            'as_tensor_output': False
        },
        'RandFlipd': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'spatial_axis': 0
        },
        # 'RandDeformGrid': {'keys': ['image', 'label']},
        # 'Spacingd': {'keys': ['image', 'label']},
        'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (0, 0.01),
            'magnitude_range': (0, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': high_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'spatial_size': None,
            # 'padding_mode': "reflection",
            'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': False
        },
        # 'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0},
        'ToTensord': {'keys': ['image', 'label']},
        # 'AddChanneld': {'keys': ['image', 'label']},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'After MONAI'},
    },
    'torchio_transform': {
        # 'PrintDim': {'keys': ['image', 'label']},
        'RandomNoise': {
            'include': ['image'],
            'mean': 0,
            'std': (0.1, 0.2),
            'p': low_prob
        },
        'RandomGhosting': {
            'include': ['image'],
            'p': low_prob,
            'num_ghosts': (4, 10)
        },
        'RandomBlur': {
            'include': ['image', 'label'],
            'std': (0.1, 0.5),
            'p': low_prob
        },
        'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 1
        },
        'RandomMotion': {
            'include': ['image', 'label'],
            'p': low_prob,
            'num_transforms': 1
        },
        'ToTensord': {'keys': ['image', 'label']},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'After TORCHIO'},
        # 'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0},
    },
    'labelonly_transform': {
        'Binarized': {'keys': ['label']},
        # 'ToTensord': {'keys': ['label']},
        # 'AddChanneld': {'keys': ['label']},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize'},
    },
    'last_transform': {
        'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size
        },
        'ToTensord': {'keys': ['image', 'label']},
        # 'AddChanneld': {'keys': ['label']},
        # 'NormalizeIntensityd': {'keys': ['image']},
    }
}

# Import all transforms from the dict
for k in hyper_dict:
    for name in hyper_dict[k]:
        try:
            eval(name)
        except NameError:
            raise ImportError('{} not imported'.format(name))

"""
Helper functions
"""


def trans_from_dict(transform_name, transform_dict):
    try:
        return eval(transform_name)(**transform_dict)
    except Exception as e:
        print('Exception found with transform {}: {}'.format(transform_name, e))


def trans_from_name(transform_name, hyper_param_dict):
    for key in hyper_param_dict:
        if transform_name in hyper_param_dict[key]:
            if transform_name in dir(torchio.transforms):
                return RandTransformWrapper(trans_from_dict(transform_name, hyper_param_dict[key]))
            else:
                return trans_from_dict(transform_name, hyper_param_dict[key])
    raise ValueError('{} not found in the hyper_param_dict'.format(transform_name))


def trans_list_from_dict(param_dict):
    trans_list = []
    for transform_name in param_dict:
        if transform_name in dir(torchio.transforms):
            trans = RandTransformWrapper(trans_from_dict(transform_name, param_dict[transform_name]))
        else:
            trans = trans_from_dict(transform_name, param_dict[transform_name])
        trans_list.append(trans)
    return trans_list


"""
Transformation compositions for the image segmentation
"""


def segmentation_train_transformd():
    train_transformd = Compose(
        trans_list_from_dict(hyper_dict['first_transform']) +
        trans_list_from_dict(hyper_dict['monai_transform']) +
        trans_list_from_dict(hyper_dict['torchio_transform']) +
        trans_list_from_dict(hyper_dict['labelonly_transform']) +
        trans_list_from_dict(hyper_dict['last_transform'])
    )
    return train_transformd


def segmentation_val_transformd():
    val_transd = Compose(
        trans_list_from_dict(hyper_dict['first_transform']) +
        trans_list_from_dict(hyper_dict['labelonly_transform']) +
        trans_list_from_dict(hyper_dict['last_transform'])
    )
    return val_transd

# def segmentation_train_transform():
#     train_imtrans = Compose(
#         trans_list_from_dict(hyper_dict['first_transform']) +
#         trans_list_from_dict(hyper_dict['intensity_transform']) +
#         trans_list_from_dict(hyper_dict['shape_transform']) +
#         trans_list_from_dict(hyper_dict['last_transform'])
#     )
#     train_segtrans = Compose(
#         trans_list_from_dict(hyper_dict['first_transform']) +
#         trans_list_from_dict(hyper_dict['shape_transform']) +
#         trans_list_from_dict(hyper_dict['seg_transform']) +
#         trans_list_from_dict(hyper_dict['last_transform'])
#     )
#     return train_imtrans, train_segtrans


# def segmentation_val_transform(resize=True):
#     val_imtrans = Compose(
#         trans_list_from_dict(hyper_dict['first_transform']) +
#         trans_list_from_dict(hyper_dict['shape_transform']) +
#         trans_list_from_dict(hyper_dict['last_transform']) if resize else [trans_from_name('ToTensor', hyper_dict)]
#     )
#     val_segtrans = Compose(
#         trans_list_from_dict(hyper_dict['first_transform']) +
#         trans_list_from_dict(hyper_dict['shape_transform']) +
#         # trans_list_from_dict(hyper_dict['seg_transform']) +
#         trans_list_from_dict(hyper_dict['last_transform']) if resize else [trans_from_name('ToTensor', hyper_dict)]
#     )
#     return val_imtrans, val_segtrans

# Augmentation params
# hyper_params['use_augmentation_in_training'] = True
# hyper_params['enable_Addblob'] = False
# hyper_params['aug_prob'] = 0.1
# hyper_params['aug_prob_rigid'] = 0.5
# hyper_params['torchio_aug_prob'] = 0.1
# hyper_params['RandHistogramShift_numcontrolpoints'] = (10, 15)
# hyper_params['RandomBlur_std'] = (0.1, 0.5)
# hyper_params['RandShiftIntensity_offset'] = 0.5
# hyper_params['RandAdjustContrast_gamma'] = 0.5
# hyper_params['RandomAffine_scales_range_frac'] = 0.3
# hyper_params['RandomAffine_max_degree_rotation'] = 360
# hyper_params['RandomAffine_max_degree_shear'] = 10
# hyper_params['RandomAffine_translate_voxels_range'] = 20
# hyper_params['RandomAffine_image_interpolation'] = 'nearest'
# hyper_params['RandZoom_mode'] = 'nearest'
# # hyper_params['RandomElastic_sigma_range'] = (0.01, 1)
# # hyper_params['RandomElasticDeformation_numcontrolpoints'] = hyper_params['resample_dims'][0] // 8
# # hyper_params['RandomElasticDeformation_maxdisplacement'] = 1.0
# # hyper_params['RandomElasticDeformation_prob'] = 0.1
# hyper_params['torchio_RandomMotion_num_transforms'] = 1
# hyper_params['torchio_RandomGhosting_num_ghosts'] = (4, 10)
# hyper_params['torchio_RandomSpike_intensity'] = (1, 3)
# hyper_params['torchio_RandomBiasField_magnitude'] = 1
# hyper_params['torchio_RandomNoise_std'] = (0.1, 0.2)
# hyper_params['torchio_RandomFlip_axes'] = (0, 1, 2)
# hyper_params['torchio_RandomFlip_per_axes_prob'] = 0.3


# transforms_resize_to_tensor = [monai_trans.Resize(hyper_params['resample_dims']),
#                                monai_trans.ToTensor()]
# train_transforms = []
# if hyper_params['use_augmentation_in_training']:
#     rot_angle_in_rads = hyper_params['RandomAffine_max_degree_rotation'] * (2 * np.pi / 360)
#     shear_angle_in_rads = hyper_params['RandomAffine_max_degree_shear'] * (2 * np.pi / 360)
#
#     # if hyper_params['enable_Addblob']:
#     #     train_transforms += [customAddblob(hyper_params['aug_prob'])]
#     train_transforms += [
#         monai_trans.RandHistogramShift(num_control_points=hyper_params['RandHistogramShift_numcontrolpoints'],
#                                        prob=hyper_params['aug_prob'])]
#     train_transforms += [monai_trans.RandAffine(prob=hyper_params['aug_prob_rigid'],
#                                                 rotate_range=rot_angle_in_rads,
#                                                 shear_range=None,
#                                                 translate_range=hyper_params['RandomAffine_translate_voxels_range'],
#                                                 scale_range=None,
#                                                 spatial_size=None,
#                                                 padding_mode="border",
#                                                 as_tensor_output=False)]
#     train_transforms += [monai_trans.RandAffine(prob=hyper_params['aug_prob'],
#                                                 rotate_range=None,
#                                                 shear_range=shear_angle_in_rads,
#                                                 translate_range=None,
#                                                 scale_range=hyper_params['RandomAffine_scales_range_frac'],
#                                                 spatial_size=None,
#                                                 padding_mode="border",
#                                                 as_tensor_output=False)]
#     # train_transforms += [monai_trans.Rand3DElastic(sigma_range=hyper_params['RandomElastic_sigma_range'],
#     #                                                 magnitude_range=(0, 1),
#     #                                                 prob=hyper_params['aug_prob'],
#     #                                                 rotate_range=(
#     #                                                 rot_angle_in_rads, rot_angle_in_rads, rot_angle_in_rads),
#     #                                                 shear_range=(
#     #                                                 shear_angle_in_rads, shear_angle_in_rads, shear_angle_in_rads),
#     #                                                 translate_range=hyper_params[
#     #                                                     'RandomAffine_translate_voxels_range'],
#     #                                                 scale_range=hyper_params['RandomAffine_scales_range_frac'],
#     #                                                 spatial_size=None,
#     #                                                 # padding_mode="reflection",
#     #                                                 padding_mode="border",
#     #                                                 # padding_mode="zeros",
#     #                                                 as_tensor_output=False)]
#     # train_transforms += [monai_trans.Rand3DElastic(sigma_range=(0, 0.01),
#     #                                                magnitude_range=(0, 5),  # hyper_params['Rand3DElastic_magnitude_range']
#     #                                                prob=1,
#     #                                                rotate_range=None,
#     #                                                shear_range=None,
#     #                                                translate_range=None,
#     #                                                scale_range=None,
#     #                                                spatial_size=None,
#     #                                                # padding_mode="reflection",
#     #                                                padding_mode="border",
#     #                                                # padding_mode="zeros",
#     #                                                as_tensor_output=False)]
#     train_transforms += [torchio_trans.RandomBlur(std=hyper_params['RandomBlur_std'],
#                                                   p=hyper_params['aug_prob'])]
#     train_transforms += [torchio_trans.RandomNoise(mean=0, std=hyper_params['torchio_RandomNoise_std'],
#                                                    p=hyper_params['torchio_aug_prob'])]
#     train_transforms += [torchio_trans.RandomFlip(axes=hyper_params['torchio_RandomFlip_axes'],
#                                                   flip_probability=hyper_params['torchio_RandomFlip_per_axes_prob'],
#                                                   p=hyper_params['torchio_aug_prob'])]
#     train_transforms += [torchio_trans.RandomMotion(p=hyper_params['torchio_aug_prob'],
#                                                     num_transforms=hyper_params['torchio_RandomMotion_num_transforms'])]
#     train_transforms += [torchio_trans.RandomGhosting(p=hyper_params['torchio_aug_prob'],
#                                                       num_ghosts=hyper_params['torchio_RandomGhosting_num_ghosts'])]
#     train_transforms += [torchio_trans.RandomSpike(p=hyper_params['torchio_aug_prob'],
#                                                    intensity=hyper_params['torchio_RandomSpike_intensity'])]
#     train_transforms += [torchio_trans.RandomBiasField(p=hyper_params['torchio_aug_prob'],
#                                                        coefficients=hyper_params['torchio_RandomBiasField_magnitude'])]
#     train_transforms = [torchvis_trans.RandomOrder(train_transforms)]  # Randomise the transforms we've just defined
#     train_transforms += [customIsotropicResampling(prob=hyper_params['aug_prob'],
#                                                    resample_dims=hyper_params['resample_dims'])]
# train_transforms += transforms_resize_to_tensor
# train_transforms = monai_trans.Compose(train_transforms)
# val_head_transforms = monai_trans.Compose(transforms_resize_to_tensor)
# val_nonhead_transforms = monai_trans.Compose(transforms_resize_to_tensor)
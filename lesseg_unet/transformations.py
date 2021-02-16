from math import radians
import logging

import numpy as np
from monai.transforms import (
    Transform,
    AddChannel,
    Compose,
    RandRotate90,
    RandFlip,
    RandSpatialCrop,
    ScaleIntensity,
    ToTensor,
    Resize,
    RandAffine,
    Rand3DElastic
)
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
Transformation classes
"""


class Binarize(Transform):
    """
    Sets every non-zero voxel to 1.0

    """

    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return np.asarray(np.where(img != 0, 1, 0), dtype=img.dtype)


"""
Transformation parameters
"""
high_prob = 0.5
low_prob = 0.1
def_spatial_size = [96, 96, 96]
def_RandAffine_dict = {'prob': low_prob,
                       'rotate_range': radians(15),
                       'shear_range': None,
                       'translate_range': None,
                       'scale_range': 0.3,
                       'spatial_size': None,
                       'padding_mode': 'border',
                       'sas_tensor_output': False}


randaffine_trans = RandAffine(**def_RandAffine_dict)
resize_trans = Resize(def_spatial_size)

def_Rand3DElastic_dict = {
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
}

rand3delastic_trans = Rand3DElastic(**def_Rand3DElastic_dict)

rand_blur_std = (0.1, 0.5)
train_transforms = RandomBlur(std=rand_blur_std, p=low_prob)
randomNoise_std = (0.1, 0.2)
RandomNoise(mean=0, std=randomNoise_std, p=low_prob)
RandomMotion_num_transforms = 1
RandomMotion(p=low_prob, num_transforms=RandomMotion_num_transforms)
RandomGhosting_num_ghosts = (4, 10)
RandomGhosting(p=low_prob, num_ghosts=RandomGhosting_num_ghosts)
RandomSpike_intensity = (1, 3)
RandomSpike(p=low_prob, intensity=RandomSpike_intensity)
RandomBiasField_magnitude = 1
RandomBiasField(p=low_prob, coefficients=RandomBiasField_magnitude)
randomize_transformations = True
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

first_transform = [AddChannel()]
last_transform = [resize_trans, ToTensor()]
seg_transform = [Binarize()]
middle_trans = []
if randomize_transformations:
    middle_trans = RandomOrder(middle_trans)

val_transform = [first_transform, last_transform]
val_transform.insert(1, randaffine_trans)
val_transform.insert(-1, randaffine_trans)
train_transform = [first_transform, last_transform]


class TransformList(list):
    pass


"""
Transformation compositions for the image segmentation
"""


def segmentation_train_transform(spatial_size=None):
    if spatial_size is None:
        spatial_size = [96, 96, 96]
    train_imtrans = Compose(
        [
            AddChannel(),
            ScaleIntensity(),
            # RandSpatialCrop(spatial_size, random_size=False),
            # RandShiftIntensity(offsets, prob=0.1),
            RandFlip(prob=0.1, spatial_axis=None),
            Resize(spatial_size),
            # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            ToTensor(),
        ]
    )
    train_segtrans = Compose(
        [
            AddChannel(),
            # RandSpatialCrop(spatial_size, random_size=False),
            RandFlip(prob=0.1, spatial_axis=None),
            Resize(spatial_size),
            # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            Binarize(),
            ToTensor(),
        ]
    )
    return train_imtrans, train_segtrans


def segmentation_val_transform(spatial_size=None):
    if spatial_size is None:
        val_imtrans = Compose([ScaleIntensity(), AddChannel(), ToTensor()])
        val_segtrans = Compose([AddChannel(),
                                Binarize(),
                                ToTensor()])
    else:
        val_imtrans = Compose([ScaleIntensity(), AddChannel(), Resize(spatial_size), ToTensor()])
        val_segtrans = Compose([AddChannel(), Resize(spatial_size),
                                Binarize(),
                                ToTensor()])
    return val_imtrans, val_segtrans

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
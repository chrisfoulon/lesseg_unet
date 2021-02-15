import math
import logging
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np

import torch
import monai
from monai.data import create_test_image_3d, list_data_collate, NiftiDataset
from torch.utils.data import DataLoader

from monai.transforms import (
    Transform,
    AddChannel,
    Compose,
    RandRotate90,
    RandFlip,
    RandSpatialCrop,
    ScaleIntensity,
    ToTensor,
    Resize
)
from monai.transforms import (
    AsChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
    Resized
)


def match_img_seg_by_names(img_path_list: Sequence, seg_path_list: Sequence,
                           img_pref: str = None) -> dict:
    img_dict = {}
    if img_pref is not None and img_pref != '':
        img_path_list = [img for img in img_path_list if img_pref in Path(img).name]
    for img in img_path_list:
        matching_les_list = [les for les in seg_path_list if Path(img).name in Path(les).name]
        if len(matching_les_list) == 0:
            raise ValueError('No matching seg file found for {}'.format(img))
        if len(matching_les_list) > 1:
            raise ValueError('Multiple matching seg file found for {}'.format(img))
        img_dict[img] = matching_les_list[0]
    print('Number of images: {}'.format(len(img_dict)))
    return img_dict


def create_file_dict_lists(raw_img_path_list: Sequence, raw_seg_path_list: Sequence,
                           img_pref: str = None, train_val_percentage: float = 75) -> Tuple[list, list]:
    img_dict = match_img_seg_by_names(raw_img_path_list, raw_seg_path_list, img_pref)
    training_end_index = math.ceil(train_val_percentage / 100 * len(img_dict))
    full_file_list = [{"img": img, "seg": img_dict[img]} for img in img_dict]
    train_files = full_file_list[:training_end_index]
    val_files = full_file_list[training_end_index:]
    return train_files, val_files


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


def segmentation_train_transform(spatial_size=None):
    if spatial_size is None:
        spatial_size = [96, 96, 96]
    train_imtrans = Compose(
        [
            ScaleIntensity(),
            AddChannel(),
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
            Resize(spatial_size),
            RandFlip(prob=0.1, spatial_axis=None),
            # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            Binarize(),
            ToTensor(),
        ]
    )
    return train_imtrans, train_segtrans


def segmentation_train_transformd(spatial_size=None):
    # define transforms for image and segmentation
    if spatial_size is None:
        spatial_size = [96, 96, 96]
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            # RandCropByPosNegLabeld(
            #     keys=["img", "seg"], label_key="seg", spatial_size=spatial_size, pos=1, neg=1, num_samples=4
            # ),
            Resized(keys=["img", "seg"], spatial_size=spatial_size),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    return train_transforms


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


def segmentation_val_transformd():
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    return val_transforms


def create_training_data_loader(train_ds: monai.data.Dataset,
                                batch_size: int = 10,
                                dataloader_workers: int = 4):
    logging.info('Creating training data loader')
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader


def create_validation_data_loader(val_ds: monai.data.Dataset,
                                  batch_size: int = 1,
                                  dataloader_workers: int = 4):
    logging.info('Creating validation data loader')
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=dataloader_workers,
                            pin_memory=torch.cuda.is_available())
    return val_loader


def data_loader_checker_first(check_ds, set_name=''):
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    img_batch, seg_batch = monai.utils.misc.first(check_loader)
    logging.info('First {} loader batch size: images {}, lesions {}'.format(
        set_name,
        img_batch.shape,
        seg_batch.shape))
    return img_batch, seg_batch

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

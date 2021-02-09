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
        val_segtrans = Compose([AddChannel(), Binarize(), ToTensor()])
    else:
        val_imtrans = Compose([ScaleIntensity(), AddChannel(), Resize(spatial_size), ToTensor()])
        val_segtrans = Compose([AddChannel(), Resize(spatial_size), Binarize(), ToTensor()])
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

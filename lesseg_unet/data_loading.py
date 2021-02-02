import math
import logging
from pathlib import Path
from collections.abc import Sequence

import torch
import monai
from monai.data import create_test_image_3d, list_data_collate
from torch.utils.data import DataLoader
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
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


def create_segmentation_data_loader(img_list: Sequence,
                                    les_list: Sequence,
                                    image_pref: str,
                                    batch_size: int = 10,
                                    training_validation_cut: float = 75,
                                    dataloader_workers: int = 4):
    img_dict = match_img_seg_by_names(img_list, les_list, image_pref)
    training_end_index = math.ceil(training_validation_cut / 100 * len(img_dict))
    full_file_list = [{"img": img, "seg": img_dict[img]} for img in img_dict]
    train_files = full_file_list[:training_end_index]
    val_files = full_file_list[training_end_index:]
    # Define transforms for image and segmentation
    logging.info('Create transformations')

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    logging.info('Create dataloader')

    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=dataloader_workers, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    logging.info('First loader batch size: images {}, lesions {}'.format(
        check_data["img"].shape, check_data["seg"].shape))

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
import math
import logging
from pathlib import Path
from typing import Sequence, Tuple

import torch
import monai
from monai.data import list_data_collate
from torch.utils.data import DataLoader
from monai.data import Dataset
from lesseg_unet import transformations


def match_img_seg_by_names(img_path_list: Sequence, seg_path_list: Sequence,
                           img_pref: str = None) -> dict:
    img_dict = {}
    if img_pref is not None and img_pref != '':
        img_path_list = [str(img) for img in img_path_list if img_pref in Path(img).name]
    for img in img_path_list:
        matching_les_list = [str(les) for les in seg_path_list if Path(img).name in Path(les).name]
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
    full_file_list = [{'image': img, 'label': img_dict[img]} for img in img_dict]
    train_files = full_file_list[:training_end_index]
    val_files = full_file_list[training_end_index:]
    return train_files, val_files


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
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(),
                              collate_fn=list_data_collate)
    first_dict = monai.utils.misc.first(check_loader)
    img_batch, seg_batch = first_dict['image'], first_dict['label']
    logging.info('First {} loader (total size: {}) batch size: images {}, lesions {}'.format(
        set_name,
        len(check_ds),
        img_batch.shape,
        seg_batch.shape))
    return img_batch, seg_batch


def init_training_data(img_path_list: Sequence,
                       seg_path_list: Sequence,
                       img_pref: str = None,
                       transform_dict=None,
                       train_val_percentage: float = 75) -> Tuple[monai.data.Dataset, monai.data.Dataset]:
    logging.info('Listing input files to be loaded')
    train_files, val_files = create_file_dict_lists(img_path_list, seg_path_list, img_pref,
                                                    train_val_percentage)
    logging.info('Create transformations')
    train_img_transforms = transformations.segmentation_train_transformd(transform_dict)
    val_img_transforms = transformations.segmentation_val_transformd(transform_dict)
    # define dataset, data loader
    logging.info('Create training monai datasets')
    train_ds = Dataset(train_files, transform=train_img_transforms)

    # define dataset, data loader
    logging.info('Create validation actual monai datasets')
    val_ds = ImageDataset(val_files, transform=val_img_transforms, )
    # We check if both the training and validation dataloaders can be created and used without immediate errors
    logging.info('Checking data loading')
    if train_val_percentage:
        data_loader_checker_first(train_ds, 'training')
    if train_val_percentage != 100:
        data_loader_checker_first(val_ds, 'validation')
    logging.info('Init training done.')
    return train_ds, val_ds

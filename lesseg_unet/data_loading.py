import math
import logging
import os
from pathlib import Path
from typing import Sequence, Tuple, Union, List

import numpy as np
import nibabel as nib
import torch
import monai
from monai.data import list_data_collate, DataLoader
from monai.data import Dataset, PersistentDataset, CacheDataset
from lesseg_unet import transformations, utils


def match_img_seg_by_names(img_path_list: Sequence, seg_path_list: Sequence,
                           img_pref: str = None, check_inputs=True) -> (dict, dict):
    # create_default_label = False
    # if default_label is not None:
    #     if Path(default_label).is_dir():
    #         create_default_label = True
    #     else:
    #         if not Path(default_label).is_file():
    #             raise ValueError('fill_up_empty_labels must be an existing nifti file ')
    controls = []
    img_dict = {}
    no_match = False
    if img_pref is not None and img_pref != '':
        img_path_list = [str(img) for img in img_path_list if img_pref in Path(img).name]
    for img in img_path_list:
        if seg_path_list is None:
            matching_les_list = []
        else:
            matching_les_list = [str(les) for les in seg_path_list if Path(img).name in Path(les).name]
        if len(matching_les_list) == 0:
            controls.append(img)
        elif len(matching_les_list) > 1:
            raise ValueError('Multiple matching seg file found for {}'.format(img))
        else:
            img_dict[img] = matching_les_list[0]
    if no_match:
        print(f'Some images did not have a label ({len(controls)} they have been added to the controls list')
    print('Number of images: {}'.format(len(img_dict)))
    if img_dict:
        print(f'First image and label in img_dict: {img_dict[list(img_dict.keys())[0]]}')
    if controls:
        print('Number of controls: {}'.format(len(controls)))
    if check_inputs:
        utils.check_inputs(img_dict)
        utils.check_inputs(controls)
        print('The inputs passed the checks')
    return img_dict, controls


def create_file_dict_lists(raw_img_path_list: Sequence, raw_seg_path_list: Sequence,
                           img_pref: str = None, train_val_percentage: float = 75) -> Tuple[list, list]:
    img_dict, _ = match_img_seg_by_names(raw_img_path_list, raw_seg_path_list, img_pref)
    training_end_index = math.ceil(train_val_percentage / 100 * len(img_dict))
    full_file_list = [{'image': str(img), 'label': str(img_dict[img])} for img in img_dict]
    train_files = full_file_list[:training_end_index]
    val_files = full_file_list[training_end_index:]
    return train_files, val_files


def create_training_data_loader(train_ds: monai.data.Dataset,
                                batch_size: int = 10,
                                dataloader_workers: int = 4,
                                persistent_workers=True):
    print('Creating training data loader')
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    return train_loader


def create_validation_data_loader(val_ds: monai.data.Dataset,
                                  batch_size: int = 1,
                                  dataloader_workers: int = 4):
    print('Creating validation data loader')
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=dataloader_workers,
                            pin_memory=torch.cuda.is_available(), persistent_workers=True)
    return val_loader


def data_loader_checker_first(check_ds, set_name=''):
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(),
                              collate_fn=list_data_collate, persistent_workers=False)
    first_dict = monai.utils.misc.first(check_loader)
    img_batch, seg_batch = first_dict['image'], first_dict['label']
    logging.info('First {} loader (total size: {}) batch size: images {}, lesions {}'.format(
        set_name,
        len(check_ds),
        img_batch.shape,
        seg_batch.shape))
    return img_batch, seg_batch


def init_training_data(
        img_path_list: Sequence,
        seg_path_list: Sequence,
        img_pref: str = None,
        transform_dict: dict = None,
        train_val_percentage: float = 75,
        clamping=None) -> Tuple[monai.data.Dataset, monai.data.Dataset]:
    print('Listing input files to be loaded')
    train_files, val_files = create_file_dict_lists(img_path_list, seg_path_list, img_pref,
                                                    train_val_percentage)
    print('Create transformations')
    train_img_transforms = transformations.segmentation_train_transformd(transform_dict, clamping)
    val_img_transforms = transformations.segmentation_val_transformd(transform_dict, clamping)
    # define dataset, data loader
    print('Create training monai datasets')
    train_ds = Dataset(train_files, transform=train_img_transforms)
    # define dataset, data loader
    logging.info('Create validation monai datasets')
    val_ds = Dataset(val_files, transform=val_img_transforms)
    # We check if both the training and validation dataloaders can be created and used without immediate errors
    print('Checking data loading')
    if train_val_percentage:
        data_loader_checker_first(train_ds, 'training')
    if train_val_percentage != 100:
        data_loader_checker_first(val_ds, 'validation')
    print('Init training done.')
    return train_ds, val_ds


def get_data_folds(img_seg_dict: dict,
                   folds_number: int = 1,
                   train_val_percentage: float = 80) -> (List[List[dict]], List[List[dict]]):
    logging.info(f'Listing input files to be loaded with {folds_number} folds')
    full_file_list = [{'image': str(img), 'label': str(img_seg_dict[img])} for img in img_seg_dict]
    if folds_number == 1:
        training_end_index = math.ceil(train_val_percentage / 100 * len(img_seg_dict))
        # Inverse the order of the splits because the validation chunk in that case will be 0
        split_lists = [full_file_list[training_end_index:], full_file_list[:training_end_index]]
    else:
        split_lists = list(np.array_split(np.array(full_file_list), folds_number))
    return split_lists


def create_fold_dataloaders(split_lists, fold, train_img_transforms, val_img_transforms, batch_size,
                            dataloader_workers, val_batch_size=1, cache_dir=None):
    train_data_list = []
    val_data_list = []
    for ind, chunk in enumerate(split_lists):
        if ind == fold:
            val_data_list = chunk
        else:
            train_data_list = np.concatenate([train_data_list, chunk])
    print(f'Create training monai dataset for fold {fold}')
    if cache_dir is not None:
        train_ds = PersistentDataset(train_data_list, transform=train_img_transforms, cache_dir=cache_dir)
    else:
        train_ds = CacheDataset(train_data_list, transform=train_img_transforms)
    # train_ds = Dataset(train_data_list, transform=train_img_transforms)
    # data_loader_checker_first(train_ds, 'training')
    # define dataset, data loader
    print(f'Create validation monai dataset')
    if cache_dir is not None:
        val_ds = PersistentDataset(val_data_list, transform=val_img_transforms, cache_dir=cache_dir)
    else:
        val_ds = CacheDataset(val_data_list, transform=val_img_transforms)
    # val_ds = Dataset(val_data_list, transform=val_img_transforms)
    # data_loader_checker_first(train_ds, 'validation')
    train_loader = create_training_data_loader(train_ds, batch_size, dataloader_workers)
    val_loader = create_validation_data_loader(val_ds, val_batch_size, dataloader_workers)
    return train_loader, val_loader


def init_segmentation(img_path_list: Sequence,
                      img_pref: str = None,
                      transform_dict: dict = None,
                      clamping=None):
    if img_pref is None:
        img_pref = ''
    print('Listing input files to be loaded')
    image_list = [{'image': str(img)} for img in img_path_list if img_pref in Path(img).name]
    print('Create transformations')
    val_img_transforms = transformations.image_only_transformd(transform_dict, training=False, clamping=clamping)
    print('Create monai dataset')
    train_ds = Dataset(image_list, transform=val_img_transforms)
    return train_ds

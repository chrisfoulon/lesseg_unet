import math
import logging
from pathlib import Path
from typing import Sequence, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import monai
from monai.data import list_data_collate, DataLoader
from monai.data import Dataset, PersistentDataset, CacheDataset
from lesseg_unet import transformations, utils
from lesseg_unet.utils import get_str_path_list


def match_img_seg_by_names(img_path_list: Sequence, seg_path_list: Sequence,
                           img_pref: str = None, image_cut_prefix: str = None,
                           image_cut_suffix: str = None, check_inputs=True) -> (dict, dict):
    unmatched_images = []
    img_dict = {}
    no_match = False
    img_path_list = get_str_path_list(img_path_list, img_pref)
    for img in img_path_list:
        if seg_path_list is None:
            matching_les_list = []
        else:
            def condition(les):
                matched = Path(img).name.split('.nii')[0] in Path(les).name.split('.nii')[0]
                if matched:
                    return True
                else:
                    if image_cut_suffix is not None and image_cut_prefix is not None:
                        return Path(img).name.split(
                            image_cut_prefix)[-1].split(image_cut_suffix)[0] in Path(les).name.split('.nii')[0]
                    if image_cut_suffix is not None:
                        return Path(img).name.split(image_cut_suffix)[0] in Path(les).name.split('.nii')[0]
                    if image_cut_prefix is not None:
                        return Path(img).name.split(image_cut_prefix)[-1] in Path(les).name.split('.nii')[0]
            # TODO make it work with both prefix and suffix
            matching_les_list = [str(les) for les in seg_path_list if condition(les)]
        if len(matching_les_list) == 0:
            unmatched_images.append(img)
        elif len(matching_les_list) > 1:
            raise ValueError('Multiple matching seg file found for {}'.format(img))
        else:
            img_dict[img] = matching_les_list[0]
    if no_match:
        print(f'Some images did not have a label ({len(unmatched_images)} they have been added to the controls list')
    print('Number of images: {}'.format(len(img_dict)))
    if img_dict:
        print(f'First image and label in img_dict: {list(img_dict.keys())[0]}, {img_dict[list(img_dict.keys())[0]]}')
    if unmatched_images:
        # print('Number of controls: {}'.format(len(controls)))
        raise ValueError(f'{len(unmatched_images)} could not be matched with a label')
    if check_inputs:
        utils.check_inputs(img_dict)
        utils.check_inputs(unmatched_images)
        print('The inputs passed the checks')
    return img_dict, unmatched_images


def create_file_dict_lists(raw_img_path_list: Sequence, raw_seg_path_list: Sequence,
                           img_pref: str = None, image_cut_prefix: str = None,
                           image_cut_suffix: str = None,
                           train_val_percentage: float = 75) -> Tuple[list, list]:
    img_dict, _ = match_img_seg_by_names(raw_img_path_list, raw_seg_path_list, img_pref,
                                         image_cut_prefix=image_cut_prefix,
                                         image_cut_suffix=image_cut_suffix)
    training_end_index = math.ceil(train_val_percentage / 100 * len(img_dict))
    full_file_list = [{'image': str(img), 'label': str(img_dict[img])} for img in img_dict]
    train_files = full_file_list[:training_end_index]
    val_files = full_file_list[training_end_index:]
    return train_files, val_files


def create_training_data_loader(train_ds: monai.data.Dataset,
                                batch_size: int = 10,
                                dataloader_workers: int = 4,
                                persistent_workers=True,
                                shuffle=True,
                                sampler=None):
    print('Creating training data loader')
    # The shuffle option is determined in the sampler
    if sampler is not None:
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=dataloader_workers,
        pin_memory=False,
        # pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        sampler=sampler,
        # TODO check which number is faster
        prefetch_factor=2
    )
    return train_loader


def create_validation_data_loader(val_ds: monai.data.Dataset,
                                  batch_size: int = 1,
                                  dataloader_workers: int = 4,
                                  sampler=None):
    print('Creating validation data loader')
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=dataloader_workers,
                            pin_memory=False,
                            # pin_memory=torch.cuda.is_available(),
                            persistent_workers=True, sampler=sampler)
    return val_loader


def data_loader_checker_first(check_ds, set_name=''):
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=False,
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
        image_cut_prefix: str = None,
        image_cut_suffix: str = None,
        transform_dict: dict = None,
        train_val_percentage: float = 75,
        clamping=None) -> Tuple[monai.data.Dataset, monai.data.Dataset]:
    print('Listing input files to be loaded')
    train_files, val_files = create_file_dict_lists(img_path_list, seg_path_list, img_pref,
                                                    image_cut_prefix,
                                                    image_cut_suffix,
                                                    train_val_percentage)
    print('Create transformations')
    train_img_transforms = transformations.train_transformd(transform_dict, clamping)
    val_img_transforms = transformations.val_transformd(transform_dict, clamping)
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


def create_fold_dataloaders(split_lists, fold, train_img_transforms, val_img_transforms, batch_size,
                            dataloader_workers, val_batch_size=1, cache_dir=None, cache_rate=0, world_size=1, rank=0,
                            shuffle_training=True, cache_num=None, training_persistent_workers=True):
    train_data_list = []
    val_data_list = []
    for ind, chunk in enumerate(split_lists):
        if ind == fold:
            val_data_list = chunk
        else:
            train_data_list = np.concatenate([train_data_list, chunk])
    utils.print_rank_0(f'Creating training monai dataset for fold {fold}', rank)
    # if cache_dir is not None:
    #     train_ds = PersistentDataset(train_data_list, transform=train_img_transforms, cache_dir=cache_dir)
    # else:
    #     if cache_num is None:
    #         cache_num = len(train_data_list)
    #     train_ds = CacheDataset(train_data_list, transform=train_img_transforms, cache_num=cache_num,
    #                             cache_rate=cache_rate)
    train_ds = Dataset(train_data_list, transform=train_img_transforms)
    # data_loader_checker_first(train_ds, 'training')
    # define dataset, data loader
    utils.print_rank_0(f'Creating validation monai dataset', rank)
    # if cache_dir is not None:
    #     val_ds = PersistentDataset(val_data_list, transform=val_img_transforms, cache_dir=cache_dir)
    # else:
    #     # val_ds = CacheDataset(val_data_list, transform=val_img_transforms, cache_num=cache_num,
    val_ds = CacheDataset(val_data_list, transform=val_img_transforms, cache_rate=cache_rate)
    print('##############################DEBUG##############################')
    print(type(train_ds))
    print('##############################DEBUG##############################')
    if world_size > 0:
        # if dataloader_workers > 1:
        #     dataloader_workers = 1
        #     print('Number of workers for the dataloader changed to 1 as DDP is activated')
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=shuffle_training,
                                           drop_last=True)
        # val_sampler = None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    val_ds = Dataset(val_data_list, transform=val_img_transforms)
    # data_loader_checker_first(train_ds, 'validation')
    train_loader = create_training_data_loader(train_ds, batch_size, dataloader_workers,
                                               sampler=train_sampler, shuffle=shuffle_training,
                                               persistent_workers=training_persistent_workers)
    val_loader = create_validation_data_loader(val_ds, val_batch_size, dataloader_workers, sampler=val_sampler)

    return train_loader, val_loader

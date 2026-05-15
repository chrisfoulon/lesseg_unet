import math
import logging
import sys
import json
import hashlib
import warnings
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import monai
from monai.data import list_data_collate, DataLoader
from monai.data import Dataset, PersistentDataset, CacheDataset
from lesseg_unet import transformations, utils
from lesseg_unet.utils import get_str_path_list


# Cache management constants
CACHE_MARKER_FILE = '.lesseg_cache_marker'
CACHE_METADATA_FILE = '.cache_metadata.json'


def create_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """
    Create cache directory with safety marker file.

    Args:
        cache_dir: Path to cache directory

    Returns:
        Path object for created cache directory
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create marker file for safe cleanup
    marker_path = cache_dir / CACHE_MARKER_FILE
    marker_path.write_text(
        f"lesseg_unet cache directory\n"
        f"Created: {datetime.now().isoformat()}\n"
        f"Safe to delete this directory and all contents.\n"
    )

    return cache_dir


def cleanup_cache_dir(cache_dir: Union[str, Path], force: bool = False) -> None:
    """
    Safely cleanup cache directory with multiple safety checks.

    Args:
        cache_dir: Path to cache directory
        force: Skip safety checks (DANGEROUS - use only for testing)

    Raises:
        ValueError: If safety checks fail
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logging.info(f"Cache directory does not exist: {cache_dir}")
        return

    if not force:
        # SAFETY CHECK 1: Marker file must exist
        marker_path = cache_dir / CACHE_MARKER_FILE
        if not marker_path.exists():
            raise ValueError(
                f"SAFETY ERROR: Cache directory not created by lesseg_unet\n"
                f"Missing marker file: {marker_path}\n"
                f"Manual cleanup required: rm -rf {cache_dir}"
            )

        # SAFETY CHECK 2: Dangerous path validation
        dangerous_paths = ['/', '/home', '/usr', '/etc', '/var', '/tmp', str(Path.home())]
        if str(cache_dir.resolve()) in dangerous_paths:
            raise ValueError(
                f"SAFETY ERROR: Refusing to delete dangerous path: {cache_dir}\n"
                f"Manual cleanup required."
            )

        # User confirmation ONLY if interactive terminal
        if sys.stdin.isatty():
            print(f"About to delete cache directory: {cache_dir}")
            print(f"This will remove all cached data.")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Cleanup cancelled by user")
                return
        else:
            # Non-interactive (script): proceed if marker exists
            logging.info(f"Non-interactive mode: Cleaning cache (marker verified): {cache_dir}")

    # Actually cleanup
    shutil.rmtree(cache_dir)
    logging.info(f"Cache directory cleaned up: {cache_dir}")


def get_cache_metadata(transform_dict: Dict, spatial_size: Optional[List[int]] = None) -> Dict:
    """
    Generate cache metadata for current configuration.

    Args:
        transform_dict: Transform dictionary with all parameters
        spatial_size: Optional spatial size override

    Returns:
        Dictionary with config hash and metadata
    """
    # Create deterministic config representation
    config = {
        'transform_dict': transform_dict,
        'monai_version': monai.__version__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    # Add spatial size override if provided
    if spatial_size:
        config['spatial_size_override'] = spatial_size

    # Create hash
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    return {
        'config_hash': config_hash,
        'config': config,
        'created': datetime.now().isoformat()
    }


def is_cache_valid(cache_dir: Union[str, Path], current_metadata: Dict) -> bool:
    """
    Check if existing cache is valid for current configuration.

    Args:
        cache_dir: Path to cache directory
        current_metadata: Metadata for current configuration

    Returns:
        True if cache is valid, False otherwise
    """
    metadata_path = Path(cache_dir) / CACHE_METADATA_FILE

    if not metadata_path.exists():
        return False

    try:
        with open(metadata_path) as f:
            cached_metadata = json.load(f)

        # Compare config hashes
        if cached_metadata['config_hash'] != current_metadata['config_hash']:
            logging.info(f"Cache invalid: Configuration changed")
            logging.debug(f"  Cached config hash: {cached_metadata['config_hash']}")
            logging.debug(f"  Current config hash: {current_metadata['config_hash']}")
            return False

        return True

    except Exception as e:
        logging.warning(f"Could not read cache metadata: {e}")
        return False


def setup_disk_cache(
    cache_dir: Union[str, Path],
    transform_dict: Dict,
    spatial_size: Optional[List[int]] = None
) -> Path:
    """
    Setup disk cache with validation and cleanup if needed.

    Args:
        cache_dir: Path to cache directory
        transform_dict: Transform dictionary
        spatial_size: Optional spatial size override

    Returns:
        Path to validated cache directory
    """
    cache_dir = Path(cache_dir)
    current_metadata = get_cache_metadata(transform_dict, spatial_size)

    if cache_dir.exists():
        if is_cache_valid(cache_dir, current_metadata):
            logging.info(f"Reusing valid cache: {cache_dir}")
            return cache_dir
        else:
            logging.info(f"Cache configuration changed, cleaning up old cache...")
            cleanup_cache_dir(cache_dir, force=False)

    # Create fresh cache
    cache_dir = create_cache_dir(cache_dir)

    # Save metadata
    metadata_path = cache_dir / CACHE_METADATA_FILE
    with open(metadata_path, 'w') as f:
        json.dump(current_metadata, f, indent=2)

    logging.info(f"Created new cache: {cache_dir}")
    return cache_dir


def create_dataset(
    data_list: List[Dict],
    transform,
    cache_mode: str = 'none',
    cache_dir: Optional[Union[str, Path]] = None,
    cache_rate: float = 1.0,
    cache_num: Optional[int] = None,
    rank: int = 0
) -> Union[Dataset, CacheDataset, PersistentDataset]:
    """
    Factory function to create appropriate dataset type based on cache mode.

    Args:
        data_list: List of data dictionaries
        transform: MONAI transform to apply
        cache_mode: 'none', 'ram', or 'disk'
        cache_dir: Directory for disk caching (required if cache_mode='disk')
        cache_rate: Fraction of data to cache (0.0-1.0, used with 'ram' mode)
        cache_num: Absolute number of samples to cache (overrides cache_rate, used with 'ram' mode)
        rank: DDP rank for logging

    Returns:
        Dataset, CacheDataset, or PersistentDataset instance

    Raises:
        ValueError: If invalid parameters provided
    """
    # Validate cache_mode
    valid_modes = ['none', 'ram', 'disk']
    if cache_mode not in valid_modes:
        raise ValueError(f"cache_mode must be one of {valid_modes}, got '{cache_mode}'")

    # Validate cache_rate for RAM mode
    if cache_mode == 'ram':
        if cache_rate < 0.0 or cache_rate > 1.0:
            raise ValueError(f"cache_rate must be in [0.0, 1.0], got {cache_rate}")
        if cache_num is not None and cache_num <= 0:
            raise ValueError(f"cache_num must be positive, got {cache_num}")

    # Mode 1: No caching
    if cache_mode == 'none':
        utils.print_rank_0("Creating Dataset (no caching)", rank)
        return Dataset(data_list, transform=transform)

    # Mode 2: RAM caching (CacheDataset)
    elif cache_mode == 'ram':
        # Calculate cache_num from cache_rate if not explicitly provided
        if cache_num is None:
            cache_num = int(len(data_list) * cache_rate)

        # Clamp to dataset size
        cache_num = min(cache_num, len(data_list))

        utils.print_rank_0(
            f"Creating CacheDataset (RAM caching {cache_num}/{len(data_list)} samples, "
            f"{cache_num/len(data_list)*100:.1f}%)",
            rank
        )
        return CacheDataset(
            data_list,
            transform=transform,
            cache_num=cache_num,
            cache_rate=1.0  # Always 1.0 since we pre-calculated cache_num
        )

    # Mode 3: Disk caching (PersistentDataset)
    elif cache_mode == 'disk':
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when cache_mode='disk'")

        utils.print_rank_0(
            f"Creating PersistentDataset (disk caching to {cache_dir}, "
            f"{len(data_list)} samples)",
            rank
        )
        return PersistentDataset(
            data_list,
            transform=transform,
            cache_dir=str(cache_dir)
        )


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

    # CacheDataset (RAM) with num_workers > 0 causes each worker to hold its own
    # copy of the cache, multiplying RAM usage by num_workers. Force single-threaded.
    # PersistentDataset (disk) uses per-item hash files and is safe with multiple
    # workers — workers read/write separate files concurrently without conflict.
    if isinstance(train_ds, CacheDataset) and not isinstance(train_ds, PersistentDataset):
        if dataloader_workers > 0:
            logging.warning(
                f'CacheDataset (RAM) detected with num_workers={dataloader_workers}. '
                f'Forcing num_workers=0 to prevent per-worker cache duplication.'
            )
        dataloader_workers = 0

    # persistent_workers requires num_workers > 0
    use_persistent = persistent_workers and dataloader_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=dataloader_workers,
        pin_memory=False,
        # pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
        sampler=sampler,
        # Reduce prefetch to lower memory/fd pressure
        prefetch_factor=2 if dataloader_workers > 0 else None
    )
    return train_loader


def create_validation_data_loader(val_ds: monai.data.Dataset,
                                  batch_size: int = 1,
                                  dataloader_workers: int = 4,
                                  sampler=None):
    print('Creating validation data loader')

    # CacheDataset (RAM) with num_workers > 0 causes each worker to hold its own
    # copy of the cache, multiplying RAM usage by num_workers. Force single-threaded.
    # PersistentDataset (disk) uses per-item hash files and is safe with multiple workers.
    if isinstance(val_ds, CacheDataset) and not isinstance(val_ds, PersistentDataset):
        if dataloader_workers > 0:
            logging.warning(
                f'CacheDataset (RAM) detected with num_workers={dataloader_workers}. '
                f'Forcing num_workers=0 to prevent per-worker cache duplication.'
            )
        dataloader_workers = 0

    # persistent_workers requires num_workers > 0
    use_persistent = dataloader_workers > 0

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=dataloader_workers,
                            pin_memory=False,
                            # pin_memory=torch.cuda.is_available(),
                            persistent_workers=use_persistent,
                            sampler=sampler)
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
                            dataloader_workers, val_batch_size=1,
                            cache_training_mode='none', cache_validation_mode='none',
                            cache_dir=None, cache_rate=1.0, cache_num=None,
                            world_size=1, rank=0, shuffle_training=True, training_persistent_workers=True):
    train_data_list = []
    val_data_list = []
    for ind, chunk in enumerate(split_lists):
        if ind == fold:
            val_data_list = chunk
        else:
            train_data_list = np.concatenate([train_data_list, chunk])
    utils.print_rank_0(f'Creating training monai dataset for fold {fold}', rank)
    train_ds = create_dataset(
        train_data_list, train_img_transforms,
        cache_mode=cache_training_mode,
        cache_dir=cache_dir,
        cache_rate=cache_rate,
        cache_num=cache_num,
        rank=rank
    )
    # data_loader_checker_first(train_ds, 'training')
    # define dataset, data loader
    utils.print_rank_0(f'Creating validation monai dataset', rank)
    val_ds = create_dataset(
        val_data_list, val_img_transforms,
        cache_mode=cache_validation_mode,
        cache_dir=cache_dir,
        cache_rate=cache_rate,
        cache_num=cache_num,
        rank=rank
    )
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
    # data_loader_checker_first(train_ds, 'validation')
    train_loader = create_training_data_loader(train_ds, batch_size, dataloader_workers,
                                               sampler=train_sampler, shuffle=shuffle_training,
                                               persistent_workers=training_persistent_workers)
    val_loader = create_validation_data_loader(val_ds, val_batch_size, dataloader_workers, sampler=val_sampler)

    return train_loader, val_loader

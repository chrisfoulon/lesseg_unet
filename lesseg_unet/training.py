import json
import os
import random
import shutil
import logging
import gc
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union, Iterable
import time
import signal
import sys
import resource

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat
import nibabel as nib
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import BCEWithLogitsLoss
from lesseg_unet import net, utils, data_loading, transformations, loss_and_metric, data_utils


# Global variable to store cleanup context
_cleanup_context = {}


def cleanup_on_exit(signum=None, frame=None):
    """
    Cleanup function called on Ctrl+C or normal exit.
    Releases memory and cleans up distributed processes.
    """
    print(f"\n[Rank {dist.get_rank() if dist.is_initialized() else 0}] Cleaning up resources...")

    # Delete model, optimizer, and scaler to free memory
    if 'model' in _cleanup_context:
        del _cleanup_context['model']
    if 'optimizer' in _cleanup_context:
        del _cleanup_context['optimizer']
    if 'scaler' in _cleanup_context and _cleanup_context['scaler'] is not None:
        del _cleanup_context['scaler']
    if 'train_loader' in _cleanup_context:
        del _cleanup_context['train_loader']
    if 'val_loader' in _cleanup_context:
        del _cleanup_context['val_loader']

    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] GPU memory cleared")

    # Force garbage collection
    gc.collect()
    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] CPU memory cleared")

    # Destroy DDP process group if it was initialized
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print(f"[Rank {dist.get_rank()}] DDP process group destroyed")
        except Exception as e:
            print(f"Warning: Could not destroy process group: {e}")

    if signum is not None:
        print(f"\n[Rank {dist.get_rank() if dist.is_initialized() else 0}] Exiting due to interrupt signal")
        sys.exit(0)


# Register signal handler for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, cleanup_on_exit)


def count_unique_parameters(parameters):
    """
    Credit: James Ruffle
    :param parameters:
    :return:
    """
    # Only counts unique params
    count = 0
    list_of_names = []
    for p in parameters:
        name = p[0]
        param = p[1]
        if name not in list_of_names:
            list_of_names.append(name)
            count += np.prod(param.size())
    return count


def testing(train_loader, output_dir):
    it = iter(train_loader)
    for i in tqdm(range(5)):
        # input_data = val_ds[i]['image']
        # print(val_ds[i]['image_meta_dict']['filename_or_obj'])
        # raw_data = nib.load(val_ds[i]['image_meta_dict']['filename_or_obj']).get_fdata()
        data = next(it)
        data_nii = nib.load(data['image_meta_dict']['filename_or_obj'][0])
        out_affine = data_nii.affine
        inputs, labels = data['image'], data['label']
        # i_data = inputs[0, 0, :, :, :].cpu().detach().numpy()
        if inputs.shape[1] > 1:
            for ind, channel in enumerate(inputs[0, :, :, :, :]):
                i_data = channel.cpu().detach().numpy()
                nib.save(nib.Nifti1Image(i_data, out_affine),
                         filename=f'{Path(output_dir)}/img_test_{i}_{ind}.nii')
        else:
            # print(np.all(i_data == raw_data))
            # i_data = inputs[0, 0, :, :, :].cpu().detach().numpy()
            # l_data = labels[0, 0, :, :, :].cpu().detach().numpy()
            # utils.save_img_lbl_seg_to_png(
            #     i_data, output_dir, 'validation_img_{}'.format(i), l_data, None)
            out_paths_list = utils.save_img_lbl_seg_to_nifti(
                inputs, labels, None, output_dir, out_affine, i)

        print(f'fsleyes {data["image_meta_dict"]["filename_or_obj"][0]} {data["label_meta_dict"]["filename_or_obj"][0]}'
              f' {out_paths_list[0]} {out_paths_list[1]} '
              f'{"/home/tolhsadum/neuro_apps/data/input_avg152T2_template.nii"}')
        print('###########VOLUMES#######')
        # orig_label = nib.load(data["label_meta_dict"]["filename_or_obj"][0]).get_fdata()
        # label = nib.load(out_paths_list[1]).get_fdata()
        # print(f'original label volume: {np.count_nonzero(orig_label)}')
        # print(f'Smoothed label volume 0.5: {len(np.where(label > 0.5)[0])}')
        # print(f'Smoothed label volume 0.25: {len(np.where(label > 0.25)[0])}')
        # print('###########ENDVOLUMES#######')
    # if np.equal(i_data, l_data).all():
    #     print('ok')
    # else:
    #     print('not ok')
    exit()


def training(img_path_list: Sequence,
             lbl_path_list: Sequence,
             output_dir: Union[str, bytes, os.PathLike],
             ctr_path_list=None,
             img_pref: str = None,
             image_cut_suffix: str = None,
             transform_dict=None,
             pretrained_point=None,
             model_type='UNETR',
             device: str = None,
             batch_size: int = 1,
             val_batch_size: int = 1,
             epoch_num: int = 50,
             gradient_accumulation_steps: int = 1,
             dataloader_workers: int = 4,
             train_val_percentage=80,
             lesion_set_clamp=None,
             # controls_clamping=None,
             resize=None,
             # label_smoothing=False,
             stop_best_epoch=-1,
             training_loss_fct='dice',
             ctr_loss_fct='binary_empty_label',
             val_loss_fct='dice',
             weight_factor=1,
             folds_number=1,
             dropout=0,
             cache_dir=None,
             cache_training_mode='none',
             cache_validation_mode='none',
             cache_rate=1.0,
             save_every_decent_best_epoch=True,
             rank=0,
             world_size=1,
             cache_num=None,
             enable_amp=True,
             learning_rate=1e-4,
             weight_decay=1e-5,
             delayed_control_training=False,
             use_ema=False,
             track_ema=False,
             no_backward_on_controls=False,
             limit_of_open_files=None,
             debug=False,
             feature_size=None,
             network_depth=None,
             **kwargs
             ):
    """
    Main training function for the UNETR and UNET models
    Parameters
    ----------
    img_path_list
    lbl_path_list
    output_dir
    ctr_path_list
    img_pref
    image_cut_suffix
    transform_dict
    pretrained_point
    model_type
    device
    batch_size
    val_batch_size
    epoch_num
    gradient_accumulation_steps
    dataloader_workers
    train_val_percentage
    lesion_set_clamp
    resize
    stop_best_epoch
    training_loss_fct
    ctr_loss_fct
    val_loss_fct
    weight_factor
    folds_number
    dropout
    cache_dir
    save_every_decent_best_epoch
    rank
    world_size
    cache_num
    enable_amp
    delayed_control_training
    use_ema
    track_ema
    no_backward_on_controls
    limit_of_open_files
    debug
    kwargs

    Returns
    -------
    Notes:
    https://mmcv.readthedocs.io/en/v1.5.2_a/_modules/torch/utils/data/dataloader.html explains
    "RuntimeError: received 0 items of ancdata" error
    """
    if limit_of_open_files is not None:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print("LIMIT before: {}".format(rlimit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit_of_open_files, rlimit[1]))
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print("LIMIT after: {}".format(rlimit))
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # print("LIMIT before: {}".format(rlimit))
    # # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    # # resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
    # # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # print("LIMIT after: {}".format(rlimit))
    shuffle_training = True
    display_training = False
    if 'display_training' in kwargs:
        v = kwargs['display_training']
        if v == 'True' or v == 1:
            print(f'Displaying training images')
            display_training = True
            shuffle_training = False
    one_loop = False
    if 'one_loop' in kwargs:
        v = kwargs['one_loop']
        if v == 'True' or v == 1:
            print(f'Stopping after one training loop')
            one_loop = True
    """MODEL PARAMETERS"""
    # Apparently it can potentially improve the performance when the model does not change its size. (Source tuto UNETR)
    torch.backends.cudnn.benchmark = True
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    cpu_device = device.type == 'cpu'

    # setup(rank, world_size, cpu=cpu_device)
    if not cpu_device:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)  # Only set CUDA device if using GPU

    logging.info(f'Torch device used for this training: {str(device)}')

    # Disable AMP on CPU (AMP only works with CUDA)
    if cpu_device and enable_amp:
        utils.print_rank_0('Warning: AMP (Automatic Mixed Precision) is not supported on CPU. Disabling AMP.',
                          dist.get_rank())
        enable_amp = False

    """
    LOSS FUNCTIONS
    """
    # Training
    if training_loss_fct.lower() in ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']:
        loss_function = DiceCELoss(sigmoid=True)
    elif training_loss_fct.lower() in ['focal', 'focalloss', 'focal_loss']:
        loss_function = FocalLoss(gamma=2.0)
    elif training_loss_fct.lower() in ['generalized_dice', 'gen_dice', 'generalised_dice', 'gen_dice_loss',
                                       'generalised_dice_loss']:
        loss_function = GeneralizedDiceLoss(sigmoid=True)
    elif training_loss_fct.lower() in ['generalized_dice_focal', 'gen_dice_focal', 'generalised_dice_focal',
                                       'gen_dice_focal_loss', 'generalised_dice_focal_loss']:
        loss_function = GeneralizedDiceFocalLoss(sigmoid=True, gamma=2.0)
    elif training_loss_fct.lower() in ['dicefocal', 'dicefocalloss', 'dice_focal_loss']:
        loss_function = DiceFocalLoss(sigmoid=True, gamma=2.0)
    else:
        loss_function = DiceLoss(sigmoid=True)
    utils.logging_rank_0(f'Training loss fct: {loss_function}', dist.get_rank())
    # Controls training losses
    if ctr_loss_fct == 'mean_sigmoid':
        ctr_loss_function = loss_and_metric.ThresholdedAverageLoss(threshold=0.0, reduction='mean')
    elif ctr_loss_fct == 'binary_empty_label':
        ctr_loss_function = loss_and_metric.BinaryEmptyLabelLoss()
    elif ctr_loss_fct == 'thresholded_average':
        ctr_loss_function = loss_and_metric.ThresholdedAverageLoss(threshold=0.5, reduction='mean')
    else:
        raise ValueError(f'Unknown controls loss function: {ctr_loss_fct}')

    # Validation
    if any([s in val_loss_fct.lower() for s in
            ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']]):
        val_loss_function = DiceCELoss(sigmoid=True)
    else:
        val_loss_function = DiceLoss(sigmoid=True)
    utils.logging_rank_0(f'Validation loss fct: {val_loss_function}', dist.get_rank())

    """
    DEBUG LOSSES
    """
    bce = BCEWithLogitsLoss(reduction='mean')
    # if resize is not None:
    #     zero_label = torch.zeros((batch_size, 1, resize[0], resize[1], resize[2])).to(device)
    # else:
    #     zero_label = torch.zeros((batch_size, 1, 96, 128, 96)).to(device)
    """
    END DEBUG LOSSES
    """

    """
    METRICS
    """
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    keep_dice_and_dist = True
    if 'keep_dice_and_dist' in kwargs:
        v = kwargs['keep_dice_and_dist']
        if v == 'False' or v == 0:
            keep_dice_and_dist = False
        if v == 'True' or v == 1:
            keep_dice_and_dist = True

    """
    DATA LOADING
    """
    if img_pref is not None and img_pref != '':
        utils.logging_rank_0(f'Abnormal images prefix: {img_pref}', dist.get_rank())
    """
    IMPORTANT!!!! THIS MUST BE DONE ONLY ONCE WHEN THE IMAGES ARE SHUFFLED!!!!
    With multi-gpu the shuffle is called multiple times and thus, the datasets are different and the fold
    splits are not respected!!!
    """
    if dist.get_rank() == 0:
        if lbl_path_list is None:
            # If no label list is provided, then it means img_path_list is a split list (split per fold)
            split_lists_to_share = [img_path_list]
        else:
            # Get the images from the image and label list and tries to match them
            img_dict, controls = data_loading.match_img_seg_by_names(img_path_list, lbl_path_list, img_pref,
                                                                     image_cut_suffix=image_cut_suffix)
            utils.logging_rank_0(f'##### Number of abnormal images for training: {len(img_dict)}', dist.get_rank())
            split_lists_to_share = [utils.split_lists_in_folds(
                img_dict, folds_number, train_val_percentage, shuffle=True)]
    else:
        split_lists_to_share = [None]
    torch.distributed.broadcast_object_list(split_lists_to_share, src=0)
    split_lists = split_lists_to_share[0]
    # Save the split_lists to easily get the content of the folds and all
    with open(Path(output_dir, 'split_lists.json'), 'w+') as f:
        json.dump(split_lists, f, indent=4)
    # TODO The controls could just be added to the split_lists and the control key can be added to the transforms
    """
    We want the same thing for the controls here. The input is control_list a list of dict of list like [{'image':[]}]
    If we need to create new variables for the controls, we add the prefix ctr_ to the name of the variable
    """
    ctr_split_lists = None
    if ctr_path_list is not None:
        if dist.get_rank() == 0:
            try:
                if len(ctr_path_list) == folds_number and isinstance(ctr_path_list[0], list):
                    utils.logging_rank_0(
                        f'##### Loading pre-split control lists for training',
                        dist.get_rank())
                    utils.logging_rank_0(
                        f'##### Number of control images for training: {np.sum([len(l) for l in ctr_path_list])}',
                        dist.get_rank())
                    split_lists_to_share = [ctr_path_list]
                else:
                    utils.logging_rank_0(f'##### Number of control images for training: {len(ctr_path_list)}',
                                         dist.get_rank())
                    split_lists_to_share = [utils.split_lists_in_folds(
                        ctr_path_list, folds_number, train_val_percentage, shuffle=True, image_key='control')]
            except IndexError as e:
                print('ERROR: The number of controls equals the number of folds! It is weird!')
                raise e
        else:
            split_lists_to_share = [None]
        torch.distributed.broadcast_object_list(split_lists_to_share, src=0)
        ctr_split_lists = split_lists_to_share[0]
        # Save the split_lists to easily get the content of the folds and all
        with open(Path(output_dir, 'control_split_lists.json'), 'w+') as f:
            json.dump(ctr_split_lists, f, indent=4)

    """
    TRANSFORMATIONS AND AUGMENTATIONS
    """
    # Attempt to send the transformations / augmentations on the GPU when possible (disabled by default)
    transformations_device = None
    if 'tr_device' in kwargs:
        v = kwargs['tr_device']
        if v == 'False' or v == 0:
            transformations_device = None
        if v == 'True' or v == 1:
            print(f'ToTensord transformation will be called on {device}')
            transformations_device = device
    utils.print_rank_0('Initialisation of the training transformations', dist.get_rank())

    """
        If resize is used, replace the crop pad transform with the resize transform
        """
    if resize is not None:
        resize_function = {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': resize}
        }
        transform_dict = transformations.replace_tr(
            transform_dict, 'ResizeWithPadOrCropd', resize_function)

    # Get any image key to determine original shape (all modalities have same spatial dims)
    first_subject = split_lists[0][0]
    image_keys = data_utils.get_category_keys(first_subject, 'image')
    first_image_key = image_keys[0] if image_keys else 'image'  # Fallback for backward compatibility
    original_image_shape = utils.get_img_size(first_subject[first_image_key])
    utils.print_rank_0(f'Original image shape: {original_image_shape}', dist.get_rank())
    # We need the training image size for the unetr as we need to know the size of the model to create it
    if list(transform_dict.keys())[-1] == 'patches':
        # TODO this might change depending on the cropping transformation
        # Try roi_size first (for patches), then spatial_size (for resize transforms)
        model_img_size = transformations.find_param_from_hyper_dict(
            transform_dict, 'roi_size', find_last=True)
        if model_img_size is None:
            model_img_size = transformations.find_param_from_hyper_dict(
                transform_dict, 'spatial_size', find_last=True)
        if model_img_size is not None:
            model_img_size = model_img_size[-3:]
        else:
            raise ValueError(
                "Could not find 'roi_size' or 'spatial_size' in transform_dict. "
                "Please ensure your transforms include a cropping or resizing operation."
            )
        transformations.setup_coord_conv(transform_dict, original_image_shape)
    else:
        model_img_size = transformations.find_param_from_hyper_dict(
            transform_dict, 'spatial_size', find_last=True)
        transformations.setup_coord_conv(transform_dict, model_img_size)

    # Adapt transform dict for multi-modal data if needed
    transform_dict = data_utils.adapt_transforms_for_multimodal(transform_dict, split_lists)

    # If we use controls, we need to add 'control' to the transform_dict every time the 'image' key is used
    if ctr_split_lists is not None:
        transform_dict = transformations.add_control_key(transform_dict)

    # Extract all the transformations from transform_dict
    train_img_transforms = transformations.train_transformd(transform_dict, lesion_set_clamp,
                                                            device=transformations_device,
                                                            writing_rank=dist.get_rank())
    # Extract only the 'first' and 'last' transformations from transform_dict ignoring the augmentations
    val_img_transforms = transformations.val_transformd(transform_dict, lesion_set_clamp,
                                                        device=transformations_device)
    """
    POST TRANSFORMATIONS
    """
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    ctr_post_trans = Compose([Activations(sigmoid=True)])

    """
    FOLDS LOOP VARIABLES
    """
    non_blocking = True
    if 'non_blocking' in kwargs:
        v = kwargs['non_blocking']
        if v == 'False' or v == 0:
            non_blocking = False
        if v == 'True' or v == 1:
            non_blocking = True
    val_interval = 1
    # val_meh_thr = 0.7
    # val_trash_thr = 0.3
    # If pretrained_point is not None, we will load the model from the checkpoint here because we need to know the
    # at what fold we are
    checkpoint_to_share = None
    starting_fold = 0
    if pretrained_point is not None:
        if dist.get_rank() == 0:
            checkpoint_to_share = [torch.load(pretrained_point, map_location="cpu")]
        else:
            checkpoint_to_share = [None]
        torch.distributed.broadcast_object_list(checkpoint_to_share, src=0)
        starting_fold = checkpoint_to_share[0]['fold']
    if stop_best_epoch != -1:
        utils.logging_rank_0(f'Will stop after {stop_best_epoch} epochs without improvement', dist.get_rank())
    for fold in range(starting_fold, folds_number):
        # if fold < starting_fold:
        #     utils.logging_rank_0(f'Skipping fold {fold}', dist.get_rank())
        #     continue
        """
        SET MODEL PARAM AND CREATE / LOAD MODEL OBJECT
        """
        utils.logging_rank_0(f'Creating monai {model_type}', dist.get_rank())
        scaler = torch.amp.GradScaler('cuda') if enable_amp else None
        starting_epoch = 0
        if checkpoint_to_share is not None:
            starting_epoch = checkpoint_to_share[0]['epoch']
            checkpoint = checkpoint_to_share[0]
            hyper_params = checkpoint['hyper_params']
            model = utils.load_model_from_checkpoint(checkpoint, device, hyper_params, model_name=model_type)
            if torch.cuda.is_available():
                model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer.load_state_dict(checkpoint['optim_dict'])
            if scaler is not None and checkpoint.get('scaler_dict') is not None:
                scaler.load_state_dict(checkpoint['scaler_dict'])
            utils.logging_rank_0(f'{model_type} created and succesfully loaded from {pretrained_point} with '
                                 f'hyper parameters: {hyper_params}',
                                 dist.get_rank())

            # Restore early stopping state from tensorboard events
            # This preserves the early stopping counter across resume, preventing wasted computation
            best_metric_epoch_from_events = None
            best_metric_dist_epoch_from_events = None

            if dist.get_rank() == 0:
                # Determine fold directory
                fold_dir_for_events = output_dir if folds_number == 1 else Path(output_dir, f'fold_{fold}')

                # Try to restore best dice epoch from events
                best_metric_epoch_from_events = utils.get_best_epoch_from_events(
                    fold_dir_for_events,
                    metric='val_mean_dice',
                    minimize=False
                )

                # If using distance metric, restore that too
                if 'dist' in val_loss_fct.lower():
                    best_metric_dist_epoch_from_events = utils.get_best_epoch_from_events(
                        fold_dir_for_events,
                        metric='val_distance',
                        minimize=True
                    )

            # Broadcast early stopping state to all ranks (for DDP)
            events_state_to_share = [best_metric_epoch_from_events, best_metric_dist_epoch_from_events]
            torch.distributed.broadcast_object_list(events_state_to_share, src=0)
            best_metric_epoch_from_events = events_state_to_share[0]
            best_metric_dist_epoch_from_events = events_state_to_share[1]
        else:
            # Not resuming, initialize to None (will use default -1 later)
            best_metric_epoch_from_events = None
            best_metric_dist_epoch_from_events = None

        # Continue with model creation if not resuming
        if checkpoint_to_share is None:
            if model_type.lower() == 'unetr' or model_type.lower() == 'swinunetr':
                if model_type.lower() == 'unetr':
                    hyper_params = net.default_unetr_hyper_params
                    hyper_params['img_size'] = model_img_size  # UNETR still requires img_size
                else:
                    hyper_params = net.default_swinunetr_hyper_params
                    # SwinUNETR: img_size removed in MONAI 1.5+ - now accepts dynamic sizes
                if feature_size is not None:
                    hyper_params['feature_size'] = int(feature_size)
                elif 'feature_size' in kwargs:
                    hyper_params['feature_size'] = int(kwargs['feature_size'])

                # Apply gradient checkpointing if specified (SwinUNETR only)
                if model_type.lower() == 'swinunetr' and 'use_checkpoint' in kwargs:
                    hyper_params['use_checkpoint'] = kwargs['use_checkpoint']
                    utils.logging_rank_0(
                        f'SwinUNETR gradient checkpointing: {kwargs["use_checkpoint"]}',
                        dist.get_rank()
                    )

                # Apply network_depth if specified (for UNETR/SwinUNETR)
                if network_depth is not None:
                    # Convert network_depth (4 or 5) to depths list
                    # Depth 4: [2, 2, 2, 2], Depth 5: [2, 2, 2, 2, 2]
                    depths_list = [2] * network_depth
                    hyper_params['depths'] = depths_list
                elif 'network_depth' in kwargs:
                    depths_list = [2] * int(kwargs['network_depth'])
                    hyper_params['depths'] = depths_list
            else:
                hyper_params = net.default_unet_hyper_params

            # Auto-detect model configuration from split_lists (multi-modal support)
            model_config = data_utils.extract_model_config(split_lists)
            hyper_params['in_channels'] = model_config['in_channels']
            hyper_params['out_channels'] = model_config['out_channels']
            utils.logging_rank_0(f'Auto-detected model config: {model_config}', dist.get_rank())

            # checking is CoordConv is used and change the input channel dimension
            if transform_dict is not None:
                for li in transform_dict:
                    for d in transform_dict[li]:
                        for t in d:
                            if t == 'CoordConvd' or t == 'CoordConvAltd':
                                hyper_params['in_channels'] = 4
            # if dropout is not None and dropout == 0:
            #     hyper_params['dropout'] = dropout
            #     logging.info(f'Dropout rate used: {dropout}')
            model, _ = net.create_model(device, hyper_params, model_class_name=model_type)
            utils.logging_rank_0(f'{model_type} created and succesfully with '
                                 f'hyper parameters: {hyper_params}',
                                 dist.get_rank())
            # print(f'[Rank {dist.get_rank()}]model created')
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # use amp to accelerate training
        if dist.get_rank() == 0:
            total_param_count = count_unique_parameters(model.named_parameters())
            utils.logging_rank_0(f'Total number of parameters in the model: {str(total_param_count)}',
                                 dist.get_rank())
        params = list(model.parameters())

        # Move model to device and wrap in DDP
        if cpu_device:
            # CPU: move to CPU device, no device_ids
            model.to(device)
            if world_size > 1:
                model = DistributedDataParallel(model, find_unused_parameters=False)
                utils.print_rank_0('Model wrapped in DDP for CPU multi-process training', dist.get_rank())
            else:
                utils.print_rank_0('Model on CPU (single process, no DDP)', dist.get_rank())
        else:
            # GPU: move to specific GPU rank, use device_ids
            model.to(device)
            model = DistributedDataParallel(model, device_ids=[rank], output_device=dist.get_rank(),
                                            find_unused_parameters=False)
            utils.print_rank_0('Model sent to GPU ranks with DDP', dist.get_rank())
        if folds_number == 1:
            output_fold_dir = output_dir
        else:
            output_fold_dir = Path(output_dir, f'fold_{fold}')
        # Tensorboard writer
        if dist.get_rank() == 0:
            writer = SummaryWriter(log_dir=str(output_fold_dir))
        else:
            writer = None
        utils.print_rank_0('Tensorboard SummaryWriter created', dist.get_rank())
        # Creates both the training and validation loaders based on the fold number
        # (e.g. fold 0 means the first sublist of split_lists will be the validation set for this fold)
        if ctr_split_lists is None or delayed_control_training:
            train_loader, val_loader = data_loading.create_fold_dataloaders(
                split_lists, fold, train_img_transforms,
                val_img_transforms, batch_size, dataloader_workers, val_batch_size,
                cache_training_mode=cache_training_mode,
                cache_validation_mode=cache_validation_mode,
                cache_dir=cache_dir,
                cache_rate=cache_rate,
                cache_num=cache_num,
                world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training
            )

        # Register objects for cleanup on Ctrl+C
        _cleanup_context['model'] = model
        _cleanup_context['optimizer'] = optimizer
        _cleanup_context['scaler'] = scaler
        if 'train_loader' in locals():
            _cleanup_context['train_loader'] = train_loader
        if 'val_loader' in locals():
            _cleanup_context['val_loader'] = val_loader

        """Initiailise EMA"""
        monitor_emas = use_ema
        normalise_by_ema = track_ema
        ema_decay_rate = 0.999
        ema_magnitude_abnormals = 1
        ema_magnitude_controls = 1

        """EPOCHS LOOP VARIABLES"""
        time_list = []
        epoch_time_list = []
        str_best_epoch = ''
        str_best_dist_epoch = ''
        epoch_suffix = ''
        best_dice = 0
        best_dice_with_dist = 0
        # only for the first epoch
        best_dist = 1000

        # Initialize early stopping counters
        # Use values from tensorboard events if resuming, otherwise start fresh
        best_metric_epoch = best_metric_epoch_from_events if best_metric_epoch_from_events is not None else -1
        best_metric_dist_epoch = best_metric_dist_epoch_from_events if best_metric_dist_epoch_from_events is not None else -1

        img_dir = Path(output_dir, 'image_dir')
        if display_training:
            if dist.get_rank() == 0:
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
                os.makedirs(img_dir, exist_ok=True)
            dist.barrier()
        stop_epoch = False
        use_controls = not delayed_control_training
        best_dice_list = []
        number_of_best_dice_intervals_to_assume_convergence = 2
        best_dice_interval_difference = 0.005
        for epoch in range(starting_epoch, epoch_num):
            utils.print_rank_0('-' * 10, dist.get_rank())
            utils.print_rank_0(f'epoch {epoch + 1}/{epoch_num}', dist.get_rank())
            # If ctr_split_lists is not None we need to create a new training loader with the controls
            if ctr_split_lists is not None and use_controls:
                # We need to add the same number of controls as the abnormal images in each fold after shuffling them
                # The new split_list has to be shared between all the ranks
                if dist.get_rank() == 0:
                    split_lists_with_ctr = []
                    for i in range(len(split_lists)):
                        ctr_fold_list = deepcopy(ctr_split_lists[i])
                        # Might be unnecessary, but I don't think it makes much a difference
                        random.shuffle(ctr_fold_list)
                        split_lists_with_ctr.append(deepcopy(split_lists[i]))
                        for img_dict in split_lists_with_ctr[i]:
                            img_dict.update(ctr_fold_list.pop())
                    split_lists_with_ctr_to_share = [split_lists_with_ctr]
                else:
                    split_lists_with_ctr_to_share = [None]
                torch.distributed.broadcast_object_list(split_lists_with_ctr_to_share, src=0)
                split_lists_with_ctr = split_lists_with_ctr_to_share[0]
                train_loader, val_loader = data_loading.create_fold_dataloaders(
                    split_lists_with_ctr, fold, train_img_transforms,
                    val_img_transforms, batch_size, dataloader_workers, val_batch_size,
                    cache_training_mode=cache_training_mode,
                    cache_validation_mode=cache_validation_mode,
                    cache_dir=cache_dir,
                    cache_rate=cache_rate,
                    cache_num=cache_num,
                    world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training,
                    training_persistent_workers=False
                )
                # Update cleanup context with new loaders
                _cleanup_context['train_loader'] = train_loader
                _cleanup_context['val_loader'] = val_loader

                # train_loader = data_loading.create_ctr_dataloader(
                #     split_lists, ctr_split_lists, fold, train_img_transforms,
                #     val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir,
                #     world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training,
                #     cache_num=cache_num
                # )
            # This is required with multi-gpu
            batches_per_epoch = len(train_loader)
            train_loader.sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0
            ctr_epoch_loss = 0
            controls_loss = None  # Initialize at epoch level to avoid UnboundLocalError
            step = 0
            start_time = time.time()
            loading_time = True

            # Initialize gradient clipping flag for UNet models
            _gradients_unscaled = False

            """
            TRAINING INITIALISATION
            """
            if dist.get_rank() == 0:
                train_iter = tqdm(train_loader, desc=f'Training[{epoch + 1}] loss/mean_loss:[N/A]')
            else:
                train_iter = train_loader
            no_progressbar_training = False
            # if 'no_progressbar_training' in kwargs:
            #     v = kwargs['no_progressbar_training']
            #     if v == 'True' or v == 0:
            #         train_iter = train_loader
            #         no_progressbar_training = True
            """
            INNER TRAINING LOOP
            """
            if display_training:
                print(train_img_transforms.transforms)
            for batch_data in train_iter:
                # TODO print image names both in training and validation loop
                # TODO Try to resume the model but mess up with the label (zero_like / one_like)
                # TODO turn off the augmentations
                if loading_time:
                    end_time = time.time()
                    load_time = end_time - start_time
                    utils.logging_rank_0(f'Loading loop Time: {load_time}', dist.get_rank())
                    time_list.append(load_time)
                    utils.logging_rank_0(
                        f'First load time: {time_list[0]} and average loading time {np.mean(time_list)}',
                        dist.get_rank())
                    loading_time = False
                step += 1
                inputs, labels = batch_data['image'].to(device, non_blocking=non_blocking), batch_data['label'].to(
                    device, non_blocking=non_blocking)
                ctr_inputs = None
                if ctr_split_lists is not None and use_controls:
                    ctr_inputs = batch_data['control'].to(device, non_blocking=non_blocking)
                """
                DEBUG AND IMAGE DISPLAY BLOCK
                """
                if display_training:
                    with torch.no_grad():
                        # print(list(batch_data.keys()))
                        # print(batch_data['image_meta_dict']['filename_or_obj'])
                        # exit()

                        img_name = Path(batch_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
                        lbl_name = Path(batch_data['label_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
                        # print(batch_data['image_meta_dict']['affine'][0].cpu().detach().numpy())
                        nii = nib.Nifti1Image(inputs[0, 0, ...].cpu().detach().numpy(),
                                              batch_data['image_meta_dict']['affine'][0].cpu().detach().numpy())
                        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
                        plot_anat(batch_data['image_meta_dict']['filename_or_obj'][0],
                                  # output_file=Path(img_dir, f'{img_name}.png'),
                                  display_mode='x', title='Original', draw_cross=False,
                                  cut_coords=(50, 54, 45), axes=axes[0]
                                  )
                        plot_anat(nii,
                                  # output_file=Path(img_dir, f'{img_name}.png'),
                                  display_mode='x', title='Augmented', draw_cross=False,
                                  cut_coords=(50, 54, 45), axes=axes[1]
                                  )
                        fig.savefig(Path(img_dir, f'{img_name}.png'))
                        data = inputs[0, 0, ...].cpu().detach().numpy()
                        print(f'Image name: {img_name}')
                        print(np.mean(data))
                        print(f'Label name: {lbl_name}')
                        nib.save(nii, Path(img_dir, f'{img_name}.nii.gz'))
                        nib.save(nib.Nifti1Image(labels[0, 0, ...].cpu().detach().numpy(),
                                                 batch_data['label_meta_dict']['affine'][0].cpu().detach().numpy()),
                                 Path(img_dir, f'{lbl_name}.nii.gz'))
                        continue

                with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
                    logit_outputs = model(inputs)
                    # In case we use CoordConv, we only take the mask of the labels without the coordinates
                    masks_only_labels = labels
                    loss = loss_function(logit_outputs, masks_only_labels)

                    writer_step = len(train_loader) * epoch + step
                    if monitor_emas:
                        ema_magnitude_abnormals = ema_decay_rate * ema_magnitude_abnormals + \
                                                  (1 - ema_decay_rate) * loss.detach()
                        utils.tensorboard_write_rank_0(writer, 'ema_abnormals',
                                                       ema_magnitude_abnormals,
                                                       writer_step, dist.get_rank())

                    # Initialize controls_loss (always, regardless of debug mode)
                    controls_loss = None

                    if not debug:
                        del inputs, labels
                    if ctr_inputs is not None:
                        ctr_logit_outputs = model(ctr_inputs)
                        # ctr_logit_outputs = torch.tensor(ctr_logit_outputs, dtype=torch.float)
                        # ctr_logit_outputs = ctr_logit_outputs[:, :1, :, :, :]
                        controls_loss = ctr_loss_function(ctr_logit_outputs) * weight_factor
                        # controls_loss += l2_reg

                        if monitor_emas:
                            ema_magnitude_controls = ema_decay_rate * ema_magnitude_controls + \
                                                     (1-ema_decay_rate) * controls_loss.detach()
                            utils.tensorboard_write_rank_0(writer, 'ema_controls',
                                                           ema_magnitude_controls,
                                                           writer_step, dist.get_rank())

                        if normalise_by_ema:
                            controls_loss /= ema_magnitude_controls
                            controls_loss *= ema_magnitude_abnormals
                            controls_loss /= 10

                        if not debug:
                            del ctr_inputs

                    # Regularisation - COMMENTED OUT: Using AdamW weight_decay instead
                    # l2_reg = utils.sum_non_bias_l2_norms(params, 1e-4)
                    # loss += l2_reg

                    """
                    DEBUG
                    """
                    if debug:
                        with torch.no_grad():
                            sigmoid_logits = ctr_post_trans(logit_outputs)

                            utils.tensorboard_write_rank_0(writer, 'loss', loss.item(),
                                                           writer_step, dist.get_rank())

                            utils.tensorboard_write_rank_0(writer, 'sum_sigmoid', torch.sum(sigmoid_logits).item(),
                                                           writer_step, dist.get_rank())

                            utils.tensorboard_write_rank_0(writer, 'mean_sigmoid', torch.mean(sigmoid_logits).item(),
                                                           writer_step, dist.get_rank())

                            utils.tensorboard_write_rank_0(writer, 'bce', bce(logit_outputs, masks_only_labels).item(),
                                                           writer_step, dist.get_rank())
                            if controls_loss is not None:
                                ctr_sigmoid_logits = ctr_post_trans(ctr_logit_outputs).as_tensor()
                                # use the sigmoid values of these voxels as the penalty (track and loss)
                                # (with thresholded_average loss)
                                utils.tensorboard_write_rank_0(writer, 'ctr_loss', controls_loss.item(),
                                                               writer_step, dist.get_rank())
                                # count number of predicted vox on controls (track)
                                utils.tensorboard_write_rank_0(writer, 'ctr_num_vox',
                                                               torch.count_nonzero(
                                                                   ctr_sigmoid_logits[ctr_sigmoid_logits > 0.5]).item(),
                                                               writer_step, dist.get_rank())
                                # percentage per bin in the sigmoid of the controls (track) (0-0.1, 0.1-0.2, ..., 0.9-1)
                                if dist.get_rank() == 0:
                                    num_voxels = torch.prod(torch.tensor(ctr_sigmoid_logits.shape)).item()
                                    writer.add_scalars(
                                        'ctr_sigmoid_bins',
                                        {'0-0.1': len(ctr_sigmoid_logits[(0 <= ctr_sigmoid_logits) &
                                                                         (ctr_sigmoid_logits < 0.1)]) / num_voxels,
                                         '0.1-0.2': len(ctr_sigmoid_logits[(0.1 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.2)]) / num_voxels,
                                         '0.2-0.3': len(ctr_sigmoid_logits[(0.2 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.3)]) / num_voxels,
                                         '0.3-0.4': len(ctr_sigmoid_logits[(0.3 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.4)]) / num_voxels,
                                         '0.4-0.5': len(ctr_sigmoid_logits[(0.4 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.5)]) / num_voxels,
                                         '0.5-0.6': len(ctr_sigmoid_logits[(0.5 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.6)]) / num_voxels,
                                         '0.6-0.7': len(ctr_sigmoid_logits[(0.6 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.7)]) / num_voxels,
                                         '0.7-0.8': len(ctr_sigmoid_logits[(0.7 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.8)]) / num_voxels,
                                         '0.8-0.9': len(ctr_sigmoid_logits[(0.8 <= ctr_sigmoid_logits) &
                                                                           (ctr_sigmoid_logits < 0.9)]) / num_voxels,
                                         '0.9-1': len(ctr_sigmoid_logits[(0.9 <= ctr_sigmoid_logits) &
                                                                         (ctr_sigmoid_logits <= 1)]) / num_voxels,
                                         },
                                        writer_step)
                                utils.tensorboard_write_rank_0(writer, 'ctr_sum_sigmoid',
                                                               torch.sum(ctr_sigmoid_logits).item(),
                                                               writer_step, dist.get_rank())
                                utils.tensorboard_write_rank_0(writer, 'ctr_mean_sigmoid',
                                                               torch.mean(ctr_sigmoid_logits).item(),
                                                               writer_step, dist.get_rank())
                                tmp_zero_label = torch.zeros_like(ctr_logit_outputs)
                                utils.tensorboard_write_rank_0(writer, 'ctr_bce',
                                                               bce(ctr_logit_outputs, tmp_zero_label).item(),
                                                               writer_step, dist.get_rank())
                                masked_ctr = ctr_sigmoid_logits[ctr_sigmoid_logits > 0.5]
                                masked_zero_label = torch.zeros_like(masked_ctr)
                                if len(masked_ctr) > 0:
                                    utils.tensorboard_write_rank_0(writer, 'ctr_masked_bce',
                                                                   bce(masked_ctr, masked_zero_label).item(),
                                                                   writer_step, dist.get_rank())
                                else:
                                    utils.tensorboard_write_rank_0(writer, 'ctr_masked_bce',
                                                                   0,
                                                                   writer_step, dist.get_rank())

                                utils.tensorboard_write_rank_0(writer, 'ctr_sum_logits',
                                                               torch.sum(ctr_logit_outputs).item(),
                                                               writer_step, dist.get_rank())
                                del ctr_sigmoid_logits

                    """
                    END DEBUG
                    """

                # No need to autocast the scaler stuff
                if controls_loss is not None and not no_backward_on_controls:
                    # mean_loss = (loss + controls_loss) / 2
                    mean_loss = (loss + controls_loss)
                    if scaler is not None:
                        scaler.scale(mean_loss).backward()
                    else:
                        mean_loss.backward()

                    # Gradient clipping for UNet models to prevent gradient explosion
                    if model_type.lower() == 'unet':
                        if scaler is not None and not _gradients_unscaled:
                            scaler.unscale_(optimizer)
                            _gradients_unscaled = True
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient clipping for UNet models to prevent gradient explosion
                    if model_type.lower() == 'unet':
                        if scaler is not None and not _gradients_unscaled:
                            scaler.unscale_(optimizer)
                            _gradients_unscaled = True
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                """
                The different ranks are coming together here
                """
                # print('############ CHECKING PARAMETERS ###########################')
                # for name, param in model.named_parameters():
                #     print(name, param.device)
                #     if 'cpu' in str(param.device):
                #         print(name, param.device)
                # print('############ CHECKING PARAMETERS ###########################')
                # # print on which device all the variables are
                # print('############ CHECKING VARIABLES ###########################')
                # for name, var in locals().items():
                #     if hasattr(var, 'device'):
                #         print(name, var.device)
                # print('############ CHECKING VARIABLES ###########################')
                # print('############ CHECKING IS CUDA ###########################')
                # for name, var in locals().items():
                #     if hasattr(var, 'is_cuda'):
                #         print(name, var.is_cuda)
                # print('############ CHECKING IS CUDA ###########################')
                if step % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    # Reset gradient unscaling flag for next accumulation cycle
                    _gradients_unscaled = False
                """
                Progress and other str formatting
                """
                epoch_loss += loss
                if controls_loss is not None:
                    ctr_epoch_loss += controls_loss
                if dist.get_rank() == 0:
                    if no_progressbar_training or dist.get_rank() != 0:
                        utils.print_rank_0(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}',
                                           dist.get_rank())
                    else:
                        ctr_desc = ''
                        if controls_loss is not None:
                            ctr_loss_str_tmp = controls_loss.item() if isinstance(controls_loss, torch.Tensor) \
                                else controls_loss
                            ctr_epoch_loss_tmp = ctr_epoch_loss.item() if isinstance(ctr_epoch_loss, torch.Tensor) \
                                else ctr_epoch_loss
                            ctr_desc = f' [ctr_loss: {ctr_loss_str_tmp:.4f} / {ctr_epoch_loss_tmp / step:.4f}]' \
                                       f' [losses sum: {loss.item() + ctr_loss_str_tmp:.4f}]'
                        train_iter.set_description(
                            f'Training[{epoch + 1}] '
                            f'batch_loss/mean_loss:[{loss.item():.4f}/{epoch_loss.item() / step:.4f}]' + ctr_desc)
            if controls_loss is not None:
                # When the controls are used, we recreate the dataloader every epoch to shuffle the controls so
                # we can delete it here
                del ctr_logit_outputs, train_loader


            del loss, controls_loss
            # Emptying cache after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if one_loop:
                exit()
            """
            GLOBAL TRAINING MEASURES HANDLING
            """
            if world_size > 1:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss /= world_size
                if ctr_split_lists is not None and use_controls:
                    dist.all_reduce(ctr_epoch_loss, op=dist.ReduceOp.SUM)
                    ctr_epoch_loss /= world_size
            mean_epoch_loss = epoch_loss.item() / step
            ctr_mean_epoch_loss = None
            if ctr_split_lists is not None and use_controls:
                ctr_mean_epoch_loss = ctr_epoch_loss.item() / step
            if dist.get_rank() == 0:
                utils.logging_rank_0(f"Epoch {epoch + 1}, average loss: {mean_epoch_loss:.4f}", dist.get_rank())
                utils.tensorboard_write_rank_0(writer, 'epoch_train_loss', mean_epoch_loss, epoch + 1, dist.get_rank())
                if ctr_mean_epoch_loss is not None:
                    utils.logging_rank_0(f"Epoch {epoch + 1}, average ctr loss: {ctr_mean_epoch_loss:.4f}",
                                         dist.get_rank())
                    utils.tensorboard_write_rank_0(writer, 'epoch_train_ctr_loss', ctr_mean_epoch_loss, epoch + 1,
                                                   dist.get_rank())
            """
            VALIDATION LOOP
            """
            if (epoch + 1) % val_interval == 0:
                # if (epoch + 1) % val_interval == 0 and dist.get_rank() == 0:
                model.eval()
                with torch.no_grad():
                    step = 0
                    val_epoch_loss = 0
                    # loss_list = []
                    # val_batch_dice_list = []
                    val_epoch_dice = 0
                    ctr_val_epoch_loss = 0
                    ctr_val_epoch_volume = 0
                    # val_batch_dist_list = None
                    if 'dist' in val_loss_fct.lower():
                        # val_batch_dist_list = []
                        val_epoch_dist = 0
                    pbar = tqdm(val_loader, desc=f'Val[{epoch + 1}] avg_metric:[N/A]')

                    """
                    VALIDATION LOOP
                    """
                    for val_data in pbar:
                        step += 1
                        val_inputs, val_labels = val_data['image'].to(
                            device, non_blocking=non_blocking), val_data['label'].to(
                            device, non_blocking=non_blocking)
                        ctr_val_inputs = None
                        if ctr_split_lists is not None and use_controls:
                            ctr_val_inputs = val_data['control'].to(device, non_blocking=non_blocking)
                        # In case CoordConv is used
                        with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
                            # masks_only_val_labels = val_labels[:, :1, :, :, :]
                            val_outputs = sliding_window_inference(val_inputs, model_img_size,
                                                                   val_batch_size, model)
                            val_loss = val_loss_function(val_outputs, val_labels)
                            # loss_list.append(val_loss.item())
                            val_epoch_loss += val_loss
                            val_outputs_list = decollate_batch(val_outputs)
                            """
                            Apply post transformations on prediction tensor
                            It can then be used with other metrics
                            """
                            val_output_convert = [
                                post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
                            ]
                            """
                            Validation of the controls if they are present
                            First we compute the same loss as for the training and then we count the number of 
                            predicted voxels (non-zero) and average it over the batch
                            """
                            ctr_val_desc = ''
                            if ctr_val_inputs is not None:
                                ctr_val_outputs = model(ctr_val_inputs)
                                # ctr_val_outputs = ctr_val_outputs[:, :1, :, :, :]
                                ctr_val_loss = ctr_loss_function(ctr_val_outputs)
                                # * weight_factor
                                ctr_val_outputs_list = decollate_batch(ctr_val_outputs)
                                ctr_val_convert = [
                                    post_trans(ctr_val_pred_tensor) for ctr_val_pred_tensor in ctr_val_outputs_list
                                ]
                                # For each element of ctr_val_convert, count non-zero voxels and average number over
                                ctr_vox_count = torch.mean(torch.tensor([torch.count_nonzero(ctr_val_pred_tensor) for
                                                                         ctr_val_pred_tensor in ctr_val_convert]),
                                                           dtype=torch.float32).to(device)
                                ctr_val_epoch_volume += ctr_vox_count
                                ctr_val_epoch_loss += ctr_val_loss
                                ctr_val_desc = f' ctr_val_loss:[{ctr_val_loss.item():.4f}]' \
                                               f' ctr_vox_count:[{ctr_vox_count.item():.4f}]'

                            dice_metric(y_pred=val_output_convert, y=val_labels)
                            dice = dice_metric.aggregate()
                            val_epoch_dice += dice
                            if 'dist' in val_loss_fct.lower():
                                hausdorff_metric(y_pred=val_output_convert, y=val_labels)
                                # For whatever reason, this metric is going back to cpu unlike the dice_metric ...
                                distance = hausdorff_metric.aggregate().to(device)
                                # val_batch_dist_list.append(distance.item())
                                val_epoch_dist += distance
                        # val_batch_dice_list.append(dice.item())
                        pbar.set_description(f'Val[{epoch + 1}] mean_loss:[{val_epoch_loss.item() / step}] '
                                             f'{ctr_val_desc}')

                    """
                    GLOBAL VALIDATION MEASURES HANDLING
                    """
                    if world_size > 1:
                        dist.all_reduce(val_epoch_loss, op=dist.ReduceOp.SUM)
                        val_epoch_loss /= world_size

                        dist.all_reduce(val_epoch_dice, op=dist.ReduceOp.SUM)
                        val_epoch_dice /= world_size
                        if 'dist' in val_loss_fct.lower():
                            dist.all_reduce(val_epoch_dist, op=dist.ReduceOp.SUM)
                            val_epoch_dist /= world_size
                        if ctr_val_inputs is not None:
                            dist.all_reduce(ctr_val_epoch_loss, op=dist.ReduceOp.SUM)
                            ctr_val_epoch_loss /= world_size
                            dist.all_reduce(ctr_val_epoch_volume, op=dist.ReduceOp.SUM)
                            ctr_val_epoch_volume /= world_size

                    val_epoch_loss /= step
                    val_epoch_dice /= step
                    mean_loss_val = val_epoch_loss
                    dice_metric.reset()
                    mean_dice_val = val_epoch_dice
                    # mean_dice_val = np.mean(val_batch_dice_list)
                    utils.tensorboard_write_rank_0(writer, 'val_mean_dice', val_epoch_dice.item(), epoch + 1,
                                                   dist.get_rank())
                    utils.tensorboard_write_rank_0(writer, 'val_mean_dice', val_epoch_dice.item(), epoch + 1,
                                                   dist.get_rank())
                    """
                    CONTROL VALIDATION MEASURES HANDLING
                    """
                    ctr_val_epoch_str = ''
                    if ctr_val_inputs is not None:
                        ctr_val_epoch_loss /= step
                        ctr_val_epoch_volume /= step
                        # mean_ctr_val_loss = ctr_val_epoch_loss
                        # mean_ctr_val_volume = ctr_val_epoch_volume
                        utils.tensorboard_write_rank_0(writer, 'ctr_val_loss', ctr_val_epoch_loss.item(), epoch + 1,
                                                       dist.get_rank())
                        utils.tensorboard_write_rank_0(writer, 'ctr_val_volume', ctr_val_epoch_volume.item(), epoch + 1,
                                                       dist.get_rank())
                        ctr_val_epoch_str = f'\nctr_val_loss:[{ctr_val_epoch_loss.item():.4f}]' \
                                            f' ctr_val_volume:[{ctr_val_epoch_volume.item():.4f}]'
                                            # f' (ctr_val_loss * {weight_factor} ' \
                                            # f':[{ctr_val_epoch_loss.item() * weight_factor:.4f}])' \

                    """
                    DISTANCE VALIDATION MEASURES HANDLING
                    """
                    mean_dist_val = None
                    mean_dist_str = ''
                    if 'dist' in val_loss_fct.lower():
                        val_epoch_dist /= step
                        mean_dist_val = val_epoch_dist
                        hausdorff_metric.reset()
                        mean_dist_str = f'/ Current mean distance {val_epoch_dist.item()}'
                        utils.tensorboard_write_rank_0(writer, 'val_distance', val_epoch_dist.item(), epoch + 1,
                                                       dist.get_rank())
                    """
                    BEST EPOCH CONDITION AND SAVE CHECKPOINT
                    """
                    if rank == 0 and best_dice < mean_dice_val:
                        if delayed_control_training:
                            best_dice_list.append(mean_dice_val)
                            if len(best_dice_list) >= number_of_best_dice_intervals_to_assume_convergence:
                                # if the last number_of_best_dice_intervals_to_assume_convergence values in the list
                                # are not separated by more than best_dice_interval_difference then change use_controls
                                # to True
                                converged = True
                                for i in range(1, number_of_best_dice_intervals_to_assume_convergence):
                                    if abs(best_dice_list[-i] - best_dice_list[-i - 1]) > best_dice_interval_difference:
                                        converged = False
                                if converged:
                                    print(f'The difference between the last '
                                          f'{number_of_best_dice_intervals_to_assume_convergence} '
                                          f'best dice values is less than {best_dice_interval_difference} '
                                          f' so we now start using the controls to train the model. ')
                                    use_controls = True

                        best_epoch_pref_str = 'Best dice epoch'
                        best_metric_epoch = epoch + 1
                        best_dice = mean_dice_val
                        best_avg_loss = mean_loss_val
                        utils.tensorboard_write_rank_0(writer, 'val_best_mean_dice', best_dice.item(),
                                                       epoch + 1, dist.get_rank())
                        utils.tensorboard_write_rank_0(writer, 'val_best_mean_loss', best_avg_loss.item(), epoch + 1,
                                                       dist.get_rank())
                        if save_every_decent_best_epoch:
                            if best_dice > 0.75:
                                epoch_suffix = '_' + str(epoch + 1)
                        # True here means that we track and keep the distance and that both dice and dist improved
                        if (mean_dist_val is not None and keep_dice_and_dist) and (
                                best_dist > mean_dist_val and best_dice_with_dist < mean_dice_val):
                            best_dice_with_dist = mean_dice_val
                            best_metric_dist_epoch = epoch + 1
                            best_epoch_pref_str = 'Best dice and best distance epoch'
                            best_dist = mean_dist_val
                            best_dist_str = f'/ Best Distance {best_dist.item()}'

                            utils.tensorboard_write_rank_0(writer, 'val_best_mean_distance', best_dist.item(),
                                                           epoch + 1, dist.get_rank())
                            checkpoint_path = utils.save_checkpoint(
                                model, epoch + 1, fold, optimizer, scaler, hyper_params,
                                output_fold_dir, model_type, transform_dict,
                                f'best_dice_and_dist_model_segmentation3d_epo{epoch_suffix}.pth')
                            utils.logging_rank_0(f'New best (dice and dist) model saved in {checkpoint_path}',
                                                 dist.get_rank())
                            str_best_dist_epoch = (
                                    f'\n{best_epoch_pref_str} {best_metric_dist_epoch} '
                                    # f'metric {best_metric:.4f}/dist {best_distance}/avgloss {best_avg_loss}\n'
                                    f'Dice metric {best_dice.item():.4f} / mean loss {val_epoch_loss.item()}'
                                    + best_dist_str + ctr_val_epoch_str
                            )
                        # Here, only dice improved
                        else:
                            checkpoint_path = utils.save_checkpoint(
                                model, epoch + 1, fold, optimizer, scaler, hyper_params,
                                output_fold_dir, model_type, transform_dict,
                                f'best_dice_model_segmentation3d_epo{epoch_suffix}.pth')
                            utils.logging_rank_0(f'New best model saved in {checkpoint_path}', dist.get_rank())
                            str_best_epoch = (
                                    f'\n{best_epoch_pref_str} {best_metric_epoch} '
                                    # f'metric {best_metric:.4f}/distance {best_distance}/avgloss {best_avg_loss}\n'
                                    f'Dice metric {best_dice.item():.4f} / mean loss {best_avg_loss.item()}'
                                    + ctr_val_epoch_str
                            )
                    if rank == 0:
                        if 'dist' in val_loss_fct.lower():
                            best_epoch_count = epoch + 1 - best_metric_dist_epoch
                        else:
                            best_epoch_count = epoch + 1 - best_metric_epoch
                        current_ctr_val_perf_str = ''
                        if ctr_val_epoch_str != '':
                            current_ctr_val_perf_str = f' current control perf: {ctr_val_epoch_str}'
                        str_current_epoch = (
                                f'[Fold: {fold}]Current epoch: {epoch + 1} current mean loss: '
                                f'{mean_loss_val.item():.4f}'
                                f' current mean dice metric: {mean_dice_val.item()}' + mean_dist_str +
                                current_ctr_val_perf_str + '\n' + str_best_epoch + str_best_dist_epoch + '\n'
                        )
                        print(str_current_epoch)
                        print(f'It has been [{best_epoch_count}] since a best epoch has been found')
                        if stop_best_epoch > -1:
                            print(f'The training will stop after [{stop_best_epoch}] epochs without improvement')
                        epoch_end_time = time.time()
                        epoch_time = epoch_end_time - start_time
                        print(f'Epoch Time: {epoch_time}')
                        epoch_time_list.append(epoch_time)
                        print(f'First epoch time: '
                              f'{epoch_time_list[0]} and average epoch time {np.mean(epoch_time_list)}')

                        if stop_best_epoch != -1:
                            if best_epoch_count > stop_best_epoch:
                                stop_epoch = True
                                print(f'More than {stop_best_epoch} without improvement')
                                # df.to_csv(Path(output_fold_dir, 'perf_measures.csv'), columns=perf_measure_names)
                                # print(f'Training completed\n')
                                # logging.info(str_best_epoch)
                                # writer.close()
                                # break
                    """
                    use_controls has to be shared with the other GPUs
                    """
                    if dist.get_rank() == 0:
                        use_controls_to_share = [use_controls]
                    else:
                        use_controls_to_share = [None]
                    torch.distributed.broadcast_object_list(use_controls_to_share, src=0)
                    use_controls = use_controls_to_share[0]
            if dist.get_rank() == 0:
                flag_to_share = [stop_epoch]
            else:
                flag_to_share = [None]
            torch.distributed.broadcast_object_list(flag_to_share, src=0)
            stop_epoch = flag_to_share[0]
            dist.barrier()

            # Strategic garbage collection to reduce memory fragmentation
            # Only every 10 epochs to minimize overhead
            if (epoch + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                utils.logging_rank_0(f'Epoch {epoch + 1}: Memory cleanup performed', dist.get_rank())

            if stop_epoch:
                break
                # utils.save_checkpoint(model, epoch + 1, optimizer, output_dir)
        # df.to_csv(Path(output_fold_dir, f'perf_measures_{fold}.csv'), columns=perf_measure_names)
        # with open(Path(output_fold_dir, f'trash_img_count_dict_{fold}.json'), 'w+') as j:
        #     json.dump(trash_seg_path_count_dict, j, indent=4)
        print(f'[Rank {dist.get_rank()}] Training completed\n')
        utils.logging_rank_0(str_best_epoch, dist.get_rank())
        if writer is not None:
            writer.close()
        utils.logging_rank_0(f'Fold {fold} finished', rank)

        # Clean up memory between folds (but don't destroy process group yet)
        if 'model' in _cleanup_context:
            del _cleanup_context['model']
        if 'optimizer' in _cleanup_context:
            del _cleanup_context['optimizer']
        if 'scaler' in _cleanup_context and _cleanup_context['scaler'] is not None:
            del _cleanup_context['scaler']
        # DIAGNOSTIC: Log memory before cleanup
        if rank == 0:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024**3  # GB
            utils.logging_rank_0(f'[DIAGNOSTIC] RAM before cleanup: {mem_before:.2f} GB', rank)

        # Explicitly shut down DataLoader workers before deletion
        if 'train_loader' in _cleanup_context:
            try:
                # Shutdown workers if they exist
                if hasattr(_cleanup_context['train_loader'], '_iterator') and _cleanup_context['train_loader']._iterator is not None:
                    _cleanup_context['train_loader']._iterator._shutdown_workers()
                    utils.logging_rank_0(f'[DIAGNOSTIC] Train loader workers shut down', rank)
            except Exception as e:
                utils.logging_rank_0(f'[DIAGNOSTIC] Train loader shutdown warning: {e}', rank)

            # Delete the underlying dataset explicitly
            if hasattr(_cleanup_context['train_loader'], 'dataset'):
                del _cleanup_context['train_loader'].dataset

            del _cleanup_context['train_loader']
            utils.logging_rank_0(f'[DIAGNOSTIC] Train loader deleted', rank)

        if 'val_loader' in _cleanup_context:
            try:
                # Shutdown workers if they exist
                if hasattr(_cleanup_context['val_loader'], '_iterator') and _cleanup_context['val_loader']._iterator is not None:
                    _cleanup_context['val_loader']._iterator._shutdown_workers()
                    utils.logging_rank_0(f'[DIAGNOSTIC] Val loader workers shut down', rank)
            except Exception as e:
                utils.logging_rank_0(f'[DIAGNOSTIC] Val loader shutdown warning: {e}', rank)

            # Delete the underlying dataset explicitly
            if hasattr(_cleanup_context['val_loader'], 'dataset'):
                del _cleanup_context['val_loader'].dataset
                utils.logging_rank_0(f'[DIAGNOSTIC] Val dataset deleted', rank)

            del _cleanup_context['val_loader']
            utils.logging_rank_0(f'[DIAGNOSTIC] Val loader deleted', rank)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force multiple GC passes to catch circular references
        utils.logging_rank_0(f'[DIAGNOSTIC] Running garbage collection...', rank)
        for _ in range(3):
            gc.collect()

        # DIAGNOSTIC: Log memory after cleanup
        if rank == 0:
            mem_after = process.memory_info().rss / 1024**3  # GB
            mem_freed = mem_before - mem_after
            utils.logging_rank_0(f'[DIAGNOSTIC] RAM after cleanup: {mem_after:.2f} GB (freed {mem_freed:.2f} GB)', rank)

        utils.logging_rank_0(f'Memory cleaned up after fold {fold}', rank)

        # Add a small delay to ensure GC completes
        time.sleep(2)
        utils.logging_rank_0(f'[DIAGNOSTIC] Waited 2s for GC to complete', rank)
    dist.destroy_process_group()

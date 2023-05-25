import json
import os
import random
import shutil
import logging
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union
import time
import signal
import sys

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat
import nibabel as nib
from monai.data import decollate_batch
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from torch.utils.tensorboard import SummaryWriter
from lesseg_unet import net, utils, data_loading, transformations
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# def handler(signum, frame):
#     res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
#     if res == 'y':
#         dist.destroy_process_group()
#         exit(1)
#
#
# signal.signal(signal.SIGINT, handler)


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
             # label_smoothing=False,
             stop_best_epoch=-1,
             training_loss_fct='dice',
             val_loss_fct='dice',
             weight_factor=1,
             folds_number=1,
             dropout=0,
             cache_dir=None,
             save_every_decent_best_epoch=True,
             rank=0,
             world_size=1,
             cache_num=None,
             debug=False,
             **kwargs
             ):
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print("LIMIT before: {}".format(rlimit))
    resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print("LIMIT after: {}".format(rlimit))
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
    torch.cuda.set_device(device)

    logging.info(f'Torch device used for this training: {str(device)}')

    """
    LOSS FUNCTIONS
    """
    # Training
    if training_loss_fct.lower() in ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']:
        loss_function = DiceCELoss(sigmoid=True)
    elif training_loss_fct.lower() in ['focal', 'focalloss', 'focal_loss']:
        loss_function = FocalLoss(gamma=2.0)
    elif training_loss_fct.lower() in ['dicefocal', 'dicefocalloss', 'dice_focal_loss']:
        loss_function = DiceFocalLoss(sigmoid=True, gamma=2.0)
    else:
        loss_function = DiceLoss(sigmoid=True)
    utils.logging_rank_0(f'Training loss fct: {loss_function}', dist.get_rank())
    # Validation
    if any([s in val_loss_fct.lower() for s in
            ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']]):
        val_loss_function = DiceCELoss(sigmoid=True)
    else:
        val_loss_function = DiceLoss(sigmoid=True)
    utils.logging_rank_0(f'Validation loss fct: {val_loss_function}', dist.get_rank())

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
            utils.logging_rank_0(f'##### Number of control images for training: {len(ctr_path_list)}', dist.get_rank())
            split_lists_to_share = [utils.split_lists_in_folds(
                ctr_path_list, folds_number, train_val_percentage, shuffle=True, image_key='control')]
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

    original_image_shape = utils.get_img_size(split_lists[0][0]['image'])
    utils.print_rank_0(f'Original image shape: {original_image_shape}', dist.get_rank())
    # We need the training image size for the unetr as we need to know the size of the model to create it
    if list(transform_dict.keys())[-1] == 'patches':
        # TODO this might change depending on the cropping transformation
        model_img_size = transformations.find_param_from_hyper_dict(
            transform_dict, 'roi_size', find_last=True)
        model_img_size = model_img_size[-3:]
        transformations.setup_coord_conv(transform_dict, original_image_shape)
    else:
        model_img_size = transformations.find_param_from_hyper_dict(
            transform_dict, 'spatial_size', find_last=True)
        transformations.setup_coord_conv(transform_dict, model_img_size)

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
    for fold in range(folds_number):
        if fold < starting_fold:
            utils.logging_rank_0(f'Skipping fold {fold}', dist.get_rank())
            continue
        """
        SET MODEL PARAM AND CREATE / LOAD MODEL OBJECT
        """
        utils.logging_rank_0(f'Creating monai {model_type}', dist.get_rank())
        scaler = torch.cuda.amp.GradScaler()
        if checkpoint_to_share is not None:
            checkpoint = checkpoint_to_share[0]
            hyper_params = checkpoint['hyper_params']
            model = utils.load_model_from_checkpoint(checkpoint, device, hyper_params, model_name=model_type)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            optimizer.load_state_dict(checkpoint['optim_dict'])
            scaler.load_state_dict(checkpoint['scaler_dict'])
            utils.logging_rank_0(f'{model_type} created and succesfully loaded from {pretrained_point} with '
                                 f'hyper parameters: {hyper_params}',
                                 dist.get_rank())
        else:
            if model_type.lower() == 'unetr' or model_type.lower() == 'swinunetr':
                if model_type.lower() == 'unetr':
                    hyper_params = net.default_unetr_hyper_params
                else:
                    hyper_params = net.default_swinunetr_hyper_params
                hyper_params['img_size'] = model_img_size
                if 'feature_size' in kwargs:
                    hyper_params['feature_size'] = int(kwargs['feature_size'])
            else:
                hyper_params = net.default_unet_hyper_params
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            # use amp to accelerate training
        if dist.get_rank() == 0:
            total_param_count = count_unique_parameters(model.named_parameters())
            utils.logging_rank_0(f'Total number of parameters in the model: {str(total_param_count)}',
                                 dist.get_rank())
        params = list(model.parameters())
        if torch.cuda.is_available():
            model.to(dist.get_rank())
            model = DistributedDataParallel(model, device_ids=[rank], output_device=dist.get_rank(),
                                            find_unused_parameters=False)
            utils.print_rank_0('Model sent to the different ranks', dist.get_rank())
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
        if ctr_split_lists is None:
            train_loader, val_loader = data_loading.create_fold_dataloaders(
                split_lists, fold, train_img_transforms,
                val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir,
                world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training, cache_num=cache_num
            )

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
        best_metric_epoch = -1
        best_metric_dist_epoch = -1
        img_dir = Path(output_dir, 'image_dir')
        if display_training:
            if dist.get_rank() == 0:
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
                os.makedirs(img_dir, exist_ok=True)
            dist.barrier()
        stop_epoch = False
        for epoch in range(epoch_num):
            utils.print_rank_0('-' * 10, dist.get_rank())
            utils.print_rank_0(f'epoch {epoch + 1}/{epoch_num}', dist.get_rank())
            # If ctr_split_lists is not None we need to create a new training loader with the controls
            if ctr_split_lists is not None:
                # We need to add the same number of controls as the abnormal images in each fold after shuffling them
                # The new split_list has to be shared between all the ranks
                if dist.get_rank() == 0:
                    split_lists_with_ctr = []
                    for i in range(len(split_lists)):
                        ctr_fold_list = deepcopy(ctr_split_lists[i])
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
                    val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir,
                    world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training, cache_num=cache_num,
                    training_persistent_workers=False
                )

                # train_loader = data_loading.create_ctr_dataloader(
                #     split_lists, ctr_split_lists, fold, train_img_transforms,
                #     val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir,
                #     world_size=world_size, rank=dist.get_rank(), shuffle_training=shuffle_training,
                #     cache_num=cache_num
                # )
            # This is required with multi-gpu
            batches_per_epoch = len(train_loader)
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0
            ctr_epoch_loss = 0
            step = 0
            start_time = time.time()
            loading_time = True

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
                if ctr_split_lists is not None:
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

                with torch.cuda.amp.autocast():
                    # TODO Look up gradient accumulation
                    logit_outputs = model(inputs)
                    # In case we use CoordConv, we only take the mask of the labels without the coordinates
                    masks_only_labels = labels[:, :1, :, :, :]
                    loss = loss_function(logit_outputs, masks_only_labels)
                    # Regularisation
                    l2_reg = utils.sum_non_bias_l2_norms(params, 1e-4)
                    loss += l2_reg

                    controls_loss = None
                    if ctr_inputs is not None:
                        ctr_logit_outputs = model(ctr_inputs)
                        outputs_batch_images_sigmoid = ctr_post_trans(ctr_logit_outputs)
                        controls_loss = torch.mean(outputs_batch_images_sigmoid) * weight_factor
                        # Regularisation
                        controls_loss += l2_reg

                # No need to autocast the scaler stuff
                if controls_loss is not None:
                    mean_loss = (loss + controls_loss) / 2
                    scaler.scale(mean_loss).backward()
                else:
                    scaler.scale(loss).backward()
                """
                The different ranks are coming together here
                """
                if epoch % gradient_accumulation_steps == 0:
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                """
                Progress and other str formatting
                """
                epoch_loss += loss
                if ctr_inputs is not None:
                    ctr_epoch_loss += controls_loss
                if dist.get_rank() == 0:
                    if no_progressbar_training or dist.get_rank() != 0:
                        utils.print_rank_0(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}',
                                           dist.get_rank())
                    else:
                        ctr_desc = ''
                        if ctr_inputs is not None:
                            ctr_desc = f' [ctr_loss: {controls_loss.item():.4f} / {ctr_epoch_loss.item()/step:.4f}]' \
                                       f' [losses sum: {loss.item() + controls_loss.item():.4f}]'
                        train_iter.set_description(
                            f'Training[{epoch + 1}] '
                            f'batch_loss/mean_loss:[{loss.item():.4f}/{epoch_loss.item()/step:.4f}]' + ctr_desc)

            if one_loop:
                exit()
            """
            GLOBAL TRAINING MEASURES HANDLING
            """
            if world_size > 1:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss /= world_size
                if ctr_split_lists is not None:
                    dist.all_reduce(ctr_epoch_loss, op=dist.ReduceOp.SUM)
                    ctr_epoch_loss /= world_size
            mean_epoch_loss = epoch_loss.item() / step
            ctr_mean_epoch_loss = None
            if ctr_split_lists is not None:
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
            if epoch % gradient_accumulation_steps == 0:
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
                        if ctr_split_lists is not None:
                            ctr_val_inputs = val_data['control'].to(device, non_blocking=non_blocking)
                        # In case CoordConv is used
                        with torch.cuda.amp.autocast():
                            masks_only_val_labels = val_labels[:, :1, :, :, :]
                            val_outputs = sliding_window_inference(val_inputs, model_img_size,
                                                                   val_batch_size, model)
                            val_loss = val_loss_function(val_outputs, masks_only_val_labels)
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
                                ctr_val_loss = torch.mean(outputs_batch_images_sigmoid).to(device)
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

                            dice_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                            dice = dice_metric.aggregate()
                            val_epoch_dice += dice
                            if 'dist' in val_loss_fct.lower():
                                hausdorff_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                                # For whatever reason, this metric is going back to cpu unlike the dice_metric ...
                                distance = hausdorff_metric.aggregate().to(device)
                                # val_batch_dist_list.append(distance.item())
                                val_epoch_dist += distance
                        # val_batch_dice_list.append(dice.item())
                        pbar.set_description(f'Val[{epoch + 1}] mean_loss:[{val_epoch_loss.item()/step}] '
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
                                            f' (ctr_val_loss * {weight_factor} ' \
                                            f':[{ctr_val_epoch_loss.item() * weight_factor:.4f}])' \
                                            f' ctr_val_volume:[{ctr_val_epoch_volume.item():.4f}]'

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
                        print(f'best_epoch_count: {best_epoch_count}')
                        print(f'best_metric_epoch: {best_metric_epoch}')
                        print(f'epoch: {epoch}')
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
            if dist.get_rank() == 0:
                flag_to_share = [stop_epoch]
            else:
                flag_to_share = [None]
            torch.distributed.broadcast_object_list(flag_to_share, src=0)
            stop_epoch = flag_to_share[0]
            dist.barrier()
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
    dist.destroy_process_group()

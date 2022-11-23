import json
import os
import shutil
import logging
from pathlib import Path
from typing import Sequence, Union
import time
import signal
import sys

from tqdm import tqdm
import numpy as np
from nilearn.plotting import plot_anat
import nibabel as nib
from monai.data import decollate_batch
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceCELoss
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


def setup(rank, world_size, port='1234', cpu=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    if not cpu:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)


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
             seg_path_list: Sequence,
             output_dir: Union[str, bytes, os.PathLike],
             img_pref: str = None,
             transform_dict=None,
             pretrained_point=None,
             device: str = None,
             batch_size: int = 1,
             val_batch_size: int = 1,
             epoch_num: int = 50,
             dataloader_workers: int = 4,
             train_val_percentage=80,
             lesion_set_clamp=None,
             # controls_clamping=None,
             # label_smoothing=False,
             stop_best_epoch=-1,
             training_loss_fct='dice',
             val_loss_fct='dice',
             # weight_factor=1,
             folds_number=1,
             dropout=0,
             cache_dir=None,
             save_every_decent_best_epoch=True,
             rank=0,
             world_size=1,
             **kwargs
             ):
    # disable logging for processes except 0 on every node
    # if rank != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f
    # rank = int(os.environ["LOCAL_RANK"])
    ####################################################################################################################
    # Checking the external parameters
    # TODO: add it to the main param when we are are sure we need them permanently
    ####################################################################################################################
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
    # Apparently it can potentially improve the performance when the model does not change its size. (Source tuto UNETR)
    torch.backends.cudnn.benchmark = True
    """MODEL PARAMETERS"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if world_size > 0:
        cpu_device = device.type == 'cpu'

        # setup(rank, world_size, cpu=cpu_device)
        if not cpu_device:
            device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    logging.info(f'Torch device used for this training: {str(device)}')

    if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
        hyper_params = net.default_unetr_hyper_params
    else:
        hyper_params = net.default_unet_hyper_params
    # checking is CoordConv is used and change the input channel dimension
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        hyper_params['in_channels'] = 4
    if dropout is not None and dropout == 0:
        hyper_params['dropout'] = dropout
        logging.info(f'Dropout rate used: {dropout}')

    """LOSS FUNCTIONS"""
    if training_loss_fct.lower() in ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']:
        loss_function = DiceCELoss(sigmoid=True)
    else:
        loss_function = DiceLoss(sigmoid=True)
    logging.info(f'Training loss fct: {loss_function}')
    if any([s in val_loss_fct.lower() for s in
            ['dice_ce', 'dicece', 'dice_ce_loss', 'diceceloss', 'dice_cross_entropy']]):
        val_loss_function = DiceCELoss(sigmoid=True)
    else:
        val_loss_function = DiceLoss(sigmoid=True)
    logging.info(f'Validation loss fct: {val_loss_function}')

    """METRICS"""
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    keep_dice_and_dist = True
    if 'keep_dice_and_dist' in kwargs:
        v = kwargs['keep_dice_and_dist']
        if v == 'False' or v == 0:
            keep_dice_and_dist = False
        if v == 'True' or v == 1:
            keep_dice_and_dist = True
    """TRANSFORMATIONS AND AUGMENTATIONS"""
    # Attempt to send the transformations / augmentations on the GPU when possible (disabled by default)
    transformations_device = None
    if 'tr_device' in kwargs:
        v = kwargs['tr_device']
        if v == 'False' or v == 0:
            transformations_device = None
        if v == 'True' or v == 1:
            print(f'ToTensord transformation will be called on {device}')
            transformations_device = device
    logging.info('Initialisation of the training transformations')
    # Extract all the transformations from transform_dict
    train_img_transforms = transformations.train_transformd(transform_dict, lesion_set_clamp,
                                                            device=transformations_device)
    # Extract only the 'first' and 'last' transformations from transform_dict ignoring the augmentations
    val_img_transforms = transformations.val_transformd(transform_dict, lesion_set_clamp,
                                                        device=transformations_device)

    """POST TRANSFORMATIONS"""
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    """DATA LOADING"""
    if img_pref is not None and img_pref != '':
        logging.info(f'Abnormal images prefix: {img_pref}')

    # Get the images from the image and label list and tries to match them
    img_dict, controls = data_loading.match_img_seg_by_names(img_path_list, seg_path_list, img_pref)

    split_lists = utils.split_lists_in_folds(img_dict, folds_number, train_val_percentage)
    # Save the split_lists to easily get the content of the folds and all
    with open(Path(output_dir, 'split_lists.json'), 'w+') as f:
        json.dump(split_lists, f, indent=4)
    # We need the training image size for the unetr as we need to know the size of the model to create it
    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    if training_img_size is None:
        training_img_size = utils.get_img_size(split_lists[0][0]['image'])

    """FOLDS LOOP VARIABLES"""
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
    if stop_best_epoch != -1:
        logging.info(f'Will stop after {stop_best_epoch} epochs without improvement')

    for fold in range(folds_number):
        if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
            hyper_params['img_size'] = training_img_size
            if 'feature_size' in kwargs:
                hyper_params['feature_size'] = int(kwargs['feature_size'])
            if pretrained_point is not None:
                model = utils.load_model_from_checkpoint(pretrained_point, device, hyper_params, model_name='unetr')
                logging.info(f'UNETR created and succesfully loaded from {pretrained_point}')
            else:
                model = net.create_unetr_model(device, hyper_params)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            # For the regularisation
            params = list(model.parameters())
        else:
            if pretrained_point is not None:
                model = utils.load_model_from_checkpoint(pretrained_point, device, hyper_params, model_name='unet')
                logging.info(f'UNet created and succesfully loaded from {pretrained_point}')
            else:
                model = net.create_unet_model(device, hyper_params)
            optimizer = torch.optim.Adam(model.parameters(), 1e-3)
            params = list(model.model.parameters())
        # TODO the segmentation might require to add 'module' after model. to access the state_dict and all
        if torch.cuda.is_available():
            model.to(rank)
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if folds_number == 1:
            output_fold_dir = output_dir
        else:
            output_fold_dir = Path(output_dir, f'fold_{fold}')
        # Tensorboard writer
        writer = SummaryWriter(log_dir=str(output_fold_dir))
        # Creates both the training and validation loaders based on the fold number
        # (e.g. fold 0 means the first sublist of split_lists will be the validation set for this fold)
        # TODO Share the data over the ranks
        # TODO https://scm4.cs.ucl.ac.uk/radiology-system/multimodalgenerativemodel/-/blob/main/modules/orchestration/training_script.py
        # line 221 to ~270
        train_loader, val_loader = data_loading.create_fold_dataloaders(
            split_lists, fold, train_img_transforms,
            val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir,
            world_size, rank, shuffle_training=shuffle_training
        )

        """EPOCHS LOOP VARIABLES"""
        batches_per_epoch = len(train_loader)
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
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
            os.makedirs(img_dir, exist_ok=True)
        for epoch in range(epoch_num):
            print('-' * 10)
            print(f'epoch {epoch + 1}/{epoch_num}')
            model.train()
            epoch_loss = 0
            step = 0
            start_time = time.time()
            loading_time = True

            """TRAINING LOOP"""
            train_iter = tqdm(train_loader, desc=f'Training[{epoch + 1}] loss/mean_loss:[N/A]')
            no_progressbar_training = False
            if 'no_progressbar_training' in kwargs:
                v = kwargs['no_progressbar_training']
                if v == 'True' or v == 0:
                    train_iter = train_loader
                    no_progressbar_training = True
            for batch_data in train_iter:
                if loading_time:
                    end_time = time.time()
                    load_time = end_time - start_time
                    print(f'Loading loop Time: {load_time}')
                    time_list.append(load_time)
                    print(f'First load time: {time_list[0]} and average loading time {np.mean(time_list)}')
                    loading_time = False
                step += 1
                optimizer.zero_grad()
                inputs, labels = batch_data['image'].to(device, non_blocking=non_blocking), batch_data['label'].to(
                    device, non_blocking=non_blocking)
                if display_training:
                    img_name = Path(batch_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
                    # print(batch_data['image_meta_dict']['affine'][0].cpu().detach().numpy())
                    nii = nib.Nifti1Image(inputs[0, 0, ...].cpu().detach().numpy(),
                                          batch_data['image_meta_dict']['affine'][0].cpu().detach().numpy())
                    plot_anat(nii,
                              output_file=Path(img_dir, f'{img_name}.png'),
                              display_mode='tiled', title=img_name, draw_cross=False,
                              cut_coords=(50, 54, 45)
                              )
                    data = inputs[0, 0, ...].cpu().detach().numpy()
                    print(img_name)
                    print(np.mean(data))
                    nib.save(nii, Path(img_dir, f'{img_name}.nii.gz'))
                    # display.savefig('pretty_brain.png')
                    # print(f'inputs shape: {inputs.shape}')
                    # plot_2d_or_3d_image(inputs[0:,...], 12, writer, tag='tr_inputs')
                    # input(img_name + '  continue??')
                logit_outputs = model(inputs)
                # In case we use CoordConv, we only take the mask of the labels without the coordinates
                masks_only_labels = labels[:, :1, :, :, :]
                loss = loss_function(logit_outputs, masks_only_labels)
                # Regularisation
                loss += utils.sum_non_bias_l2_norms(params, 1e-4)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item()
                # if no_progressbar_training:
                #     print(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}')
                # else:
                #     train_iter.set_description(
                #         f'Training[{epoch + 1}] batch_loss/mean_loss:[{loss.item():.4f}/{epoch_loss / step:.4f}]')

                loss = 0
                epoch_loss = 0
                # with torch.no_grad():
                #     epoch_loss += loss
                #     if no_progressbar_training:
                #         print(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}')
                #     else:
                #         train_iter.set_description(
                #             f'Training[{epoch + 1}] batch_loss/mean_loss:[{loss.item():.4f}/{epoch_loss / step:.4f}]')

            if world_size > 1:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss /= world_size
            epoch_loss = epoch_loss.item()

            print(f"[{dist.get_rank()}] " + f"epoch {epoch + 1}, average loss: {epoch_loss / step:.4f}")
            if dist.get_rank() == 0:
                epoch_loss /= step
                print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')
                writer.add_scalar('epoch_train_loss', epoch_loss, epoch + 1)
            if one_loop:
                return
            """VALIDATION"""
            if (epoch + 1) % val_interval == 0:
                # if (epoch + 1) % val_interval == 0 and dist.get_rank() == 0:
                model.eval()
                with torch.no_grad():
                    step = 0
                    loss_list = []
                    val_batch_dice_list = []
                    val_batch_dist_list = None
                    if 'dist' in val_loss_fct.lower():
                        val_batch_dist_list = []
                    pbar = tqdm(val_loader, desc=f'Val[{epoch + 1}] avg_metric:[N/A]')

                    """VALIDATION LOOP"""
                    for val_data in pbar:
                        step += 1
                        val_inputs, val_labels = val_data['image'].to(
                            device, non_blocking=non_blocking), val_data['label'].to(
                            device, non_blocking=non_blocking)
                        # In case CoordConv is used
                        masks_only_val_labels = val_labels[:, :1, :, :, :]
                        val_outputs = sliding_window_inference(val_inputs, training_img_size,
                                                               val_batch_size, model)
                        loss_list.append(val_loss_function(val_outputs, masks_only_val_labels).item())
                        val_outputs_list = decollate_batch(val_outputs)
                        val_output_convert = [
                            post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
                        ]
                        dice_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                        dice = dice_metric.aggregate().item()

                        if val_batch_dist_list is not None:
                            hausdorff_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                            distance = hausdorff_metric.aggregate().item()
                            val_batch_dist_list.append(distance)
                        val_batch_dice_list.append(dice)
                        pbar.set_description(f'Val[{epoch + 1}] mean_loss:[{np.mean(loss_list)}]')
                    mean_loss_val = np.mean(loss_list)
                    dice_metric.reset()
                    mean_dice_val = np.mean(val_batch_dice_list)
                    writer.add_scalar('val_mean_dice', mean_dice_val, epoch + 1)
                    writer.add_scalar('val_mean_loss', mean_loss_val, epoch + 1)

                    mean_dist_val = None
                    mean_dist_str = ''
                    # # TODO
                    # if world_size > 1:
                    #     dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                    #     epoch_loss /= world_size
                    # epoch_loss = epoch_loss.item()

                    if val_batch_dist_list is not None:
                        mean_dist_val = np.mean(val_batch_dist_list)
                        hausdorff_metric.reset()
                        mean_dist_str = f'/ Current mean distance {mean_dist_val}'
                        writer.add_scalar('val_distance', mean_dist_val, epoch + 1)
                    """IF NEW BEST EPOCH"""
                    if best_dice < mean_dice_val and rank == 0:
                        best_epoch_pref_str = 'Best dice epoch'
                        best_metric_epoch = epoch + 1
                        best_dice = mean_dice_val
                        best_avg_loss = mean_loss_val
                        writer.add_scalar('val_best_mean_dice', best_dice, epoch + 1)
                        writer.add_scalar('val_best_mean_loss', best_avg_loss, epoch + 1)
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
                            best_dist_str = f'/ Best Distance {best_dist}'
                            writer.add_scalar('val_best_mean_distance', best_dist, epoch + 1)
                            # TODO add dist_barrier
                            checkpoint_path = utils.save_checkpoint(
                                model, epoch + 1, optimizer, output_fold_dir,
                                f'best_dice_and_dist_model_segmentation3d_epo{epoch_suffix}.pth')
                            print(f'New best (dice and dist) model saved in {checkpoint_path}')
                            str_best_dist_epoch = (
                                    f'\n{best_epoch_pref_str} {best_metric_dist_epoch} '
                                    # f'metric {best_metric:.4f}/dist {best_distance}/avgloss {best_avg_loss}\n'
                                    f'Dice metric {best_dice:.4f} / mean loss {best_avg_loss}'
                                    + best_dist_str
                            )
                        # Here, only dice improved
                        else:
                            checkpoint_path = utils.save_checkpoint(
                                model, epoch + 1, optimizer, output_fold_dir,
                                f'best_dice_model_segmentation3d_epo{epoch_suffix}.pth')
                            print(f'New best model saved in {checkpoint_path}')
                            str_best_epoch = (
                                f'\n{best_epoch_pref_str} {best_metric_epoch} '
                                # f'metric {best_metric:.4f}/distance {best_distance}/avgloss {best_avg_loss}\n'
                                f'Dice metric {best_dice:.4f} / mean loss {best_avg_loss}'
                            )
                    dist.barrier()
                    if keep_dice_and_dist:
                        best_epoch_count = epoch + 1 - best_metric_dist_epoch
                    else:
                        best_epoch_count = epoch + 1 - best_metric_epoch
                    str_current_epoch = (
                            f'[Fold: {fold}]Current epoch: {epoch + 1} current mean loss: {mean_loss_val:.4f}'
                            f' current mean dice metric: {mean_dice_val}' + mean_dist_str + '\n'
                            + str_best_epoch + str_best_dist_epoch + '\n'
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
                            print(f'More than {stop_best_epoch} without improvement')
                            # df.to_csv(Path(output_fold_dir, 'perf_measures.csv'), columns=perf_measure_names)
                            print(f'Training completed\n')
                            logging.info(str_best_epoch)
                            writer.close()
                            break
                    # utils.save_checkpoint(model, epoch + 1, optimizer, output_dir)
        # df.to_csv(Path(output_fold_dir, f'perf_measures_{fold}.csv'), columns=perf_measure_names)
        # with open(Path(output_fold_dir, f'trash_img_count_dict_{fold}.json'), 'w+') as j:
        #     json.dump(trash_seg_path_count_dict, j, indent=4)
        print(f'Training completed\n')
        logging.info(str_best_epoch)
        writer.close()
        dist.destroy_process_group()

import json
import os
import shutil
import logging
from operator import lt, gt
from pathlib import Path
from typing import Sequence, Tuple, Union

import pandas as pd
from tqdm import tqdm
import numpy as np
import monai
from monai.data import Dataset
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as BCE
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, TverskyLoss, FocalLoss, DiceFocalLoss, DiceCELoss
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from torch.utils.tensorboard import SummaryWriter
from lesseg_unet import net, utils, data_loading, transformations


def testing(train_loader, output_dir):
    it = iter(train_loader)
    import nibabel as nib
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
        orig_label = nib.load(data["label_meta_dict"]["filename_or_obj"][0]).get_fdata()
        label = nib.load(out_paths_list[1]).get_fdata()
        print(f'original label volume: {np.count_nonzero(orig_label)}')
        print(f'Smoothed label volume 0.5: {len(np.where(label > 0.5)[0])}')
        print(f'Smoothed label volume 0.25: {len(np.where(label > 0.25)[0])}')
        print('###########ENDVOLUMES#######')
    # if np.equal(i_data, l_data).all():
    #     print('ok')
    # else:
    #     print('not ok')
    exit()


def training_loop(img_path_list: Sequence,
                  seg_path_list: Sequence,
                  output_dir: Union[str, bytes, os.PathLike],
                  ctr_path_list: Sequence = None,
                  img_pref: str = None,
                  transform_dict=None,
                  device: str = None,
                  batch_size: int = 10,
                  epoch_num: int = 50,
                  dataloader_workers: int = 4,
                  train_val_percentage=80,
                  label_smoothing=False,
                  stop_best_epoch=-1,
                  default_label=None,
                  training_loss_fct='dice',
                  val_loss_fct='dice',
                  weight_factor=1,
                  folds_number=1,
                  dropout=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    # checking is CoordConv is used and change the input channel dimension
    unet_hyper_params = net.default_unet_hyper_params
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        unet_hyper_params['in_channels'] = 4
    # TODO outdated
    # for t in val_ds.transform.transforms:
    #     if isinstance(t, transformations.CoordConvd) or isinstance(t, transformations.CoordConvAltd):
    #         unet_hyper_params = net.coord_conv_unet_hyper_params

    if dropout is not None and dropout == 0:
        unet_hyper_params['dropout'] = dropout
    model = net.create_unet_model(device, unet_hyper_params)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    # surface_metric = SurfaceDistanceMetric(include_background=True, reduction="mean", symmetric=True)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    loss_function = DiceLoss(sigmoid=True)
    tversky_function = TverskyLoss(sigmoid=True, alpha=2, beta=0.5)
    # focal_function = FocalLoss()
    # df_loss = DiceFocalLoss(sigmoid=True)
    # dce_loss = DiceCELoss(sigmoid=True, ce_weight=torch.Tensor([1]).cuda())
    # loss_function = BCE
    # val_loss_function = BCE
    val_loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # Save training parameters info
    logging.info(f'Abnormal images prefix: {img_pref}')
    logging.info(f'Device: {device}')
    logging.info(f'Training loss fct: {training_loss_fct}')
    logging.info(f'Validation loss fct: {val_loss_fct}')
    print('check ok')

    # utils.save_tensor_to_nifti(
    #     inputs, Path('/home/tolhsadum/neuro_apps/data/', 'nib_input_{}.nii'.format('test')), val_output_affine)
    # utils.save_tensor_to_nifti(
    #     labels, Path('/home/tolhsadum/neuro_apps/data/', 'nib_label_{}.nii'.format('test')), val_output_affine)
    # ##################################################################################
    # # Code for restoring!
    # state_dict_fullpath = os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt')
    # checkpoint = torch.load(state_dict_fullpath)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # starting_epoch = checkpoint['epoch'] + 1
    # scaler.load_state_dict(checkpoint['scaler'])
    #
    # # Code for saving (to be used in the validation)
    # checkpoint_dict = {'epoch': epoch,
    #                    'model_state_dict': model.state_dict(),
    #                    'optimizer_state_dict': optimizer.state_dict(),
    #                    'loss_tally_train': loss_tally_train,
    #                    'nifti_t1_paths': nifti_t1_paths,}
    # checkpoint_dict['scaler'] = scaler.state_dict()
    # torch.save(checkpoint_dict, os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt'))
    # ##################################################################################

    # start a typical PyTorch training

    val_interval = 1
    best_metric = -1
    best_distance = -1
    best_avg_loss = -1
    best_metric_epoch = -1
    val_meh_thr = 0.7
    val_trash_thr = 0.3
    metric_select_fct = gt
    control_weight_factor = weight_factor  # Experiment with different weightings!
    if stop_best_epoch != -1:
        logging.info(f'Will stop after {stop_best_epoch} epochs without improvement')
    if label_smoothing:
        logging.info('Label smoothing activated')
    """
    Measure tracking init
    """
    # val_images_dir = Path(output_dir, 'val_images')
    # if not val_images_dir.is_dir():
    #     val_images_dir.mkdir(exist_ok=True)
    # trash_val_images_dir = Path(output_dir, 'trash_val_images')
    # if not trash_val_images_dir.is_dir():
    #     trash_val_images_dir.mkdir(exist_ok=True)
    # best_val_images_dir = Path(output_dir, 'best_epoch_val_images')
    # if not best_val_images_dir.is_dir():
    #     best_val_images_dir.mkdir(exist_ok=True)
    # best_trash_images_dir = Path(output_dir, 'best_epoch_val_images')
    # if not best_trash_images_dir.is_dir():
    #     best_trash_images_dir.mkdir(exist_ok=True)
    # TODO add the new measures if useful to the csv
    perf_measure_names = ['avg_train_loss',
                          'val_mean_metric',
                          'val_median_metric',
                          'val_std_metric',
                          'trash_img_nb',
                          'val_min_metric',
                          'val_max_metric',
                          'val_best_mean_metric']
    df = pd.DataFrame(columns=perf_measure_names)
    # Gather data
    img_dict, controls = data_loading.match_img_seg_by_names(img_path_list, seg_path_list, img_pref, default_label)

    # If the default_label option is used, we try to find controls images and create a second set of datasets and a
    # second set of loaders
    if default_label is not None:
        if ctr_path_list is not None and ctr_path_list != []:
            # _, controls_2 = data_loading.match_img_seg_by_names(img_path_list, [], img_pref, default_label)
            _, controls_2 = data_loading.match_img_seg_by_names(ctr_path_list, [], None, default_label)
            controls.update(controls_2)
    else:
        if ctr_path_list is not None and ctr_path_list != []:
            _, controls_2 = data_loading.match_img_seg_by_names(ctr_path_list, [], None,
                                                                # _, controls_2 = data_loading.match_img_seg_by_names(ctr_path_list, [], img_pref,
                                                                default_label=output_dir)
            controls.update(controls_2)
        else:
            # If default_label is not given, it is not intended to have unmatched images, thus we throw an error
            # if some images don't have a match
            if controls:
                raise ValueError('The default label for controls was not given but some images do not have a '
                                 'matched lesion')

    split_lists = utils.split_lists_in_folds(img_dict, folds_number, train_val_percentage)
    logging.info('Initialisation of the training transformations')
    train_img_transforms = transformations.segmentation_train_transformd(transform_dict)
    val_img_transforms = transformations.segmentation_val_transformd(transform_dict)

    controls_lists = []
    trash_seg_path_count_dict = {}
    ctr_img_transforms = None
    ctr_val_img_transforms = None
    if controls:
        print('Initialisation of the control dataset')
        if len(controls) < len(img_dict):
            raise ValueError(f'Only {len(controls)} controls for {len(img_dict)} patient images')
        else:
            controls_lists = utils.split_lists_in_folds(controls, folds_number, train_val_percentage)
        logging.info('Initialisation of the control training transformations')
        ctr_img_transforms = transformations.image_only_transformd(transform_dict, training=True)
        logging.info('Initialisation of the control validation transformations')
        ctr_val_img_transforms = transformations.image_only_transformd(transform_dict, training=False)
        logging.info(f'Control training loss: mean(sigmoid(outputs)) * {control_weight_factor}')
    for fold in range(folds_number):
        if folds_number == 1:
            output_fold_dir = output_dir
        else:
            output_fold_dir = Path(output_dir, f'fold_{fold}')
        writer = SummaryWriter(log_dir=str(output_fold_dir))
        trash_list_path = Path(output_fold_dir, 'trash_images_list.csv')
        if trash_list_path.is_file():
            os.remove(trash_list_path)
        # TODO outdated
        # train_ds, val_ds = data_loading.init_training_data(img_path_list, seg_path_list, img_pref,
        #                                                    transform_dict=transform_dict,
        #                                                    train_val_percentage=train_val_percentage,
        #                                                    default_label=default_label)
        train_loader, val_loader = data_loading.create_fold_dataloaders(
            split_lists, fold, train_img_transforms,
            val_img_transforms, batch_size, dataloader_workers
        )
        output_spatial_size = None
        max_distance = None
        batches_per_epoch = len(train_loader)
        val_batches_per_epoch = len(val_loader)
        """
        Training loop
        """
        str_best_epoch = ''
        # Cumulated values might reach max float so storing it into lists
        ctr_loss = []
        ctr_vol = []
        for epoch in range(epoch_num):
            print('-' * 10)
            print(f'epoch {epoch + 1}/{epoch_num}')
            ctr_train_loader = None
            ctr_val_loader = None
            if controls_lists:
                print('Reshuffling control data')
                ctr_train_loader, ctr_val_loader = data_loading.create_fold_dataloaders(
                    controls_lists, fold, ctr_img_transforms,
                    ctr_val_img_transforms, batch_size, dataloader_workers
                )
            best_epoch_count = 0
            model.train()
            epoch_loss = 0
            step = 0
            # for batch_data in tqdm(train_loader, desc=f'training_loop{epoch}'):
            # Time test
            # import time
            # # batch_data = next(iter(train_loader))
            # # inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            # # optimizer.zero_grad()
            # start_time = time.time()
            # # a bit less that 2sec/loop on laptop cpu
            # # for i in tqdm(range(100)):
            # #     outputs = model(inputs)
            # for i in tqdm(range(100)):
            #     next(iter(train_loader))
            # end_time = time.time()
            # print(f'model Time: {end_time - start_time}')
            # # start_time = time.time()
            # # for i, b in enumerate(train_loader):
            # #     if i == 99:
            # #         break
            # #     continue
            # # end_time = time.time()
            # # print(f'Loop Time: {end_time - start_time}')
            # exit()
            ctr_train_iter = None
            if ctr_train_loader:
                ctr_train_iter = iter(ctr_train_loader)
            for batch_data in train_loader:
                # with torch.autograd.profiler.profile(use_cuda=False) as prof:
                step += 1
                optimizer.zero_grad()
                inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
                if output_spatial_size is None:
                    output_spatial_size = inputs.shape
                    max_distance = torch.as_tensor(
                        [torch.linalg.norm(torch.as_tensor(output_spatial_size, dtype=torch.float16))])
                outputs = model(inputs)
                # TODO smoothing?
                y = labels[:, :1, :, :, :]
                if label_smoothing:
                    s = .1
                    y = y * (1 - s) + 0.5 * s

                if training_loss_fct.lower() == 'BCE':
                    loss = BCE(outputs, y)
                elif training_loss_fct.lower() == 'tversky_loss':
                    loss = tversky_function(outputs, y)
                elif training_loss_fct.lower() == 'dist_dice':
                    loss = loss_function(outputs, y)
                    # Just trying some dark magic
                    distance, _ = hausdorff_metric(y_pred=post_trans(outputs), y=y)
                    # distance, _ = surface_metric(y_pred=post_trans(outputs), y=labels[:, :1, :, :, :])
                    distance = torch.minimum(distance, max_distance)
                    print(f'Distance {distance}')
                    loss += torch.mean(distance)
                else:
                    loss = loss_function(outputs, y)
                controls_loss = 0
                controls_loss_str = ''
                # inputs_controls = None
                # labels_controls = None
                # outputs_controls = None
                if ctr_train_iter:
                    batch_data_controls = next(ctr_train_iter)
                    inputs_controls = batch_data_controls['image'].to(device)
                    # labels_controls = torch.zeros_like(inputs_controls).to(device)
                    outputs_controls = model(inputs_controls)
                    batch_mean = torch.mean(outputs_controls[:, :1, :, :, :], 0)
                    batch_mean_sigmoid = torch.sigmoid(batch_mean)
                    # It's basically torch.mean(torch.sigmoid(normal_logits) > 0.5)
                    controls_vol = utils.volume_metric(batch_mean_sigmoid,
                                                       sigmoid=False, discrete=True)
                    controls_loss = torch.mean(batch_mean_sigmoid) * control_weight_factor
                    # controls_loss = utils.percent_vox_loss(outputs_controls[:, :1, :, :, :], divide_max_vox=100)
                    # controls_loss = controls_vol
                    ctr_loss += [controls_loss]
                    ctr_vol += [controls_vol]
                    controls_loss_str = f'Controls loss: {controls_loss}, controls volume: {controls_vol}'
                    writer.add_scalar('control_loss', controls_loss.item(), batches_per_epoch * epoch + step)
                    writer.add_scalar('mean_control_vol', controls_vol, batches_per_epoch * epoch + step)
                loss = loss + controls_loss
                # TODO check that
                # distance, _ = surface_metric(y_pred=Activations(sigmoid=True)(outputs), y=labels[:, :1, :, :, :])
                # hausdorff, _ = hausdorff_metric(y_pred=post_trans(outputs), y=labels[:, :1, :, :, :])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # PRINTS
                print(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}, ' +
                      controls_loss_str
                      # f'| tversky_loss: {tversky_function(outputs, y).item():.4f}'
                      # f'| dicefocal: {df_loss(outputs, y).item():.4f}'
                      # f'| BCE: {BCE(outputs, y).item():.4f}'
                      # f'| Surface distance: {distance.item():.4f}'
                      # f'| Hausdorff distance: {hausdorff.item():.4f}'
                      # f'| BCE: {BCE(outputs, y, reduction="mean"):.4f}'
                      # f'| focal_loss: {focal_function(outputs, y).item():.4f}'
                      )
                writer.add_scalar('train_loss', loss.item(), batches_per_epoch * epoch + step)
                # print(prof.total_average())
                # print(prof.key_averages().table(row_limit=0))
            epoch_loss /= step
            print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')
            # ########## training EPOCH LEVEL WRITER ###########
            writer.add_scalar('epoch_train_loss', epoch_loss, epoch + 1)
            writer.add_scalar('epoch_ctr_loss', torch.mean(torch.tensor(ctr_loss), dtype=torch.float), epoch + 1)
            writer.add_scalar('epoch_ctr_volume', torch.mean(torch.tensor(ctr_vol), dtype=torch.float), epoch + 1)
            # if (epoch + 1) % val_interval == 1:
            """
            Validation loop
            """
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    metric_sum = 0.0
                    metric_count = 0
                    img_count = 0
                    meh_count = 0
                    trash_count = 0
                    distance_sum = 0.0
                    distance_count = 0
                    controls_vol = None
                    trash_seg_paths_list = []
                    if controls_lists:
                        controls_vol = 0
                    val_dice_list = []
                    loss_list = []
                    ctr_loss = []
                    ctr_vol = []
                    step = 0
                    pbar = tqdm(val_loader, desc=f'Val[{epoch + 1}] avg_metric:[N/A]')
                    controls_mean_loss = -1
                    controls_mean_vol = -1
                    ctr_val_iter = None
                    if ctr_val_loader:
                        ctr_val_iter = iter(ctr_val_loader)
                    for val_data in pbar:
                        step += 1
                        inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
                        outputs = model(inputs)
                        controls_loss = 0
                        if ctr_val_iter:
                            batch_data_controls = next(ctr_val_iter)
                            inputs_controls = batch_data_controls['image'].to(device)
                            # labels_controls = torch.zeros_like(inputs_controls).to(device)
                            outputs_controls = model(inputs_controls)
                            batch_mean = torch.mean(outputs_controls[:, :1, :, :, :], 0)
                            batch_mean_sigmoid = torch.sigmoid(batch_mean)
                            # It's basically torch.mean(torch.sigmoid(normal_logits) > 0.5)
                            controls_vol = utils.volume_metric(batch_mean_sigmoid,
                                                               sigmoid=False, discrete=True)
                            controls_loss = torch.mean(batch_mean_sigmoid) * control_weight_factor
                            ctr_loss += [controls_loss]
                            ctr_vol += [controls_vol]

                        outputs = post_trans(outputs)
                        dice_value, _ = dice_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
                        distance, _ = hausdorff_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
                        distance = torch.minimum(distance, max_distance)
                        loss = val_loss_function(outputs, labels[:, :1, :, :, :]).cpu()
                        if val_loss_fct == 'dist':
                            # In that case we want the distance to be smaller
                            metric_select_fct = lt
                            metric = distance
                        elif val_loss_fct == 'dist_dice':
                            # In that case we want the loss to be smaller
                            metric_select_fct = lt
                            metric = distance + loss
                        elif val_loss_fct == 'dice_plus_ctr':
                            # In that case we want the loss to be smaller
                            metric_select_fct = lt
                            # TODO meh?
                            metric = loss + controls_vol
                        else:
                            metric = dice_value.item()

                        # distance, _ = surface_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
                        # TODO the len(value) thing is really confusing and most likely useless here get rid of it!
                        distance_sum += distance.item() * len(distance)
                        distance_count += len(distance)

                        val_dice_list.append(dice_value.item())
                        metric_count += len(dice_value)
                        metric_sum += metric.item() * len(dice_value)

                        loss_list.append(loss.item())
                        if dice_value.item() > val_meh_thr:
                            img_count += 1
                        elif dice_value.item() > val_trash_thr:
                            meh_count += 1
                        else:
                            if epoch > 25:
                                p = val_data['image_meta_dict']['filename_or_obj'][0]
                                trash_seg_paths_list.append(p)
                                if p in trash_seg_path_count_dict:
                                    trash_seg_path_count_dict[p] += 1
                                else:
                                    trash_seg_path_count_dict[p] = 0
                            trash_count += 1
                        pbar.set_description(f'Val[{epoch}] avg_loss:[{metric_sum / metric_count}]')
                    mean_metric = metric_sum / metric_count
                    val_mean_loss = np.mean(loss_list)
                    mean_dice = np.mean(np.array(val_dice_list))
                    median = np.median(np.array(val_dice_list))
                    std = np.std(np.array(val_dice_list))
                    min_score = np.min(np.array(val_dice_list))
                    max_score = np.max(np.array(val_dice_list))
                    if ctr_loss:
                        controls_mean_loss = torch.mean(torch.tensor(ctr_loss, dtype=torch.float))
                        controls_mean_vol = torch.mean(torch.tensor(ctr_vol, dtype=torch.float))

                    writer.add_scalar('val_mean_metric', mean_metric, epoch + 1)
                    writer.add_scalar('val_mean_dict', mean_dice, epoch + 1)
                    writer.add_scalar('val_distance', distance_sum / distance_count, epoch + 1)
                    writer.add_scalar('val_mean_loss', val_mean_loss, epoch + 1)
                    writer.add_scalar('trash_img_nb', trash_count, epoch + 1)
                    writer.add_scalar('val_ctr_loss', torch.mean(torch.tensor(ctr_loss), dtype=torch.float), epoch + 1)
                    writer.add_scalar('val_ctr_volume', torch.mean(torch.tensor(ctr_vol), dtype=torch.float), epoch + 1)
                    writer.add_scalar('val_median_metric', median, epoch + 1)
                    writer.add_scalar('val_min_metric', min_score, epoch + 1)
                    writer.add_scalar('val_max_metric', max_score, epoch + 1)
                    writer.add_scalar('val_std_metric', std, epoch + 1)
                    # TODO check if that corresponds
                    df.loc[epoch + 1] = pd.Series({
                        'avg_train_loss': epoch_loss,
                        'val_mean_metric': mean_metric,
                        'val_mean_dice': mean_dice,
                        'val_distance': distance_sum / distance_count,
                        'trash_img_nb': trash_count,
                        'val_mean_loss': val_mean_loss,
                        'val_median_metric': median,
                        'val_std_metric': std,
                        'meh_img_nb': meh_count,
                        'good_img_nb': img_count,
                        'val_min_metric': min_score,
                        'val_max_metric': max_score,
                        # 'composite1': mean_metric + distance_sum / distance_count,
                        # 'controls_metric': controls_mean_loss,
                        'controls_vol': torch.mean(torch.tensor(ctr_vol), dtype=torch.float),
                        'val_best_mean_metric': 0
                    })
                    # TODO maybe find a better way so it would also save the first epoch. Even though it is no big deal
                    if epoch == 0:
                        best_metric = mean_metric
                    if metric_select_fct(mean_metric, best_metric):
                        best_metric = mean_metric
                        best_distance = distance_sum / distance_count
                        best_avg_loss = val_mean_loss
                        best_metric_epoch = epoch + 1
                        utils.save_checkpoint(model, epoch + 1, optimizer, output_fold_dir,
                                              'best_metric_model_segmentation3d_array_epo.pth')
                        print('saved new best metric model')
                        str_best_epoch = (
                            f'Best epoch {best_metric_epoch} '
                            f'metric {best_metric:.4f}/dist {best_distance}/avgloss {best_avg_loss}'
                        )
                        writer.add_scalar('val_best_mean_metric', mean_metric, epoch + 1)
                        df.at[epoch + 1, 'val_best_mean_metric'] = mean_metric
                        if trash_seg_paths_list:
                            with open(trash_list_path, 'a+') as f:
                                f.write(','.join([str(epoch + 1)] + trash_seg_paths_list) + '\n')
                    best_epoch_count = epoch + 1 - best_metric_epoch
                    str_img_count = (
                            f'Trash (<{val_trash_thr}|'.rjust(12, ' ') +
                            f'Meh (<{val_meh_thr})|'.rjust(12, ' ') + f'Good\n'.rjust(12, ' ') +
                            f'{trash_count}|'.rjust(12, ' ') + f'{meh_count}|'.rjust(12, ' ') +
                            f'{img_count}'.rjust(12, ' ') + '\n\n'
                    )
                    str_current_epoch = (
                            f'[Fold: {fold}]Current epoch: {epoch + 1} current mean metric: {mean_metric:.4f}\n'
                            f'and an average distance of [{distance_sum / distance_count}];\n'
                            f'Controls loss [{controls_mean_loss}] / volume[{controls_mean_vol}] ;\n\n'
                            + str_img_count + str_best_epoch
                    )
                    print(str_current_epoch)
                    print(f'It has been [{best_epoch_count}] since a best epoch has been found'
                          f'\nThe training will stop after [{stop_best_epoch}] epochs without improvement')
                    if stop_best_epoch != -1:
                        if best_epoch_count > stop_best_epoch:
                            print(f'More than {stop_best_epoch} without improvement')
                            df.to_csv(Path(output_fold_dir, 'perf_measures.csv'), columns=perf_measure_names)
                            print(f'train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
                            writer.close()
                            break
                    # utils.save_checkpoint(model, epoch + 1, optimizer, output_dir)
        df.to_csv(Path(output_fold_dir, f'perf_measures_{fold}.csv'), columns=perf_measure_names)
        with open(Path(output_fold_dir, f'trash_img_count_dict_{fold}.json'), 'w+') as j:
            json.dump(trash_seg_path_count_dict, j, indent=4)
        print(f'train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
        writer.close()

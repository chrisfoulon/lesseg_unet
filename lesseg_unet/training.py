import os
import shutil
import logging
from pathlib import Path
from typing import Sequence, Tuple, Union

import pandas as pd
import numpy as np
import monai
import torch
from monai.metrics import DiceMetric
from monai.data import Dataset
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from lesseg_unet import data_loading, net, utils, transformations


def init_training_data(img_path_list: Sequence,
                       seg_path_list: Sequence,
                       img_pref: str = None,
                       transform_dict=None,
                       train_val_percentage: float = 75) -> Tuple[monai.data.Dataset, monai.data.Dataset]:
    logging.info('Listing input files to be loaded')
    train_files, val_files = data_loading.create_file_dict_lists(img_path_list, seg_path_list, img_pref,
                                                                 train_val_percentage)
    logging.info('Create transformations')
    train_img_transforms = transformations.segmentation_train_transformd(transform_dict)
    val_img_transforms = transformations.segmentation_val_transformd(transform_dict)
    # define dataset, data loader
    logging.info('Create training monai datasets')
    train_ds = Dataset(train_files, transform=train_img_transforms)

    # define dataset, data loader
    logging.info('Create validation actual monai datasets')
    val_ds = Dataset(val_files, transform=val_img_transforms)
    # We check if both the training and validation dataloaders can be created and used without immediate errors
    logging.info('Checking data loading')
    data_loading.data_loader_checker_first(train_ds, 'training')
    data_loading.data_loader_checker_first(val_ds, 'validation')
    logging.info('Init training done.')
    return train_ds, val_ds


def training_loop(img_path_list: Sequence,
                  seg_path_list: Sequence,
                  output_dir: Union[str, bytes, os.PathLike],
                  img_pref: str = None,
                  transform_dict=None,
                  device: str = None,
                  batch_size: int = 10,
                  epoch_num: int = 50,
                  dataloader_workers: int = 4,
                  num_nifti_save=25):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    train_ds, val_ds = init_training_data(img_path_list, seg_path_list, img_pref, transform_dict=transform_dict)
    train_loader = data_loading.create_training_data_loader(train_ds, batch_size, dataloader_workers)
    val_loader = data_loading.create_validation_data_loader(val_ds, dataloader_workers=dataloader_workers)
    model = net.create_unet_model(device, net.default_unet_hyper_params)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    print('check ok')
    # for i in range(25):
    #     data = next(iter(train_loader))
    #     inputs, labels = data['image'], data['label']
    #     # i_data = inputs[0, 0, :, :, :].cpu().detach().numpy()
    #     # l_data = labels[0, 0, :, :, :].cpu().detach().numpy()
    #     utils.save_img_lbl_seg_to_png(
    #         inputs, '/home/tolhsadum/neuro_apps/data', 'validation_img_{}'.format(i), labels, None)
    #     utils.save_img_lbl_seg_to_nifti(
    #         inputs, labels, None, '/home/tolhsadum/neuro_apps/data', val_output_affine, i)
    # # if np.equal(i_data, l_data).all():
    # #     print('ok')
    # # else:
    # #     print('not ok')
    # exit()

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
    # # Code for saving (to be used in the validaiton vlock)
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
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    """
    Measure tracking init
    """
    writer = SummaryWriter(log_dir=str(output_dir))
    val_images_dir = Path(output_dir, 'val_images')
    if not val_images_dir.is_dir():
        val_images_dir.mkdir(exist_ok=True)
    trash_val_images_dir = Path(output_dir, 'trash_val_images')
    if not trash_val_images_dir.is_dir():
        trash_val_images_dir.mkdir(exist_ok=True)
    best_val_images_dir = Path(output_dir, 'best_epoch_val_images')
    if not best_val_images_dir.is_dir():
        best_val_images_dir.mkdir(exist_ok=True)
    best_trash_images_dir = Path(output_dir, 'best_epoch_val_images')
    if not best_trash_images_dir.is_dir():
        best_trash_images_dir.mkdir(exist_ok=True)
    perf_measure_names = ['avg_train_loss',
                          'val_mean_dice',
                          'val_median_dice',
                          'val_std_dice',
                          'trash_img_nb',
                          'val_min_dice',
                          'val_max_dice',
                          'val_best_mean_dice']
    df = pd.DataFrame(columns=perf_measure_names)
    batches_per_epoch = len(train_loader)
    """
    Training loop
    """
    for epoch in range(epoch_num):
        print('-' * 10)
        print(f'epoch {epoch + 1}/{epoch_num}')
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            # print('inputs size: {}'.format(inputs.size()))
            # print('labels size: {}'.format(labels.size()))
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs size: {}'.format(outputs.size()))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f'{step}/{batches_per_epoch}, train_loss: {loss.item():.4f}')
            writer.add_scalar('train_loss', loss.item(), batches_per_epoch * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')

        # if (epoch + 1) % val_interval == 1:
        """
        Validation loop
        """
        if (epoch + 1) % val_interval == 0:
            for f in val_images_dir.iterdir():
                os.remove(f)
            for f in trash_val_images_dir.iterdir():
                os.remove(f)
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                img_count = 0
                trash_count = 0
                # img_max_num = len(train_ds) + len(val_ds)
                # inputs = None
                # labels = None
                # outputs = None
                # saver = NiftiSaver(output_dir=output_dir)
                val_score_list = []
                for val_data in val_loader:
                    inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
                    outputs = model(inputs)
                    outputs = post_trans(outputs)

                    value, _ = dice_metric(y_pred=outputs, y=labels)
                    val_score_list.append(value.item())
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                    if best_metric > 0.7:
                        if value.item() * len(value) > 0.7:
                            print('Saving good image #{}'.format(img_count))
                            img_count += 1
                            if img_count < num_nifti_save:
                                utils.save_img_lbl_seg_to_png(
                                    inputs, val_images_dir, 'validation_img_{}'.format(img_count), labels, outputs)
                                utils.save_img_lbl_seg_to_nifti(
                                    inputs, labels, outputs, val_images_dir, val_output_affine, img_count)
                        if value.item() * len(value) < 0.1:
                            trash_count += 1
                            print('Saving trash image #{}'.format(trash_count))
                            if trash_count < num_nifti_save:
                                utils.save_img_lbl_seg_to_png(
                                    inputs, trash_val_images_dir, 'trash_img_{}'.format(trash_count), labels, outputs)
                                utils.save_img_lbl_seg_to_nifti(
                                    inputs, labels, outputs, trash_val_images_dir, val_output_affine, trash_count)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                median = np.median(np.array(val_score_list))
                std = np.std(np.array(val_score_list))
                min_score = np.min(np.array(val_score_list))
                max_score = np.max(np.array(val_score_list))
                writer.add_scalar('val_mean_dice', metric, epoch + 1)
                writer.add_scalar('val_median_dice', median, epoch + 1)
                writer.add_scalar('trash_img_nb', trash_count, epoch + 1)
                writer.add_scalar('val_min_dice', min_score, epoch + 1)
                writer.add_scalar('val_max_dice', max_score, epoch + 1)
                writer.add_scalar('val_std_dice', std, epoch + 1)
                df.loc[0] = pd.Series({
                    'avg_train_loss': epoch_loss,
                    'val_mean_dice': metric,
                    'val_median_dice': median,
                    'val_std_dice': std,
                    'trash_img_nb': trash_count,
                    'val_min_dice': min_score,
                    'val_max_dice': max_score,
                    'val_best_mean_dice': 0
                })
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               Path(output_dir,
                                    'best_metric_model_segmentation3d_array_epo_{}.pth'.format(best_metric_epoch)))
                    print('saved new best metric model')
                    writer.add_scalar('val_best_mean_dice', metric, epoch + 1)
                    df.at[epoch + 1, 'val_best_mean_dice'] = metric
                    print('Replacing best images')
                    for f in best_val_images_dir.iterdir():
                        os.remove(f)
                    for f in best_trash_images_dir.iterdir():
                        os.remove(f)
                    for f in val_images_dir.iterdir():
                        shutil.copy(f, best_val_images_dir)
                    for f in trash_val_images_dir.iterdir():
                        shutil.copy(f, best_trash_images_dir)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                    # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                    # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
                    # plot_2d_or_3d_image(inputs, epoch + 1, writer, index=0, tag="image")
                    # plot_2d_or_3d_image(labels, epoch + 1, writer, index=0, tag="label")
                    # plot_2d_or_3d_image(outputs, epoch + 1, writer, index=0, tag="output")
                print(
                    'current epoch: {} current mean dice: {:.4f} with {} '
                    'trash images best mean dice: {:.4f} at epoch {}'.format(
                        epoch + 1, metric, trash_count, best_metric, best_metric_epoch
                    )
                )

                utils.save_checkpoint(model, epoch + 1, optimizer, output_dir)
    df.to_csv(Path(output_dir, 'perf_measures.csv'), columns=perf_measure_names)
    print(f'train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
    writer.close()

import os
import json
from pathlib import Path
from typing import Sequence, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from lesseg_unet import data_loading, utils, net, transformations
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)


def segmentation_loop(img_path_list: Sequence,
                      output_dir: Union[str, bytes, os.PathLike],
                      checkpoint_path: Union[str, bytes, os.PathLike],
                      img_pref: str = None,
                      transform_dict: dict = None,
                      device: str = None,
                      batch_size: int = 1,
                      dataloader_workers: int = 8,
                      original_size=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    original_size = original_size
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    val_ds = data_loading.init_segmentation(img_path_list, img_pref, transform_dict)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)
    unet_hyper_params = net.default_unet_hyper_params
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        unet_hyper_params['in_channels'] = 4
    model = utils.load_eval_from_checkpoint(checkpoint_path, device, unet_hyper_params)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    model.eval()
    img_count = 0
    img_vol_dict = {}
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc=f'Segmentation '):
            img_count += 1
            inputs = val_data['image'].to(device)
            outputs = model(inputs)
            outputs = post_trans(outputs)
            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            vol_output = utils.volume_metric(outputs, False, False)
            input_filename += f'_v{vol_output}v'
            if original_size:
                output_dict_data = deepcopy(val_data)
                output_dict_data['image'] = val_data['image'].to(device)[0]
                output_dict_data['label'] = outputs[0]
                inverted_output_dict = val_ds.transform.inverse(output_dict_data)
                inv_inputs = inverted_output_dict['image']
                inv_outputs = inverted_output_dict['label']
                inputs_np = inv_inputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_inputs, torch.Tensor) \
                    else inv_inputs[0, :, :, :]
                outputs_np = inv_outputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_outputs, torch.Tensor) \
                    else inv_outputs[0, :, :, :]
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, output_dir,
                #     '{}_segmentation_{}'.format(input_filename, img_count), outputs_np)
                tmp = None
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            else:
                inputs_np = inputs[0, 0, :, :, :].cpu().detach().numpy() if isinstance(inputs, torch.Tensor) \
                    else inputs[0, :, :, :]
                outputs_np = outputs[0, 0, :, :, :].cpu().detach().numpy() if isinstance(outputs, torch.Tensor) \
                    else outputs[0, :, :, :]

                tmp = None
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))

            img_vol_dict[output_path_list[-1]] = vol_output
            with open(Path(output_dir, f'output_image_volumes.json'), 'w+') as j:
                json.dump(img_vol_dict, j, indent=4)
            del inputs
            del outputs
            del inputs_np
            del outputs_np


def validation_loop(img_path_list: Sequence,
                    seg_path_list: Sequence,
                    output_dir: Union[str, bytes, os.PathLike],
                    checkpoint_path: Union[str, bytes, os.PathLike],
                    img_pref: str = None,
                    transform_dict: dict = None,
                    device: str = None,
                    batch_size: int = 1,
                    dataloader_workers: int = 8,
                    # num_nifti_save: int = -1,
                    bad_dice_treshold: float = 0.1,
                    train_val_percentage=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])

    _, val_ds = data_loading.init_training_data(img_path_list, seg_path_list, img_pref,
                                                transform_dict=transform_dict,
                                                train_val_percentage=train_val_percentage)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)
    unet_hyper_params = net.default_unet_hyper_params
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        unet_hyper_params['in_channels'] = 4
    model = utils.load_eval_from_checkpoint(checkpoint_path, device, unet_hyper_params)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    # start a typical PyTorch training
    # val_interval = 1
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = list()
    metric_values = list()
    # val_save_thr = 0.7
    """
    Measure tracking init
    """
    val_images_dir = Path(output_dir, 'val_images')
    trash_val_images_dir = Path(output_dir, 'trash_val_images')
    if not val_images_dir.is_dir():
        val_images_dir.mkdir(exist_ok=True)
    for f in val_images_dir.iterdir():
        os.remove(f)
    if not trash_val_images_dir.is_dir():
        trash_val_images_dir.mkdir(exist_ok=True)
    for f in trash_val_images_dir.iterdir():
        os.remove(f)
    perf_measure_names = ['val_mean_dice',
                          'val_median_dice',
                          'val_std_dice',
                          'trash_img_nb',
                          'val_min_dice',
                          'val_max_dice']
    df = pd.DataFrame(columns=perf_measure_names)
    img_vol_dict = {}
    model.eval()
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        img_count = 0
        trash_count = 0
        # img_max_num = len(train_ds) + len(val_ds)
        val_score_list = []
        for val_data in tqdm(val_loader, desc=f'Validation '):
            inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            outputs = model(inputs)
            outputs = post_trans(outputs)
            vol_output = utils.volume_metric(outputs, False, False)
            input_filename += f'_v{vol_output}v'
            output_dict_data = deepcopy(val_data)
            value = dice_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
            val_score_list.append(value.item())
            metric_count += len(value)
            metric_sum += value.item() * len(value)

            val_data['image'] = val_data['image'].to(device)[0]
            val_data['label'] = val_data['label'].to(device)[0]
            output_dict_data['image'] = val_data['image']
            output_dict_data['label'] = outputs[0]

            inverted_dict = val_ds.transform.inverse(val_data)
            inv_inputs, inv_labels = inverted_dict['image'], inverted_dict['label']
            inv_outputs = val_ds.transform.inverse(output_dict_data)['label']
            inputs_np = inv_inputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_inputs, torch.Tensor) \
                else inv_inputs[0, :, :, :]
            labels_np = inv_labels[0, :, :, :].cpu().detach().numpy() if isinstance(inv_labels, torch.Tensor) \
                else inv_labels[0, :, :, :]
            outputs_np = inv_outputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_outputs, torch.Tensor) \
                else inv_outputs[0, :, :, :]
            # inputs_np = inv_inputs[0, :, :, :].detach().numpy()
            # labels_np = inv_labels[0, :, :, :].detach().numpy()
            # outputs_np = inv_outputs[0, :, :, :].detach().numpy()
            if value.item() < bad_dice_treshold:
                trash_count += 1
                # print('Saving trash image #{}'.format(trash_count))
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, trash_val_images_dir,
                #     '{}_trash_img_{}'.format(input_filename, trash_count), labels_np, outputs_np)
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, labels_np, outputs_np, trash_val_images_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(trash_count)))
            else:
                img_count += 1
                # print('Saving good image #{}'.format(img_count))
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, val_images_dir,
                #     '{}_validation_{}'.format(input_filename, img_count), labels_np, outputs_np)
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, labels_np, outputs_np, val_images_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            img_vol_dict[output_path_list[-1]] = vol_output
        metric = metric_sum / metric_count
        metric_values.append(metric)
        median = np.median(np.array(val_score_list))
        std = np.std(np.array(val_score_list))
        min_score = np.min(np.array(val_score_list))
        max_score = np.max(np.array(val_score_list))
        df.loc[0] = pd.Series({
            'val_mean_dice': metric,
            'val_median_dice': median,
            'val_std_dice': std,
            'trash_img_nb': trash_count,
            'val_min_dice': min_score,
            'val_max_dice': max_score,
            'val_best_mean_dice': 0
        })
    with open(Path(output_dir, f'output_image_volumes.json'), 'w+') as j:
        json.dump(img_vol_dict, j, indent=4)
    df.to_csv(Path(output_dir, 'val_perf_measures.csv'), columns=perf_measure_names)

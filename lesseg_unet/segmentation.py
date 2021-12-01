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
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
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
                      original_size=True,
                      clamping: tuple = None,
                      segmentation_area=True,
                      **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    original_size = original_size
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    val_ds = data_loading.init_segmentation(img_path_list, img_pref, transform_dict, clamping=clamping)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)
    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    if training_img_size is None:
        training_img_size = utils.get_img_size(img_path_list[0])
    model_name = 'unet'
    hyper_params = net.default_unet_hyper_params
    if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
        hyper_params = net.default_unetr_hyper_params
        hyper_params['img_size'] = training_img_size
        model_name = 'unetr'
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        hyper_params['in_channels'] = 4
    model = utils.load_model_from_checkpoint(checkpoint_path, device, hyper_params, model_name=model_name)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    model.eval()
    les_area_finder = None
    if segmentation_area:
        les_area_finder = utils.LesionAreaFinder()
    img_count = 0
    img_vol_dict = {}
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc=f'Segmentation '):
            img_count += 1
            inputs = val_data['image'].to(device)
            # TODO
            val_outputs = sliding_window_inference(inputs, training_img_size,
                                                   1, model, overlap=0.8)
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]

            vol_output = utils.volume_metric(val_output_convert[0], False, False)

            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            input_filename += f'_v{vol_output}v'
            # TODO
            # if 'entropy' in kwargs and (kwargs['entropy'] == 'True' or kwargs['entropy'] == 1):
            #     print(utils.entropy_metric(val_outputs_list[0], sigmoid=True))
            #     exit()
            #     input_filename += f'_e{utils.entropy_metric(val_outputs_list[0], sigmoid=True)}e'
            if original_size:
                output_dict_data = deepcopy(val_data)
                output_dict_data['image'] = val_data['image'].to(device)[0]
                inverted_output_dict = val_ds.transform.inverse(output_dict_data)
                inv_inputs = inverted_output_dict['image']
                output_dict_data['image'] = val_output_convert[0]
                inverted_output_dict = val_ds.transform.inverse(output_dict_data)
                inv_outputs = inverted_output_dict['image']
                inputs_np = inv_inputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_inputs, torch.Tensor) \
                    else inv_inputs[0, :, :, :]
                outputs_np = inv_outputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_outputs, torch.Tensor) \
                    else inv_outputs[0, :, :, :]
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, output_dir,
                #     '{}_segmentation_{}'.format(input_filename, img_count), outputs_np)
                tmp = None
                if les_area_finder is not None:
                    if vol_output == 0:
                        output_subdir = Path(output_dir, 'empty_prediction')
                    else:
                        cluster_name = les_area_finder.get_img_area(outputs_np)
                        output_subdir = Path(output_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                else:
                    output_subdir = output_dir
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_subdir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            else:
                inputs_np = inputs[0, 0, :, :, :].cpu().detach().numpy() if isinstance(
                    inputs, torch.Tensor) else inputs[0, :, :, :]
                outputs_np = val_output_convert[0, 0, :, :, :].cpu().detach().numpy() if isinstance(
                    val_output_convert, torch.Tensor) else val_output_convert[0, :, :, :]

                tmp = None
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))

            img_vol_dict[output_path_list[-1]] = vol_output
            with open(Path(output_dir, f'__output_image_volumes.json'), 'w+') as j:
                json.dump(img_vol_dict, j, indent=4)
            pd.DataFrame().from_dict(img_vol_dict, orient='index').to_csv(
                Path(output_dir, f'__output_image_volumes.csv'))
            del inputs
            del val_output_convert
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
                    bad_dice_treshold: float = 0.1,
                    clamping: tuple = None,
                    segmentation_area=True,
                    **kwargs
                    ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])

    _, val_ds = data_loading.init_training_data(img_path_list, seg_path_list, img_pref,
                                                transform_dict=transform_dict,
                                                train_val_percentage=0, clamping=clamping)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)

    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    if training_img_size is None:
        training_img_size = utils.get_img_size(img_path_list[0])
    model_name = 'unet'
    hyper_params = net.default_unet_hyper_params
    if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
        hyper_params = net.default_unetr_hyper_params
        hyper_params['img_size'] = training_img_size
        model_name = 'unetr'
    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        hyper_params['in_channels'] = 4
    model = utils.load_model_from_checkpoint(checkpoint_path, device, hyper_params, model_name=model_name)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    les_area_finder = None
    if segmentation_area:
        les_area_finder = utils.LesionAreaFinder()
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
        img_count = 0
        trash_count = 0
        # img_max_num = len(train_ds) + len(val_ds)
        val_score_list = []
        for val_data in tqdm(val_loader, desc=f'Validation '):
            inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            # outputs = model(inputs)
            # outputs = post_trans(outputs)

            masks_only_val_labels = labels[:, :1, :, :, :]
            # TODO
            val_outputs = sliding_window_inference(inputs, training_img_size,
                                                   1, model, overlap=0.8)
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=masks_only_val_labels)
            dice = dice_metric.aggregate().item()

            vol_output = utils.volume_metric(val_output_convert[0], False, False)
            input_filename += f'_v{vol_output}v'
            # if 'entropy' in kwargs and (kwargs['entropy'] == 'True' or kwargs['entropy'] == 1):
            #     input_filename += f'_e{utils.entropy_metric(val_outputs_list[0], sigmoid=True)}e'
            output_dict_data = deepcopy(val_data)
            # value = dice_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
            val_score_list.append(dice)

            val_data['image'] = val_data['image'].to(device)[0]
            val_data['label'] = val_data['label'].to(device)[0]
            output_dict_data['image'] = val_data['image']
            output_dict_data['label'] = val_output_convert[0]

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
            if dice < bad_dice_treshold:
                trash_count += 1
                # print('Saving trash image #{}'.format(trash_count))
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, trash_val_images_dir,
                #     '{}_trash_img_{}'.format(input_filename, trash_count), labels_np, outputs_np)
                if les_area_finder is not None:
                    cluster_name = les_area_finder.get_img_area(outputs_np)
                    output_subdir = Path(trash_val_images_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                else:
                    output_subdir = trash_val_images_dir
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, labels_np, outputs_np, output_subdir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(trash_count)))
            else:
                img_count += 1
                # print('Saving good image #{}'.format(img_count))
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, val_images_dir,
                #     '{}_validation_{}'.format(input_filename, img_count), labels_np, outputs_np)
                if les_area_finder is not None:
                    if vol_output == 0:
                        output_subdir = Path(val_images_dir, 'empty_prediction')
                    else:
                        cluster_name = les_area_finder.get_img_area(outputs_np)
                        output_subdir = Path(val_images_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                else:
                    output_subdir = val_images_dir
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, labels_np, outputs_np, output_subdir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            img_vol_dict[output_path_list[-1]] = vol_output
        mean_metric = np.mean(np.array(val_score_list))
        median = np.median(np.array(val_score_list))
        std = np.std(np.array(val_score_list))
        min_score = np.min(np.array(val_score_list))
        max_score = np.max(np.array(val_score_list))
        df.loc[0] = pd.Series({
            'val_mean_dice': mean_metric,
            'val_median_dice': median,
            'val_std_dice': std,
            'trash_img_nb': trash_count,
            'val_min_dice': min_score,
            'val_max_dice': max_score,
            'val_best_mean_dice': 0
        })
        # Not necessary but I prefer it there
        dice_metric.reset()
    with open(Path(output_dir, f'__output_image_volumes.json'), 'w+') as j:
        json.dump(img_vol_dict, j, indent=4)
    pd.DataFrame().from_dict(img_vol_dict, orient='index').to_csv(Path(output_dir, f'__output_image_volumes.csv'))
    df.to_csv(Path(output_dir, 'val_perf_measures.csv'), columns=perf_measure_names)

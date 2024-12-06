import os
import json
from pathlib import Path
from typing import Sequence, Union, List
from copy import deepcopy
import logging
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
from lesseg_unet.utils import save_tensor_to_nifti
from tqdm import tqdm
import torch
from bcblib.tools.general_utils import save_json
from lesseg_unet import data_loading, utils, net, transformations
from monai.transforms.utils import allow_missing_keys_mode
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, list_data_collate
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    SaveImaged
)

from lesseg_unet.loss_and_metric import DistanceRatioMetric


def get_images_np_on_cpu(batch_data, batch_images, compute_device, dataset_obj, output_size=None):
    # batch_data_image is typically batch_data['image'] or batch_data['label'] or outputs of the model
    if output_size is not None and output_size != batch_images.shape[-3:]:
        for i in batch_images.shape[0]:
            img = batch_images[i, ...].to(compute_device)
            inv_img = dataset_obj.transform.inverse(img)
            # The first dimension should already be 1 but as we want a 3D images as an output I preferred to ensure it
            ing_img__np = inv_img[0, :, :, :].cpu().detach().numpy() if isinstance(inv_img, torch.Tensor) \
                else inv_img[0, :, :, :]


def invert_transformations_from_dict(img_dict, val_ds, inv_label=False):
    inv_input_dict = val_ds.transform.inverse(img_dict)
    inv_input = inv_input_dict['image']
    label_np = np.array([])
    if inv_label:
        inv_label = inv_input_dict['label']
        label_np = inv_label[0, :, :, :].cpu().detach().numpy() if isinstance(
            inv_label, torch.Tensor) else inv_label[0, :, :, :]
    input_np = inv_input[0, :, :, :].cpu().detach().numpy() if isinstance(
        inv_input, torch.Tensor) else inv_input[0, :, :, :]
    return input_np, label_np


def segmentation(img_path_list: Sequence,
                 seg_path_list: Sequence,
                 output_dir: Union[str, bytes, os.PathLike],
                 checkpoint_list: List[Union[str, bytes, os.PathLike]],
                 img_pref: str = None,
                 image_cut_suffix=None,
                 transform_dict: dict = None,
                 device: Union[str, List[str]] = None,
                 batch_size: int = 1,
                 dataloader_workers: int = 8,
                 bad_dice_treshold: float = 0.1,
                 clamping: tuple = None,
                 segmentation_area=True,
                 ensemble_operation='mean',
                 **kwargs
                 ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if seg_path_list is None or not seg_path_list:
        perform_validation = False
    else:
        perform_validation = True

    if perform_validation:
        _, val_ds = data_loading.init_training_data(img_path_list, seg_path_list, img_pref,
                                                    transform_dict=transform_dict,
                                                    train_val_percentage=0, clamping=clamping)
        val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                                dataloader_workers=dataloader_workers)
    else:
        val_ds = data_loading.init_segmentation(img_path_list, img_pref, transform_dict, clamping=clamping)
        val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                                dataloader_workers=dataloader_workers)
    # Either the transformations contain a crop/pad/resize function with "spatial_size"
    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    # Or the transformations don't change the size of the input images
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
    # Could be used to send the models on different gpus
    # devices_repeated = utils.repeat_array_to_dim(devices, np.array(checkpoint_list).shape)

    models = [
        utils.load_model_from_checkpoint(torch.load(checkpoint_path, map_location="cpu"),
                                         device, None, model_name=model_name)
        for checkpoint_path in checkpoint_list
    ]
    for model in models:
        model.to(device)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    activation_fct = Activations(sigmoid=True)
    threshold_fct = AsDiscrete(threshold=0.5)
    try:
        ensemble_operation = getattr(torch, ensemble_operation)
    except AttributeError as e:
        raise Exception(f'ERROR: {ensemble_operation} is not an existing function from torch\n {e}')

    les_area_finder = None
    if segmentation_area:
        les_area_finder = utils.LesionAreaFinder()
    # start a typical PyTorch training
    # val_interval = 1
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = list()
    # metric_values = list()
    # val_save_thr = 0.7
    """
    Measure tracking init
    """
    dice_metric = None
    hausdorff_metric = None
    if perform_validation:
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
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
                              'val_mean_dist',
                              'pred_volume',
                              'val_median_dice',
                              'val_std_dice',
                              'trash_img_nb',
                              'val_min_dice',
                              'val_max_dice']
        global_dict = {}
    loop_df_columns = ['dice_metric', 'volume', 'distance']
    loop_df = pd.DataFrame(columns=loop_df_columns)
    img_vol_dict = {}
    for model in models:
        model.eval()
    with torch.no_grad():
        img_count = 0
        trash_count = 0
        # img_max_num = len(train_ds) + len(val_ds)
        val_score_list = []
        val_dist_list = []
        for batch in tqdm(val_loader, desc=f'Validation '):
            inputs = batch['image']
            batch_model_channel_image = torch.zeros((batch_size, len(models)) + inputs.shape[1:])
            # First we send the input image through every model
            for ind, model in enumerate(models):
                outputs = sliding_window_inference(
                    inputs, training_img_size, batch_size, model, overlap=0.8)
                batch_model_channel_image[:, ind, ...] = outputs
            # Then we calculate the ensemble average of the predictions from all the model for each image
            decollated_batch_ensembled = []
            for decollated_model_chan_img in decollate_batch(batch_model_channel_image):
                activation_decollated_model_chan_img = activation_fct(decollated_model_chan_img)
                # We calculate the ensemble prediction from the sigmoid values, before thresholding in discrete values
                decollated_batch_ensembled.append(
                    threshold_fct(ensemble_operation(activation_decollated_model_chan_img, dim=0))
                )

            # Here, decollated_batch_ensembled should have the dimensions as decollated_inputs (except for the channels)
            # decollated_batch_ensembled contains binary values
            output_dict_data = deepcopy(batch)
            for ind, decollated_image_dict in enumerate(output_dict_data):
                if 'labels' in batch:
                    decollated_image_dict['label'] = decollated_batch_ensembled[ind]
                else:
                    decollated_image_dict['image'] = decollated_batch_ensembled[ind]

            for img_dict, output_dict in zip(decollate_batch(batch), decollate_batch(output_dict_data)):
                output_affine = img_dict['image_meta_dict']['affine']
                input_filename = Path(img_dict['image_meta_dict']['filename_or_obj']).name.split('.nii')[0]
                input_filename += f'_v{vol_output}v'
                one_image_measures = {}
                if perform_validation:
                    label = img_dict['label']
                    # If perform_validation the output prediction was put at the "label" key of output_dict_data
                    output_pred = output_dict['label']
                    dice_metric(y_pred=output_pred, y=label)
                    dice = dice_metric.aggregate().item()
                    hausdorff_metric(y_pred=output_pred, y=label)
                    dist = hausdorff_metric.aggregate().item()
                    # Loop dataframe filling
                    one_image_measures = {'dice_metric': dice, 'distance': dist}
                    val_score_list.append(dice)
                    val_dist_list.append(dist)
                    # Maybe not necessary but I prefer it there
                    dice_metric.reset()
                    hausdorff_metric.reset()

                input_np, label_np = invert_transformations_from_dict(img_dict, val_ds, False)
                if 'labels' in batch:
                    _, output_np = invert_transformations_from_dict(output_dict, val_ds, False)
                else:
                    output_np, _ = invert_transformations_from_dict(output_dict, val_ds, False)
                vol_output = utils.volume_metric(output_np, False, False)
                if les_area_finder is not None:
                    if vol_output == 0:
                        output_subdir = Path(output_dir, 'empty_prediction')
                    else:
                        cluster_name = les_area_finder.get_img_area(output_np)
                        output_subdir = Path(output_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                else:
                    output_subdir = output_dir
                # The function returns the path of every image saved (input, [label], output)
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    input_np, label_np, output_np, output_subdir, output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
                # TODO img_vol_dict becomes useless
                img_vol_dict[output_path_list[-1]] = vol_output
                one_image_measures['volume'] = vol_output
                one_image_measures['image'] = img_vol_dict[output_path_list[-1]]
                loop_df = loop_df.append(one_image_measures, ignore_index=True)
            mean_metric = np.mean(np.array(val_score_list))
            median = np.median(np.array(val_score_list))
            std = np.std(np.array(val_score_list))
            min_score = np.min(np.array(val_score_list))
            max_score = np.max(np.array(val_score_list))
            global_df = global_dict.from_dict({
                'val_mean_dice': mean_metric,
                'val_mean_dist': np.mean(val_dist_list),
                'pred_volume': vol_output,
                'val_median_dice': median,
                'val_std_dice': std,
                'trash_img_nb': trash_count,
                'val_min_dice': min_score,
                'val_max_dice': max_score,
                'val_best_mean_dice': 0
            }, ignore_index=True)
        with open(Path(output_dir, f'__output_image_volumes.json'), 'w+') as j:
            json.dump(img_vol_dict, j, indent=4)
        pd.DataFrame().from_dict(img_vol_dict, orient='index').to_csv(
            Path(output_dir, f'__output_image_volumes.csv'))
        if perform_validation:
            global_df.to_csv(Path(output_dir, 'val_perf_global_measures.csv'), columns=perf_measure_names)
        loop_df.to_csv(Path(output_dir, 'val_perf_individual_measures.csv'), columns=loop_df_columns)


def segmentation_loop(img_path_list: Sequence,
                      output_dir: Union[str, bytes, os.PathLike],
                      checkpoint_path: Union[str, bytes, os.PathLike],
                      img_pref: str = None,
                      transform_dict: dict = None,
                      output_mode='segmentation',
                      device: str = None,
                      batch_size: int = 1,
                      dataloader_workers: int = 8,
                      original_size=True,
                      clamping: tuple = None,
                      segmentation_area=True,
                      use_parent_folder=False,
                      **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if transform_dict is None:
        transform_dict = checkpoint['transform_dict']
    logging.info(f'Torch device used for this segmentation: {str(device)}')
    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    val_ds = data_loading.init_segmentation(img_path_list, img_pref,
                                            transform_dict, clamping=clamping)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)
    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    if training_img_size is None:
        training_img_size = utils.get_img_size(img_path_list[0])

    hyper_params = net.default_unet_hyper_params
    if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
        hyper_params = net.default_unetr_hyper_params
        hyper_params['img_size'] = training_img_size

    if transform_dict is not None:
        for li in transform_dict:
            for d in transform_dict[li]:
                for t in d:
                    if t == 'CoordConvd' or t == 'CoordConvAltd':
                        hyper_params['in_channels'] = 4
    model = utils.load_model_from_checkpoint(checkpoint, device, checkpoint['hyper_params'],
                                             model_name=checkpoint['model_name'])
    model.to(device)
    if output_mode == 'sigmoid':
        post_trans = Compose([Activationsd(keys=['pred'], sigmoid=True),
                              Invertd(
                                  keys=['pred'],  # invert the `pred` data field, also support multiple fields
                                  transform=val_ds.transform,
                                  orig_keys='image',
                              )])
    elif output_mode == 'logits':
        post_trans = Compose([Invertd(
                                  keys=['pred'],  # invert the `pred` data field, also support multiple fields
                                  transform=val_ds.transform,
                                  orig_keys='image',
                              )])
    else:
        post_trans = Compose([Activationsd(keys=['pred'], sigmoid=True), AsDiscreted(keys=['pred'], threshold=0.5),
                              Invertd(
                                  keys=['pred'],  # invert the `pred` data field, also support multiple fields
                                  transform=val_ds.transform,
                                  orig_keys='image',
                              ),
                              # SaveImaged(keys="image", output_dir=output_subdir, output_postfix="seg", resample=False)
                              # ,
                              # SaveImaged(keys="pred", output_dir=output_subdir, output_postfix="seg", resample=False)
                              ])

    # TODO maybe have 2 sets of post transformations. The first sigmoid and the second binarises a **copy** of the first
    model.eval()
    les_area_finder = None
    if segmentation_area:
        les_area_finder = utils.LesionAreaFinder()
    img_count = 0
    img_vol_dict = {}
    input_output_paths_dict = {}
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc=f'Segmentation '):
            img_count += 1
            inputs = val_data['image'].to(device)
            with torch.cuda.amp.autocast():
                # if training_img_size < inputs size we need several patches to cover the inputs
                # if training_img_size > inputs sliding_window_inference pads for the inference and then crops back
                val_data['pred'] = sliding_window_inference(inputs, training_img_size,
                                                            1, model, overlap=0.8)
                val_data['pred'].applied_operations = deepcopy(val_data['image'].applied_operations)
                val_outputs_list = decollate_batch(val_data)
                val_output_convert = [
                    post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

            vol_output = utils.volume_metric(val_output_convert[0]['pred'], False, False)

            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            input_filename += f'_v{vol_output}v'
            # TODO
            # if 'entropy' in kwargs and (kwargs['entropy'] == 'True' or kwargs['entropy'] == 1):
            #     print(utils.entropy_metric(val_outputs_list[0], sigmoid=True))
            #     exit()
            #     input_filename += f'_e{utils.entropy_metric(val_outputs_list[0], sigmoid=True)}e'
            if original_size:
                inv_outputs = val_output_convert[0]['pred']
                inputs_np = nib.load(Path(val_data['image_meta_dict']['filename_or_obj'][0])).get_fdata()
                outputs_np = inv_outputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_outputs, torch.Tensor) \
                    else inv_outputs[0, :, :, :]
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, output_dir,
                #     '{}_segmentation_{}'.format(input_filename, img_count), outputs_np)
                tmp = np.array([])
                if les_area_finder is not None:
                    if vol_output == 0:
                        output_subdir = Path(output_dir, 'empty_prediction')
                    else:
                        cluster_name = les_area_finder.get_img_area(outputs_np)
                        output_subdir = Path(output_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                    if not use_parent_folder:
                        parent_folder = Path(val_data['image_meta_dict']['filename_or_obj'][0]).parent.name
                        output_subdir = Path(output_subdir, parent_folder)
                        os.makedirs(output_subdir, exist_ok=True)
                else:
                    if not use_parent_folder:
                        output_subdir = output_dir
                    else:
                        parent_folder = Path(val_data['image_meta_dict']['filename_or_obj'][0]).parent.name
                        output_subdir = Path(output_dir, parent_folder)
                        os.makedirs(output_subdir, exist_ok=True)
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_subdir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            else:
                inputs_np = inputs[0, 0, :, :, :].cpu().detach().numpy() if isinstance(
                    inputs, torch.Tensor) else inputs[0, :, :, :]
                outputs_np = val_output_convert[0, 0, :, :, :].cpu().detach().numpy() if isinstance(
                    val_output_convert, torch.Tensor) else val_output_convert[0, :, :, :]

                tmp = np.array([])
                output_path_list = utils.save_img_lbl_seg_to_nifti(
                    inputs_np, tmp, outputs_np, output_dir, val_output_affine,
                    '{}_{}'.format(str(input_filename), str(img_count)))
            img_vol_dict[output_path_list[-1]] = vol_output
            for i, input_image_path in enumerate(val_data['image_meta_dict']['filename_or_obj']):
                input_output_paths_dict[input_image_path] = output_path_list[i]
            save_json(Path(output_dir, f'__input_output_paths_dict.json'), input_output_paths_dict)
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
                    image_cut_prefix=None,
                    image_cut_suffix=None,
                    transform_dict: dict = None,
                    device: str = None,
                    batch_size: int = 1,
                    dataloader_workers: int = 8,
                    bad_dice_treshold: float = 0,
                    clamping: tuple = None,
                    segmentation_area=True,
                    only_save_seg=False,
                    **kwargs
                    ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
    if transform_dict is None:
        transform_dict = checkpoint['transform_dict']

    _, val_ds = data_loading.init_training_data(img_path_list, seg_path_list, img_pref,
                                                image_cut_prefix=image_cut_prefix,
                                                image_cut_suffix=image_cut_suffix,
                                                transform_dict=transform_dict,
                                                train_val_percentage=0, clamping=clamping)
    val_loader = data_loading.create_validation_data_loader(val_ds, batch_size=batch_size,
                                                            dataloader_workers=dataloader_workers)

    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
    if training_img_size is None:
        training_img_size = utils.get_img_size(img_path_list[0])
    # original_size = utils.get_img_size(img_path_list[0])
    # model_name = 'unet'
    # hyper_params = net.default_unet_hyper_params
    # if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
    #     hyper_params = net.default_unetr_hyper_params
    #     hyper_params['img_size'] = training_img_size
    #     model_name = 'unetr'
    # if transform_dict is not None:
    #     for li in transform_dict:
    #         for d in transform_dict[li]:
    #             for t in d:
    #                 if t == 'CoordConvd' or t == 'CoordConvAltd':
    #                     hyper_params['in_channels'] = 4
    model = utils.load_model_from_checkpoint(checkpoint, device, checkpoint['hyper_params'],
                                             model_name=checkpoint['model_name'])
    model.to(device)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    dist_ratio = DistanceRatioMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    les_area_finder = None
    if segmentation_area:
        les_area_finder = utils.LesionAreaFinder()
    # start a typical PyTorch training
    # val_interval = 1
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = list()
    # metric_values = list()
    # val_save_thr = 0.7
    """
    Measure tracking init
    """
    val_images_dir = Path(output_dir, 'val_images')
    trash_val_images_dir = Path(output_dir, 'trash_val_images')
    if not val_images_dir.is_dir():
        val_images_dir.mkdir(exist_ok=True)
    for f in val_images_dir.iterdir():
        shutil.rmtree(f)
    if not trash_val_images_dir.is_dir():
        trash_val_images_dir.mkdir(exist_ok=True)
    for f in trash_val_images_dir.iterdir():
        os.remove(f)
    perf_measure_names = ['val_mean_dice',
                          'val_mean_dist',
                          'pred_volume',
                          'val_median_dice',
                          'val_std_dice',
                          'trash_img_nb',
                          'val_min_dice',
                          'val_max_dice']
    loop_dicts_list = []
    img_vol_dict = {}
    model.eval()
    with torch.no_grad():
        img_count = 0
        trash_count = 0
        # img_max_num = len(train_ds) + len(val_ds)
        val_score_list = []
        val_dist_list = []
        input_output_paths_dict = {}
        for val_data in tqdm(val_loader, desc=f'Validation '):
            inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
            input_filename = Path(val_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
            # outputs = model(inputs)
            # outputs = post_trans(outputs)
            with torch.cuda.amp.autocast():
                masks_only_val_labels = labels[:, :1, :, :, :]
                # TODO
                # print(type(model.module.weight))
                val_outputs = sliding_window_inference(inputs, training_img_size,
                                                       1, model, overlap=0.8)
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                dice = dice_metric.aggregate().item()
                hausdorff_metric(y_pred=val_output_convert, y=masks_only_val_labels)
                dist = hausdorff_metric.aggregate().item()
                # dist_ratio(y_pred=val_output_convert, y=masks_only_val_labels)
                # distance_ratio = dist_ratio.aggregate().item()
                # TODO make it work ...
                distance_ratio = np.NAN

            # if 'entropy' in kwargs and (kwargs['entropy'] == 'True' or kwargs['entropy'] == 1):
            #     input_filename += f'_e{utils.entropy_metric(val_outputs_list[0], sigmoid=True)}e'
            output_dict_data = deepcopy(val_data)
            # value = dice_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
            val_score_list.append(dice)
            val_dist_list.append(dist)
            val_data['image'] = val_data['image'].to(device)[0]
            val_data['label'] = val_data['label'].to(device)[0]
            output_dict_data['image'] = deepcopy(val_data['image'])
            # del output_dict_data['image']
            output_dict_data['label'] = deepcopy(val_output_convert[0])
            # Loop dataframe filling


            # Maybe not necessary but I prefer it there
            dice_metric.reset()
            hausdorff_metric.reset()

            second_tr = deepcopy(val_ds.transform)
            output_dict_data['label'].applied_operations = deepcopy(val_data['label'].applied_operations)
            output_dict_data['image'].applied_operations = deepcopy(val_data['image'].applied_operations)
            second_tr.transforms = val_ds.transform.transforms
            inverted_dict = val_ds.transform.inverse(val_data)
            inv_inputs, inv_labels = inverted_dict['image'], inverted_dict['label']
            with allow_missing_keys_mode(second_tr):
                inv_outputs = second_tr.inverse(output_dict_data)['label']
            inputs_np = inv_inputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_inputs, torch.Tensor) \
                else inv_inputs[0, :, :, :]
            labels_np = inv_labels[0, :, :, :].cpu().detach().numpy() if isinstance(inv_labels, torch.Tensor) \
                else inv_labels[0, :, :, :]
            outputs_np = inv_outputs[0, :, :, :].cpu().detach().numpy() if isinstance(inv_outputs, torch.Tensor) \
                else inv_outputs[0, :, :, :]
            # inputs_np = val_data['image'][0].cpu().detach().numpy()
            # labels_np = labels
            # outputs_np = val_output_convert[0][0, :, :, :].cpu().detach().numpy()
            vol_output = len(outputs_np[np.where(outputs_np)])
            input_filename += f'_v{vol_output}v'
            loop_dicts_list.append({'core_filename': input_filename.split('input_')[-1],
                                    'dice_metric': dice,
                                    'volume': vol_output,
                                    'distance': dist,
                                    'distance_ratio': distance_ratio})
            if dice < bad_dice_treshold:
                trash_count += 1
                # print('Saving trash image #{}'.format(trash_count))
                # TODO This is slow AF because of the imshow, maybe resetting the plot would work
                # utils.save_img_lbl_seg_to_png(
                #     inputs_np, trash_val_images_dir,
                #     '{}_trash_img_{}'.format(input_filename, trash_count), labels_np, outputs_np)
                if les_area_finder is not None:
                    if vol_output == 0:
                        output_subdir = Path(val_images_dir, 'empty_prediction')
                    else:
                        cluster_name = les_area_finder.get_img_area(outputs_np)
                        output_subdir = Path(trash_val_images_dir, cluster_name)
                    os.makedirs(output_subdir, exist_ok=True)
                else:
                    output_subdir = trash_val_images_dir
                if not only_save_seg:
                    output_path_list = utils.save_img_lbl_seg_to_nifti(
                        inputs_np, labels_np, outputs_np, output_subdir, val_output_affine,
                        '{}_{}'.format(str(input_filename), str(trash_count)))
                else:
                    out_input_path = Path(output_dir, 'input_{}.nii.gz'.format(str(trash_count)))
                    save_tensor_to_nifti(outputs_np, out_input_path, val_output_affine)
                    output_path_list = [str(out_input_path)]
            else:
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

                if not only_save_seg:
                    output_path_list = utils.save_img_lbl_seg_to_nifti(
                        inputs_np, labels_np, outputs_np, output_subdir, val_output_affine,
                        '{}_{}'.format(str(input_filename), str(img_count)))
                else:
                    out_input_path = Path(output_subdir, f'output_{input_filename}_{trash_count}.nii.gz')
                    save_tensor_to_nifti(outputs_np, out_input_path, val_output_affine)
                    output_path_list = [str(out_input_path)]
                # It's just easier to start at 0 for EVERYTHING
                img_count += 1
            img_vol_dict[output_path_list[-1]] = vol_output
            # TODO this needs to change if we want to increase the batch size > 1
            for i, input_image_path in enumerate(val_data['image_meta_dict']['filename_or_obj']):
                input_output_paths_dict[input_image_path] = output_path_list[-1]
            save_json(Path(output_dir, f'__input_output_paths_dict.json'), input_output_paths_dict)
        mean_metric = np.mean(np.array(val_score_list))
        median = np.median(np.array(val_score_list))
        std = np.std(np.array(val_score_list))
        min_score = np.min(np.array(val_score_list))
        max_score = np.max(np.array(val_score_list))
        df = pd.DataFrame.from_dict({
            'val_mean_dice': mean_metric,
            'val_mean_dist': np.mean(val_dist_list),
            'val_mean_dist_ratio': np.mean([d['distance_ratio'] for d in loop_dicts_list]),
            'pred_volume': vol_output,
            'val_median_dice': median,
            'val_std_dice': std,
            'trash_img_nb': trash_count,
            'val_min_dice': min_score,
            'val_max_dice': max_score,
            'val_best_mean_dice': 0
        }, orient='index')
    with open(Path(output_dir, f'__output_image_volumes.json'), 'w+') as j:
        json.dump(img_vol_dict, j, indent=4)
    pd.DataFrame().from_dict(img_vol_dict, orient='index').to_csv(Path(output_dir, f'__output_image_volumes.csv'))
    df.to_csv(Path(output_dir, 'val_perf_global_measures.csv'))  # , columns=perf_measure_names)
    # loop_df_columns = ['core_filename', 'dice_metric', 'volume', 'distance', 'distance_ratio']
    loop_df = pd.DataFrame().from_records(loop_dicts_list)
    loop_df.to_csv(Path(output_dir, 'val_perf_individual_measures.csv'))  # , columns=loop_df_columns)

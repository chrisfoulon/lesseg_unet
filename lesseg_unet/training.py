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


def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        dist.destroy_process_group()
        exit(1)


signal.signal(signal.SIGINT, handler)


def setup(rank, world_size, port='1234', cpu=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    if not cpu:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)


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


# def training_loop(img_path_list: Sequence,
#                   seg_path_list: Sequence,
#                   output_dir: Union[str, bytes, os.PathLike],
#                   ctr_path_list: Sequence = None,
#                   img_pref: str = None,
#                   transform_dict=None,
#                   device: str = None,
#                   batch_size: int = 10,
#                   val_batch_size: int = 1,
#                   epoch_num: int = 50,
#                   dataloader_workers: int = 4,
#                   train_val_percentage=80,
#                   lesion_set_clamp=None,
#                   controls_clamping=None,
#                   label_smoothing=False,
#                   stop_best_epoch=-1,
#                   training_loss_fct='dice',
#                   val_loss_fct='dice',
#                   weight_factor=1,
#                   folds_number=1,
#                   dropout=0,
#                   cache_dir=None,
#                   save_every_decent_best_epoch=True,
#                   **kwargs):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)
#     logging.info(f'Torch device used for this training: {str(device)}')
#     # Apparently it can potentially improve the performance when the model does not change its size. (Source tuto UNETR)
#     torch.backends.cudnn.benchmark = True
#     # val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
#     # checking is CoordConv is used and change the input channel dimension
#     if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
#         unet_hyper_params = net.default_unetr_hyper_params
#         loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#     else:
#         unet_hyper_params = net.default_unet_hyper_params
#         loss_function = DiceLoss(sigmoid=True)
#     if transform_dict is not None:
#         for li in transform_dict:
#             for d in transform_dict[li]:
#                 for t in d:
#                     if t == 'CoordConvd' or t == 'CoordConvAltd':
#                         unet_hyper_params['in_channels'] = 4
#     # TODO outdated
#     # for t in val_ds.transform.transforms:
#     #     if isinstance(t, transformations.CoordConvd) or isinstance(t, transformations.CoordConvAltd):
#     #         unet_hyper_params = net.coord_conv_unet_hyper_params
#
#     if dropout is not None and dropout == 0:
#         unet_hyper_params['dropout'] = dropout
#
#     regularisation = True
#     if 'regularisation' in kwargs:
#         v = kwargs['regularisation']
#         if v == 'False' or v == 0:
#             regularisation = False
#         if v == 'True' or v == 1:
#             regularisation = True
#
#     non_blocking = True
#     if 'non_blocking' in kwargs:
#         v = kwargs['non_blocking']
#         if v == 'False' or v == 0:
#             non_blocking = False
#         if v == 'True' or v == 1:
#             non_blocking = True
#
#     transformations_device = None
#     if 'tr_device' in kwargs:
#         v = kwargs['tr_device']
#         if v == 'False' or v == 0:
#             transformations_device = None
#         if v == 'True' or v == 1:
#             print(f'ToTensord transformation will be called on {device}')
#             transformations_device = device
#
#     no_ctr_trainloss = False
#     if 'no_ctr_trainloss' in kwargs:
#         v = kwargs['no_ctr_trainloss']
#         if v == 'False' or v == 0:
#             no_ctr_trainloss = False
#         if v == 'True' or v == 1:
#             if ctr_path_list is not None and ctr_path_list != []:
#                 print('The control loss is tracked but not computed within the model')
#             no_ctr_trainloss = True
#
#     dice_metric = DiceMetric(include_background=True, reduction="mean")
#     # surface_metric = SurfaceDistanceMetric(include_background=True, reduction="mean", symmetric=True)
#     hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
#     post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
#     tversky_function = TverskyLoss(sigmoid=True, alpha=2, beta=0.5)
#     # focal_function = FocalLoss()
#     # df_loss = DiceFocalLoss(sigmoid=True)
#     # dce_loss = DiceCELoss(sigmoid=True, ce_weight=torch.Tensor([1]).cuda())
#     # loss_function = BCE
#     # val_loss_function = BCE
#     val_loss_function = DiceLoss(sigmoid=True)
#     # Save training parameters info
#     logging.info(f'Abnormal images prefix: {img_pref}')
#     logging.info(f'Device: {device}')
#     logging.info(f'Training loss fct: {training_loss_fct}')
#     logging.info(f'Validation loss fct: {val_loss_fct}')
#
#     # utils.save_tensor_to_nifti(
#     #     inputs, Path('/home/tolhsadum/neuro_apps/data/', 'nib_input_{}.nii'.format('test')), val_output_affine)
#     # utils.save_tensor_to_nifti(
#     #     labels, Path('/home/tolhsadum/neuro_apps/data/', 'nib_label_{}.nii'.format('test')), val_output_affine)
#     # ##################################################################################
#     # # Code for restoring!
#     # state_dict_fullpath = os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt')
#     # checkpoint = torch.load(state_dict_fullpath)
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     # starting_epoch = checkpoint['epoch'] + 1
#     # scaler.load_state_dict(checkpoint['scaler'])
#     #
#     # # Code for saving (to be used in the validation)
#     # checkpoint_dict = {'epoch': epoch,
#     #                    'model_state_dict': model.state_dict(),
#     #                    'optimizer_state_dict': optimizer.state_dict(),
#     #                    'loss_tally_train': loss_tally_train,
#     #                    'nifti_t1_paths': nifti_t1_paths,}
#     # checkpoint_dict['scaler'] = scaler.state_dict()
#     # torch.save(checkpoint_dict, os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt'))
#     # ##################################################################################
#
#     # start a typical PyTorch training
#
#     val_interval = 1
#     best_metric = -1
#     best_controls_mean_loss = -1
#     # best_distance = -1
#     best_avg_loss = -1
#     best_metric_epoch = -1
#     val_meh_thr = 0.7
#     val_trash_thr = 0.3
#     metric_select_fct = gt
#     control_weight_factor = weight_factor  # Experiment with different weightings!
#     nb_patches = None
#     if stop_best_epoch != -1:
#         logging.info(f'Will stop after {stop_best_epoch} epochs without improvement')
#     if label_smoothing:
#         logging.info('Label smoothing activated')
#     """
#     Measure tracking init
#     """
#     # TODO add the new measures if useful to the csv
#     perf_measure_names = ['avg_train_loss',
#                           'val_mean_metric',
#                           'val_median_metric',
#                           'val_std_metric',
#                           'trash_img_nb',
#                           'val_min_metric',
#                           'val_max_metric',
#                           'val_best_mean_metric']
#     df = pd.DataFrame(columns=perf_measure_names)
#     # Gather data
#     img_dict, controls = data_loading.match_img_seg_by_names(img_path_list, seg_path_list, img_pref)
#
#     # # If the default_label option is used, we try to find controls images and create a second set of datasets and a
#     # # second set of loaders
#     # if default_label is not None:
#     #     if ctr_path_list is not None and ctr_path_list != []:
#     #         # _, controls_2 = data_loading.match_img_seg_by_names(img_path_list, [], img_pref, default_label)
#     #         _, controls_2 = data_loading.match_img_seg_by_names(ctr_path_list, [], None, default_label)
#     #         controls.update(controls_2)
#     # else:
#     if ctr_path_list is not None and ctr_path_list != []:
#         _, controls_2 = data_loading.match_img_seg_by_names(ctr_path_list, [], None)
#         controls += controls_2
#     else:
#         # If default_label is not given, it is not intended to have unmatched images, thus we throw an error
#         # if some images don't have a match
#         if controls:
#             raise ValueError('Not control file list provided but images with missing labels found')
#
#     split_lists = utils.split_lists_in_folds(img_dict, folds_number, train_val_percentage)
#     # TESTING
#     # for i, li in enumerate(split_lists):
#     #     split_lists[i] = li[0:20]
#     # split_lists = [split_lists[0], split_lists[0]]
#     # model = utils.load_eval_from_checkpoint(
#     #     '/home/tolhsadum/neuro_apps/data/'
#     #     'filt_zetas_ctr_cc_weight_100_clamp_halfpercent_best_metric_model_segmentation3d_epo.pth',
#     #     device, unet_hyper_params)
#     # TESTING
#     logging.info('Initialisation of the training transformations')
#     train_img_transforms = transformations.train_transformd(transform_dict, lesion_set_clamp,
#                                                             device=transformations_device)
#     val_img_transforms = transformations.val_transformd(transform_dict, lesion_set_clamp,
#                                                         device=transformations_device)
#     training_img_size = transformations.find_param_from_hyper_dict(
#         transform_dict, 'spatial_size', find_last=True)
#     if training_img_size is None:
#         training_img_size = utils.get_img_size(split_lists[0][0]['image'])
#     controls_lists = []
#     trash_seg_path_count_dict = {}
#     ctr_img_transforms = None
#     ctr_val_img_transforms = None
#     if controls:
#         print('Initialisation of the control dataset')
#         if len(controls) < len(img_dict):
#             raise ValueError(f'Only {len(controls)} controls for {len(img_dict)} patient images')
#         else:
#             controls_lists = utils.split_lists_in_folds(controls, folds_number, train_val_percentage)
#         logging.info('Initialisation of the control training transformations')
#         ctr_img_transforms = transformations.image_only_transformd(transform_dict, training=True,
#                                                                    clamping=controls_clamping,
#                                                                    device=transformations_device)
#         logging.info('Initialisation of the control validation transformations')
#         ctr_val_img_transforms = transformations.image_only_transformd(transform_dict, training=False,
#                                                                        clamping=controls_clamping,
#                                                                        device=transformations_device)
#         logging.info(f'Control training loss: mean(sigmoid(outputs)) * {control_weight_factor}')
#     for fold in range(folds_number):
#         if 'unetr' in kwargs and (kwargs['unetr'] == 'True' or kwargs['unetr'] == 1):
#             unet_hyper_params['img_size'] = training_img_size
#             model = net.create_unetr_model(device, unet_hyper_params)
#             optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#             params = list(model.parameters())
#             unetr = True
#         else:
#             model = net.create_unet_model(device, unet_hyper_params)
#             optimizer = torch.optim.Adam(model.parameters(), 1e-3)
#             params = list(model.model.parameters())
#             unetr = False
#         # with torch.no_grad():
#         #     regularisation_val = utils.sum_non_bias_l2_norms(params, 1e-4)
#         # Get model parameters
#         # parameters = list(model.model.parameters())
#         # parameters = list(model.model.named_parameters())
#         # def count_unique_parameters(parameters):
#         #     # Only counts unique params
#         #     count = 0
#         #     list_of_names = []
#         #     for p in parameters:
#         #         name = p[0]
#         #         param = p[1]
#         #         if name not in list_of_names:
#         #             list_of_names.append(name)
#         #             count += np.prod(param.size())
#         #     return count
#         #
#         # print(count_unique_parameters(parameters))
#         # exit()
#         if folds_number == 1:
#             output_fold_dir = output_dir
#         else:
#             output_fold_dir = Path(output_dir, f'fold_{fold}')
#         writer = SummaryWriter(log_dir=str(output_fold_dir))
#         trash_list_path = Path(output_fold_dir, 'trash_images_list.csv')
#         if trash_list_path.is_file():
#             os.remove(trash_list_path)
#         train_loader, val_loader = data_loading.create_fold_dataloaders(
#             split_lists, fold, train_img_transforms,
#             val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir
#         )
#         ctr_train_loader = None
#         ctr_val_loader = None
#         if controls_lists:
#             print('Creating control data loader')
#             ctr_train_loader, ctr_val_loader = data_loading.create_fold_dataloaders(
#                 controls_lists, fold, ctr_img_transforms,
#                 ctr_val_img_transforms, batch_size, dataloader_workers, val_batch_size, cache_dir
#             )
#         output_spatial_size = None
#         # max_distance = None
#         batches_per_epoch = len(train_loader)
#         # val_batches_per_epoch = len(val_loader)
#         """
#         Training loop
#         """
#         str_best_epoch = ''
#         # Cumulated values might reach max float so storing it into lists
#         ctr_loss = []
#         # batch_ctr_vol = []
#         time_list = []
#         epoch_time_list = []
#         for epoch in range(epoch_num):
#             print('-' * 10)
#             print(f'epoch {epoch + 1}/{epoch_num}')
#             model.train()
#             epoch_loss = 0
#             step = 0
#             # testing
#             # testing(train_loader, output_dir)
#             # import time
#             # for i in range(5):
#             #     step = 0
#             #     start_time = time.time()
#             #     val_output_affine = utils.nifti_affine_from_dataset(img_path_list[0])
#             #     for batch_data in tqdm(train_loader, desc=f'training_loop{epoch}'):
#             #         # batch_data = next(iter(train_loader))
#             #         step += 1
#             #         # if step > 5:
#             #         #     break
#             #         # inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
#             #         input_filename = Path(batch_data['image_meta_dict']['filename_or_obj'][0]).name.split('.nii')[0]
#             #         print(f'Step {step}: {input_filename}')
#             #     end_time = time.time()
#             #     print(f'Loop Time: {end_time - start_time}')
#             # inputs_np = inputs[0, 0, :, :, :].cpu().detach().numpy() if isinstance(inputs, torch.Tensor) \
#             #     else inputs[0, :, :, :]
#             # labels_np = labels[0, 0, :, :, :].cpu().detach().numpy() if isinstance(labels, torch.Tensor) \
#             #     else labels[0, :, :, :]
#             # output_path_list = utils.save_img_lbl_seg_to_nifti(
#             #     inputs_np, labels_np, None, output_dir, val_output_affine,
#             #     'input_test_{}'.format(str(step)))
#             # print(f'image min/max : {torch.min(inputs)}/{torch.max(inputs)}')
#             # Time test
#             # import time
#             # # batch_data = next(iter(train_loader))
#             # # inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
#             # # optimizer.zero_grad()
#             # start_time = time.time()
#             # # a bit less that 2sec/loop on laptop cpu
#             # # for i in tqdm(range(100)):
#             # #     outputs = model(inputs)
#             # for i in tqdm(range(100)):
#             #     next(iter(train_loader))
#             # end_time = time.time()
#             # print(f'model Time: {end_time - start_time}')
#             # # start_time = time.time()
#             # # for i, b in enumerate(train_loader):
#             # #     if i == 99:
#             # #         break
#             # #     continue
#             # # end_time = time.time()
#             # # print(f'Loop Time: {end_time - start_time}')
#             # exit()
#             start_time = time.time()
#             time_loading = True
#             ctr_train_iter = None
#             if ctr_train_loader:
#                 ctr_train_iter = iter(ctr_train_loader)
#             for batch_data in train_loader:
#                 if time_loading:
#                     end_time = time.time()
#                     load_time = end_time - start_time
#                     print(f'Loading loop Time: {load_time}')
#                     time_list.append(load_time)
#                     print(f'First load time: {time_list[0]} and average loading time {np.mean(time_list)}')
#                     time_loading = False
#                 # with torch.autograd.profiler.profile(use_cuda=False) as prof:
#                 step += 1
#                 optimizer.zero_grad()
#                 inputs, labels = batch_data['image'].to(device, non_blocking=non_blocking), batch_data['label'].to(
#                     device, non_blocking=non_blocking)
#                 if nb_patches is None:
#                     nb_patches = int(inputs.shape[0]/batch_size)
#                 # print(inputs.shape)
#                 # print(batch_data['image_meta_dict']['spatial_shape'])
#                 # print(batch_data['image_meta_dict']['original_channel_dim'])
#                 # exit()
#                 # if output_spatial_size is None:
#                 #     output_spatial_size = inputs.shape
#                 #     max_distance = torch.as_tensor(
#                 #         [torch.linalg.norm(torch.as_tensor(output_spatial_size, dtype=torch.float16))])
#                 outputs = model(inputs)
#                 # print(f'Inputs size: {inputs.shape}')
#                 # print(f'Labels size: {labels.shape}')
#                 # print(f'Outputs size: {outputs.shape}')
#                 # TODO smoothing?
#                 y = labels[:, :1, :, :, :]
#                 if label_smoothing:
#                     s = .1
#                     y = y * (1 - s) + 0.5 * s
#                 # TODO make sure all the different measures still work
#                 if training_loss_fct.lower() == 'BCE':
#                     loss = BCE(outputs, y)
#                 elif training_loss_fct.lower() == 'tversky_loss':
#                     loss = tversky_function(outputs, y)
#                 elif training_loss_fct.lower() == 'dist_dice':
#                     loss = loss_function(outputs, y)
#                     # Just trying some dark magic
#                     distance = hausdorff_metric(y_pred=post_trans(outputs), y=y)
#                     # distance = surface_metric(y_pred=post_trans(outputs), y=labels[:, :1, :, :, :])
#                     # distance = torch.minimum(distance, max_distance)
#                     loss += torch.mean(distance)
#                 else:
#                     loss = loss_function(outputs, y)
#                 controls_loss = 0
#                 controls_loss_str = ''
#                 # inputs_controls = None
#                 # labels_controls = None
#                 # outputs_controls = None
#                 if ctr_train_iter is not None:
#                     with torch.no_grad():
#                         batch_data_controls = next(ctr_train_iter)
#                     if not no_ctr_trainloss:
#
#                         inputs_controls = batch_data_controls['image'].to(device, non_blocking=non_blocking)
#                         # labels_controls = torch.zeros_like(inputs_controls).to(device)
#                         outputs_controls = model(inputs_controls)
#                         outputs_batch_images = outputs_controls[:, :1, :, :, :]
#                         outputs_batch_images_sigmoid = torch.sigmoid(outputs_batch_images)
#                         # controls_vol = utils.volume_metric(outputs_batch_images_sigmoid,
#                         #                                    sigmoid=False, discrete=True)
#                         # The volume (number of voxels with 1) needs to be averaged on the batch size
#                         # batch_ctr_vol += [controls_vol / outputs_controls.shape[0]]
#                         controls_loss = torch.mean(outputs_batch_images_sigmoid) * control_weight_factor
#                         ctr_loss += [controls_loss]
#                         # controls_loss = utils.percent_vox_loss(outputs_controls[:, :1, :, :, :], divide_max_vox=100)
#                         # controls_loss = controls_vol
#                         # TODO it's just to save a bit of time
#                         # batch_ctr_vol += [controls_vol]
#                         # controls_loss_str = f'Controls loss: {controls_loss}, controls volume: {controls_vol}'
#                         controls_loss_str = f'Controls loss: {controls_loss}'
#                         # TODO it's just to save a bit of time
#                         # writer.add_scalar('batch_control_loss', controls_loss.item(), batches_per_epoch * epoch + step)
#                         # writer.add_scalar('batch_control_vol', controls_vol, batches_per_epoch * epoch + step)
#                     else:
#                         # In that case the control_loss does not pass through the backward function and overloads memory
#                         with torch.no_grad():
#                             inputs_controls = batch_data_controls['image'].to(device, non_blocking=non_blocking)
#                             outputs_controls = model(inputs_controls)
#                             outputs_batch_images = outputs_controls[:, :1, :, :, :]
#                             outputs_batch_images_sigmoid = torch.sigmoid(outputs_batch_images)
#                             controls_loss = torch.mean(outputs_batch_images_sigmoid) * control_weight_factor
#                             ctr_loss += [controls_loss]
#                             controls_loss_str = f'Controls loss: {controls_loss}'
#                 if not no_ctr_trainloss:
#                     loss = loss + controls_loss
#                 # Regularisation
#                 if regularisation:
#                     # regularisation_val = utils.sum_non_bias_l2_norms(params, 1e-4)
#                     loss += utils.sum_non_bias_l2_norms(params, 1e-4)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#                 # PRINTS
#                 print(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}, ' +
#                       controls_loss_str + str(inputs.shape) + ' ' + str(outputs.shape)
#                       # f'| tversky_loss: {tversky_function(outputs, y).item():.4f}'
#                       # f'| dicefocal: {df_loss(outputs, y).item():.4f}'
#                       # f'| BCE: {BCE(outputs, y).item():.4f}'
#                       # f'| Surface distance: {distance.item():.4f}'
#                       # f'| Hausdorff distance: {hausdorff.item():.4f}'
#                       # f'| BCE: {BCE(outputs, y, reduction="mean"):.4f}'
#                       # f'| focal_loss: {focal_function(outputs, y).item():.4f}'
#                       )
#                 # TODO it's just to save a bit of time
#                 # writer.add_scalar('train_loss', loss.item(), batches_per_epoch * epoch + step)
#                 # print(prof.total_average())
#                 # print(prof.key_averages().table(row_limit=0))
#             epoch_loss /= step
#             print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')
#             # ########## training EPOCH LEVEL WRITER ###########
#             writer.add_scalar('epoch_train_loss', epoch_loss, epoch + 1)
#             writer.add_scalar('epoch_ctr_loss', torch.mean(torch.tensor(ctr_loss), dtype=torch.float), epoch + 1)
#             # TODO it's just to save a bit of time
#             # writer.add_scalar('epoch_ctr_volume', torch.mean(torch.tensor(batch_ctr_vol), dtype=torch.float), epoch + 1)
#             # if (epoch + 1) % val_interval == 1:
#             """
#             Validation loop
#             """
#             if (epoch + 1) % val_interval == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     metric_sum = 0.0
#                     metric_count = 0
#                     img_count = 0
#                     meh_count = 0
#                     trash_count = 0
#                     # distance_sum = 0.0
#                     # distance_count = 0
#                     controls_vol = None
#                     trash_seg_paths_list = []
#                     if controls_lists:
#                         controls_vol = 0
#                     val_dice_list = []
#                     loss_list = []
#                     ctr_loss = []
#                     batch_ctr_vol = []
#                     step = 0
#                     pbar = tqdm(val_loader, desc=f'Val[{epoch + 1}] avg_metric:[N/A]')
#                     controls_mean_loss = -1
#                     controls_mean_vol = -1
#                     ctr_val_iter = None
#                     if ctr_val_loader:
#                         ctr_val_iter = iter(ctr_val_loader)
#                     for val_data in pbar:
#                         step += 1
#                         inputs, labels = val_data['image'].to(device, non_blocking=non_blocking), val_data['label'].to(
#                             device, non_blocking=non_blocking)
#
#                         print(f'Nb patches: {nb_patches}')
#                         val_outputs = sliding_window_inference(inputs, training_img_size,
#                                                                nb_patches, model)
#                                                                # ,overlap=0.8) for the segmentation
#                         outputs = model(inputs)
#
#                         # In case CoordConv is used, we only want the measures on the images, not the coordinates
#                         # inputs = inputs[:, :1, :, :, :]
#                         labels = labels[:, :1, :, :, :]
#                         outputs = outputs[:, :1, :, :, :]
#                         controls_loss = 0
#                         if ctr_val_iter:
#                             batch_data_controls = next(ctr_val_iter)
#                             inputs_controls = batch_data_controls['image'].to(device, non_blocking=non_blocking)
#                             # labels_controls = torch.zeros_like(inputs_controls).to(device)
#                             outputs_controls = model(inputs_controls)
#                             outputs_batch_images = outputs_controls[:, :1, :, :, :]
#                             outputs_batch_images_sigmoid = torch.sigmoid(outputs_batch_images)
#                             controls_vol = utils.volume_metric(outputs_batch_images_sigmoid,
#                                                                sigmoid=False, discrete=True)
#                             # The volume (number of voxels with 1) needs to be averaged on the batch size
#                             batch_ctr_vol += [controls_vol/outputs_controls.shape[0]]
#                             # It's basically torch.mean(torch.sigmoid(normal_logits) > 0.5)
#                             ctr_loss += [torch.mean(outputs_batch_images_sigmoid) * control_weight_factor]
#                             # TODO clean once above tested
#                             # batch_mean = torch.mean(outputs_controls[:, :1, :, :, :], 0)
#                             # batch_mean_sigmoid = torch.sigmoid(batch_mean)
#                             # controls_vol = utils.volume_metric(batch_mean_sigmoid,
#                             #                                    sigmoid=True, discrete=True)
#                             # controls_loss = torch.mean(batch_mean_sigmoid) * control_weight_factor
#                             #
#                             # ctr_vol += [controls_vol]
#
#                         loss = val_loss_function(outputs, labels)
#                         # discrete_outputs = post_trans(outputs)
#                         discrete_outputs = [post_trans(i) for i in decollate_batch(outputs)]
#                         decollated_labels = decollate_batch(labels)
#                         # TODO break down the metric calculation as we use it to count the good and bad segmentations
#                         # for ind, output in enumerate(discrete_outputs):
#                         batch_dice_metric_list = []
#                         for ind, output in enumerate(discrete_outputs):
#                             dice_value = dice_metric(y_pred=[output], y=[decollated_labels[ind]])
#                             val_dice_list.append(dice_value.item())
#                             batch_dice_metric_list.append(dice_value.item())
#                             if dice_value.item() > val_meh_thr:
#                                 img_count += 1
#                             elif dice_value.item() > val_trash_thr:
#                                 meh_count += 1
#                             else:
#                                 if best_metric > val_meh_thr:
#                                     p = val_data['image_meta_dict']['filename_or_obj'][0]
#                                     trash_seg_paths_list.append(p)
#                                     if p in trash_seg_path_count_dict:
#                                         trash_seg_path_count_dict[p] += 1
#                                     else:
#                                         trash_seg_path_count_dict[p] = 0
#                                 trash_count += 1
#                             # distance = hausdorff_metric(y_pred=[output], y=[decollated_labels[ind]])
#                             # distance = torch.minimum(distance, max_distance).mean()
#                             # distance = surface_metric(y_pred=outputs, y=labels[:, :1, :, :, :])
#                             # distance_sum += distance.item()
#                             # distance_count += 1
#                         # dice_value = dice_metric(y_pred=discrete_outputs[i, :1, :, :, :],
#                         #                          y=labels[i, :1, :, :, :]) # .mean()
#                         # distance = hausdorff_metric(y_pred=discrete_outputs[i, :1, :, :, :],
#                         #                             y=labels[i, :1, :, :, :]).mean()
#                         # dice_value = dice_metric(y_pred=discrete_outputs, y=labels[:, :1, :, :, :]).mean()
#                         # distance = hausdorff_metric(y_pred=discrete_outputs, y=labels[:, :1, :, :, :]).mean()
#
#                         if val_loss_fct == 'dist':
#                             # In that case we want the distance to be smaller
#                             metric_select_fct = lt
#                             metric = distance
#                         elif val_loss_fct == 'dist_loss':
#                             # In that case we want the loss to be smaller
#                             metric_select_fct = lt
#                             metric = distance + loss
#                         elif val_loss_fct == 'dice_ctr_loss':
#                             # In that case we want the loss to be smaller
#                             metric_select_fct = lt
#                             metric = loss + controls_loss
#                             # print(f'Batch dice metric: {batch_dice_metric_list}')
#                         elif val_loss_fct == 'dice_ctr_vol':
#                             # In that case we want the loss to be smaller
#                             metric_select_fct = lt
#                             metric = loss + controls_vol
#                         elif val_loss_fct == 'dice_loss':
#                             metric_select_fct = lt
#                             metric = loss
#                         else:
#                             metric = torch.mean(torch.tensor(batch_dice_metric_list, dtype=torch.float))
#
#                         # The metric is already averaged over the batch so no need to average it further
#                         metric_count += 1
#                         metric_sum += metric.item()
#
#                         loss_list.append(loss.item())
#
#                         pbar.set_description(f'Val[{epoch + 1}] avg_loss:[{np.mean(loss_list)}]')
#                     mean_metric = metric_sum / metric_count
#                     val_mean_loss = np.mean(loss_list)
#                     mean_dice = np.mean(np.array(val_dice_list))
#                     # median = np.median(np.array(val_dice_list))
#                     std = np.std(np.array(val_dice_list))
#                     # min_score = np.min(np.array(val_dice_list))
#                     # max_score = np.max(np.array(val_dice_list))
#                     val_ctr_str = ''
#                     if ctr_loss:
#                         controls_mean_loss = torch.mean(torch.tensor(ctr_loss, dtype=torch.float))
#                         controls_mean_vol = torch.mean(torch.tensor(batch_ctr_vol, dtype=torch.float))
#                         val_ctr_str = f'Controls loss [{controls_mean_loss}] / volume[{controls_mean_vol}];\n\n'
#
#                     writer.add_scalar('val_mean_metric', mean_metric, epoch + 1)
#                     writer.add_scalar('val_mean_dice', mean_dice, epoch + 1)
#                     writer.add_scalar('val_mean_loss', val_mean_loss, epoch + 1)
#                     # writer.add_scalar('val_distance', distance_sum / distance_count, epoch + 1)
#                     writer.add_scalar('trash_img_nb', trash_count, epoch + 1)
#                     writer.add_scalar('val_ctr_loss', controls_mean_loss, epoch + 1)
#                     writer.add_scalar('val_ctr_volume', controls_mean_vol, epoch + 1)
#                     # writer.add_scalar('val_median_metric', median, epoch + 1)
#                     # writer.add_scalar('val_min_metric', min_score, epoch + 1)
#                     # writer.add_scalar('val_max_metric', max_score, epoch + 1)
#                     writer.add_scalar('val_std_metric', std, epoch + 1)
#                     # TODO check if that corresponds
#                     df.loc[epoch + 1] = pd.Series({
#                         'avg_train_loss': epoch_loss,
#                         'val_mean_metric': mean_metric,
#                         'val_mean_dice': mean_dice,
#                         # 'val_distance': distance_sum / distance_count,
#                         'trash_img_nb': trash_count,
#                         'val_mean_loss': val_mean_loss,
#                         # 'val_median_metric': median,
#                         'val_std_metric': std,
#                         'meh_img_nb': meh_count,
#                         'good_img_nb': img_count,
#                         # 'val_min_metric': min_score,
#                         # 'val_max_metric': max_score,
#                         # 'composite1': mean_metric + distance_sum / distance_count,
#                         # 'controls_metric': controls_mean_loss,
#                         'controls_vol': torch.mean(torch.tensor(batch_ctr_vol), dtype=torch.float),
#                         'val_best_mean_metric': 0
#                     })
#                     str_img_count = (
#                             f'Trash (<{val_trash_thr}|'.rjust(12, ' ') +
#                             f'Meh (<{val_meh_thr})|'.rjust(12, ' ') + f'Good\n'.rjust(12, ' ') +
#                             f'{trash_count}|'.rjust(12, ' ') + f'{meh_count}|'.rjust(12, ' ') +
#                             f'{img_count}'.rjust(12, ' ') + '\n\n'
#                     )
#                     # TODO maybe find a better way so it would also save the first epoch. Even though it is no big deal
#                     if epoch == 0:
#                         # best_metric = mean_metric
#                         best_metric = val_mean_loss
#                         best_metric_epoch = 0
#                         best_controls_mean_loss = controls_mean_loss
#                     # if metric_select_fct(val_mean_loss, best_metric) or (
#                     #         not no_ctr_trainloss and controls_mean_loss < best_controls_mean_loss):
#                     if metric_select_fct(mean_metric, best_metric):
#                         # best_metric = mean_metric
#                         best_metric = val_mean_loss
#                         # best_distance = distance_sum / distance_count
#                         # TODO clean up all that ....
#                         best_avg_loss = val_mean_loss
#                         best_metric_epoch = epoch + 1
#                         epoch_suffix = ''
#                         if save_every_decent_best_epoch:
#                             if mean_dice > 0.75:
#                                 epoch_suffix = '_' + str(epoch)
#                         utils.save_checkpoint(model, epoch + 1, optimizer, output_fold_dir,
#                                               f'best_metric_model_segmentation3d_epo{epoch_suffix}.pth')
#                         print('saved new best metric model')
#                         str_best_epoch = (
#                             f'Best epoch {best_metric_epoch} '
#                             # f'metric {best_metric:.4f}/dist {best_distance}/avgloss {best_avg_loss}\n'
#                             f'metric {best_metric:.4f} / avgloss {best_avg_loss}\n'
#                             + val_ctr_str + 'Img count of best epoch: \n'
#                             + str_img_count
#                         )
#                         writer.add_scalar('val_best_mean_metric', mean_metric, epoch + 1)
#                         df.at[epoch + 1, 'val_best_mean_metric'] = mean_metric
#                         if trash_seg_paths_list:
#                             with open(trash_list_path, 'a+') as f:
#                                 f.write(','.join([str(epoch + 1)] + trash_seg_paths_list) + '\n')
#                     best_epoch_count = epoch + 1 - best_metric_epoch
#                     str_current_epoch = (
#                             f'[Fold: {fold}]Current epoch: {epoch + 1} current mean metric: {mean_metric:.4f}'
#                             f' current mean dice metric: {mean_dice}\n'
#                             # f'and an average distance of [{distance_sum / distance_count}];\n'
#                             f'Controls loss [{controls_mean_loss}] / volume[{controls_mean_vol}] ;\n\n'
#                             + str_img_count + str_best_epoch
#                     )
#                     print(str_current_epoch)
#                     print(f'It has been [{best_epoch_count}] since a best epoch has been found')
#                     if stop_best_epoch > -1:
#                         print(f'The training will stop after [{stop_best_epoch}] epochs without improvement')
#                     epoch_end_time = time.time()
#                     epoch_time = epoch_end_time - start_time
#                     print(f'Epoch Time: {epoch_time}')
#                     epoch_time_list.append(epoch_time)
#                     print(f'First epoch time: '
#                           f'{epoch_time_list[0]} and average epoch time {np.mean(epoch_time_list)}')
#                     if stop_best_epoch != -1:
#                         if best_epoch_count > stop_best_epoch:
#                             print(f'More than {stop_best_epoch} without improvement')
#                             df.to_csv(Path(output_fold_dir, 'perf_measures.csv'), columns=perf_measure_names)
#                             print(f'train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
#                             logging.info(str_best_epoch)
#                             writer.close()
#                             break
#                     # utils.save_checkpoint(model, epoch + 1, optimizer, output_dir)
#         df.to_csv(Path(output_fold_dir, f'perf_measures_{fold}.csv'), columns=perf_measure_names)
#         with open(Path(output_fold_dir, f'trash_img_count_dict_{fold}.json'), 'w+') as j:
#             json.dump(trash_seg_path_count_dict, j, indent=4)
#         print(f'train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
#         logging.info(str_best_epoch)
#         writer.close()


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
    # disable logging for processes except 0 on every node
    if rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    # rank = int(os.environ["LOCAL_RANK"])
    print(f'################RANK : {rank}####################')
    """MODEL PARAMETERS"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    # TODO just see what the performance looks like on only one GPU with DDP
    if world_size > 0:
        cpu_device = device.type == 'cpu'

        setup(rank, world_size, cpu=cpu_device)
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
                # TODO might be a cause of crash with DDP
                loss += utils.sum_non_bias_l2_norms(params, 1e-4)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if no_progressbar_training:
                    print(f'[{fold}]{step}/{batches_per_epoch}, train loss: {loss.item():.4f}')
                else:
                    train_iter.set_description(
                        f'Training[{epoch + 1}] batch_loss/mean_loss:[{loss.item():.4f}/{epoch_loss/step:.4f}]')
            print(f"[{dist.get_rank()}] " + f"epoch {epoch + 1}, average loss: {epoch_loss/step:.4f}")
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
                    if val_batch_dist_list is not None:
                        mean_dist_val = np.mean(val_batch_dist_list)
                        hausdorff_metric.reset()
                        mean_dist_str = f'/ Current mean distance {mean_dist_val}'
                        writer.add_scalar('val_distance', mean_dist_val, epoch + 1)
                    """IF NEW BEST EPOCH"""
                    if best_dice < mean_dice_val:
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

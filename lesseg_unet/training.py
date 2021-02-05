import os
import logging
import math
from pathlib import Path
from typing import Sequence, Tuple, Union

import monai
import torch
from monai.metrics import DiceMetric
from monai.data import NiftiDataset
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.visualize import plot_2d_or_3d_image
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from lesseg_unet import data_loading, net


def init_training_data(img_path_list: Sequence,
                       seg_path_list: Sequence,
                       img_pref: str = None,
                       train_val_percentage: float = 75) -> Tuple[monai.data.NiftiDataset, monai.data.NiftiDataset]:
    logging.info('Listing input files to be loaded')
    img_seg_dict = data_loading.match_img_seg_by_names(img_path_list, seg_path_list, img_pref)
    train_img = []
    train_seg = []
    val_img = []
    val_seg = []
    training_end_index = math.ceil(train_val_percentage / 100 * len(img_seg_dict))
    for ind, img in enumerate(list(img_seg_dict.keys())):
        if ind < training_end_index:
            train_img.append(img)
            train_seg.append(img_seg_dict[img])
        else:
            val_img.append(img)
            val_seg.append(img_seg_dict[img])
    logging.info('Create transformations')
    train_img_transforms, train_seg_transforms = data_loading.segmentation_train_transform(spatial_size=[96, 96, 96])
    val_img_transforms, val_seg_transforms = data_loading.segmentation_val_transform()

    # define dataset, data loader
    logging.info('Create training monai datasets')
    train_ds = NiftiDataset(train_img, train_seg, transform=train_img_transforms, seg_transform=train_seg_transforms)

    # define dataset, data loader
    logging.info('Create validation actual monai datasets')
    val_ds = NiftiDataset(val_img, val_seg, transform=val_img_transforms, seg_transform=val_seg_transforms)
    # We check if both the training and validation dataloaders can be created and used without immediate errors
    logging.info('Checking data loading')
    data_loading.data_loader_checker_first(train_ds, 'training')
    data_loading.data_loader_checker_first(val_ds, 'validation')
    logging.info('Init training done.')
    return train_ds, val_ds


def training_loop(img_path_list: Sequence, seg_path_list: Sequence, output_dir: Union[str, bytes, os.PathLike],
                  img_pref: str = None, device: str = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    train_ds, val_ds = init_training_data(img_path_list, seg_path_list, img_pref)
    train_loader = data_loading.create_training_data_loader(train_ds)
    val_loader = data_loading.create_validation_data_loader(val_ds)
    model = net.create_unet_model(device, net.default_unet_hyper_params)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=str(output_dir))
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 1:
        # if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = post_trans(val_outputs)
                    value, _ = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), Path(output_dir, "best_metric_model_segmentation3d_array.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

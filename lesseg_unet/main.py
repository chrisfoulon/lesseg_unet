import glob
import logging
import os
import shutil
import sys
import argparse
from pathlib import Path
import math

import ignite
import nibabel as nib
import numpy as np
import torch

from monai.config import print_config
from monai.data import ArrayDataset, create_test_image_3d
from monai.handlers import MeanDice, StatsHandler, TensorBoardImageHandler, TensorBoardStatsHandler
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from monai.utils import first
from lesseg_unet import utils, data_loading, tensorboard_utils
from bcblib.tools.nifti_utils import file_to_list

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # Script arguments
    parser = argparse.ArgumentParser(description='Monai unet')
    parser.add_argument('-o', '--output', type=str, help='output folder')
    nifti_paths_group = parser.add_mutually_exclusive_group(required=True)
    nifti_paths_group.add_argument('-p', '--input_path', type=str, help='Root folder of the b1000 dataset')
    nifti_paths_group.add_argument('-li', '--input_list', type=str, help='Text file containing the list of b1000')
    lesion_paths_group = parser.add_mutually_exclusive_group(required=True)
    lesion_paths_group.add_argument('-lp', '--lesion_input_path', type=str,
                                    help='Root folder of the b1000 dataset')
    lesion_paths_group.add_argument('-lli', '--lesion_input_list', type=str,
                                    help='Text file containing the list of b1000')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    parser.add_argument('-nw', '--num_workers', default=8, type=float, help='Number of torch workers')
    parser.add_argument('-ne', '--num_epochs', default=5, type=float, help='Number of epochs')
    parser.add_argument('-tb', '--show_tensorboard', action='store_true', help='Show tensorboard in the web browser')
    parser.add_argument('-tbp', '--tensorboard_port', help='Tensorboard port')
    parser.add_argument('-tbnw', '--tensorboard_new_window', action='store_true',
                        help='Open tensorboard in a new browser window')
    args = parser.parse_args()
    # print MONAI config
    print_config()
    # logs init
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Gather input data and setup based on script arguments
    output_root = Path(args.output)
    if not output_root.is_dir():
        raise ValueError('{} is not an existing directory'.format(output_root))
    logging.info('loading input dwi path list')
    if args.input_path is not None:
        img_list = utils.create_input_path_list_from_root(args.input_path)
        if args.input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.input_list is not None
    else:
        img_list = file_to_list(args.input_list)
    if args.output in img_list:
        raise ValueError("The output directory CANNOT be one of the input directories")
    logging.info('loading input lesion label path list')
    if args.lesion_input_path is not None:
        les_list = utils.create_input_path_list_from_root(args.lesion_input_path)
        if args.lesion_input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.lesion_input_list is not None
    else:
        les_list = file_to_list(args.lesion_input_list)
    if args.output in les_list:
        raise ValueError("The output directory CANNOT be one of the input directories")

    # match the lesion labels with the images
    logging.info('Matching the dwi and lesions')
    if args.image_prefix is not None:
        b1000_pref = args.image_prefix
    else:
        b1000_pref = None
        # b1000_pref = 'wodctH25_b1000'
    # images = []
    # images = [str(p) for p in img_list if b1000_pref in Path(p).name]
    # segs = []
    # segs = [str(p) for p in les_list if b1000_pref in Path(p).name]
    # segs = [str(p) for p in les_list if [Path(p).name in Path(pp).name for pp in images]]

    images, segs = data_loading.match_img_seg_by_names(img_list, les_list, b1000_pref)
    training_validation_cut = 75
    training_end_index = math.ceil(training_validation_cut / 100 * len(images))
    # Define transforms for image and segmentation
    logging.info('Create transformations')
    imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            AddChannel(),
            Resize((96, 96, 96)),
            # RandSpatialCrop((96, 96, 96), random_size=False),
            ToTensor(),
        ]
    )
    segtrans = Compose(
        [
            LoadImage(image_only=True),
            AddChannel(),
            Resize((96, 96, 96)),
            # RandSpatialCrop((96, 96, 96), random_size=False),
            ToTensor(),
        ]
    )
    # Define nifti dataset, dataloader
    logging.info('Create dataloader')
    ds = ArrayDataset(images, imtrans, segs, segtrans)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    im, seg = first(loader)

    logging.info('First loader batch size: images {}, lesions {}'.format(im.shape, seg.shape))
    # Create UNet, DiceLoss and Adam optimizer
    # device = torch.device("cuda:0")
    device = torch.device("cpu:0")
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss = DiceLoss(sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)
    logging.info('Create UNet, DiceLoss and Adam optimizer')
    # Create trainer
    logging.info('Create trainer')
    trainer = ignite.engine.create_supervised_trainer(net, opt, loss, device, False)
    # optional section for checkpoint and tensorboard logging
    # adding checkpoint handler to save models (network params and optimizer stats) during training
    logging.info('Optional checkpoint logging stuff')
    log_dir = Path(output_root, "logs")
    if not log_dir.is_dir():
        os.makedirs(log_dir)
    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        log_dir, "net", n_saved=10, require_empty=False
    )
    trainer.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not a loss value
    train_stats_handler = StatsHandler(name="trainer")
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir)
    train_tensorboard_stats_handler.attach(trainer)

    # If the tensorboard show option is selected we open a new tensorboard window/tab in the browser
    tb_port = '8008'
    tb_window = False
    if args.tensorboard_port is not None:
        tb_port = args.tensorboard_port
        args.show_tensorboard = True
    if args.tensorboard_new_window:
        tb_window = True
    if args.show_tensorboard:
        tensorboard_utils.open_tensorboard_page(log_dir, tb_port, tb_window)
    # optional section for model validation during training
    validation_every_n_epochs = 1
    # Set parameters for validation
    metric_name = "Mean_Dice"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: MeanDice()}
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    post_label = AsDiscrete(threshold_values=True)
    # Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = ignite.engine.create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: (post_pred(y_pred), post_label(y)),
    )
    logging.info('Validation data loader')
    # create a validation data loader
    val_imtrans = Compose(
        [LoadImage(image_only=True), ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), ToTensor()]
    )
    val_segtrans = Compose([LoadImage(image_only=True), AddChannel(), Resize((96, 96, 96)), ToTensor()])
    val_ds = ArrayDataset(images[training_end_index+1:], val_imtrans, segs[training_end_index+1:], val_segtrans)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available()
    )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
    def run_validation(engine):
        evaluator.run(val_loader)

    # Add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,  # fetch global epoch number from trainer
    )
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,  # fetch global epoch number from trainer
    )
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding label and model output in the last batch
    # here we draw the 3D output as GIF format along Depth axis, at every validation epoch
    val_tensorboard_image_handler = TensorBoardImageHandler(
        log_dir=log_dir,
        batch_transform=lambda batch: (batch[0], batch[1]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda x: trainer.state.epoch,
    )
    evaluator.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED, handler=val_tensorboard_image_handler,
    )

    # create a training data loader
    logging.info('Running loop!')
    train_ds = ArrayDataset(images[:training_end_index], imtrans, segs[:training_end_index], segtrans)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=5, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    train_epochs = args.num_epochs
    state = trainer.run(train_loader, train_epochs)


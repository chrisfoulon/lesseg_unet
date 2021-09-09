import logging
import sys
import argparse
from pathlib import Path
import os
import json
from datetime import datetime

from monai.config import print_config
from lesseg_unet import utils, training, segmentation
from lesseg_unet.data import transform_dicts
from bcblib.tools.nifti_utils import file_to_list
import lesseg_unet.data.transform_dicts as tr_dicts

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # Script arguments
    parser = argparse.ArgumentParser(description='Monai unet training')
    # Paths
    parser.add_argument('-o', '--output', type=str, help='output folder', required=True)
    nifti_paths_group = parser.add_mutually_exclusive_group(required=True)
    nifti_paths_group.add_argument('-p', '--input_path', type=str, help='Root folder of the b1000 dataset')
    nifti_paths_group.add_argument('-li', '--input_list', type=str, help='Text file containing the list of b1000')
    nifti_paths_group.add_argument('-sid', '--seg_input_dict', type=str,
                                   help='[Segmentation only] The path to a json dict containing the keys of the '
                                        'different populations (subfolder names) with the list of images paths'
                                        '[Cannot be used for Validation]')
    # # Hidden param, do not use
    # nifti_paths_group.add_argument('-bm', '--benchmark', action='store_true', help=argparse.SUPPRESS)
    lesion_paths_group = parser.add_mutually_exclusive_group(required=False)
    lesion_paths_group.add_argument('-lp', '--lesion_input_path', type=str,
                                    help='Root folder of the b1000 dataset')
    lesion_paths_group.add_argument('-lli', '--lesion_input_list', type=str,
                                    help='Text file containing the list of b1000')
    control_paths_group = parser.add_mutually_exclusive_group(required=False)
    control_paths_group.add_argument('-ctr', '--controls_path', type=str,
                                     help='folder containing the control images (image_prefix not applied)')
    control_paths_group.add_argument('-lctr', '--controls_list', type=str,
                                     help='file path of the list of control images (image_prefix not applied)')
    parser.add_argument('-trs', '--transform_dict', type=str,
                                     help='file path to a json dictionary of transformations')
    parser.add_argument('-lfct', '--loss_function', type=str, default='dice',
                        help='Loss function used for training')
    parser.add_argument('-wf', '--weight_factor', type=float, default=1,
                        help='Multiply the control loss by this factor')
    parser.add_argument('-vlfct', '--val_loss_function', type=str, default='dice',
                        help='Loss function used for validation')
    default_label_group = parser.add_mutually_exclusive_group(required=False)
    default_label_group.add_argument('-cdl', '--create_default_label', action='store_true',
                                     help='Creates a default image filled with zeros for control images label')
    default_label_group.add_argument('-dl', '--default_label', type=str,
                                     help='Path to a default mask that will be used in case no '
                                          'lesion mask is found for the b1000')
    parser.add_argument('-lab_smo', '--label_smoothing', action='store_true',
                        help='Apply a label smoothing during the training')
    parser.add_argument('-pt', '--checkpoint', type=str, help='file path to a torch checkpoint file')
    parser.add_argument('-d', '--torch_device', type=str, help='Device type and number given to'
                                                               'torch.device()')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    # TODO maybe
    # parser.add_argument('-ctr_pref', '--control_image_prefix', type=str,
    #                     help='Define a prefix to filter the control images')
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('-bs', '--batch_size', default=10, type=int, help='Batch size for the training loop')
    parser.add_argument('-vbs', '--val_batch_size', default=10, type=int, help='Batch size for the validation loop')
    parser.add_argument('-ne', '--num_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-tv', '--train_val', type=int, help='Training / validation percentage cut')
    parser.add_argument('-sbe', '--stop_best_epoch', type=int, help='Number of epochs without improvement before it '
                                                                    'stops')
    # parser.add_argument('-ns', '--num_nifti_save', default=25, type=int, help='Number of niftis saved '
    #                                                                           'during validation')
    parser.add_argument('-dropout', type=float, help='Set a dropout value for the model')
    parser.add_argument('-nf', '--folds_number', default=1, type=int, help='Set a dropout value for the model')
    parser.add_argument('-kmos', '--keep_model_output_size', action='store_true', help='Keep the output of the '
                                                                                       'segmentation in the '
                                                                                       'spatial_size of the model')
    args = parser.parse_args()
    # print MONAI config
    print_config()
    # logs init
    if args.stop_best_epoch is None:
        stop_best_epoch = -1
    else:
        stop_best_epoch = args.stop_best_epoch

    # Gather input data and setup based on script arguments
    output_root = Path(args.output)
    os.makedirs(output_root, exist_ok=True)

    log_file_path = str(Path(output_root, '__logging_training.txt'))
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO)
    # if args.default_label is not None:
    #     logging.info(f'{args.default_label} will be used to fill up missing labels')
    # if args.create_default_label:
    #     args.default_label = output_root
    #     print(f'Missing labels will be replace by an zero-filled default image')
    if not output_root.is_dir():
        raise ValueError('{} is not an existing directory and could not be created'.format(output_root))
    print('loading input dwi path list')
    seg_input_dict = {}
    if args.input_path is not None:
        logging.info(f'Input image directory : {args.input_path}')
        img_list = utils.create_input_path_list_from_root(args.input_path)
        if args.input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.input_list is not None
    elif args.input_list is not None:
        logging.info(f'Input image list : {args.input_list}')
        img_list = file_to_list(args.input_list)
    else:
        seg_input_dict = json.load(open(args.seg_input_dict, 'r'))
        img_list = [img_path for sublist in seg_input_dict for img_path in sublist]
    if args.output in img_list:
        raise ValueError("The output directory CANNOT be one of the input directories")
    print('loading input lesion label path list')
    if args.lesion_input_path is not None:
        logging.info(f'Input lesion directory : {args.lesion_input_path}')
        les_list = utils.create_input_path_list_from_root(args.lesion_input_path)
        if args.lesion_input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.lesion_input_list is not None
    elif args.lesion_input_list is not None:
        logging.info(f'Input lesion list : {args.input_list}')
        les_list = file_to_list(args.lesion_input_list)
    else:
        les_list = None

    if args.controls_path is not None:
        ctr_list = utils.create_input_path_list_from_root(args.controls_path)
        if args.controls_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.lesion_input_list is not None
    elif args.controls_list is not None:
        ctr_list = file_to_list(args.controls_list)
    else:
        ctr_list = None

    # match the lesion labels with the images
    print('Matching the dwi and lesions')
    if args.image_prefix is not None:
        b1000_pref = args.image_prefix
    else:
        b1000_pref = None
        # b1000_pref = 'wodctH25_b1000'

    if args.transform_dict is not None:
        td = args.transform_dict
        if Path(td).is_file():
            transform_dict = utils.load_json_transform_dict(args.transform_dict)
        else:
            if td in dir(tr_dicts):
                transform_dict = getattr(tr_dicts, td)
            else:
                raise ValueError('{} is not an existing dict file or is not '
                                 'in lesseg_unet/data/transform_dicts.py'.format(args.transform_dict))
    else:
        print('Using default transformation dictionary')
        transform_dict = transform_dicts.minimal_hyper_dict
    train_val_percentage = None
    if args.train_val is not None:
        train_val_percentage = args.train_val
    if args.checkpoint is None and args.seg_input_dict is None:
        if train_val_percentage is None:
            train_val_percentage = 75
        # if les_list is None and args.default_label is None:
        #     parser.error(message='For the training, there must be a list of labels')
        logging.info(f'Output training folder : {output_root}')
        training.training_loop(img_path_list=img_list,
                               seg_path_list=les_list,
                               output_dir=output_root,
                               ctr_path_list=ctr_list,
                               img_pref=b1000_pref,
                               transform_dict=transform_dict,
                               device=args.torch_device,
                               batch_size=args.batch_size,
                               val_batch_size=args.val_batch_size,
                               epoch_num=args.num_epochs,
                               dataloader_workers=args.num_workers,
                               # num_nifti_save=args.num_nifti_save,
                               train_val_percentage=train_val_percentage,
                               label_smoothing=args.label_smoothing,
                               stop_best_epoch=stop_best_epoch,
                               # default_label=args.default_label,
                               training_loss_fct=args.loss_function,
                               val_loss_fct=args.val_loss_function,
                               weight_factor=args.weight_factor,
                               folds_number=args.folds_number,
                               dropout=args.dropout)
    else:
        if args.checkpoint is None:
            raise ValueError('A checkpoint must be given for the segmentation or validation')
        if train_val_percentage is None:
            train_val_percentage = 0
        if args.seg_input_dict is not None:
            if les_list is not None:
                raise ValueError('seg_input_dict cannot be used for the Validation')
        if les_list is None:
            if seg_input_dict:
                for sub_folder in seg_input_dict:
                    logging.info(f'Input image subfolder : {sub_folder}')
                    logging.info(f'Output segmentation folder : {Path(output_root, sub_folder)}')
                    os.makedirs(Path(output_root, sub_folder), exist_ok=True)
                    segmentation.segmentation_loop(seg_input_dict[sub_folder],
                                                   Path(output_root, sub_folder),
                                                   args.checkpoint,
                                                   b1000_pref,
                                                   transform_dict=transform_dict,
                                                   device=args.torch_device,
                                                   dataloader_workers=args.num_workers)
            else:
                logging.info(f'Output segmentation folder : {output_root}')
                segmentation.segmentation_loop(img_list,
                                               output_root,
                                               args.checkpoint,
                                               b1000_pref,
                                               transform_dict=transform_dict,
                                               device=args.torch_device,
                                               dataloader_workers=args.num_workers)
        else:
            logging.info(f'Output validation folder : {output_root}')
            segmentation.validation_loop(img_list, les_list,
                                         output_root,
                                         args.checkpoint,
                                         b1000_pref,
                                         transform_dict=transform_dict,
                                         device=args.torch_device,
                                         dataloader_workers=args.num_workers,
                                         train_val_percentage=train_val_percentage,
                                         # default_label=args.default_label
                                         )


if __name__ == "__main__":
    main()

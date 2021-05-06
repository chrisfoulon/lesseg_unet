import logging
import sys
import argparse
from pathlib import Path
import os

from monai.config import print_config
from lesseg_unet import utils, training, segmentation
from lesseg_unet.data import transform_dicts
from bcblib.tools.nifti_utils import file_to_list
import lesseg_unet.data.transform_dicts as tr_dicts

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # Script arguments
    parser = argparse.ArgumentParser(description='Monai unet training')
    parser.add_argument('-o', '--output', type=str, help='output folder', required=True)
    nifti_paths_group = parser.add_mutually_exclusive_group(required=True)
    nifti_paths_group.add_argument('-p', '--input_path', type=str, help='Root folder of the b1000 dataset')
    nifti_paths_group.add_argument('-li', '--input_list', type=str, help='Text file containing the list of b1000')
    lesion_paths_group = parser.add_mutually_exclusive_group(required=False)
    lesion_paths_group.add_argument('-lp', '--lesion_input_path', type=str,
                                    help='Root folder of the b1000 dataset')
    lesion_paths_group.add_argument('-lli', '--lesion_input_list', type=str,
                                    help='Text file containing the list of b1000')
    parser.add_argument('-trs', '--transform_dict', type=str, help='file path to a json dictionary of transformations')
    parser.add_argument('-lab_smo', '--label_smoothing', action='store_true',
                        help='Apply a label smoothing during the training')
    parser.add_argument('-pt', '--checkpoint', type=str, help='file path to a torch checkpoint file')
    parser.add_argument('-d', '--torch_device', type=str, help='Device type and number given to'
                                                               'torch.device()')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('-ne', '--num_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-tv', '--train_val', type=int, help='Training / validation percentage cut')
    # parser.add_argument('-ns', '--num_nifti_save', default=25, type=int, help='Number of niftis saved '
    #                                                                           'during validation')
    nifti_paths_group.add_argument('-bm', '--benchmark', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    # print MONAI config
    print_config()
    # logs init
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if args.benchmark:
        output_root = Path(args.output)
        os.makedirs(output_root, exist_ok=True)
        if not output_root.is_dir():
            raise ValueError('{} is not an existing directory and could not be created'.format(output_root))
        param_dict = {
            '-o': output_root,
            '-p': '/media/chrisfoulon/DATA1/z_Zetas/a_pre_processing_lesions_fil_norm/',
            '-lp': '/media/chrisfoulon/DATA1/z_Zetas/w_les_2mm_H25_kde_zeta/',
            # '-p': '/data/UCL_Data/a_pre_processing_lesions_fil_norm/',
            # '-lp': '/data/UCL_Data/w_les_2mm_H25_kde_zeta/',
            '-nw': 16,
            '-pref': 'wodctH25_b1000',
            '-ne': 2000,
        }
        value_dict = {
            '-trs': ['minimal_hyper_dict', 'minimal_hyper_dict_cc', 'full_hyper_dict', 'full_hyper_dict_cc',
                     'new_full_dict_cc'],
            '-o': ['minimal', 'minimal_cc', 'full', 'full_cc', 'new_full_dict_cc']
        }
        img_list = utils.create_input_path_list_from_root(param_dict['-p'])
        les_list = utils.create_input_path_list_from_root(param_dict['-lp'])
        b1000_pref = param_dict['-pref']

        for ind, tr_dict in enumerate(value_dict['-trs']):
            transform_dict = getattr(tr_dicts, tr_dict)
            output_dir = Path(output_root, value_dict['-o'][ind])
            checkpoint = Path(output_dir, 'best_metric_model_segmentation3d_array_epo.pth')
            if args.checkpoint is None:
                if not checkpoint.is_file():
                    training.training_loop(img_list, les_list, output_dir, b1000_pref,
                                           transform_dict=transform_dict,
                                           epoch_num=param_dict['-ne'],
                                           dataloader_workers=param_dict['-nw'],
                                           label_smoothing=False,
                                           stop_best_epoch=25)
            else:
                segmentation.validation_loop(img_list, les_list,
                                             output_dir,
                                             checkpoint,
                                             b1000_pref,
                                             transform_dict=transform_dict,
                                             dataloader_workers=param_dict['-nw'],
                                             train_val_percentage=75)
    else:
        # Gather input data and setup based on script arguments
        output_root = Path(args.output)
        os.makedirs(output_root, exist_ok=True)
        if not output_root.is_dir():
            raise ValueError('{} is not an existing directory and could not be created'.format(output_root))
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
        elif args.lesion_input_list is not None:
            les_list = file_to_list(args.lesion_input_list)
        else:
            les_list = None
        if les_list is not None and args.output in les_list:
            raise ValueError("The output directory CANNOT be one of the input directories")

        # match the lesion labels with the images
        logging.info('Matching the dwi and lesions')
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
        label_smoothing = False
        if args.label_smoothing:
            label_smoothing = True
        if args.checkpoint is None:
            if train_val_percentage is None:
                train_val_percentage = 75
            training.training_loop(img_list, les_list, output_root, b1000_pref,
                                   transform_dict=transform_dict,
                                   device=args.torch_device,
                                   epoch_num=args.num_epochs,
                                   dataloader_workers=args.num_workers,
                                   # num_nifti_save=args.num_nifti_save,
                                   train_val_percentage=train_val_percentage,
                                   label_smoothing=True)
        else:
            if train_val_percentage is None:
                train_val_percentage = 0
            if les_list is None:
                segmentation.segmentation_loop(img_list,
                                               output_root,
                                               args.checkpoint,
                                               b1000_pref,
                                               transform_dict=transform_dict,
                                               device=args.torch_device,
                                               dataloader_workers=args.num_workers)
            else:
                segmentation.validation_loop(img_list, les_list,
                                             output_root,
                                             args.checkpoint,
                                             b1000_pref,
                                             transform_dict=transform_dict,
                                             device=args.torch_device,
                                             dataloader_workers=args.num_workers,
                                             train_val_percentage=train_val_percentage)


if __name__ == "__main__":
    main()

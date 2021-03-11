import logging
import sys
import argparse
from pathlib import Path

from monai.config import print_config
from lesseg_unet import utils, training
from lesseg_unet.data import transform_dicts
from bcblib.tools.nifti_utils import file_to_list
import lesseg_unet.data.transform_dicts as tr_dicts

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
    parser.add_argument('-trs', '--transform_dict', type=str, help='file path to a json dictionary of transformations')
    parser.add_argument('-d', '--torch_device', type=str, help='Device type and number given to'
                                                               'torch.device()')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('-ne', '--num_epochs', default=50, type=int, help='Number of epochs')
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

    training.training_loop(img_list, les_list, output_root, b1000_pref,
                           transform_dict=transform_dict,
                           device=args.torch_device,
                           epoch_num=args.num_epochs,
                           dataloader_workers=args.num_workers)


if __name__ == "__main__":
    main()

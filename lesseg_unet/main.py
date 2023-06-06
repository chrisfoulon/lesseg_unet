import logging
import sys
import argparse
from pathlib import Path
import os
import json
import re

from monai.config import print_config
from lesseg_unet import utils, training, segmentation
from bcblib.tools.nifti_utils import file_to_list, overlaps_subfolders, nifti_overlap_images
import lesseg_unet.data.transform_dicts as tr_dicts
import nibabel as nib
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
    nifti_paths_group.add_argument('-psl', '--pretrained_split_list', type=str,
                                   help='File containing split paths lists of the k-fold')

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
    # Tranformation
    parser.add_argument('-trs', '--transform_dict', type=str,
                        help='file path to a json dictionary of transformations')
    parser.add_argument('-clamp', action='store_true', help='Apply intensity clamping (with default value if not given'
                                                            ' with --clamp_low and --clamp_high)')
    parser.add_argument('-cl', '--clamp_low', type=float, help='Define the low quantile of intensity clamping')
    parser.add_argument('-ch', '--clamp_high', type=float, help='Define the high quantile of intensity clamping')
    parser.add_argument('-clamp_lesion_set', action='store_true',
                        help='Apply intensity clamping on the training lesioned set as well'
                             ' (with default value if not given with --clamp_low and --clamp_high)')
    parser.add_argument('--resize', type=str, help='Resize the images to the given spatial size')
    # Losses and metric parameters
    parser.add_argument('-lfct', '--loss_function', type=str, default='dice',
                        help='Loss function used for training')
    parser.add_argument('-ctrfct', '--ctr_loss_function', type=str, default='binary_empty_label',
                        help='Loss function used for training on controls')
    parser.add_argument('-vlfct', '--val_loss_function', type=str, default='dice',
                        help='Loss function used for validation')
    parser.add_argument('-wf', '--weight_factor', type=float, default=1,
                        help='Multiply the control loss by this factor')
    parser.add_argument('-ema', action="store_true", help='Use EMA')
    parser.add_argument('-tema', '--track_ema', action="store_true", help='Track EMA')
    parser.add_argument('-nboc', '--no_backward_on_controls', action="store_true", help='No backward on controls')

    # Parameters for control data
    parser.add_argument('-ctr_pref', '--ctrl_image_prefix', type=str,
                        help='Define a prefix to filter the control images')
    # Segmentation parameters
    parser.add_argument('-pt', '--checkpoint', type=str, help='file path to a torch checkpoint file'
                                                              ' or directory (in that case, the most recent '
                                                              'checkpoint will be used)')
    parser.add_argument('-om', '--output_mode', type=str, default='segmentation',
                        help='Select the segmentation output mode (segmentation, sigmoid, logits)')
    parser.add_argument('-sa', '--segmentation_area', action='store_true', help='Associate the segmentated masks'
                                                                                ' to lesion areas in a csv file')
    parser.add_argument('-overlap', action='store_true', help='Create the overlap of the segmentations')
    parser.add_argument('-kmos', '--keep_model_output_size', action='store_true', help='Keep the output of the '
                                                                                       'segmentation in the '
                                                                                       'spatial_size of the model')
    parser.add_argument('-kifs', '--keep_input_folder_structure',
                        action='store_true', help='Segmentation output will reflect the folder structure from'
                                                  ' the given root_folder')

    # Model parameters
    parser.add_argument('-pp', '--pretrained_point', type=str, help='[Training opt]file path to a torch checkpoint file'
                                                                    ' or directory (in that case, the most recent '
                                                                    'checkpoint will be used)')
    parser.add_argument('-mt', '--model_type', type=str, default='UNETR',
                        help='Select the model architecture (UNet, UNETR, SWINUNETR)')
    parser.add_argument('-d', '--torch_device', type=str, help='Device type and number given to'
                                                               'torch.device()')
    parser.add_argument('-dropout', type=float, help='Set a dropout value for the model')
    parser.add_argument('-fs', '--feature_size', type=int, help='Set the feature size for (SWIN)UNETR')
    # Gradient accumulation
    parser.add_argument('-ga', '--gradient_accumulation', type=int, default=1,
                        help='Number of batches to accumulate before performing a backward/update pass')
    # Mixed precision
    parser.add_argument('-dmp', '--disable_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    # Files split and matching options
    parser.add_argument('-tv', '--train_val', type=int, help='Training / validation percentage cut')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    parser.add_argument('-ics', '--image_cut_suffix', type=str,
                        help='Suffix to cut the lesion file name (keep left part) in case the filename without the '
                             '.nii extension cannot be found')
    parser.add_argument('-nf', '--folds_number', default=1, type=int, help='Set a dropout value for the model')
    # Datasets and Loaders parameters
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('-bs', '--batch_size', default=10, type=int, help='Batch size for the training loop')
    parser.add_argument('-vbs', '--val_batch_size', default=10, type=int, help='Batch size for the validation loop')
    parser.add_argument('-cache', '--cache', action='store_true',
                        help='Cache the non-random transformation in cache in output directory')
    parser.add_argument('-cn', '--cache_num', type=int, default=0,
                        help='Number of images to be cached with CacheDataset (default)')
    parser.add_argument('-cr', '--cache_rate', type=int, default=0,
                        help='Rate cached images with CacheDataset (default)')
    # Epochs parameters
    parser.add_argument('-ne', '--num_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-sbe', '--stop_best_epoch', type=int, help='Number of epochs without improvement before it '
                                                                    'stops')
    # DDP arguments
    # parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--local_rank", type=int, help="node rank for distributed training")
    parser.add_argument("-cvd", "--cuda_visible_devices", type=str, help="List of visible devices for cuda")
    # DEBUG options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('-din', '--debug_img_num', type=int, help='Number of images from the input list')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    kwargs = {}

    # Gather input data and setup based on script arguments
    if args.output_mode != 'segmentation':
        output_root = Path(args.output + '_' + args.output_mode)
    else:
        output_root = Path(args.output)
    args.output = output_root
    os.makedirs(output_root, exist_ok=True)
    # TODO FIND A SOLUTION TO WRITE IN A FILE
    if args.debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    log_file_path = str(Path(output_root, '__logging_training.txt'))
    logging.basicConfig(filename=log_file_path, level=logging_level)
    file_handler = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(file_handler)
    logging.info('log file stored in {}'.format(log_file_path))
    if not Path(log_file_path).is_file():
        print(f'{log_file_path} was not created! Thanks ddp ....')
    if unknown:
        kwargs = utils.kwargs_argparse(unknown)
        print(f'Unlisted arguments : {kwargs}')
        # resp = input('If the additional parameters you entered are not what you wanted type quit/q/stop/s/no/n')
        # if resp.lower() in ['quit', 'q', 'stop', 's', 'no', 'n']:
        #     print('Sorry little parameter but, your parents never wanted you. Good bye.')
        #     exit()
    # print MONAI config
    print_config()
    with open(Path(args.output, 'run_command.txt'), 'w+') as f:
        f.write(' '.join(sys.argv))

    if args.checkpoint is not None:
        args.local_rank = 0
    """
    Set enrivonment variables to overwrite torchrun default config and allow 
    1) more open files allowed
    2) Max OpenMP threads increased to the number of workers
    3) also set the number of threads in torch to the number of workers 
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    torch.set_num_threads(args.num_workers)
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    if args.local_rank is not None:
        # Starting the model manually
        local_rank = args.local_rank
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            if args.checkpoint is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['0'])
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['0', '1'])
        print('Using local_rank directly')
    else:
        # Starting the model using torchrun
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['0', '1'])
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['0'])
        print('Using torchrun handler')
        local_rank = int(os.environ["LOCAL_RANK"])

    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        # os.environ['WORLD_SIZE'] = '2'
    main_worker(local_rank=local_rank, args=args, kwargs=kwargs)


def main_worker(local_rank, args, kwargs):
    # logging.basicConfig(filename=log_file_path, level=logging_level, encoding='utf-8', filemode='w', force=True)
    # file_handler = logging.StreamHandler(sys.stdout)
    # logging.getLogger().addHandler(file_handler)
    # TODO Organise the code the same way as the argparse
    # scaler = torch.cuda.amp.GradScaler()
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '1234'
    args.world_size = int(os.environ['WORLD_SIZE'])
    print(f'WORLD SIZE: {args.world_size}')
    print(f'################RANK : {local_rank}####################')

    print("Waiting for all DDP processes to establish contact...")
    if args.torch_device != 'cpu':
        dist.init_process_group('nccl', rank=local_rank, world_size=args.world_size)
    else:
        dist.init_process_group('gloo', rank=local_rank, world_size=args.world_size)
    print('Contact established!')
    # logs init
    if args.stop_best_epoch is None:
        stop_best_epoch = -1
    else:
        stop_best_epoch = args.stop_best_epoch
    if args.debug:
        print(torch.__config__.parallel_info())

    output_root = Path(args.output)
    if not output_root.is_dir():
        raise ValueError('{} is not an existing directory and could not be created'.format(output_root))
    cache_dir = None
    if args.cache:
        cache_dir = Path(output_root, 'cache')
    utils.print_rank_0('loading input dwi path list', dist.get_rank())
    seg_input_dict = {}
    if args.input_path is not None:
        utils.logging_rank_0(f'Input image directory : {args.input_path}', dist.get_rank())
        img_list = utils.create_input_path_list_from_root(args.input_path)
        if args.input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.input_list is not None
    elif args.input_list is not None:
        utils.logging_rank_0(f'Input image list : {args.input_list}', dist.get_rank())
        img_list = file_to_list(args.input_list)
    elif args.seg_input_dict is not None:
        with open(args.seg_input_dict, 'r') as f:
            seg_input_dict = json.load(f)
        img_list = [img_path for sublist in seg_input_dict for img_path in sublist]
    else:
        img_list = utils.open_json(args.pretrained_split_list)
    # TODO not very pretty in the case of pretrained_split_list ...
    if args.output in img_list:
        raise ValueError("The output directory CANNOT be one of the input directories")
    if args.debug_img_num is not None:
        img_list = img_list[:args.debug_img_num]
    utils.print_rank_0('loading input lesion label path list', dist.get_rank())
    if args.lesion_input_path is not None:
        logging.info(f'Input lesion directory : {args.lesion_input_path}')
        les_list = utils.create_input_path_list_from_root(args.lesion_input_path)
        if args.lesion_input_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.lesion_input_list is not None
    elif args.lesion_input_list is not None:
        utils.print_rank_0(f'Input lesion list : {args.lesion_input_list}', dist.get_rank())
        les_list = file_to_list(args.lesion_input_list)
    else:
        les_list = None
    # if args.debug_img_num is not None and les_list is not None:
    #     img_list = les_list[:args.debug_img_num]
    if args.controls_path is not None:
        ctr_list = utils.create_input_path_list_from_root(args.controls_path, pref=args.ctrl_image_prefix)
        if args.controls_path == args.output:
            raise ValueError("The output directory CANNOT be the input directory")
    # So args.lesion_input_list is not None
    elif args.controls_list is not None:
        ctr_list = file_to_list(args.controls_list)
        ctr_list = utils.get_str_path_list(ctr_list, pref=args.ctrl_image_prefix)
    else:
        ctr_list = None
    if args.image_prefix is not None:
        b1000_pref = args.image_prefix
    else:
        b1000_pref = None

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
        transform_dict = None
    # Clamping or not clamping
    if args.clamp_low is not None:
        if args.clamp_high is not None:
            clamp_tuple = (args.clamp_low, args.clamp_high)
        else:
            clamp_tuple = (args.clamp_low, 1)
    elif args.clamp_high is not None:
        clamp_tuple = (0, args.clamp_high)
    else:
        if args.clamp:
            clamp_tuple = (.00005, .99995)
        else:
            clamp_tuple = None
    clamp_lesion_set = None
    if args.clamp_lesion_set:
        if clamp_tuple is None:
            clamp_lesion_set = (.00005, .99995)
        else:
            clamp_lesion_set = clamp_tuple
    if clamp_lesion_set is not None:
        utils.logging_rank_0(f'Clamping of training set : {clamp_lesion_set}', dist.get_rank())
    if clamp_tuple is not None:
        utils.logging_rank_0(f'Clamping of control set: {clamp_tuple}', dist.get_rank())
    if args.resize is not None:
        args.resize = [int(val) for val in re.split(r'\D+', args.resize)]
    train_val_percentage = None
    if args.train_val is not None:
        train_val_percentage = args.train_val
    if args.checkpoint is None and args.seg_input_dict is None:
        if train_val_percentage is None:
            train_val_percentage = 75
        # if les_list is None and args.default_label is None:
        #     parser.error(message='For the training, there must be a list of labels')
        utils.logging_rank_0(f'Output training folder : {output_root}', dist.get_rank())
        if args.pretrained_point is not None:
            if Path(args.pretrained_point).is_dir():
                pretrained_point = utils.get_best_epoch_from_folder(args.pretrained_point)
                if pretrained_point == '':
                    raise ValueError(f'Checkpoint could not be found in {args.pretrained_point}')
                else:
                    utils.logging_rank_0(f'Latest checkpoint found is: {pretrained_point}', dist.get_rank())
            else:
                # So it quickly breaks if the checkpoint does not exist
                pretrained_point = str(Path(args.pretrained_point))
        else:
            pretrained_point = None
        training.training(img_path_list=img_list,
                          lbl_path_list=les_list,
                          output_dir=output_root,
                          ctr_path_list=ctr_list,
                          img_pref=b1000_pref,
                          image_cut_suffix=args.image_cut_suffix,
                          transform_dict=transform_dict,
                          pretrained_point=pretrained_point,
                          model_type=args.model_type,
                          device=args.torch_device,
                          batch_size=args.batch_size,
                          val_batch_size=args.val_batch_size,
                          epoch_num=args.num_epochs,
                          gradient_accumulation_steps=args.gradient_accumulation,
                          dataloader_workers=args.num_workers,
                          train_val_percentage=train_val_percentage,
                          lesion_set_clamp=clamp_lesion_set,
                          controls_clamping=clamp_tuple,
                          resize=args.resize,
                          # label_smoothing=args.label_smoothing,
                          stop_best_epoch=stop_best_epoch,
                          training_loss_fct=args.loss_function,
                          ctr_loss_fct=args.ctr_loss_function,
                          val_loss_fct=args.val_loss_function,
                          weight_factor=args.weight_factor,
                          folds_number=args.folds_number,
                          dropout=args.dropout,
                          cache_dir=cache_dir,
                          world_size=args.world_size,
                          rank=local_rank,
                          cache_num=args.cache_num,
                          enable_amp=not args.disable_mixed_precision,
                          use_ema=args.ema,
                          track_ema=args.track_ema,
                          no_backward_on_controls=args.no_backward_on_controls,
                          debug=args.debug,
                          **kwargs)
    else:
        if args.checkpoint is None:
            raise ValueError('A checkpoint must be given for the segmentation or validation')
        else:
            if Path(args.checkpoint).is_dir():
                checkpoint = utils.get_best_epoch_from_folder(args.checkpoint)
                if checkpoint == '':
                    raise ValueError(f'Checkpoint could not be found in {args.checkpoint}')
                else:
                    utils.logging_rank_0(f'Latest checkpoint found is: {checkpoint}', dist.get_rank())
            else:
                # So it quickly breaks if the checkpoint does not exist
                checkpoint = str(Path(args.checkpoint))
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
                                                   checkpoint,
                                                   b1000_pref,
                                                   image_cut_suffix=args.image_cut_suffix,
                                                   transform_dict=transform_dict,
                                                   output_mode=args.output_mode,
                                                   device=args.torch_device,
                                                   dataloader_workers=args.num_workers,
                                                   clamping=clamp_tuple,
                                                   segmentation_area=args.segmentation_area,
                                                   keep_input_folder_structure=args.keep_input_folder_structure,
                                                   **kwargs)
                if args.overlap:
                    overlaps_subfolders(output_root, 'output_')
            else:
                logging.info(f'Output segmentation folder : {output_root}')
                segmentation.segmentation_loop(img_list,
                                               output_root,
                                               checkpoint,
                                               b1000_pref,
                                               image_cut_suffix=args.image_cut_suffix,
                                               transform_dict=transform_dict,
                                               output_mode=args.output_mode,
                                               device=args.torch_device,
                                               dataloader_workers=args.num_workers,
                                               clamping=clamp_tuple,
                                               segmentation_area=args.segmentation_area,
                                               keep_input_folder_structure=args.keep_input_folder_structure,
                                               **kwargs)
                if args.overlap:
                    nib.save(nifti_overlap_images(output_root, 'output_', recursive=True),
                             Path(output_root, 'overlap_segmentation.nii'))
            # if args.segmentation_area:
            #     output_img_list = [p for p in Path(output_root).rglob('*')
            #                        if p.name.startswith('output_') and bcblib.tools.nifti_utils.is_nifti(p)]
            #     segmentation_areas_dict = utils.get_segmentation_areas(
            #         output_img_list, comp_meth='dice', cluster_thr=0.1, root_dir_path=output_root)
            #     pd.DataFrame().from_dict(segmentation_areas_dict).to_csv(Path(output_root, 'segmentation_areas.csv'))

        else:
            logging.info(f'Output validation folder : {output_root}')
            segmentation.validation_loop(img_list, les_list,
                                         output_root,
                                         checkpoint,
                                         b1000_pref,
                                         image_cut_suffix=args.image_cut_suffix,
                                         transform_dict=transform_dict,
                                         device=args.torch_device,
                                         dataloader_workers=args.num_workers,
                                         # train_val_percentage=train_val_percentage,
                                         # default_label=args.default_label
                                         clamping=clamp_tuple,
                                         segmentation_area=args.segmentation_area,
                                         keep_input_folder_structure=args.keep_input_folder_structure,
                                         **kwargs)
            if args.overlap:
                nib.save(nifti_overlap_images(output_root, 'output_', recursive=True),
                         Path(output_root, 'overlap_segmentation.nii'))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

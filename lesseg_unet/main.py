import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime
import os
import json
import re

from monai.config import print_config
from lesseg_unet import utils, training, segmentation
from lesseg_unet.data_utils import (
    # Legacy functions (deprecated)
    folder_mode_to_split_lists,
    list_nifti_from_folders,
    match_modalities_by_subject,
    # New staged pipeline functions
    read_folder,
    match_lists_to_dicts,
    shuffle_and_split_subjects,
)
from lesseg_unet.hardware import get_hardware_profile
from lesseg_unet.auto_config import AutoConfigurator, DatasetProfile, TrainingConfig
from bcblib.tools.nifti_utils import file_to_list, overlaps_subfolders, nifti_overlap_images
import lesseg_unet.data.transform_dicts as tr_dicts
import nibabel as nib
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def validate_modality_names(names, reserved):
    """Validate modality names are valid identifiers and not reserved.

    Parameters
    ----------
    names : list of str
        List of modality names to validate
    reserved : set of str
        Set of reserved names that cannot be used

    Raises
    ------
    ValueError
        If any name is invalid (not a Python identifier), is reserved, or duplicates exist

    Examples
    --------
    >>> validate_modality_names(['dwi', 'adc'], reserved={'label', 'control'})
    # Passes without error

    >>> validate_modality_names(['label'], reserved={'label'})
    ValueError: Modality name 'label' is reserved.
    """
    for name in names:
        if not name.isidentifier():
            raise ValueError(
                f"Invalid modality name '{name}'. Must be a valid Python identifier "
                f"(alphanumeric and underscores only, cannot start with number)."
            )
        if name in reserved:
            raise ValueError(
                f"Modality name '{name}' is reserved. Choose a different name."
            )

    # Check for duplicates
    if len(set(names)) != len(names):
        duplicates = [name for name in set(names) if names.count(name) > 1]
        raise ValueError(
            f"Duplicate modality names detected: {duplicates}"
        )


def validate_multimodal_arguments(args):
    """Validate multi-value argument combinations.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    argparse.Namespace
        Validated arguments (same object, returned for chaining)

    Raises
    ------
    ValueError
        If argument combinations are invalid (e.g., name count mismatch)

    Examples
    --------
    Multi-modal images with explicit names:
    >>> args.input_path = ['/data/dwi', '/data/adc']
    >>> args.image_modality_names = ['dwi', 'adc']
    >>> validate_multimodal_arguments(args)  # Passes

    Missing names (will use defaults):
    >>> args.input_path = ['/data/mod1', '/data/mod2']
    >>> args.image_modality_names = None
    >>> validate_multimodal_arguments(args)  # Passes, will auto-generate names
    """
    # Validate image modalities
    if args.input_path and len(args.input_path) > 1:
        # Filter out empty strings from paths
        args.input_path = [p for p in args.input_path if p]
        if args.image_modality_names:
            # Filter out empty strings
            args.image_modality_names = [name for name in args.image_modality_names if name]
            if len(args.image_modality_names) != len(args.input_path):
                raise ValueError(
                    f"Image modality count mismatch: {len(args.image_modality_names)} names provided "
                    f"but {len(args.input_path)} image paths specified.\n"
                    f"Example: -p /data/dwi /data/adc -imn dwi adc"
                )
            validate_modality_names(args.image_modality_names, reserved={'label', 'control'})

    # Validate label classes
    if args.lesion_input_path and len(args.lesion_input_path) > 1:
        # Filter out empty strings from paths
        args.lesion_input_path = [p for p in args.lesion_input_path if p]
        if args.label_modality_names:
            # Filter out empty strings
            args.label_modality_names = [name for name in args.label_modality_names if name]
            if len(args.label_modality_names) != len(args.lesion_input_path):
                raise ValueError(
                    f"Label class count mismatch: {len(args.label_modality_names)} names provided "
                    f"but {len(args.lesion_input_path)} label paths specified.\n"
                    f"Example: -lp /labels/lesion /labels/edema -lmn lesion edema"
                )
            validate_modality_names(args.label_modality_names, reserved={'image', 'control'})

    # Validate control modalities
    if args.controls_path and len(args.controls_path) > 1:
        if args.control_modality_names:
            # Filter out empty strings
            args.control_modality_names = [name for name in args.control_modality_names if name]
            if len(args.control_modality_names) != len(args.controls_path):
                raise ValueError(
                    f"Control modality count mismatch: {len(args.control_modality_names)} names provided "
                    f"but {len(args.controls_path)} control paths specified.\n"
                    f"Example: -ctr /controls/dwi /controls/adc -cmn dwi adc"
                )
            validate_modality_names(args.control_modality_names, reserved={'image', 'label'})

    return args


def build_modality_dict(paths, names, default_prefix):
    """Build modality dictionary from paths and optional names.

    Parameters
    ----------
    paths : list of str or None
        List of folder paths for each modality
    names : list of str or None
        Optional list of modality names. If None, generates alphabetic names (a, b, c, ...)
    default_prefix : str
        Prefix for auto-generated names (e.g., 'image', 'label', 'control')

    Returns
    -------
    dict or None
        Dictionary mapping modality names to Path objects, or None if paths is None/empty

    Examples
    --------
    With explicit names:
    >>> build_modality_dict(['/data/dwi', '/data/adc'], ['dwi', 'adc'], 'image')
    {'dwi': Path('/data/dwi'), 'adc': Path('/data/adc')}

    With auto-generated alphabetic names:
    >>> build_modality_dict(['/data/mod1', '/data/mod2'], None, 'image')
    {'a': Path('/data/mod1'), 'b': Path('/data/mod2')}

    Single path:
    >>> build_modality_dict(['/data/images'], None, 'image')
    {'a': Path('/data/images')}
    """
    if not paths:
        return None

    if not isinstance(paths, list):
        paths = [paths]

    # Generate alphabetic names if not provided
    if names is None:
        # Use lowercase letters: a, b, c, ..., z, aa, ab, ...
        import string
        alphabet = string.ascii_lowercase
        names = []
        for i in range(len(paths)):
            if i < 26:
                names.append(alphabet[i])
            else:
                # For > 26 paths, use aa, ab, ac, ... ba, bb, ...
                names.append(alphabet[i // 26 - 1] + alphabet[i % 26])

    return {name: Path(path) for name, path in zip(names, paths)}


def main():
    # Script arguments
    parser = argparse.ArgumentParser(description='Monai unet training')
    # Paths
    parser.add_argument('-o', '--output', type=str, help='output folder', required=True)
    nifti_paths_group = parser.add_mutually_exclusive_group(required=True)
    nifti_paths_group.add_argument(
        '-p', '--input_path',
        type=str,
        nargs='*',
        help='Image folder path(s). Single path: scans folder for NIfTI files. '
             'Multiple paths: folder-per-modality mode (use -imn to name modalities, default: a, b, c, ...)'
    )
    nifti_paths_group.add_argument('-li', '--input_list', type=str, help='Text file containing the list of b1000')
    nifti_paths_group.add_argument('-sid', '--seg_input_dict', type=str,
                                   help='[Segmentation only] The path to a json dict containing the keys of the '
                                        'different populations (subfolder names) with the list of images paths'
                                        '[Cannot be used for Validation]')
    nifti_paths_group.add_argument('-psl', '--pretrained_split_list', type=str,
                                   help='File containing split paths lists of the k-fold')

    # Matching patterns for folder-based multi-modal mode
    # Two mechanisms: STRIP (remove what differs) or EXTRACT (find what's the same)
    parser.add_argument(
        '--strip-pattern', type=str, default=None,
        help='Pattern to REMOVE from filenames for residual matching. '
             'After stripping, files with identical residuals are matched. '
             'Can be literal string or regex. Example: --strip-pattern "dwi" '
             'Default: None (exact filename matching - filenames must be identical)'
    )
    parser.add_argument(
        '--extract-pattern', type=str, default=None,
        help='Regex pattern to EXTRACT from filenames as matching key. '
             'Files with identical extracted keys are matched. '
             'Example: --extract-pattern "sub-\\d+" extracts "sub-001" from "scan_sub-001_dwi.nii". '
             'Cannot be used together with --strip-pattern.'
    )
    parser.add_argument(
        '--control-strip-pattern', type=str, default=None,
        help='Strip pattern for control files. Default: same as --strip-pattern'
    )
    parser.add_argument(
        '--control-extract-pattern', type=str, default=None,
        help='Extract pattern for control files. Cannot be used with --control-strip-pattern.'
    )

    # Filter patterns for loading files
    parser.add_argument(
        '--image-filter', type=str, nargs='*', default=None,
        help='Glob pattern(s) to filter image files during loading. '
             'If 1 pattern: applied to all image modalities. '
             'If N patterns: must match N image modalities (one per folder). '
             'Example: --image-filter "patient*" or --image-filter "*dwi*" "*adc*"'
    )
    parser.add_argument(
        '--label-filter', type=str, nargs='*', default=None,
        help='Glob pattern(s) to filter label files. Same rules as --image-filter.'
    )
    parser.add_argument(
        '--control-filter', type=str, nargs='*', default=None,
        help='Glob pattern(s) to filter control files. Same rules as --image-filter.'
    )

    lesion_paths_group = parser.add_mutually_exclusive_group(required=False)
    lesion_paths_group.add_argument(
        '-lp', '--lesion_input_path',
        type=str,
        nargs='*',
        help='Label folder path(s). Single path: one label class. '
             'Multiple paths: multi-class labels (use -lmn to name classes, default: a, b, c, ...)'
    )
    lesion_paths_group.add_argument('-lli', '--lesion_input_list', type=str,
                                    help='Text file containing the list of b1000')

    control_paths_group = parser.add_mutually_exclusive_group(required=False)
    control_paths_group.add_argument(
        '-ctr', '--controls_path',
        type=str,
        nargs='*',
        help='Control folder path(s). Single path: one control modality. '
             'Multiple paths: multi-modal controls (use -cmn to name modalities, default: a, b, c, ...)'
    )
    control_paths_group.add_argument('-lctr', '--controls_list', type=str,
                                     help='file path of the list of control images (image_prefix not applied)')
    control_paths_group.add_argument('-ctr_psl', '--controls_pretrained_split_list', type=str,
                                   help='File containing split paths lists of the k-fold for the controls')

    # Modality/class naming arguments
    parser.add_argument(
        '-imn', '--image_modality_names',
        type=str,
        nargs='+',
        help='Names for image modalities (must match -p count). Example: dwi adc flair. '
             'Default: alphabetic suffixes (a, b, c, ...)'
    )
    parser.add_argument(
        '-lmn', '--label_modality_names',
        type=str,
        nargs='+',
        help='Names for label classes (must match -lp count). Example: lesion edema. '
             'Default: alphabetic suffixes (a, b, c, ...)'
    )
    parser.add_argument(
        '-cmn', '--control_modality_names',
        type=str,
        nargs='+',
        help='Names for control modalities (must match -ctr count). Example: dwi adc. '
             'Default: alphabetic suffixes (a, b, c, ...)'
    )

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
    parser.add_argument('-ctrfct', '--ctr_loss_function', type=str, default='thresholded_average',
                        help='Loss function used for training on controls')
    parser.add_argument('-vlfct', '--val_loss_function', type=str, default='dice',
                        help='Loss function used for validation')
    parser.add_argument('-wf', '--weight_factor', type=float, default=1.,
                        help='Multiply the control loss by this factor')
    parser.add_argument('-ema', action="store_true", help='Use EMA')
    parser.add_argument('-tema', '--track_ema', action="store_true", help='Track EMA')
    parser.add_argument('-nboc', '--no_backward_on_controls', action="store_true", help='No backward on controls')

    # Parameters for control data
    parser.add_argument('-ctr_pref', '--ctrl_image_prefix', type=str,
                        help='Define a prefix to filter the control images')
    # add delayed_control_training
    parser.add_argument('-dct', '--delayed_control_training', action="store_true",
                        help='Train on controls after the the model converged on the lesioned set')
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
    # TODO modify that to instead take an int setting the number of subfolders to recreate in the output folder
    parser.add_argument('-upf',
                        '--use_parent_folder',
                        action='store_true',
                        help='Create a folder in the output with the parent folder name of the input file')
    # only_save_seg is a boolean used in validation_loop to save only the segmentation image
    parser.add_argument('-oss', '--only_save_seg', action='store_true', help='Save only the segmentation image')

    # Model parameters
    parser.add_argument('-pp', '--pretrained_point', type=str,
                        help='[Training opt]file path to a torch checkpoint file'
                             ' or directory (in that case, the most recent checkpoint will be used)')
    parser.add_argument('-mt', '--model_type', type=str, default='UNETR',
                        help='Select the model architecture (UNet, UNETR, SWINUNETR)')
    parser.add_argument('-d', '--torch_device', type=str, default=None,
                        help='Device type and number given to torch.device()')
    parser.add_argument('-dropout', type=float, help='Set a dropout value for the model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the AdamW optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                        help='Weight decay for the AdamW optimizer')
    # Learning rate scheduler
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Enable learning rate scheduling to prevent gradient explosion')
    parser.add_argument('--lr_scheduler_patience', type=int, default=10,
                        help='Epochs to wait before reducing LR when validation loss plateaus')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='Factor to multiply LR by when reducing (0.5 = half LR)')
    parser.add_argument('-fs', '--feature_size', type=int, help='Set the feature size for (SWIN)UNETR')
    # Gradient accumulation
    parser.add_argument('-ga', '--gradient_accumulation', type=int, default=1,
                        help='Number of batches to accumulate before performing a backward/update pass')
    # Mixed precision
    parser.add_argument('-dmp', '--disable_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    # Output clamping for numerical stability
    parser.add_argument('-oc', '--output_clamping', action='store_true',
                        help='Enable output clamping to prevent float16 overflow and NaN values')
    parser.add_argument('-ocr', '--output_clamp_range', type=float, default=10.0,
                        help='Range for output clamping: [-range, +range] (default: 10.0)')
    # Files split and matching options
    parser.add_argument('-tv', '--train_val', type=int, help='Training / validation percentage cut')
    parser.add_argument('-pref', '--image_prefix', type=str, help='Define a prefix to filter the input images')
    # add image_cut_prefix
    parser.add_argument('-icp', '--image_cut_prefix', type=str,
                        help='Prefix to cut the lesion file name (keep right part) in case the filename without the '
                             '.nii extension cannot be found')
    parser.add_argument('-ics', '--image_cut_suffix', type=str,
                        help='Suffix to cut the lesion file name (keep left part) in case the filename without the '
                             '.nii extension cannot be found')
    parser.add_argument('-nf', '--folds_number', default=5, type=int, help='Number of folds for cross-validation (default: 5)')
    # Datasets and Loaders parameters
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('-bs', '--batch_size', default=10, type=int, help='Batch size for the training loop')
    parser.add_argument('-vbs', '--val_batch_size', default=10, type=int, help='Batch size for the validation loop')

    # Cache mode arguments
    parser.add_argument('--cache-mode', type=str, choices=['none', 'ram', 'disk'],
                        help='Cache mode for both training and validation datasets')
    parser.add_argument('--cache-training-mode', type=str, choices=['none', 'ram', 'disk'],
                        help='Cache mode for training set only (overrides --cache-mode)')
    parser.add_argument('--cache-validation-mode', type=str, choices=['none', 'ram', 'disk'],
                        help='Cache mode for validation set only (overrides --cache-mode)')

    # Cache parameters (apply to both datasets unless overridden)
    parser.add_argument('-cn', '--cache_num', type=int, default=None,
                        help='Absolute number of samples to cache (overrides rate, for RAM mode)')
    parser.add_argument('-cr', '--cache_rate', type=float, default=1.0,
                        help='Fraction of dataset to cache (0.0-1.0, for RAM mode)')
    # Epochs parameters
    parser.add_argument('-ne', '--num_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-sbe', '--stop_best_epoch', type=int, help='Number of epochs without improvement before it '
                                                                    'stops')
    # DDP arguments
    # parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--local_rank", type=int, help="node rank for distributed training")
    parser.add_argument("-cvd", "--cuda_visible_devices", type=str, help="List of visible devices for cuda")
    # add a parameter to increase the number of open files
    parser.add_argument('-loof', '--limit_of_open_files', type=int,
                        help='Limit of open files allowed (ulimit)')
    # Memory management
    parser.add_argument('--disable-expandable-segments', action='store_true',
                        help='Disable CUDA expandable segments memory management '
                             '(default: enabled; use this flag if experiencing H100/A100 compatibility issues)')
    # DEBUG options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('-din', '--debug_img_num', type=int, help='Number of images from the input list')

    # Auto-configuration options
    parser.add_argument('--auto_config', action='store_true',
                        help='Enable hardware-aware auto-configuration for training parameters')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use for training (default: 1 if --auto_config, else all available)')
    parser.add_argument('--network_depth', type=int, choices=[4, 5],
                        help='Network depth (4 or 5 layers). Auto-configured if --auto_config is set')
    parser.add_argument('--vram_safety_margin', type=float, default=0.95,
                        help='VRAM safety margin for auto-configuration (0.0-1.0, default: 0.95)')
    parser.add_argument('--auto_config_target', type=str, default='balanced', choices=['speed', 'memory', 'balanced'],
                        help='Auto-configuration optimization target (default: balanced)')
    parser.add_argument('--no_dryrun', action='store_true',
                        help='Skip dry-run validation before training (default: perform dry-run)')
    parser.add_argument('--override_batch_size', type=int,
                        help='Override auto-configured batch size')
    parser.add_argument('--override_patch_size', type=str,
                        help='Override auto-configured patch size (format: H,W,D, e.g., 96,96,96)')
    parser.add_argument('--override_num_workers', type=int,
                        help='Override auto-configured num_workers')
    parser.add_argument('--override_network_depth', type=int, choices=[4, 5],
                        help='Override auto-configured network depth')
    parser.add_argument('--override_feature_size', type=int,
                        help='Override auto-configured feature size')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Validate multi-modal argument combinations
    args = validate_multimodal_arguments(args)

    # Configure CUDA memory management (must be before any CUDA operations)
    if not args.disable_expandable_segments and torch.cuda.is_available():
        try:
            # Enable expandable segments to reduce memory fragmentation
            # This helps prevent OOM errors during long training runs
            # See: https://pytorch.org/docs/stable/notes/cuda.html
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
            print("✓ Enabled CUDA expandable_segments for memory management")
            print("  (Use --disable-expandable-segments if you experience compatibility issues)")
        except Exception as e:
            logging.warning(f"Could not enable expandable_segments: {e}")

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
    if args.checkpoint and args.world_size == 1:
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        print("Distributed process group initialized with one process.")
    else:
        if args.world_size > 1:
            print("Waiting for all DDP processes to establish contact...")
            if args.torch_device != 'cpu':
                dist.init_process_group('nccl', rank=local_rank, world_size=args.world_size)
            else:
                dist.init_process_group('gloo', rank=local_rank, world_size=args.world_size)
            print('Contact established!')
        else:
            dist.init_process_group(backend='gloo', world_size=1, rank=0)
            print("Distributed process group initialized with one process.")
    if args.torch_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.torch_device)
    logging.info(f'Using device: {device}')
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
    # Cache mode resolution logic
    base_cache_mode = args.cache_mode or 'none'

    # Setup cache directory if disk mode is used
    cache_dir = None
    if base_cache_mode == 'disk' or args.cache_training_mode == 'disk' or args.cache_validation_mode == 'disk':
        cache_dir = Path(output_root, 'cache')

    # Per-dataset cache mode (overrides base mode)
    cache_training_mode = args.cache_training_mode or base_cache_mode
    cache_validation_mode = args.cache_validation_mode or base_cache_mode

    utils.print_rank_0('loading input dwi path list', dist.get_rank())
    seg_input_dict = {}
    # Detect if multi-modal mode (multiple paths in any of the input arguments)
    is_multi_modal = (
        (args.input_path and len(args.input_path) > 1) or
        (args.lesion_input_path and len(args.lesion_input_path) > 1) or
        (args.controls_path and len(args.controls_path) > 1)
    )

    if args.input_path is not None:
        if is_multi_modal:
            # Multi-modal folder-per-modality mode
            utils.logging_rank_0('Using folder-per-modality mode', dist.get_rank())

            # Build modality dictionaries
            image_folders_dict = build_modality_dict(
                args.input_path,
                args.image_modality_names,
                default_prefix='image'
            )

            label_folders_dict = build_modality_dict(
                args.lesion_input_path,
                args.label_modality_names,
                default_prefix='label'
            )

            control_folders_dict = build_modality_dict(
                args.controls_path,
                args.control_modality_names,
                default_prefix='control'
            )

            # Validate that multi-class labels and multi-modal controls aren't used yet
            if label_folders_dict and len(label_folders_dict) > 1:
                raise NotImplementedError(
                    f"\nMulti-class label segmentation is not yet implemented.\n"
                    f"You provided {len(label_folders_dict)} label classes: {list(label_folders_dict.keys())}\n"
                    f"Currently only single-class segmentation is supported.\n\n"
                    f"To implement multi-class support, the following pipeline components need updates:\n"
                    f"  - Transform pipeline (how to handle multiple label keys)\n"
                    f"  - Loss functions (per-class or combined loss)\n"
                    f"  - Validation metrics (per-class Dice, IoU, etc.)\n"
                    f"  - Output saving and visualization\n\n"
                    f"See implementation_docs/FUTURE_WORK.md for details."
                )

            if control_folders_dict and len(control_folders_dict) > 1:
                raise NotImplementedError(
                    f"\nMulti-modal control subjects are not yet implemented.\n"
                    f"You provided {len(control_folders_dict)} control modalities: {list(control_folders_dict.keys())}\n"
                    f"Currently only single-modality controls are supported.\n\n"
                    f"To implement multi-modal control support, the control handling pipeline needs updates.\n"
                    f"See implementation_docs/FUTURE_WORK.md for details."
                )

            # ===== STAGE 0: LOGGING =====
            utils.logging_rank_0(f'Image modalities: {list(image_folders_dict.keys())}', dist.get_rank())
            if label_folders_dict:
                utils.logging_rank_0(f'Label classes: {list(label_folders_dict.keys())}', dist.get_rank())
            if control_folders_dict:
                utils.logging_rank_0(f'Control modalities: {list(control_folders_dict.keys())}', dist.get_rank())

            # Log matching pattern info
            if args.strip_pattern:
                utils.logging_rank_0(f'Strip pattern: {args.strip_pattern}', dist.get_rank())
            elif args.extract_pattern:
                utils.logging_rank_0(f'Extract pattern: {args.extract_pattern}', dist.get_rank())
            else:
                utils.logging_rank_0('Matching mode: exact filename match', dist.get_rank())

            # ===== STAGE 0: LOADING FILES =====
            # Helper to load files with optional filter pattern
            def load_modality_files(folders_dict, filters, type_name):
                """Load NIfTI files from folders with optional filtering."""
                modality_lists = {}
                modalities = list(folders_dict.keys())

                # Determine filter to use for each modality
                if filters is None:
                    # No filtering, load all NIfTI files
                    for mod in modalities:
                        modality_lists[mod] = read_folder(folders_dict[mod])
                elif len(filters) == 1:
                    # Same filter for all modalities
                    for mod in modalities:
                        modality_lists[mod] = read_folder(folders_dict[mod], pattern=filters[0])
                elif len(filters) == len(modalities):
                    # One filter per modality
                    for mod, filt in zip(modalities, filters):
                        modality_lists[mod] = read_folder(folders_dict[mod], pattern=filt)
                else:
                    raise ValueError(
                        f"Filter count mismatch for {type_name}: got {len(filters)} filters "
                        f"for {len(modalities)} {type_name}s. Provide 0, 1, or {len(modalities)} filters."
                    )

                return modality_lists

            # Load image files
            image_lists = load_modality_files(
                image_folders_dict, args.image_filter, 'image modality'
            )

            # Load label files (if provided)
            label_lists = None
            if label_folders_dict:
                label_lists = load_modality_files(
                    label_folders_dict, args.label_filter, 'label class'
                )

            # Load control files (if provided)
            control_lists = None
            if control_folders_dict:
                control_lists = load_modality_files(
                    control_folders_dict, args.control_filter, 'control modality'
                )

            # ===== STAGE 1: MATCHING =====
            # Determine control pattern (default to subject pattern if not specified)
            ctrl_strip = args.control_strip_pattern or args.strip_pattern
            ctrl_extract = args.control_extract_pattern or args.extract_pattern

            subject_dicts, control_dicts = match_lists_to_dicts(
                image_lists=image_lists,
                label_lists=label_lists,
                control_lists=control_lists,
                strip_pattern=args.strip_pattern,
                extract_pattern=args.extract_pattern,
                control_strip_pattern=ctrl_strip,
                control_extract_pattern=ctrl_extract
            )

            utils.logging_rank_0(
                f'Matched {len(subject_dicts)} subjects, {len(control_dicts)} controls',
                dist.get_rank()
            )

            # ===== STAGE 2: SPLIT (training) or FLAT LIST (inference) =====
            is_training = args.checkpoint is None

            if is_training:
                # Training mode: combine subjects + controls, then shuffle and split
                all_subjects = subject_dicts + control_dicts
                img_list = shuffle_and_split_subjects(
                    subject_dicts=all_subjects,
                    n_folds=args.folds_number,
                    shuffle=True,
                    random_seed=42
                )

                les_list = None  # Labels embedded in subject dicts
                ctr_list = None  # Controls embedded in subject dicts
                utils.logging_rank_0(
                    f'Training mode: {len(all_subjects)} subjects split into {args.folds_number} folds',
                    dist.get_rank()
                )
            else:
                # Inference mode: flat list, no shuffle
                utils.logging_rank_0('Inference mode: listing files without shuffling', dist.get_rank())

                # Format: [dict1, dict2, ...] NOT [[fold0], [fold1], ...]
                img_list = subject_dicts + control_dicts

                les_list = None  # Labels embedded in subject dicts
                ctr_list = None  # Controls embedded in subject dicts

                utils.logging_rank_0(
                    f'Inference mode: {len(img_list)} subjects (order preserved)',
                    dist.get_rank()
                )
        else:
            # Single folder scan mode (backward compatible)
            single_path = args.input_path[0] if isinstance(args.input_path, list) else args.input_path
            utils.logging_rank_0(f'Input image directory : {single_path}', dist.get_rank())
            img_list = utils.create_input_path_list_from_root(single_path)
            if single_path == args.output:
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

    # Skip lesion loading if using multi-modal folder mode (already matched in SplitLists)
    if not is_multi_modal:
        utils.print_rank_0('loading input lesion label path list', dist.get_rank())
        if args.lesion_input_path is not None:
            # Extract single path from list (nargs='*' always returns list)
            single_lesion_path = args.lesion_input_path[0] if isinstance(args.lesion_input_path, list) else args.lesion_input_path
            logging.info(f'Input lesion directory : {single_lesion_path}')
            les_list = utils.create_input_path_list_from_root(single_lesion_path)
            if single_lesion_path == args.output:
                raise ValueError("The output directory CANNOT be the input directory")
        # So args.lesion_input_list is not None
        elif args.lesion_input_list is not None:
            utils.print_rank_0(f'Input lesion list : {args.lesion_input_list}', dist.get_rank())
            les_list = file_to_list(args.lesion_input_list)
        else:
            les_list = None

        # Controls for non-multi-modal mode
        if args.controls_path is not None:
            # Extract single path from list (nargs='*' always returns list)
            single_control_path = args.controls_path[0] if isinstance(args.controls_path, list) else args.controls_path
            ctr_list = utils.create_input_path_list_from_root(single_control_path, pref=args.ctrl_image_prefix)
            if single_control_path == args.output:
                raise ValueError("The output directory CANNOT be the input directory")
        # So args.controls_list is not None
        elif args.controls_list is not None:
            ctr_list = file_to_list(args.controls_list)
            ctr_list = utils.get_str_path_list(ctr_list, pref=args.ctrl_image_prefix)
        elif args.controls_pretrained_split_list is not None:
            ctr_list = utils.open_json(args.controls_pretrained_split_list)
        else:
            ctr_list = None
    if args.image_prefix is not None:
        b1000_pref = args.image_prefix
    else:
        b1000_pref = None

    if args.transform_dict is not None:
        if callable(transform_dict):
            td = args.transform_dict()
        else:
            td = args.transform_dict
        if Path(td).is_file():
            transform_dict = utils.load_json_transform_dict(td)
        else:
            if td in dir(tr_dicts):
                transform_dict = getattr(tr_dicts, td)
            else:
                transform_dict = None
                for d in dir(tr_dicts):
                    if d in td.lower():
                        tr_function_params = td.split(d)[-1].split('_')[1:]

                        transform_dict_fct = getattr(tr_dicts, d)
                        if callable(transform_dict_fct):
                            transform_dict = transform_dict_fct(*tr_function_params)
                if transform_dict is None:
                    raise ValueError('{} is not an existing dict file or is not '
                                     'in lesseg_unet/data/transform_dicts.py'.format(args.transform_dict))
    else:
        utils.logging_rank_0('Using default transformation dictionary', dist.get_rank())
        # Build default transform_dict from patch_size
        # Use None (wildcard) if auto-config enabled, otherwise use args.patch_size or fallback
        if args.auto_config:
            # Auto-config will tune patch_size - use wildcard (None)
            roi_size = None
            utils.logging_rank_0('Created default transform_dict with wildcard patch_size (auto-config will tune)', dist.get_rank())
        elif hasattr(args, 'patch_size') and args.patch_size is not None:
            # User specified patch_size via -ps
            if isinstance(args.patch_size, (list, tuple)) and len(args.patch_size) == 3:
                roi_size = list(args.patch_size)
            else:
                roi_size = [96, 96, 96]  # Fallback default
            utils.logging_rank_0(f'Created default transform_dict with user-specified patch_size: {roi_size}', dist.get_rank())
        else:
            # No auto-config, no user patch_size - use default
            roi_size = [96, 96, 96]
            utils.logging_rank_0(f'Created default transform_dict with default patch_size: {roi_size}', dist.get_rank())

        # Create minimal default transform dict with patches
        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}},
                {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
            ],
            'monai_transform': [],
            'patches': [
                {'RandCropByPosNegLabeld': {
                    'keys': ['image', 'label'],
                    'label_key': 'label',
                    'spatial_size': roi_size,  # Can be None (wildcard) or list
                    'pos': 1,
                    'neg': 1,
                    'num_samples': 4}
                },
            ],
            'last_transform': []  # Empty last transform for post-processing
        }
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

    # ===== AUTO-CONFIGURATION LOGIC =====
    if args.auto_config and args.checkpoint is None:
        utils.logging_rank_0('=' * 70, dist.get_rank())
        utils.logging_rank_0('Hardware-Aware Auto-Configuration', dist.get_rank())
        utils.logging_rank_0('=' * 70, dist.get_rank())

        # Detect hardware
        hw_profile = get_hardware_profile()

        # Override to CPU mode if user specified -d cpu
        if args.torch_device == 'cpu':
            utils.logging_rank_0('User forced CPU mode (-d cpu), ignoring GPU hardware', dist.get_rank())
            hw_profile.gpus = []  # Clear GPU list for CPU-only configuration
            hw_profile.device_type = 'cpu'

        utils.logging_rank_0(f'Device: {hw_profile.device_type}', dist.get_rank())
        utils.logging_rank_0(f'GPUs detected: {len(hw_profile.gpus)}', dist.get_rank())
        for gpu in hw_profile.gpus:
            utils.logging_rank_0(f'  - {gpu.name}: {gpu.total_memory_mb / 1024:.1f} GB', dist.get_rank())
        utils.logging_rank_0(f'CPU cores: {hw_profile.cpu.available_cores}', dist.get_rank())
        utils.logging_rank_0(f'RAM: {hw_profile.cpu.total_ram_gb:.1f} GB', dist.get_rank())

        # Analyze dataset to get median image size
        utils.logging_rank_0('\nAnalyzing dataset characteristics...', dist.get_rank())
        import nibabel as nib
        import numpy as np
        sample_images = []
        if isinstance(img_list[0], list):
            # Split lists format
            for fold in img_list[:min(3, len(img_list))]:  # Sample first 3 folds
                sample_images.extend(fold[:min(10, len(fold))])  # Up to 10 images per fold
        else:
            # Simple list format
            sample_images = img_list[:min(30, len(img_list))]  # Sample up to 30 images

        image_shapes = []
        for img_path in sample_images:
            try:
                # Handle multi-modal dictionaries
                if isinstance(img_path, dict):
                    # Extract first image path (not label)
                    image_keys = [k for k in img_path.keys() if k.startswith('image_')]
                    if image_keys:
                        actual_path = img_path[image_keys[0]]
                    else:
                        # Fallback to first key if no image_ prefix found
                        actual_path = list(img_path.values())[0]
                else:
                    actual_path = img_path

                img = nib.load(actual_path)
                image_shapes.append(img.shape[:3])  # Only spatial dims
            except Exception as e:
                utils.logging_rank_0(f'Warning: Could not load {img_path}: {e}', dist.get_rank())

        if not image_shapes:
            raise ValueError("Could not load any images to analyze dataset")

        median_shape = tuple(int(np.median([s[i] for s in image_shapes])) for i in range(3))
        utils.logging_rank_0(f'Median image size: {median_shape}', dist.get_rank())

        # Count in_channels from multi-modal setup
        if is_multi_modal and hasattr(img_list, '__len__') and len(img_list) > 0:
            # Handle split lists format (list of folds)
            sample_item = img_list[0]
            if isinstance(sample_item, list) and len(sample_item) > 0:
                sample_item = sample_item[0]  # Get first item from first fold

            if isinstance(sample_item, dict):
                # Count image_ keys only (not labels)
                in_channels = len([k for k in sample_item.keys() if k.startswith('image_')])
            else:
                in_channels = 1  # Single modality
        else:
            in_channels = 1

        # Count out_channels
        out_channels = 1  # Default binary segmentation
        if les_list is not None and isinstance(les_list, list) and len(les_list) > 0:
            # Handle split lists format (list of folds)
            sample_label = les_list[0]
            if isinstance(sample_label, list) and len(sample_label) > 0:
                sample_label = sample_label[0]  # Get first item from first fold

            if isinstance(sample_label, dict):
                # Count label_ keys only
                out_channels = len([k for k in sample_label.keys() if k.startswith('label_')])
                if out_channels == 0:
                    out_channels = 1  # Fallback

        # Determine storage type (simplified - assume SSD)
        storage_type = 'ssd'  # Default

        # Create dataset profile
        dataset_profile = DatasetProfile(
            median_image_size=median_shape,
            num_subjects=len(sample_images),
            in_channels=in_channels,
            out_channels=out_channels,
            storage_type=storage_type
        )

        # Normalize model_type to lowercase
        model_type_normalized = args.model_type.lower()

        # Create configurator
        # num_samples=4 matches default transform_dict RandCropByPosNegLabeld setting
        configurator = AutoConfigurator(
            hardware_profile=hw_profile,
            dataset_profile=dataset_profile,
            model_type=model_type_normalized,
            target=args.auto_config_target,
            vram_safety_margin=args.vram_safety_margin,
            num_gpus=args.num_gpus,
            num_samples=4  # Must match RandCropByPosNegLabeld num_samples in transform_dict
        )

        # Parse override_patch_size if provided
        override_patch_size = None
        if args.override_patch_size:
            try:
                override_patch_size = tuple(int(x) for x in args.override_patch_size.split(','))
                if len(override_patch_size) != 3:
                    raise ValueError("Patch size must have 3 dimensions")
            except Exception as e:
                raise ValueError(f"Invalid patch size format: {args.override_patch_size}. Use H,W,D (e.g., 96,96,96)")

        # Get suggested configuration
        auto_config_result = configurator.suggest_config(
            override_batch_size=args.override_batch_size,
            override_patch_size=override_patch_size,
            override_num_workers=args.override_num_workers,
            override_network_depth=args.override_network_depth,
            override_feature_size=args.override_feature_size
        )

        # Display configuration
        utils.logging_rank_0('\nSuggested Configuration:', dist.get_rank())
        utils.logging_rank_0(f'  batch_size: {auto_config_result.batch_size}', dist.get_rank())
        utils.logging_rank_0(f'  val_batch_size: {auto_config_result.val_batch_size}', dist.get_rank())
        utils.logging_rank_0(f'  patch_size: {auto_config_result.patch_size}', dist.get_rank())
        utils.logging_rank_0(f'  num_workers: {auto_config_result.num_workers}', dist.get_rank())
        utils.logging_rank_0(f'  network_depth: {auto_config_result.network_depth}', dist.get_rank())
        utils.logging_rank_0(f'  feature_size: {auto_config_result.feature_size}', dist.get_rank())
        utils.logging_rank_0(f'  use_amp: {auto_config_result.use_amp}', dist.get_rank())
        utils.logging_rank_0(f'  use_checkpoint: {auto_config_result.use_checkpoint}', dist.get_rank())
        utils.logging_rank_0(f'  num_gpus: {auto_config_result.num_gpus}', dist.get_rank())

        mem = auto_config_result.memory_estimate
        utils.logging_rank_0(f'\nMemory Estimate: {mem["total_gb"]:.2f} GB', dist.get_rank())
        utils.logging_rank_0(f'  Parameters:   {mem["params_mb"]:>8.1f} MB', dist.get_rank())
        utils.logging_rank_0(f'  Optimizer:    {mem["optimizer_mb"]:>8.1f} MB', dist.get_rank())
        utils.logging_rank_0(f'  Activations:  {mem["activations_mb"]:>8.1f} MB', dist.get_rank())
        utils.logging_rank_0(f'  Gradients:    {mem["gradients_mb"]:>8.1f} MB', dist.get_rank())

        if hw_profile.gpus:
            vram_gb = hw_profile.gpus[0].total_memory_mb / 1024
            usage_pct = (mem['total_gb'] / vram_gb) * 100
            utils.logging_rank_0(f'  VRAM Usage: {usage_pct:.1f}% of {vram_gb:.1f} GB', dist.get_rank())

        utils.logging_rank_0('\nReasoning:', dist.get_rank())
        for key, reason in auto_config_result.reasoning.items():
            utils.logging_rank_0(f'  {key}: {reason}', dist.get_rank())

        # Apply configuration to args
        args.batch_size = auto_config_result.batch_size
        args.val_batch_size = auto_config_result.val_batch_size
        args.patch_size = auto_config_result.patch_size  # For default transform_dict
        args.num_workers = auto_config_result.num_workers
        args.network_depth = auto_config_result.network_depth
        args.feature_size = auto_config_result.feature_size
        args.disable_mixed_precision = not auto_config_result.use_amp

        # Pass use_checkpoint to training via kwargs
        if auto_config_result.use_checkpoint:
            kwargs['use_checkpoint'] = True
            utils.logging_rank_0(
                'Gradient checkpointing enabled (VRAM optimization)',
                dist.get_rank()
            )

        # Constrain batch_size for small datasets (edge case: toy datasets with k-fold CV)
        # Rule: batch_size ≤ split_size / 2 (ensures minimum 2 batches per epoch with drop_last=True)
        # This only affects toy datasets; real datasets (1000+ samples) are never constrained.
        if isinstance(img_list, list) and img_list and len(img_list) > 1:
            # Multi-modal folder mode returns list of lists (folds)
            total_subjects = sum(len(fold) for fold in img_list)
            min_fold_size = min(len(fold) for fold in img_list)
            max_fold_size = max(len(fold) for fold in img_list)

            # Training uses (k-1) folds; validation uses 1 fold
            # Use min_fold_size for training (worst case: largest fold is validation)
            min_training_size = total_subjects - max_fold_size
            val_set_size = max_fold_size  # Worst case: largest fold is validation

            # Constrain batch_size ≤ training_set_size / 2
            max_train_batch = max(1, min_training_size // 2)
            if args.batch_size > max_train_batch:
                old_batch = args.batch_size
                args.batch_size = max_train_batch
                utils.logging_rank_0(
                    f'\nWARNING: Batch size reduced {old_batch}→{args.batch_size} due to small dataset '
                    f'(with {args.folds_number}-fold CV, training has ~{min_training_size} subjects). '
                    f'Consider using fewer folds (e.g., -nf 2) to increase training set size.',
                    dist.get_rank()
                )

            # Constrain val_batch_size ≤ val_set_size / 2
            max_val_batch = max(1, val_set_size // 2)
            if args.val_batch_size > max_val_batch:
                old_val_batch = args.val_batch_size
                args.val_batch_size = max_val_batch
                utils.logging_rank_0(
                    f'WARNING: Validation batch size reduced {old_val_batch}→{args.val_batch_size} '
                    f'due to small validation set (~{val_set_size} subjects).',
                    dist.get_rank()
                )

        # Fill wildcards in transform_dict with auto-configured values
        if transform_dict is not None:
            from lesseg_unet.auto_config import fill_wildcards
            transform_dict = fill_wildcards(transform_dict, auto_config_result.patch_size)
            utils.logging_rank_0(
                f'Filled transform_dict wildcards with auto-configured patch_size: {auto_config_result.patch_size}',
                dist.get_rank()
            )

        # Save configuration
        training_config = TrainingConfig.from_auto_config(
            auto_config_result,
            hardware_profile=hw_profile.to_dict(),
            command=' '.join(sys.argv),
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            model_type=args.model_type,
            user_overrides={
                'batch_size': args.override_batch_size,
                'patch_size': override_patch_size,
                'num_workers': args.override_num_workers,
                'network_depth': args.override_network_depth,
                'feature_size': args.override_feature_size
            } if any([args.override_batch_size, override_patch_size, args.override_num_workers,
                     args.override_network_depth, args.override_feature_size]) else {}
        )
        training_config.save(output_root / 'auto_config.yaml', overwrite=True)
        utils.logging_rank_0(f'\nConfiguration saved to: {output_root / "auto_config.yaml"}', dist.get_rank())
        utils.logging_rank_0('=' * 70, dist.get_rank())
    elif args.checkpoint is None:
        # ===== SAVE MANUAL CONFIGURATION =====
        # Even without auto_config, save the training configuration for reproducibility
        utils.logging_rank_0('Saving manual training configuration...', dist.get_rank())

        # Get hardware profile for reference
        hw_profile = get_hardware_profile()

        # Try to extract patch_size from transform_dict name (e.g., 'p64' -> (64, 64, 64))
        patch_size = None
        if args.transform_dict is not None:
            import re
            # Check if transform_dict is a simple name like 'p64'
            match = re.search(r'p(\d+)', str(args.transform_dict))
            if match:
                size = int(match.group(1))
                patch_size = (size, size, size)

        # Create manual training config
        manual_config = TrainingConfig(
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            patch_size=patch_size if patch_size else (64, 64, 64),  # Default if unknown
            num_workers=args.num_workers,
            network_depth=args.network_depth if hasattr(args, 'network_depth') and args.network_depth else None,
            feature_size=args.feature_size if args.feature_size else 48,  # Default
            use_amp=not args.disable_mixed_precision,
            num_gpus=args.num_gpus if hasattr(args, 'num_gpus') and args.num_gpus else 1,
            vram_safety_margin=args.vram_safety_margin if hasattr(args, 'vram_safety_margin') else 0.95,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            model_type=args.model_type,
            hardware_profile=hw_profile.to_dict(),
            command=' '.join(sys.argv),
            timestamp=datetime.now().isoformat(),
            reasoning={'source': 'Manual configuration (not auto-configured)'},
            memory_estimate={},
            user_overrides={}
        )

        manual_config.save(output_root / 'training_config.yaml', overwrite=True)
        utils.logging_rank_0(f'Manual configuration saved to: {output_root / "training_config.yaml"}', dist.get_rank())

    # ===== END AUTO-CONFIGURATION LOGIC =====

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

        # Auto-load split_lists.json when resuming to prevent data leakage
        if pretrained_point is not None:
            checkpoint_path = Path(pretrained_point)
            # Checkpoint is at: output_dir/fold_X/checkpoint.pth
            # split_lists.json is at: output_dir/split_lists.json
            output_dir_from_checkpoint = checkpoint_path.parent.parent
            split_lists_json_path = output_dir_from_checkpoint / 'split_lists.json'

            if split_lists_json_path.exists():
                utils.logging_rank_0(
                    f'Resume mode: Loading splits from {split_lists_json_path}',
                    dist.get_rank()
                )

                # Load pre-existing splits
                img_list = utils.open_json(str(split_lists_json_path))

                # Set les_list to None to trigger pre-split mode in training.py line 365-367
                # This prevents re-shuffling and split recreation
                les_list = None

                # Also handle controls if they exist
                control_split_lists_json_path = output_dir_from_checkpoint / 'control_split_lists.json'
                if control_split_lists_json_path.exists():
                    ctr_list = utils.open_json(str(control_split_lists_json_path))
                    utils.logging_rank_0(
                        f'Resume mode: Loaded control splits from {control_split_lists_json_path}',
                        dist.get_rank()
                    )

                utils.logging_rank_0(
                    f'Loaded {len(img_list)} folds from existing split_lists.json',
                    dist.get_rank()
                )
            else:
                # Backward compatibility: old checkpoints without split_lists.json
                utils.logging_rank_0(
                    f'Warning: split_lists.json not found at {split_lists_json_path}. '
                    f'Using provided input paths (splits will be recreated).',
                    dist.get_rank()
                )

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
                          cache_training_mode=cache_training_mode,
                          cache_validation_mode=cache_validation_mode,
                          cache_rate=args.cache_rate,
                          cache_num=args.cache_num,
                          world_size=args.world_size,
                          rank=local_rank,
                          enable_amp=not args.disable_mixed_precision,
                          learning_rate=args.learning_rate,
                          weight_decay=args.weight_decay,
                          use_lr_scheduler=args.use_lr_scheduler,
                          lr_scheduler_patience=args.lr_scheduler_patience,
                          lr_scheduler_factor=args.lr_scheduler_factor,
                          delayed_control_training=args.delayed_control_training,
                          use_ema=args.ema,
                          track_ema=args.track_ema,
                          no_backward_on_controls=args.no_backward_on_controls,
                          output_clamping=args.output_clamping,
                          output_clamp_range=args.output_clamp_range,
                          limit_of_open_files=args.limit_of_open_files,
                          debug=args.debug,
                          feature_size=args.feature_size,
                          network_depth=args.network_depth if hasattr(args, 'network_depth') else None,
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

        # Detect if labels are embedded in img_list (multi-modal folder mode with labels)
        has_embedded_labels = False
        if isinstance(img_list, list) and len(img_list) > 0:
            # img_list can be:
            # - Split lists: [[fold0_subjects], [fold1_subjects], ...] (training)
            # - Flat list: [dict1, dict2, ...] (inference)
            # Get first subject from appropriate format
            first_item = img_list[0]
            if isinstance(first_item, list) and len(first_item) > 0:
                # Split lists format - get first subject from first fold
                first_item = first_item[0]

            # Check if it's a dictionary with label keys
            if isinstance(first_item, dict):
                has_embedded_labels = any(k.startswith('label_') for k in first_item.keys())

        if les_list is None and not has_embedded_labels:
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
                                                   use_parent_folder=args.use_parent_folder,
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
                                               use_parent_folder=args.use_parent_folder,
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
            if has_embedded_labels:
                # Multi-modal folder mode: labels embedded in subject dicts
                # Use new function that handles dict format directly
                segmentation.validation_loop_split_lists(
                    img_list,  # Already contains subject dicts with embedded labels
                    output_root,
                    checkpoint,
                    transform_dict=transform_dict,
                    device=args.torch_device,
                    dataloader_workers=args.num_workers,
                    clamping=clamp_tuple,
                    segmentation_area=args.segmentation_area,
                    only_save_seg=args.only_save_seg,
                    **kwargs
                )
            else:
                # Legacy mode: separate img_list and les_list
                segmentation.validation_loop(img_list, les_list,
                                             output_root,
                                             checkpoint,
                                             b1000_pref,
                                             image_cut_prefix=args.image_cut_prefix,
                                             image_cut_suffix=args.image_cut_suffix,
                                             transform_dict=transform_dict,
                                             device=args.torch_device,
                                             dataloader_workers=args.num_workers,
                                             clamping=clamp_tuple,
                                             segmentation_area=args.segmentation_area,
                                             use_parent_folder=args.use_parent_folder,
                                             only_save_seg=args.only_save_seg,
                                             **kwargs)
            if args.overlap:
                nib.save(nifti_overlap_images(output_root, 'output_', recursive=True),
                         Path(output_root, 'overlap_segmentation.nii'))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
    dist.destroy_process_group()  # Cleanly shut it down

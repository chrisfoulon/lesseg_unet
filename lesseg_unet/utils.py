import os
from pathlib import Path
from typing import Union
import json

from bcblib.tools.nifti_utils import is_nifti
from lesseg_unet.visualisation_utils import plot_seg
import nibabel as nib
import torch


def create_input_path_list_from_root(root_folder_path, recursive_search=False):
    root_folder_path = Path(root_folder_path)
    if not root_folder_path.is_dir():
        raise ValueError('{} does not exist or is not a directory'.format(root_folder_path))
    input_path_list = []
    if not recursive_search:
        input_path_list = [p for p in root_folder_path.iterdir() if is_nifti(p)]
    if recursive_search or len(input_path_list) == 0:
        input_path_list = [p for p in root_folder_path.rglob('*') if is_nifti(p)]
    return input_path_list


def nifti_affine_from_dataset(nifti_path: Union[str, bytes, os.PathLike]):
    if not Path(nifti_path).is_file():
        raise ValueError('{} is not an existing file')
    return nib.load(nifti_path).affine


def save_checkpoint(model, epoch, optimizer, output_folder):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optim_dict': optimizer.state_dict(),
             }
    torch.save(state, Path(output_folder, 'state_dictionary_{}.pt'.format(epoch)))
    # utils.save_checkpoint(state,
    #                       is_best=is_best,  # True if this is the model with best metrics
    #                       checkpoint=model_dir)  # path to folder
    return


def save_tensor_to_nifti(tensor, output_path, val_output_affine):
    np_tensor = tensor[0, 0, :, :, :].cpu().detach().numpy()
    nib.save(nib.Nifti1Image(np_tensor, val_output_affine), output_path)


def save_img_lbl_seg_to_nifti(input_tensor, label_tensor, seg_tensor, output_dir, val_output_affine, suffix):
    save_tensor_to_nifti(input_tensor, Path(output_dir, 'nib_input_{}.nii'.format(suffix)), val_output_affine)
    if label_tensor is not None:
        save_tensor_to_nifti(label_tensor, Path(output_dir, 'nib_label_{}.nii'.format(suffix)), val_output_affine)
    if seg_tensor is not None:
        save_tensor_to_nifti(seg_tensor, Path(output_dir, 'nib_output_{}.nii'.format(suffix)), val_output_affine)


def save_img_lbl_seg_to_png(input_tensor, output_dir, filename, label_tensor=None, seg_tensor=None,):
    input_np = input_tensor[0, 0, :, :, :].cpu().detach().numpy()
    if label_tensor is not None:
        label_np = label_tensor[0, 0, :, :, :].cpu().detach().numpy()
    else:
        label_np = None
    if seg_tensor is not None:
        seg_np = seg_tensor[0, 0, :, :, :].cpu().detach().numpy()
    else:
        seg_np = None
    plot_seg(input_np, label_np, seg_np, Path(output_dir, filename + '.png'))


def save_json_transform_dict(transform_dict, output_path):
    with open(output_path, 'w') as json_fd:
        json.dump(transform_dict, json_fd, indent=4)


def load_json_transform_dict(json_path):
    if not Path(json_path).is_file():
        raise ValueError('{} is not an existing json file'.format(json_path))
    return json.load(open(json_path, 'r'))

import os
from pathlib import Path
from typing import Union
import json

from bcblib.tools.nifti_utils import is_nifti
from lesseg_unet.visualisation_utils import plot_seg
from lesseg_unet.net import create_unet_model
import nibabel as nib
import torch
import numpy as np


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


def save_checkpoint(model, epoch, optimizer, output_folder, filename=None):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optim_dict': optimizer.state_dict(),
             }
    if filename is None:
        torch.save(state, Path(output_folder, 'state_dictionary_{}.pt'.format(epoch)))
    else:
        torch.save(state, Path(output_folder, filename))
    return


def load_eval_from_checkpoint(checkpoint_path, device, unet_hyper_params=None):
    checkpoint = torch.load(checkpoint_path)
    model = create_unet_model(device, unet_hyper_params)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # bottom_up_graph.model.load_state_dict(checkpoint['bottom_up_graph_state_dict'])
    return model


def save_tensor_to_nifti(image: Union[np.ndarray, torch.Tensor],
                         output_path: Union[str, bytes, os.PathLike],
                         val_output_affine: np.ndarray) -> None:
    if isinstance(image, torch.Tensor):
        image_np = image[0, 0, :, :, :].cpu().detach().numpy()
    else:
        image_np = image
    nib.save(nib.Nifti1Image(image_np, val_output_affine), output_path)


def save_img_lbl_seg_to_nifti(image: Union[np.ndarray, torch.Tensor],
                              label: Union[np.ndarray, torch.Tensor],
                              seg: Union[np.ndarray, torch.Tensor],
                              output_dir: Union[str, bytes, os.PathLike],
                              val_output_affine: np.ndarray,
                              suffix: str) -> None:
    save_tensor_to_nifti(image, Path(output_dir, 'nib_input_{}.nii'.format(suffix)), val_output_affine)
    if label is not None:
        save_tensor_to_nifti(label, Path(output_dir, 'nib_label_{}.nii'.format(suffix)), val_output_affine)
    if seg is not None:
        save_tensor_to_nifti(seg, Path(output_dir, 'nib_output_{}.nii'.format(suffix)), val_output_affine)


def save_img_lbl_seg_to_png(image: Union[np.ndarray, torch.Tensor],
                            output_dir: Union[str, bytes, os.PathLike],
                            filename: str,
                            label: Union[np.ndarray, torch.Tensor] = None,
                            seg: Union[np.ndarray, torch.Tensor] = None) -> None:
    input_np = image
    if isinstance(image, torch.Tensor):
        input_np = image[0, 0, :, :, :].cpu().detach().numpy()
    label_np = label
    if isinstance(label, torch.Tensor):
        label_np = label[0, 0, :, :, :].cpu().detach().numpy()
    seg_np = seg
    if isinstance(seg, torch.Tensor):
        seg_np = seg[0, 0, :, :, :].cpu().detach().numpy()
    plot_seg(input_np, label_np, seg_np, Path(output_dir, filename + '.png'))


def save_json_transform_dict(transform_dict, output_path):
    with open(output_path, 'w') as json_fd:
        json.dump(transform_dict, json_fd, indent=4)


def load_json_transform_dict(json_path):
    if not Path(json_path).is_file():
        raise ValueError('{} is not an existing json file'.format(json_path))
    return json.load(open(json_path, 'r'))

import os
from pathlib import Path
from typing import Union

from bcblib.tools.nifti_utils import is_nifti
import nibabel as nib
import torch


def create_input_path_list_from_root(root_folder_path, recursive_search=False):
    root_folder_path = Path(root_folder_path)
    if not root_folder_path.is_dir():
        raise ValueError(root_folder_path + ' does not exist or is not a directory')
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
    save_tensor_to_nifti(label_tensor, Path(output_dir, 'nib_label_{}.nii'.format(suffix)), val_output_affine)
    save_tensor_to_nifti(seg_tensor, Path(output_dir, 'nib_output_{}.nii'.format(suffix)), val_output_affine)


def fsleyes_seg_comparison(img_folder, image_ind):
    # fsleyes / home / tolhsadum / neuro_apps / data / lesseg_unet_output_first_good /${folder} / nib_in
    # put_${i}.nii / home / tolhsadum / neuro_apps / data / lesseg_unet_output_first_good /${folder} / nib_label_${
    # i}.nii / home / tolhsadum / neuro_apps / data / lesseg_unet_outp
    # ut_first_good /${folder} / nib_output_${i}.nii
    pass

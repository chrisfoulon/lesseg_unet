import os
import shutil
from pathlib import Path
from typing import Union, List
import json
import math

from bcblib.tools.nifti_utils import is_nifti, centre_of_mass_difference
from lesseg_unet.visualisation_utils import plot_seg
from lesseg_unet.net import create_unet_model
import nibabel as nib
import torch
import numpy as np


validation_input_pref = 'nib_input_'
validation_label_pref = 'nib_label_'
validation_output_pref = 'nib_output_'


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
                              suffix: str) -> List[str]:
    out_input_path = Path(output_dir, 'input_{}.nii'.format(suffix))
    save_tensor_to_nifti(image, out_input_path, val_output_affine)
    out_paths_list = [str(out_input_path)]
    if label is not None:
        out_label_path = Path(output_dir, 'label_{}.nii'.format(suffix))
        save_tensor_to_nifti(label, out_label_path, val_output_affine)
        out_paths_list.append(str(out_label_path))
    if seg is not None:
        out_output_path = Path(output_dir, 'output_{}.nii'.format(suffix))
        save_tensor_to_nifti(seg, out_output_path, val_output_affine)
        out_paths_list.append(str(out_output_path))
    return out_paths_list


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


def get_output_paths(output_dir, img_num, split_string='_segmentation', reference_centre_of_mass=None):
    if not isinstance(output_dir, list):
        file_list = [f for f in Path(output_dir).iterdir()]
    else:
        file_list = output_dir
    output_images = [f for f in file_list if '.png' in f.name
                     and str(img_num) == f.name.split('.png')[0].split('_')[-1]]
    if len(output_images) == 0:
        raise ValueError(f'Cannot find png image number {img_num} in {output_dir}')
    if split_string is not None and split_string != '':
        fname = output_images[0].name.split(split_string)[0] + f'_{img_num}'
    else:
        fname = output_images[0].name.split('.png')[0]
    img_dict = {'png': str(output_images[0])}
    match = [f for f in file_list if fname in f.name and f != output_images[0]]
    for m in match:
        if validation_input_pref in m.name:
            img_dict['input'] = str(m)
        if validation_label_pref in m.name:
            img_dict['label'] = str(m)
        if validation_output_pref in m.name:
            img_dict['output'] = str(m)
    if reference_centre_of_mass is not None:
        # print(f'Calculating {img_dict["input"]} centre of mass')
        centre = centre_of_mass_difference(img_dict['input'], reference_centre_of_mass)
        # print('Input centre of mass distance from reference: {}'.format(centre))
        img_dict['centre'] = str(centre)
    if not output_images:
        return None
    return img_dict


def create_output_dict(output_dir, split_string='_segmentation', reference_centre_of_mass=None):
    if reference_centre_of_mass is None:
        reference_centre_of_mass = (47.535904629937065, 63.16388127843436, 46.78187843241487)
    file_list = [f for f in Path(output_dir).iterdir()]
    png_list = [f for f in file_list if '.png' in f.name]
    img_numb_list = [f.name.split('.png')[0].split('_')[-1] for f in png_list]
    output_dict = {}
    for num in img_numb_list:
        output_dict[num] = get_output_paths(file_list, num, split_string, reference_centre_of_mass)
    return output_dict


def get_big_seg(output_dict, vox_nb_thr=80):
    return np.random.choice(
        [k for k in output_dict if np.count_nonzero(nib.load(output_dict[k]['output']).get_fdata()) > vox_nb_thr])


def split_output_fname(path, prefix=None, suffix=None):
    fname = Path(path).name
    if prefix is not None and prefix != '':
        fname = fname.split(prefix)[1]
    if suffix is not None and suffix != '':
        s_split = fname.split(suffix)
        fname = s_split[0] + s_split[1]
    return fname


def split_output_files(source_dir, dest_dir, png_split_string='_segmentation', reference_centre_of_mass=None):
    output_dict = create_output_dict(source_dir, png_split_string, reference_centre_of_mass)
    output_json_path = Path(dest_dir, '__output_dict.json')
    for num in output_dict:
        for k in output_dict[num]:
            if k != 'centre':
                if k == 'png':
                    fname = split_output_fname(output_dict[num][k], suffix=png_split_string)
                else:
                    fname = split_output_fname(output_dict[num][k], prefix=k)
                cp_dir = Path(dest_dir, k)
                if not cp_dir.is_dir():
                    os.makedirs(cp_dir, exist_ok=True)
                out_path = str(Path(cp_dir, fname))
                print(fname)
                print(output_dict[num][k])
                shutil.copyfile(output_dict[num][k], out_path)
                output_dict[num][k] = out_path
    with open(output_json_path, 'w+') as out_file:
        json.dump(output_dict, out_file, indent=4)
    return output_dict


def split_lists_in_folds(img_dict: dict,
                         folds_number: int = 1,
                         train_val_percentage: float = 80):
    full_file_list = [{'image': str(img), 'label': str(img_dict[img])} for img in img_dict]
    if folds_number == 1:
        training_end_index = math.ceil(train_val_percentage / 100 * len(img_dict))
        # Inverse the order of the splits because the validation chunk in that case will be 0
        split_lists = [full_file_list[training_end_index:], full_file_list[:training_end_index]]
    else:
        split_lists = list(np.array_split(np.array(full_file_list), folds_number))
    return split_lists


def percent_vox_loss(img, half_total=True):
    max_vox = torch.prod(torch.as_tensor(img.shape))
    if half_total:
        max_vox = max_vox // 2
    return len(img[torch.where(img != 0)]) / max_vox

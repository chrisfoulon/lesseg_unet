import os
import shutil
from pathlib import Path
from typing import Union, List
import json
import math
import random
import importlib.resources as rsc
import logging
from operator import lt, gt
from multiprocessing.dummy import Pool
import multiprocessing
from scipy.stats import entropy

import monai.transforms
from monai.metrics import HausdorffDistanceMetric, DiceMetric
from bcblib.tools.nifti_utils import is_nifti, centre_of_mass_difference
from lesseg_unet.net import create_model
import nibabel as nib
import torch
import numpy as np
from bcblib.tools.nifti_utils import load_nifti


validation_input_pref = 'nib_input_'
validation_label_pref = 'nib_label_'
validation_output_pref = 'nib_output_'


def open_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def save_json(path, d):
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4)


def save_list(path, li):
    np.savetxt(path, li, delimiter='\n', fmt='%s')


def print_rank_0(string, rank):
    if rank == 0:
        print(string)


def logging_rank_0(string, rank, level='info'):
    if rank == 0:
        if level == 'info':
            logging.info(string)
        if level == 'debug':
            logging.debug(string)


def tensorboard_write_rank_0(writer, key, value, epoch, rank):
    if rank == 0:
        writer.add_scalar(key, value, epoch)


def kwargs_argparse(unknown_param_list):
    kwargs_dict = {}
    key = None
    for el in unknown_param_list:
        if '=' in el:
            spl = el.split('=')
            try:
                val = float(spl[1])
            except ValueError:
                val = spl[1]
            kwargs_dict[spl[0]] = val
        else:
            if key is not None:
                try:
                    val = float(el)
                except ValueError:
                    val = el
                kwargs_dict[key] = val
                key = None
            else:
                key = el
    return kwargs_dict


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


def save_checkpoint(model, epoch, optimizer, scaler, hyper_params, output_folder, model_name,
                    transform_dict, filename=None):
    state_dict = {'epoch': epoch,
                  'state_dict': model.module.state_dict(),
                  'optim_dict': optimizer.state_dict(),
                  'hyper_params': hyper_params,
                  'scaler_dict': scaler.state_dict(),
                  'model_name': model_name,
                  'transform_dict': transform_dict
                  }
    if filename is None:
        out_path = Path(output_folder, 'checkpoint_dictionary_{}.pt'.format(epoch))
    else:
        out_path = Path(output_folder, filename)
    torch.save(state_dict, out_path, _use_new_zipfile_serialization=False)
    return out_path


def load_model_from_checkpoint(loaded_checkpoint, device, hyper_params=None, model_name='UNETR'):
    if hyper_params is None:
        hyper_params = loaded_checkpoint['hyper_params']
    # print(list(loaded_checkpoint.keys()))
    # print(list(loaded_checkpoint['state_dict'].keys()))
    model, hyper_params = create_model(device, hyper_params, model_name)
    model.load_state_dict(loaded_checkpoint['state_dict'])
    # model.eval()
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
    if len(label) > 0:
        out_label_path = Path(output_dir, 'label_{}.nii'.format(suffix))
        save_tensor_to_nifti(label, out_label_path, val_output_affine)
        out_paths_list.append(str(out_label_path))
    if len(seg) > 0:
        out_output_path = Path(output_dir, 'output_{}.nii'.format(suffix))
        save_tensor_to_nifti(seg, out_output_path, val_output_affine)
        out_paths_list.append(str(out_output_path))
    return out_paths_list


def save_json_transform_dict(transform_dict, output_path):
    with open(output_path, 'w') as json_fd:
        json.dump(transform_dict, json_fd, indent=4)


def load_json_transform_dict(json_path):
    if not Path(json_path).is_file():
        raise ValueError('{} is not an existing json file'.format(json_path))
    with open(json_path, 'r') as f:
        return json.load(f)


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
    """
    Return the paths of the segmentations with more than vox_nb_thr voxels
    Parameters
    ----------
    output_dict
    vox_nb_thr

    Returns
    -------

    """
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


def split_lists_in_folds(img_dict: Union[dict, list],
                         folds_number: int = 1,
                         train_val_percentage: float = 80,
                         shuffle=True):
    if isinstance(img_dict, dict):
        full_file_list = [{'image': str(img), 'label': str(img_dict[img])} for img in img_dict]
    else:
        full_file_list = [{'image': str(img)} for img in img_dict]
    if shuffle:
        random.shuffle(full_file_list)
    if folds_number == 1:
        training_end_index = math.ceil(train_val_percentage / 100 * len(img_dict))
        # Inverse the order of the splits because the validation chunk in that case will be 0
        split_lists = [full_file_list[training_end_index:], full_file_list[:training_end_index]]
    else:
        split_lists = list(np.array_split(np.array(full_file_list), folds_number))
        split_lists = [list(li) for li in split_lists]
    return split_lists


def percent_vox_loss(img, sigmoid=True, discrete=True, divide_max_vox=1):
    if sigmoid:
        img = torch.sigmoid(img)
    if discrete:
        img = monai.transforms.AsDiscrete(threshold=0.5)(img)
    max_vox = torch.prod(torch.as_tensor(img.shape))
    if divide_max_vox != 1:
        max_vox = max_vox // divide_max_vox
    return len(img[torch.where(img)]) / max_vox


def volume_metric(img, sigmoid=True, discrete=True):
    if sigmoid:
        img = torch.sigmoid(img)
    if discrete:
        img = monai.transforms.AsDiscrete(threshold_values=True)(img)
    return len(img[torch.where(img)])


def entropy_metric(img, sigmoid=True):
    if sigmoid:
        img = torch.sigmoid(img)
    return entropy(img, base=2)


def check_inputs(inputs, img_string=None, min_file_size=300):
    """

    Parameters
    ----------
    inputs
    img_string
    min_file_size : int
        default == 300B which is lower than an empty image with just a nifti header

    Returns
    -------

    """
    if isinstance(inputs, list):
        if img_string is None:
            img_string = 'control'
        for img in inputs:
            if not Path(img).is_file():
                raise ValueError(f'The {img_string} image [{img}] does not exist')
            if Path(img).stat().st_size <= min_file_size:
                raise ValueError(f'The {img_string} image [{img}] has a size of zero')
            try:
                nib.load(img)
            except Exception as e:
                print(f'The {img_string} image [{img}] cannot be loaded. See Exception:')
                raise e
    elif isinstance(inputs, dict):
        if img_string is None:
            img_string = 'input'
        for img in inputs:
            if not Path(img).is_file():
                raise ValueError(f'The {img_string} image [{img}] does not exist')
            if Path(img).stat().st_size <= min_file_size:
                raise ValueError(f'The {img_string} image [{img}] has a size of zero')
            if not Path(inputs[img]).is_file():
                raise ValueError(f'The {img_string} label [{inputs[img]}] does not exist')
            if Path(img).stat().st_size <= min_file_size:
                raise ValueError(f'The {img_string} label [{inputs[img]}] has a size of zero')
            try:
                nib.load(img)
            except Exception as e:
                print(f'The {img_string} image [{img}] cannot be loaded. See Exception:')
                raise e
            try:
                nib.load(inputs[img])
            except Exception as e:
                print(f'The {img_string} label [{inputs[img]}] cannot be loaded. See Exception:')
                raise e
    else:
        raise ValueError('inputs must either be a list of paths or a dict of {image_paths: label_paths}')


def filter_outside_brain_voxels(img, output=None, template=None):
    nii = load_nifti(img)
    if output is not None:
        if Path(output).is_dir():
            output_path = Path(output, Path(img).name)
        else:
            output_path = output
    else:
        output_path = img
    if template is None:
        with rsc.path('lesseg_unet.data', 'brain_mask.nii') as p:
            template = str(p.resolve())
    data = nii.get_fdata()
    data[np.where(nib.load(template).get_fdata() == 0)] = 0
    nib.save(nib.Nifti1Image(data, nii.affine), output_path)


def count_outside_brain_voxels(img, template=None):
    nii = load_nifti(img)
    if template is None:
        with rsc.path('lesseg_unet.data', 'brain_mask.nii') as p:
            template = str(p.resolve())
    data = nii.get_fdata()
    count = np.count_nonzero(data[np.where(nib.load(template).get_fdata() != 0)])
    return count


def sum_non_bias_l2_norms(parameters, multiplier=None):
    """
    Given parameters=model.parameters() where model is a PyTorch model, this iterates through the list and tallies
    the L2 norms of all the non-bias tensors.
    """
    l2_reg = 0
    for param in parameters:
        if len(list(param.size())) > 1:
            l2_reg += torch.mean(torch.square(param))

    if multiplier is not None:
        l2_reg = multiplier * l2_reg
    return l2_reg


def prompt_choice(choice1, choice2, previous_choices_list=None):
    if previous_choices_list is None:
        previous_choices_list = []
    if choice1 == choice2:
        return choice1, previous_choices_list
    for cell in previous_choices_list:
        if cell[0] == {choice1, choice2}:
            return cell[1], previous_choices_list
    resp = input(f'Item found for both {choice1} and {choice2}')
    while not (resp == choice1 or resp == choice2):
        resp = input(f'Item found for both {choice1} and {choice2} (Choose between the 2)')
    previous_choices_list.append([{choice1, choice2}, resp])
    return resp, previous_choices_list


def get_fname_from_sorted_images(root_dir, pref='', split_suff=None):
    subdir_fnames_dict = {}
    for subdir in [p for p in Path(root_dir).iterdir() if p.is_dir()]:
        subdir_fnames_dict[subdir.name] = {p.name.split(split_suff)[0].split(pref)[-1] for p in subdir.iterdir()
                                           if p.name.startswith(pref)}
    for subdir in subdir_fnames_dict:
        fname_set = subdir_fnames_dict[subdir]
        for inner_subdir in subdir_fnames_dict:
            if inner_subdir != subdir:
                inner_set = subdir_fnames_dict[inner_subdir]
                intersection = fname_set.intersection(inner_set)
                if intersection:
                    resp = input(f'Item found for both {inner_subdir} and {subdir}, which subdir keeps the files?')
                    while not (resp == inner_subdir or resp == subdir):
                        resp = input(f'Item found for both {inner_subdir} and {subdir}, which subdir keeps the files?')
                    if resp == subdir:
                        subdir_fnames_dict[inner_subdir] = inner_set - intersection
                    else:
                        subdir_fnames_dict[subdir] = fname_set - intersection
    return subdir_fnames_dict


def get_best_epoch_from_folder(folder):
    best_epoch_time = 0
    best_epoch_path = ''
    for p in [pp for pp in Path(folder).iterdir() if pp.name.endswith('.pth')]:
        t = p.stat().st_ctime
        if best_epoch_time < t:
            best_epoch_time = t
            best_epoch_path = str(p)
    return best_epoch_path


def compare_img_to_cluster(img, cluster, comp_meth='dice', cluster_thr=None, flip=False):
    if is_nifti(img):
        img = load_nifti(img)
        img_data = img.get_fdata()
    elif isinstance(img, np.ndarray):
        img_data = img
    else:
        raise TypeError('img must be a path or a numpy array')
    if len(np.unique(img_data)) > 2:
        ValueError(f'The image must contain binary values')
    if is_nifti(cluster):
        cluster = load_nifti(cluster)
        cluster_data = cluster.get_fdata()
    elif isinstance(cluster, np.ndarray):
        cluster_data = cluster
    else:
        raise TypeError('img must be a path or a numpy array')
    if flip:
        cluster_data = np.flip(cluster_data, 0)
    if cluster_thr is not None and len(np.unique(cluster_data)) > 2:
        cluster_data[cluster_data < cluster_thr] = 0
        cluster_data[cluster_data >= cluster_thr] = 1
    # Apparently it is faster?
    img_data = np.array([img_data])
    img_data = torch.tensor(img_data)
    cluster_data = np.array([cluster_data])
    cluster_data = torch.tensor(cluster_data)
    if comp_meth == 'distance':
        hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
        return hausdorff_metric(y_pred=[img_data], y=[cluster_data]).item()
    elif comp_meth == 'dice':
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        return dice_metric(y_pred=[img_data], y=[cluster_data]).item()
    else:
        raise ValueError(f'{comp_meth} not recognized (available: "distance" or "dice")')


def get_image_area(img, comp_meth='dice', cluster_thr=0.1):
    with rsc.path('lesseg_unet.data', 'Final_lesion_clusters') as p:
        clusters_dir = str(p.resolve())
    best_val = None
    best_clu = None
    best_is_flipped = False
    if comp_meth == 'distance':
        op = lt
    else:
        op = gt
    for clu in [p for p in Path(clusters_dir).iterdir() if is_nifti(p)]:
        comp_val = compare_img_to_cluster(img, clu, comp_meth=comp_meth, cluster_thr=cluster_thr)
        flipped_comp_val = compare_img_to_cluster(img, clu, comp_meth=comp_meth, cluster_thr=cluster_thr, flip=True)
        if op(flipped_comp_val, comp_val):
            comp_val = flipped_comp_val
            flipped = True
        else:
            flipped = False
        if best_val is None or op(comp_val, best_val):
            best_val = comp_val
            best_clu = clu
            best_is_flipped = flipped
    return Path(best_clu.stem).stem, best_is_flipped


class LesionAreaFinder:
    def __init__(self, cluster_thr=0.1, cluster_list=None):
        if cluster_list is None:
            with rsc.path('lesseg_unet.data', 'Final_lesion_clusters') as p:
                clusters_dir = str(p.resolve())
            cluster_list = list(Path(clusters_dir).iterdir())
        else:
            if not isinstance(cluster_list, list):
                if Path(cluster_list).is_dir():
                    cluster_list = list(Path(cluster_list).iterdir())
                elif Path(cluster_list).is_file():
                    cluster_list = [cluster_list]
                else:
                    ValueError('If cluster_list given, it must be a list of files, a folder or a file path')
        data_names_dict = {Path(p.stem).stem: nib.load(p).get_fdata() for p in
                           cluster_list if is_nifti(p)}
        self.cluster_data_names_dict = {}
        for clu_name in data_names_dict:
            cluster_data = data_names_dict[clu_name]
            cluster_data[cluster_data < cluster_thr] = 0
            cluster_data[cluster_data >= cluster_thr] = 1
            self.cluster_data_names_dict['R_' + clu_name] = cluster_data
            self.cluster_data_names_dict['L_' + clu_name] = np.flip(cluster_data, 0)

    def get_img_area(self, img, comp_meth='dice', nb_cores=-1):
        def partial_compare(clu):
            return compare_img_to_cluster(img, clu, comp_meth=comp_meth, cluster_thr=None)
        if nb_cores == -1:
            nb_cores = multiprocessing.cpu_count()
        pool_obj = Pool(nb_cores)
        sim_val_list = pool_obj.map(partial_compare, self.cluster_data_names_dict.values())
        pool_obj.close()
        pool_obj.join()
        if comp_meth == 'distance':
            best_clu_ind = np.argmin(np.array(sim_val_list))
        else:
            best_clu_ind = np.argmax(np.array(sim_val_list))
            if sim_val_list[best_clu_ind] == 0:
                return 'outside_clusters'
        return list(self.cluster_data_names_dict.keys())[best_clu_ind]


def get_segmentation_areas(img_list, comp_meth='dice', cluster_thr=0.1, root_dir_path=None):
    img_cluster_isflipped_dict = {}
    for img in img_list:
        clu, flipped = get_image_area(img, comp_meth=comp_meth, cluster_thr=cluster_thr)
        path = img
        if root_dir_path is not None:
            path = Path(img).relative_to(root_dir_path)
        img_cluster_isflipped_dict[path] = {'cluster': Path(clu).name.split('.nii')[0],
                                            'flipped': flipped}
    return img_cluster_isflipped_dict


def get_img_size(img):
    return load_nifti(img).shape


def repeat_array_to_dim(arr, dim):
    # credit to @scieronomic on https://stackoverflow.com/questions/64061832/numpy-tile-up-to-specific-size
    return np.tile(arr, np.array(dim) // np.array(np.shape(arr)) + 1)[tuple(map(slice, dim))]


def get_fold_splits(split_list_file_path, output_dir, replace_root='', by_this_root='', pref=''):
    if Path(split_list_file_path).is_dir():
        split_data_list = open_json(Path(split_list_file_path, 'split_lists.json'))
    else:
        split_data_list = open_json(split_list_file_path)
    image_list = []
    label_list = []
    for i in range(5):
        first_val_set = split_data_list[i]
        image_list = [img_dict['image'].replace(replace_root, by_this_root) for img_dict in
                      first_val_set]
        label_list = [img_dict['label'].replace(replace_root, by_this_root) for img_dict in
                      first_val_set]
        save_list(Path(output_dir, f'{pref}fold_{i}_img_list.csv'), image_list)
        save_list(Path(output_dir, f'{pref}fold_{i}_lbl_list.csv'), label_list)
    return image_list, label_list

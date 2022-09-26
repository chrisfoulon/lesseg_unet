import os
import shutil
import webbrowser
import subprocess
from pathlib import Path
import re
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.ndimage import center_of_mass
from bcblib.tools.nifti_utils import is_nifti
from lesseg_unet.utils import LesionAreaFinder, open_json, save_json
# from bcblib.tools.visualisation import mricron_display


def open_tensorboard_page(log_dir, port='8008', new_browser_window=False):
    if not Path(log_dir).is_dir():
        raise ValueError('{} is not an existing directory with the logs'.format(log_dir))
    return_value = subprocess.call(['tensorboard', '--logdir', log_dir, '--port', port])
    print('###############')
    print('Return value:', return_value)
    tb_url = 'http://localhost:{}'.format(port)
    if new_browser_window:
        webbrowser.open_new(tb_url)
    else:
        webbrowser.open_new_tab(tb_url)


""" TODO
Make a function going from the bottom up to display each slice with voxels in the segmentation mask(s) to compare them
from nilearn import plotting
plotting.find_xyz_cut_coords could be useful
"""


def plot_one_slice(data_to_plot, title=None, rot90=True, normalise=True):
    if title is not None:
        plt.title(title)
    plt.axis('off')
    current = np.squeeze(data_to_plot)
    if rot90:
        current = np.rot90(current)
    # if normalise:
    #     current = visuals.norm_zero_to_one(current)
    plt.imshow(current)


def plot_seg(img, label=None, seg=None, save_path=None):
    if np.isnan(np.sum(img)):
        img = np.nan_to_num(img)
    if label is not None and np.isnan(np.sum(label)):
        label = np.nan_to_num(label)
    if seg is not None and np.isnan(np.sum(seg)):
        seg = np.nan_to_num(seg)
    if label is not None:
        _, _, z = center_of_mass(label)
        # nilearn doesn't to work here -__-"
        # _, _, z = plotting.find_xyz_cut_coords(nib.Nifti1Image(label, np.eye(4)), activation_threshold=1)
        if not np.isnan(z):
            z = round(z)
        else:
            z = round(len(img[0, 0, :]) / 2)
    else:
        z = round(len(img[0, 0, :]) / 2)
    # fig = plt.figure()
    # plt.subplot(1,1,1).imshow(img[:, :, z], cmap='gray', interpolation='none')
    z = int(z)
    plt.imshow(img[:, :, z], cmap='gray', interpolation='none')
    if label is not None:
        plt.imshow(label[:, :, z], cmap='Reds', alpha=label[:, :, z], interpolation='none')
    if seg is not None:
        plt.imshow(seg[:, :, z], cmap='Blues', alpha=label[:, :, z]*.5, interpolation='none')
    if save_path is not None:
        plt.savefig(str(save_path))
    else:
        plt.show()
    return


def display_img(img, over1=None, over2=None, display='mricron'):
    data = nib.load(img).get_fdata()
    if display == 'mricron':
        img_opt = ['-x', '-c', '-0',
                   '-l', '{:.4f}'.format(np.min(data)), '-h', '{:.4f}'.format(np.max(data)), '-b', '60']
    elif display == 'fsleyes':
        # img_opt = ['-cm', 'red', '-a', '40', ]
        fsleyes_command = ['fsleyes', str(img)]
        if over1 is not None:
            fsleyes_command += [str(over1), '-cm', 'red', '-a', '40']
        if over2 is not None:
            fsleyes_command += [str(over2), '-cm', 'green', '-a', '40']
        fsleyes_command = fsleyes_command  # + img_opt
        print('Fsleyes command: "{}"'.format(' '.join(fsleyes_command)))
        process = subprocess.run(fsleyes_command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True
                                 )
        return process
    else:
        raise ValueError(f'{display} display tool unknown')
    label_opt = []
    change_dir = True
    current_dir = os.getcwd()
    relative_dir = Path(img).parent
    if over1:
        change_dir = change_dir and Path(over1).parent == Path(img).parent
        print(f'Non zero voxels in the label: {np.count_nonzero(nib.load(over1).get_fdata())})')
    if over2:
        change_dir = change_dir and Path(over2).parent == Path(img).parent
    if change_dir:
        img = Path(img).name
        over1 = Path(over1).name if over1 else None
        over2 = Path(over2).name if over2 else None
    if over1:
        # -c -1 means red
        label_opt = ['-o', str(over1), '-c', '-1', '-t', '-1']
    seg_opt = []
    if over2:
        # -c -3 means green
        seg_opt = ['-o', str(over2), '-c', '-3', '-t', '-1']
    mricron_options = img_opt + label_opt + seg_opt
    mricron_command = ['mricron', str(img)] + mricron_options
    print('Mricron command: "{}"'.format(mricron_command))
    os.chdir(relative_dir)
    process = subprocess.run(mricron_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    os.chdir(current_dir)
    return process


def display_one_output(output_dir, number=None, string=None, recursive=False, display='mricron'):
    # from monai.transforms import AddChannel, LoadImage
    # from monai.metrics import DiceMetric
    # import torch
    # add_channel = AddChannel()
    # load = LoadImage()
    # dice_metric = DiceMetric(include_background=True, reduction="mean")
    if number is None and string is None:
        raise ValueError('Provide either an image number or a string to find')
    if number is not None:
        if not recursive:
            f_list = [p for p in Path(output_dir).iterdir() if is_nifti(p) and
                      re.search(r'_{}.nii'.format(number), p.name)]
        else:
            f_list = [p for p in Path(output_dir).rglob('*') if is_nifti(p) and
                      re.search(r'_{}.nii'.format(number), p.name)]
    else:
        if not recursive:
            f_list = [p for p in Path(output_dir).iterdir() if is_nifti(p) and string in p.name]
        else:
            f_list = [p for p in Path(output_dir).rglob('*') if is_nifti(p) and string in p.name]
    img = None
    label = None
    seg = None
    for f in f_list:
        if f.name.startswith('input'):
            img = f
        if f.name.startswith('label'):
            print(f'Non zero voxels in the label: {np.count_nonzero(nib.load(f).get_fdata())})')
            label = f
        if f.name.startswith('output'):
            print(f'Non zero voxels in the output prediction: {np.count_nonzero(nib.load(f).get_fdata())}')
            seg = f

    # ldata = torch.tensor(nib.load(label).get_fdata())
    # odata = torch.tensor(nib.load(seg).get_fdata())
    # dice_metric(y_pred=add_channel(odata),
    #             y=add_channel(ldata))
    # dice = dice_metric.aggregate().item()
    # print(f'Dice score: {dice}')
    # dice_metric.reset()
    if img is None:
        raise ValueError('input image not found')
    process = display_img(img, over1=label, over2=seg, display=display)
    return process


def sort_output_images(files, dest_dir, labels=None, subfolder_list=None, tmp_copy=True, skip_already_sorted=True):
    os.makedirs(dest_dir, exist_ok=True)
    if subfolder_list is None:
        subfolder_list = []
    tmp_dir = None
    if tmp_copy:
        tmp_dir = Path(dest_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
    if skip_already_sorted:
        # could do it with tmp but maybe tmp would get deleted at some point so it might be safer this way
        subfolder_paths = [p for p in Path(dest_dir).iterdir() if p.is_dir() and p.name != 'tmp']
        subfolder_list = subfolder_list + [p.name for p in subfolder_paths]
        subfolder_list = list(set(subfolder_list))
        already_sorted_files = [f.name for d in subfolder_paths for f in d.iterdir()]
        to_sort_list = []
        for f in files:
            if Path(f).name not in already_sorted_files:
                to_sort_list.append(f)
        files = to_sort_list
        if labels is not None and labels:
            to_sort_list = []
            for f in labels:
                if Path(f).name not in already_sorted_files:
                    to_sort_list.append(f)
            labels = to_sort_list
    for i, file in tqdm(enumerate(files)):
        print(f'Image {i}/{len(files)}')
        if tmp_dir:
            tmp_f = Path(tmp_dir, Path(file).name)
            shutil.copy(file, tmp_f)
            file = tmp_f
        if labels is not None and labels:
            label = labels[i]
            if tmp_dir:
                tmp_f = Path(tmp_dir, Path(label).name)
                shutil.copy(label, tmp_f)
                label = tmp_f
        else:
            label = None
        display_img(file, label)
        resp = input('Select a subfolder to copy the images into:\n'
                     f'(Current subfolder list: {subfolder_list}\n')
        if resp not in subfolder_list:
            subfolder_list.append(resp)
        os.makedirs(Path(dest_dir, resp), exist_ok=True)
        shutil.move(file, Path(dest_dir, resp, Path(file).name))
        if label is not None:
            shutil.move(label, Path(dest_dir, resp, Path(label).name))
        print(f'Image writen in {Path(dest_dir, resp)}')
    if tmp_dir is not None:
        os.rmdir(tmp_dir)
    return subfolder_list


def sort_seg_following_folder_tree(root_dir, output_dir, img_pref='input_', label_pref='output_',
                                   ignore_folder_names=None,
                                   output_subfolder_depth=3):
    if ignore_folder_names is None:
        ignore_folder_names = []
    output_subfolder_list = list({di.name for di in Path(output_dir).rglob('*')
                                  if di.is_dir() and di.name != 'tmp' and len(
            list(di.relative_to(output_dir).parents)) == output_subfolder_depth})
    if output_subfolder_list:
        print(f'Resuming from precedent sorting with output subfolder names:\n {output_subfolder_list}')
    for d in [p for p in Path(root_dir).rglob('*') if p.is_dir() and p.name not in ignore_folder_names
              and not set(ignore_folder_names).intersection(set([pp.name for pp in p.parents]))]:
        f_list = sorted([f for f in d.iterdir() if is_nifti(f) and f.name.startswith(img_pref)])
        l_list = sorted([f for f in d.iterdir() if is_nifti(f) and f.name.startswith(label_pref)])
        if f_list:
            if len(f_list) != len(l_list):
                raise ValueError(f'The image list and the label list in {d} do not have the same length')
            print(f'Sorting images from {d}')
            output_subfolder_list = sort_output_images(f_list, Path(output_dir, d.relative_to(root_dir)), l_list,
                                                       output_subfolder_list)


def perf_per_cluster(perf_df, seg_folder, cluster_by_labels=False, save_path=None):
    """

    Parameters
    ----------
    perf_df : Union[pd.DataFrame, str, pathlib.Path]
        Usually, 'val_perf_individual_measures.csv' from the segmentation pipeline. Dataframe/csv file path containing
        the columns 'core_filename', 'dice_metric', 'volume', and 'distance'
    seg_folder : Union[str, pathlib.Path]
        Path to the segmentation folder containing 'input_', 'label_', and 'output_' files (they can be in separate
        directories)
    cluster_by_labels : bool
        If True, the clusters will be calculated on the ground truth labels instead of the prediction
    save_path : Union[None, str, pathlib.Path]
        [Optional] Path to a file or to a folder (in which case the filename will be 'pred_cluster_to_img_perf.npy',
         it will be 'label_cluster_to_img_perf.npy' if cluster_by_labels is True)
    Returns
    -------
    cluster_to_img_perf : list of dict
        list of dictionaries with the following keys: {'b1000', 'label', 'segmentation', 'dice', 'volume', 'distance'}
        for each segmentation
    """
    if isinstance(perf_df, str) or isinstance(perf_df, Path):
        perf_df = pd.read_csv(perf_df, header=0, index_col=0)
    file_list = [f for f in Path(seg_folder).rglob('*') if is_nifti(f)]
    cluster_to_img_perf = defaultdict(list)
    for row in tqdm(perf_df.iloc):
        core_name = row['core_filename']
        img = [f for f in file_list if f.name.startswith('input_') and core_name in f.name][0]
        lbl = [f for f in file_list if f.name.startswith('label_') and core_name in f.name][0]
        seg = [f for f in file_list if f.name.startswith('output_') and core_name in f.name][0]
        cluster_name = img.parent.name
        cluster_to_img_perf[cluster_name].append({'b1000': str(img), 'label': str(lbl), 'segmentation': str(seg),
                                                  'dice': row['dice_metric'], 'volume': row['volume'],
                                                  'distance': row['distance']})
    pref = 'pred'
    if cluster_by_labels:
        les_area_finder = LesionAreaFinder()
        labels_cluster_dict = defaultdict(list)
        img_dict_list = []
        for cluster in cluster_to_img_perf:
            img_dict_list += cluster_to_img_perf[cluster]
        for d in tqdm(img_dict_list):
            data = nib.load(d['label']).get_fdata()
            cluster_name = les_area_finder.get_img_area(data)
            labels_cluster_dict[cluster_name].append(d)
        cluster_to_img_perf = dict(labels_cluster_dict)
        pref = 'label'
    if save_path is not None:
        if Path(save_path).is_dir():
            np.save(str(Path(save_path, f'{pref}_cluster_to_img_perf.npy')), cluster_to_img_perf)
        else:
            np.save(save_path, cluster_to_img_perf)
    return cluster_to_img_perf


def get_bilateral_cluster_dict(cluster_dict):
    bilateral_cluster_dict = {}
    for cluster in cluster_dict:
        if cluster.startswith('L_') or cluster.startswith('R_'):
            bilat_clu_name = cluster[2:]
            if bilat_clu_name not in bilateral_cluster_dict:
                bilateral_cluster_dict[bilat_clu_name] = \
                    cluster_dict['L_' + bilat_clu_name] + cluster_dict['R_' + bilat_clu_name]
        else:
            bilateral_cluster_dict[cluster] = cluster_dict[cluster]
    return bilateral_cluster_dict


def plot_perf_per_cluster(cluster_dicts, set_names, output_path, perf_measure='dice', bilateral=True):
    if isinstance(cluster_dicts, dict):
        cluster_dicts = [cluster_dicts]
    if isinstance(set_names, str):
        set_names = [set_names]
    if bilateral:
        bilateral_cluster_dicts = []
        for cluster_dict in cluster_dicts:
            bilateral_cluster_dicts.append(get_bilateral_cluster_dict(cluster_dict))
        cluster_dicts = bilateral_cluster_dicts

    pp = PdfPages(output_path)
    for cluster in cluster_dicts[0]:
        fig, axes = plt.subplots(len(cluster_dicts)//4 + 1, len(cluster_dicts), figsize=(15, 5), sharey='none')
        if not isinstance(axes, np.ndarray):
            axes = np.ndarray([axes])
        fig.suptitle(cluster)
        for ind, cluster_dict in enumerate(cluster_dicts):
            perf_list = [d[perf_measure] for d in cluster_dict[cluster]]
            sns.violinplot(ax=axes[ind], data=perf_list)
            axes[ind].set_title(set_names[ind])
            axes[ind].set_xlabel(f'{len(cluster_dict[cluster])} images | mean: {np.mean(perf_list)}')
            axes[ind].set_ylabel(perf_measure)

        # plt.show()
        pp.savefig(fig)
    pp.close()


def check_and_exclude(seg_dict, root_seg_path, seg_dict_path=None, exclude_json_path=None):
    if not isinstance(seg_dict, dict):
        seg_dict_path = seg_dict
        seg_dict = open_json(seg_dict_path)
    if Path(exclude_json_path).is_file():
        exclude_dict = open_json(exclude_json_path)
    else:
        if not Path(exclude_json_path).parent.is_dir():
            os.makedirs(Path(exclude_json_path).parent)
        exclude_dict = {}
    to_check_keys = []
    for k in tqdm(list(seg_dict.keys())):
        if k not in exclude_dict:
            pred_data = nib.load(root_seg_path + seg_dict[k]['segmentation']).get_fdata()
            if np.count_nonzero(pred_data):
                to_check_keys.append(k)
    for k in tqdm(to_check_keys):
        display_img(root_seg_path + seg_dict[k]['b1000'], root_seg_path + seg_dict[k]['segmentation'],
                    display='fsleyes')
        resp = input(f'Keep[keep] or exclude[{set([exclude_dict[kk]["exclude"] for kk in exclude_dict])}]?')
        if resp.lower() in ['quit', 'q', 'exit']:
            break
        if resp.lower() != 'keep':
            exclude_dict[k] = seg_dict[k]
            exclude_dict[k]['exclude'] = resp
            del seg_dict[k]
    print(len(exclude_dict))
    save_json(exclude_json_path, exclude_dict)
    save_json(seg_dict_path, seg_dict)
    return exclude_dict


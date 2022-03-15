import os
import shutil
import webbrowser
import subprocess
from pathlib import Path
import re
from tqdm import tqdm

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from bcblib.tools.nifti_utils import is_nifti
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


def display_img(img, over1=None, over2=None):
    data = nib.load(img).get_fdata()
    img_opt = ['-x', '-c', '-0',
               '-l', '{:.4f}'.format(np.min(data)), '-h', '{:.4f}'.format(np.max(data)), '-b', '60']
    # img_opt = ['-x', '-c', '-0', '-l', '0', '-h', '1000', '-b', '60']
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


def display_one_output(output_dir, number=None, string=None, recursive=False):
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
    process = display_img(img, over1=label, over2=seg)
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


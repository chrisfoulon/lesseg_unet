import os
import shutil
import webbrowser
import subprocess
from pathlib import Path
import re

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from bcblib.tools.nifti_utils import is_nifti
from bcblib.tools.visualisation import mricron_display


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


def plot_3d_recons_v2(data_to_plot, titles, recon_folder, epoch='', subjects_to_show=5, hyper_params=None, prefix='',
                      per_subject_prefix=None, suppress_progress_bar=False, normalise=True):
    """
    Plot 3 slices per axis for each item in the list 'data_to_plot'
    """
    plot_types = len(data_to_plot)
    subjects_to_show = min(int(data_to_plot[0].shape[0]), subjects_to_show)
    for i in range(subjects_to_show):
        plt.close("all")
        if epoch:
            plt.suptitle("Epoch: " + str(epoch) + "; subject: " + str(i+1), fontsize=10)
        else:
            plt.suptitle("Subject: " + str(i + 1), fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(24, 14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.gray()
        for j in range(plot_types):
            current = np.squeeze(data_to_plot[j][i])
            axis_length = current.shape[0]
            axis_length = axis_length // 4
            plt.subplot(plot_types, 9, j * 9 + 1)
            plot_one_slice(current[axis_length, :, :], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 2)
            plot_one_slice(current[2*axis_length, :, :], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 3)
            plot_one_slice(current[3*axis_length, :, :], normalise=normalise)
            axis_length = current.shape[1]
            axis_length = axis_length // 4
            plt.subplot(plot_types, 9, j * 9 + 4)
            plot_one_slice(current[:, axis_length, :], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 5)
            plot_one_slice(current[:, 2*axis_length, :], titles[j], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 6)
            plot_one_slice(current[:, 3*axis_length, :], normalise=normalise)
            axis_length = current.shape[2]
            axis_length = axis_length // 4
            plt.subplot(plot_types, 9, j * 9 + 7)
            plot_one_slice(current[:, :, axis_length], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 8)
            plot_one_slice(current[:, :, 2*axis_length], normalise=normalise)
            plt.subplot(plot_types, 9, j * 9 + 9)
            plot_one_slice(current[:, :, 3*axis_length], normalise=normalise)
        if per_subject_prefix:
            plt.savefig(Path(recon_folder, prefix + per_subject_prefix[i] + '.png'))
        else:
            plt.savefig(Path(recon_folder, prefix + 'subject_' + str(i+1) + '.png'))
        # if not suppress_progress_bar:
        #     progress_bar(i+1, subjects_to_show, prefix='Plotting:')


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
    if over2:
        change_dir = change_dir and Path(over2).parent == Path(img).parent
    if change_dir:
        img = Path(img).name
        over1 = Path(over1).name if over1 else None
        over2 = Path(over2).name if over2 else None
    if over1:
        label_opt = ['-o', str(over1), '-c', '-1', '-t', '-1']
    seg_opt = []
    if over2:
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


def display_one_output(output_dir, number=None, string=None):
    if number is None and string is None:
        raise ValueError('Provide either an image number or a string to find')
    if number is not None:
        f_list = [p for p in Path(output_dir).iterdir() if is_nifti(p) and re.search(r'_{}.nii'.format(number), p.name)]
    else:
        f_list = [p for p in Path(output_dir).iterdir() if is_nifti(p) and string in p.name]
    img = None
    label = None
    seg = None
    for f in f_list:
        if 'input' in f.name:
            img = f
        if 'label' in f.name:
            label = f
        if 'output' in f.name:
            seg = f
    if img is None:
        raise ValueError('input image not found')
    process = display_img(img, over1=label, over2=seg)
    return process


def sort_output_images(looking_dir, dest_dir, number=None, string=None):
    if not Path(looking_dir).is_dir():
        raise ValueError(f'{looking_dir} is not an existing directory')
    os.makedirs(dest_dir, exist_ok=True)
    display_one_output(looking_dir, number, string)
    subfolder_list = []
    resp = input('Select a subfolder to copy the images into:\n'
                 f'(Current subfolder list: {subfolder_list}')
    shutil.copy()
    print()

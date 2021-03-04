import webbrowser
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib


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
        _, _, z = plotting.find_xyz_cut_coords(nib.Nifti1Image(label, np.eye(4)), activation_threshold=1)
        z = round(z)
    else:
        z = round(len(img[0, 0, :]) / 2)
    # fig = plt.figure()
    # plt.subplot(1,1,1).imshow(img[:, :, z], cmap='gray', interpolation='none')
    plt.imshow(img[:, :, z], cmap='gray', interpolation='none')
    if label is not None:
        plt.imshow(label[:, :, z], cmap='Reds', alpha=label[:, :, z], interpolation='none')
    if seg is not None:
        plt.imshow(seg[:, :, z], cmap='Blues', alpha=label[:, :, z]*.5, interpolation='none')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    return

# l_img = lambda x: '/home/tolhsadum/neuro_apps/data/lesseg_unet_output/val_images/nib_input_{}.nii'.format(x)
# l_label = lambda x: '/home/tolhsadum/neuro_apps/data/lesseg_unet_output/val_images/nib_label_{}.nii'.format(x)
# l_seg = lambda x: '/home/tolhsadum/neuro_apps/data/lesseg_unet_output/val_images/nib_output_{}.nii'.format(x)
# # plot_seg(nib.load(l_img(2)).get_fdata(), nib.load(l_label(2)).get_fdata(), nib.load(l_seg(2)).get_fdata())
# plot_seg(nib.load(l_img(2)).get_fdata(), nib.load(l_label(2)).get_fdata(), nib.load(l_seg(2)).get_fdata(), '/home/tolhsadum/neuro_apps/data/fig.pdf')
def nil_plot_img(img, label=None, seg=None):
    smap = None
    if label is not None:
        smap = nib.load(label)
    if seg is not None:
        if label is not None:
            stack = np.stack((smap.get_fdata(), nib.load(seg).get_fdata()), axis=3)
            smap = nib.Nifti1Image(stack, smap.affine)
        else:
            smap = nib.load(seg)
    if smap:
        display = plotting.plot_prob_atlas(smap, img, colorbar=True)
    else:
        display = plotting.plot_anat(img)

    plotting.show()
    return

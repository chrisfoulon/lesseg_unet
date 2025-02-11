import os
import shutil
import webbrowser
import subprocess
from pathlib import Path
import re
from tqdm import tqdm
from collections import defaultdict
import importlib.resources as rsc
from pprint import pprint

import numpy as np
import pandas as pd
import nibabel as nib
import dask.array as da
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.ndimage import center_of_mass
from bcblib.tools.nifti_utils import is_nifti, file_to_list
from bcblib.tools.spreadsheet_io_utils import import_spreadsheet
from bcblib.tools.general_utils import open_json
from bcblib.tools.nifti_utils import nifti_overlap_images
from lesseg_unet.utils import LesionAreaFinder


def open_tensorboard_page(log_dir, port='8008', new_browser_window=False):
    if not Path(log_dir).is_dir():
        raise ValueError('{} is not an existing directory with the logs'.format(log_dir))
    return_value = subprocess.call(['python', '-m', 'tensorboard.main', '--logdir', log_dir, '--port', port])
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


def display_img(img, over1=None, over2=None, display='mricron', coord=None):
    data = nib.load(img).get_fdata()
    if display == 'mricron':
        img_opt = ['-x', '-c', '-0',
                   '-l', '{:.4f}'.format(np.min(data)), '-h', '{:.4f}'.format(np.max(data)), '-b', '60']
    elif display == 'fsleyes':
        # img_opt = ['-cm', 'red', '-a', '40', ]
        if coord is not None:
            coord_str = '-vl ' + ' '.join([str(c) for c in coord])
            fsleyes_command = ['fsleyes', coord_str, str(img)]
        else:
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


def seg_dict_to_cluster_dict(seg_dict, seg_key='segmentation'):
    cluster_lists_dict = defaultdict(list)
    for k in seg_dict:
        cluster_name = Path(seg_dict[k][seg_key]).parent.name
        cluster_lists_dict[cluster_name].append(seg_dict[k])
    return cluster_lists_dict


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


def plot_perf_per_cluster(cluster_dicts, set_names, output_path, display_cluster_overlap=True,
                          perf_measure='dice', bilateral=True):
    if isinstance(cluster_dicts, dict):
        cluster_dicts = [cluster_dicts]
    if isinstance(set_names, str):
        set_names = [set_names]
    if bilateral:
        bilateral_cluster_dicts = []
        for cluster_dict in cluster_dicts:
            bilateral_cluster_dicts.append(get_bilateral_cluster_dict(cluster_dict))
        cluster_dicts = bilateral_cluster_dicts

    archetypes = None
    if display_cluster_overlap:
        with rsc.path('lesseg_unet.data', 'Final_lesion_clusters') as p:
            clusters_dir = str(p.resolve())
        archetypes = [p for p in Path(clusters_dir).iterdir() if is_nifti(p)]

    pp = PdfPages(output_path)
    for cluster in cluster_dicts[0]:
        img_plot_num = 0
        cluster_archetype = None
        overlap_image = None
        seg_path_list = []
        if archetypes is not None:
            img_plot_num = 2
            cluster_archetype = [p for p in archetypes if p.name == cluster + '.nii.gz']
            seg_path_list = [d['segmentation'] for d in cluster_dicts[0][cluster]]
            overlap_image = nifti_overlap_images(seg_path_list, mean=False)

        fig, axes = plt.subplots((len(cluster_dicts) + img_plot_num)//4 + 1, len(cluster_dicts) + img_plot_num,
                                 figsize=(15, 5), sharey='none')

        flat_axes = axes.flatten()
        if not isinstance(axes, np.ndarray):
            flat_axes = np.ndarray([axes])
        fig.suptitle(cluster)
        if archetypes is not None:
            if cluster == 'outside_clusters' or len(seg_path_list) == 0:
                flat_axes[0].axis('off')
            else:
                plot_stat_map(nib.load(cluster_archetype[0]), display_mode='yz', axes=flat_axes[0], draw_cross=False)
            flat_axes[0].set_title('Archetype')
            if len(seg_path_list) != 0:
                plot_stat_map(overlap_image, display_mode='yz', axes=flat_axes[1], draw_cross=False)
                flat_axes[1].set_title('Prediction overlap')
            else:
                flat_axes[1].axis('off')
        for ind, cluster_dict in enumerate(cluster_dicts):
            perf_list = [d[perf_measure] for d in cluster_dict[cluster]]
            if len(perf_list) == 0:
                flat_axes[ind + img_plot_num].axis('off')
                continue
            sns.violinplot(ax=flat_axes[ind + img_plot_num], data=perf_list)
            # Fix the value range to have the same for every cluster
            flat_axes[ind + img_plot_num].set(ylim=(-0.2, 1.2))
            flat_axes[ind + img_plot_num].set_title(set_names[ind])
            flat_axes[ind + img_plot_num].set_xlabel(
                f'{len(cluster_dict[cluster])} images | mean: {np.mean(perf_list)}')
            flat_axes[ind + img_plot_num].set_ylabel(perf_measure)

        # plt.show()
        pp.savefig(fig)
    pp.close()


def plot_archetype_and_cluster_seg(cluster_dict, output_path, key='segmentation', bilateral=True, input_folder=''):
    if not isinstance(cluster_dict, dict):
        cluster_dict = open_json(cluster_dict)
    if bilateral:
        bilateral_cluster_dicts = get_bilateral_cluster_dict(cluster_dict)
        cluster_dict = bilateral_cluster_dicts

    with rsc.path('lesseg_unet.data', 'Final_lesion_clusters') as p:
        clusters_dir = str(p.resolve())
    archetypes = [p for p in Path(clusters_dir).iterdir() if is_nifti(p)]

    pp = PdfPages(output_path)
    for cluster in cluster_dict:
        cluster_archetype = [p for p in archetypes if p.name == cluster + '.nii.gz']
        seg_path_list = [Path(input_folder, d[key]) for d in cluster_dict[cluster]]
        overlap_image = nifti_overlap_images(seg_path_list, mean=False)

        fig, axes = plt.subplots(1, 2,
                                 figsize=(15, 5), sharey='none')
        fig.suptitle(cluster)
        if cluster == 'outside_clusters' or len(seg_path_list) == 0:
            axes[0].axis('off')
        else:
            plot_stat_map(nib.load(cluster_archetype[0]), display_mode='yz', axes=axes[0], draw_cross=False, )
        axes[0].set_title('Archetype')
        if len(seg_path_list) != 0:
            plot_stat_map(overlap_image, display_mode='yz', axes=axes[1], draw_cross=False)
            axes[1].set_title('Prediction overlap')
        else:
            axes[1].axis('off')
        pp.savefig(fig)
    pp.close()


def plot_perf_violin_per_cluster(perf_seg_dict, output_folder, metric_name='dice_metric', y_val_range=(0, 1)):
    lesion_cluster_list = list(
        set([perf_seg_dict[k]['lesion_cluster'].split('L_')[-1].split('R_')[-1] for k in perf_seg_dict]))
    # For each lesion_cluster, create a list of metric values
    metric_list = []
    for lesion_cluster in lesion_cluster_list:
        metric_list.append([perf_seg_dict[k][metric_name] for k in perf_seg_dict if
                            lesion_cluster in perf_seg_dict[k]['lesion_cluster']])
    # sort the cluster list and the metric list by the mean of the metric (decreasing)
    lesion_cluster_list, metric_list = zip(
        *sorted(zip(lesion_cluster_list, metric_list), key=lambda x: np.mean(x[1]), reverse=True))
    # Then, plot the distribution of the metric for each lesion_cluster (one violinplot per lesion_cluster) and compute the correlation between the lesion_cluster and the metric and display all the plots in one figure (save it as a png in the fold_root_folder)
    # fig, axes = plt.subplots(len(lesion_cluster_list) // 4 + 1, 4)
    fig, axes = plt.subplots(1, len(lesion_cluster_list), sharey=True)
    # make plots taller and larger
    plt.rcParams['figure.figsize'] = [20, 15]
    # Add space between rows
    fig.subplots_adjust(hspace=0.5)
    # fix error: AttributeError: 'numpy.ndarray' object has no attribute 'fill_betweenx'
    axes = axes.flatten()
    # all plots y axis should be between 0 and 1
    # if y_val_range is not None:
    if y_val_range is not None:
        for ax in axes:
            ax.set_ylim(y_val_range)
    else:
        # compute the min and max of the metric_list
        min_metric = min([min(metric) for metric in metric_list])
        max_metric = max([max(metric) for metric in metric_list])
        for ax in axes:
            ax.set_ylim(min_metric, max_metric)
    # # if the number of lesion_cluster is not a multiple of 4, the last row of plots should be empty
    # if len(lesion_cluster_list) % 4 != 0:
    #     for ax in axes[-(4 - len(lesion_cluster_list) % 4):]:
    #         ax.axis('off')

    for ind, lesion_cluster in enumerate(lesion_cluster_list):
        sns.violinplot(ax=axes[ind], data=metric_list[ind])
        # instead of title, the lesion_cluster, number of images in the cluster and
        # the mean metric should be displayed on the x axis in diagonal
        axes[ind].set_xlabel(f'{lesion_cluster}', rotation=45, ha='right')
        # axes[ind].set_xlabel(f'{lesion_cluster} / {len(metric_list[ind])} / {np.mean(metric_list[ind]):.2f}',
        # rotation=45, ha='right')

    plt.savefig(Path(output_folder, f'{metric_name}_lesion_cluster_correlation.png'))
    plt.show()


def perf_dataset_overlap(input_images, spreadsheet, filename_col='core_filename', perf_col='dice_metric',
                         operation='mean', filter_pref='', recursive=False, non_zero_only=False, header=0,
                         window_size=-1):
    if not isinstance(input_images, list):
        if Path(input_images).is_file():
            input_images = [str(p) for p in file_to_list(input_images) if is_nifti(p)]
        elif Path(input_images).is_dir():
            if recursive:
                input_images = [str(p) for p in Path(input_images).rglob('*') if is_nifti(p)]
            else:
                input_images = [str(p) for p in Path(input_images).iterdir() if is_nifti(p)]
        else:
            raise ValueError('Wrong input (must be a file/directory path of a list of paths)')
    if filter_pref:
        input_images = [p for p in input_images if Path(p).name.startswith(filter_pref)]
    if not input_images:
        print(' The image list is empty')
        return None
    df = import_spreadsheet(spreadsheet, header)
    temp_overlap = None
    temp_overlap_data = None
    chunk_length = 0
    for img in tqdm(input_images):
        # find the row in the dataframe
        perf_val = None
        for row in df.iloc:
            if row[filename_col] in Path(img).name:
                perf_val = row[perf_col]
                break
        if perf_val is None:
            ValueError(f'{img} not found in spreadsheet in {filename_col} column')
        nii = nib.load(img)
        nii_data = nii.get_fdata() * perf_val
        if non_zero_only:
            nii_data[nii_data == 0] = np.nan
        if temp_overlap_data is None:
            temp_overlap_data = da.from_array(nii_data)
            temp_overlap = nii
            temp_overlap_data = da.stack([temp_overlap_data], axis=0)
        else:
            temp_overlap_data = da.concatenate([temp_overlap_data, [da.from_array(nii_data)]])
    #     else:
    #         if operation in ['mean', 'std']:
    #             temp_overlap_data.append(nii)
    #             # temp_overlap_data = np.std(stack, axis=0)
    #         else:
    #             raise ValueError(f'{operation} not implemented yet')
    #     chunk_length += 1
    #     # if chunk_length == window_size:
    # temp_overlap_data_stack = np.stack(temp_overlap_data)
    #         # if operation == 'mean':
    #         #     temp_overlap_data = np.nanmean(temp_overlap_data, axis=0)
    if operation == 'mean':
        # temp_overlap_data = np.nanmean(temp_overlap_data, axis=0)
        temp_overlap_data = da.nanmean(temp_overlap_data, axis=0).compute()
    if operation == 'std':
        # temp_overlap_data = np.nanstd(temp_overlap_data, axis=0)
        temp_overlap_data = da.nanstd(temp_overlap_data, axis=0).compute()
    temp_overlap = nib.Nifti1Image(temp_overlap_data, temp_overlap.affine)
    return temp_overlap


def perf_dataset_overlap_from_json(json_dict, filename_key='segmentation', perf_key='dice_metric',
                                   operation='mean', non_zero_only=False):
    if isinstance(json_dict, str):
        json_dict = open_json(json_dict)
    temp_overlap = None
    temp_overlap_data = None
    for k in tqdm(json_dict):
        nii = nib.load(json_dict[k][filename_key])
        nii_data = nii.get_fdata() * json_dict[k][perf_key]
        if non_zero_only:
            nii_data[nii_data == 0] = np.nan
        if temp_overlap_data is None:
            temp_overlap_data = da.from_array(nii_data)
            temp_overlap = nii
            temp_overlap_data = da.stack([temp_overlap_data], axis=0)
        else:
            temp_overlap_data = da.concatenate([temp_overlap_data, [da.from_array(nii_data)]])
    #     else:
    #         if operation in ['mean', 'std']:
    #             temp_overlap_data.append(nii)
    #             # temp_overlap_data = np.std(stack, axis=0)
    #         else:
    #             raise ValueError(f'{operation} not implemented yet')
    #     chunk_length += 1
    #     # if chunk_length == window_size:
    # temp_overlap_data_stack = np.stack(temp_overlap_data)
    #         # if operation == 'mean':
    #         #     temp_overlap_data = np.nanmean(temp_overlap_data, axis=0)
    if operation == 'mean':
        # temp_overlap_data = np.nanmean(temp_overlap_data, axis=0)
        temp_overlap_data = da.nanmean(temp_overlap_data, axis=0).compute()
    if operation == 'std':
        # temp_overlap_data = np.nanstd(temp_overlap_data, axis=0)
        temp_overlap_data = da.nanstd(temp_overlap_data, axis=0).compute()
    temp_overlap = nib.Nifti1Image(temp_overlap_data, temp_overlap.affine)
    return temp_overlap

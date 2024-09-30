from pathlib import Path
import os
import sys

from bcblib.tools.umap_utils import nifti_dataset_to_matrix, train_umap
from bcblib.tools.general_utils import open_json
import nibabel as nib
import joblib
import numpy as np
from tqdm import tqdm


def compute_umap_embeddings(label_paths, model_paths_dict, output_folder, random_state=42):
    # Load the labels
    label_list = [nib.load(path) for path in tqdm(label_paths, desc='Loading labels')]

    # Create the output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Compute the UMAP input matrix for the labels
    umap_input_matrix = nifti_dataset_to_matrix(label_list)

    # Check if the UMAP model for the labels already exists
    output_label_umap_path = output_folder / '/labels_trained_umap.joblib'

    if not os.path.exists(output_label_umap_path):
        # Train a new UMAP model for the labels
        trained_umap = train_umap(umap_input_matrix, n_neighbors=15, min_dist=0.1, n_components=3,
                                  random_state=random_state)
        # Save the trained UMAP model
        joblib.dump(trained_umap, output_label_umap_path)
    else:
        # Load the existing UMAP model
        trained_umap = joblib.load(output_label_umap_path)

    # Transform the UMAP input matrix for the labels
    transformed_umap = trained_umap.transform(umap_input_matrix)

    # Store the UMAP embeddings in a dictionary
    embeddings_dict = {}
    embeddings_dict['labels'] = transformed_umap

    # Save the labels embedding
    np.save(output_folder / '/labels_embedding.npy', transformed_umap)

    # Iterate over each model
    for model_name, segmentation_paths in model_paths_dict.items():
        # Load the segmentations for the model
        segmentation_list = [nib.load(path) for path in tqdm(segmentation_paths, desc='Loading segmentations')]
        # Compute the UMAP input matrix for the model
        umap_input_matrix = nifti_dataset_to_matrix(segmentation_list)
        # Train a new UMAP model for the model
        trained_umap = train_umap(umap_input_matrix, n_neighbors=15, min_dist=0.1, n_components=3,
                                  random_state=random_state)
        # Save the trained UMAP model
        joblib.dump(trained_umap, Path(output_folder, f'{model_name}_trained_umap.joblib'))
        # Transform the UMAP input matrix for the model
        transformed_umap = trained_umap.transform(umap_input_matrix)
        # Store the UMAP embeddings for the model
        embeddings_dict[model_name] = transformed_umap
        # Save the model's embedding
        np.save(output_folder / f'/{model_name}_embedding.npy', transformed_umap)

    # Return the dictionary containing the UMAP embeddings
    return embeddings_dict


def train_and_embed_from_image_dataset(image_paths, output_folder, prefix, random_state=None, disable_progress=True,
                                       umap_params=None):
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'random_state': random_state,
            'verbose': True,
        }
    print(f'Loading images for [{prefix} / {str(random_state)}]')
    # Load the images
    image_list = [nib.load(path) for path in tqdm(image_paths, desc='Loading images', disable=disable_progress)]
    print(f'Loaded all images ({len(image_list)}) for [{prefix} / {str(random_state)}]')
    # if the umap exists, load it, otherwise create the input matrix and train it
    output_name_umap_path = Path(output_folder, f'{prefix}_trained_umap.joblib')
    output_name_embedding_path = Path(output_folder, f'{prefix}_embedding.npy')
    umap_input_matrix = None
    if output_name_umap_path.exists():
        # if the embeddings already exist, load them and return them
        if output_name_embedding_path.exists():
            print(f'Loading embedding for [{prefix} / {str(random_state)}] at {output_name_embedding_path}')
            return np.load(output_name_embedding_path)
        # Load the existing UMAP model
        trained_umap = joblib.load(output_name_umap_path)
        print(f'Loaded UMAP for [{prefix} / {str(random_state)}] at {output_name_umap_path}')
    else:
        # Compute the UMAP input matrix for the images
        umap_input_matrix = nifti_dataset_to_matrix(image_list, disable_progress=disable_progress)
        print(f'Size of umap_input_matrix in memory (in Mb): {sys.getsizeof(umap_input_matrix) / 1024 / 1024} Mb')
        print(f'Computed UMAP input matrix for [{prefix} / {str(random_state)}]')
        # Train a new UMAP model for the images
        trained_umap = train_umap(umap_input_matrix, **umap_params)
        # Save the trained UMAP model
        joblib.dump(trained_umap, Path(output_folder, f'{prefix}_trained_umap.joblib'))
        print(
            f'Trained UMAP for [{prefix} / {str(random_state)}] at {Path(output_folder, f"{prefix}_trained_umap.joblib")}')

    if umap_input_matrix is None:
        # Compute the UMAP input matrix for the images
        umap_input_matrix = nifti_dataset_to_matrix(image_list, disable_progress=disable_progress)
        print(f'Size of umap_input_matrix in memory (in Mb): {sys.getsizeof(umap_input_matrix) / 1024 / 1024} Mb')
        print(f'Computed UMAP input matrix for [{prefix} / {str(random_state)}]')
    # Transform the UMAP input matrix for the images
    transformed_umap = trained_umap.transform(umap_input_matrix)
    # Save the images embedding
    np.save(Path(output_folder, f'{prefix}_embedding.npy'), transformed_umap)
    print(f'Saved embedding for [{prefix} / {str(random_state)}] at {Path(output_folder, f"{prefix}_embedding.npy")}')
    # Return the embedding
    return transformed_umap


if __name__ == '__main__':

    segmentation_folders = [
        '/gamma/Dropbox (GIN)/UCL_Data/segmentations_all_models/abnormal_training_no_augs_unet',
        '/gamma/Dropbox (GIN)/UCL_Data/segmentations_all_models/control_training_wf_div1000_dicefocal',
        '/gamma/Dropbox (GIN)/UCL_Data/segmentations_all_models/abnormal_segmentation_dice_focal'
    ]

    perf_dict_filename = 'perf_seg_dict.json'

    label_dict_path = '/gamma/Dropbox (GIN)/UCL_Data/cleaned_abnormal_b1000_info_dict_label_size_with_lesion_cluster.json'
    label_dict = open_json(label_dict_path)

    # Define a list of seeds
    seeds = [42, 91, 117, 66, 1337]

    # Define the base output folder
    base_output_folder = '/gamma/Dropbox (GIN)/UCL_Data/umaps_2d_seed_'

    # Define the label paths and model paths dictionary
    label_paths = [
        label_dict[k]['label'].replace('/media/chrisfoulon/HDD2/final_training_set/', '/gamma/Dropbox (GIN)/UCL_Data/')
        for k in label_dict
    ]

    # try to load all the label paths
    for label_path in label_paths:
        if not os.path.exists(label_path):
            raise ValueError(f'File {label_path} does not exist')
        # if the path is not a nifti file
        if not label_path.endswith('.nii') and not label_path.endswith('.nii.gz'):
            raise ValueError(f'File {label_path} is not a nifti file')

    # model_paths_dict should associate model names with the list of 'segmentation' paths in the perf dict json (need to load it)
    model_paths_dict = {
        'labels': label_paths
    }
    for segmentation_folder in segmentation_folders:
        perf_dict = open_json(Path(segmentation_folder, perf_dict_filename))
        model_name = Path(segmentation_folder).name
        model_paths_dict[model_name] = [
            perf_dict[k]['segmentation'].replace('/media/chrisfoulon/HDD2/final_training_set/',
                                                 '/gamma/Dropbox (GIN)/UCL_Data/segmentations_all_models/')
            for k in perf_dict
        ]

        # try to load all the model paths
        for model_path in model_paths_dict[model_name]:
            if not os.path.exists(model_path):
                raise ValueError(f'File {model_path} does not exist')
            # if the path is not a nifti file
            if not model_path.endswith('.nii') and not model_path.endswith('.nii.gz'):
                raise ValueError(f'File {model_path} is not a nifti file')

    # print the length of the lists in model_paths_dict
    for model_name, model_paths in model_paths_dict.items():
        print(f'{model_name}: {len(model_paths)}')

    # create list of lists [model_name, seed, images_paths]
    model_paths_list = []
    for model_name, model_paths in model_paths_dict.items():
        for seed in seeds:
            model_paths_list.append([model_name, seed, model_paths])

    seed_embeddings_dict = {seed: {} for seed in seeds}
    # shorten the list to only 2 seeds
    # model_paths_list = model_paths_list[:2]
    # use train_and_embed_from_image_dataset
    # if not parallel:
    for model_name, seed, model_paths in model_paths_list:
        print(f'Processing {model_name} with seed {seed}')
        seed_embeddings_dict[seed][model_name] = train_and_embed_from_image_dataset(model_paths,
                                                                                    base_output_folder + str(seed),
                                                                                    model_name, random_state=seed,
                                                                                    disable_progress=False)
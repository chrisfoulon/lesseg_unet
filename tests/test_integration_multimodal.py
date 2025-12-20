"""Integration tests for multi-modal pipeline.

End-to-end tests that verify the complete multi-modal workflow from
folder-based input through transform adaptation and model configuration.
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from lesseg_unet.data_utils import (
    folder_mode_to_split_lists,
    adapt_transforms_for_multimodal,
    extract_model_config,
)


@pytest.fixture
def temp_multimodal_dataset(tmp_path):
    """Create temporary multi-modal NIfTI dataset for testing.

    Creates a realistic dataset structure:
    - 6 subjects (sub-001 to sub-006)
    - 2 modalities (DWI, ADC)
    - 1 label per subject
    - Small 10x10x10 volumes to keep tests fast
    """
    # Create folder structure
    dwi_folder = tmp_path / "dwi"
    adc_folder = tmp_path / "adc"
    label_folder = tmp_path / "labels"

    dwi_folder.mkdir()
    adc_folder.mkdir()
    label_folder.mkdir()

    # Create 6 subjects with both modalities and labels
    subject_ids = [f'sub-{i:03d}' for i in range(1, 7)]

    for subject_id in subject_ids:
        # Create small 10x10x10 volumes
        img_shape = (10, 10, 10)

        # DWI image
        dwi_data = np.random.rand(*img_shape).astype(np.float32)
        dwi_img = nib.Nifti1Image(dwi_data, affine=np.eye(4))
        nib.save(dwi_img, dwi_folder / f"{subject_id}_dwi.nii.gz")

        # ADC image
        adc_data = np.random.rand(*img_shape).astype(np.float32)
        adc_img = nib.Nifti1Image(adc_data, affine=np.eye(4))
        nib.save(adc_img, adc_folder / f"{subject_id}_adc.nii.gz")

        # Label mask (binary)
        label_data = np.random.randint(0, 2, img_shape).astype(np.uint8)
        label_img = nib.Nifti1Image(label_data, affine=np.eye(4))
        nib.save(label_img, label_folder / f"{subject_id}_mask.nii.gz")

    return {
        'dwi_folder': dwi_folder,
        'adc_folder': adc_folder,
        'label_folder': label_folder,
        'subject_ids': subject_ids,
        'n_subjects': len(subject_ids)
    }


@pytest.fixture
def temp_single_modality_dataset(tmp_path):
    """Create temporary single-modality NIfTI dataset for backward compatibility testing.

    Creates:
    - 4 subjects (sub-001 to sub-004)
    - 1 modality (T1)
    - 1 label per subject
    """
    img_folder = tmp_path / "t1"
    label_folder = tmp_path / "labels"

    img_folder.mkdir()
    label_folder.mkdir()

    subject_ids = [f'sub-{i:03d}' for i in range(1, 5)]

    for subject_id in subject_ids:
        img_shape = (10, 10, 10)

        # T1 image
        img_data = np.random.rand(*img_shape).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine=np.eye(4))
        nib.save(img, img_folder / f"{subject_id}_t1.nii.gz")

        # Label mask
        label_data = np.random.randint(0, 2, img_shape).astype(np.uint8)
        label_img = nib.Nifti1Image(label_data, affine=np.eye(4))
        nib.save(label_img, label_folder / f"{subject_id}_mask.nii.gz")

    return {
        'img_folder': img_folder,
        'label_folder': label_folder,
        'subject_ids': subject_ids,
        'n_subjects': len(subject_ids)
    }


class TestMultiModalPipelineIntegration:
    """End-to-end integration tests for multi-modal pipeline."""

    def test_full_multimodal_pipeline(self, temp_multimodal_dataset):
        """Test complete pipeline: folder → split_lists → transforms → model config."""
        # Step 1: Convert folders to split_lists
        split_lists = folder_mode_to_split_lists(
            image_folders={
                'dwi': temp_multimodal_dataset['dwi_folder'],
                'adc': temp_multimodal_dataset['adc_folder']
            },
            label_folder=temp_multimodal_dataset['label_folder'],
            n_folds=3,
            subject_pattern=r'(sub-\d+)',
            random_seed=42
        )

        # Verify split_lists structure
        assert len(split_lists) == 3  # 3 folds
        assert sum(len(fold) for fold in split_lists) == 6  # 6 subjects total

        # Verify all subjects have both modalities
        for fold in split_lists:
            for subject in fold:
                assert 'image_adc' in subject
                assert 'image_dwi' in subject
                assert 'label' in subject
                assert len(subject) == 3  # Exactly 3 keys

        # Step 2: Create a transform dict
        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}},
            ]
        }

        # Step 3: Adapt transforms for multi-modal
        adapted_dict = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # Verify transform adaptation
        first_transform = adapted_dict['first_transform']

        # LoadImaged should have image_adc, image_dwi, label
        load_keys = first_transform[0]['LoadImaged']['keys']
        assert 'image_adc' in load_keys
        assert 'image_dwi' in load_keys
        assert 'label' in load_keys
        assert 'image' not in load_keys  # Original 'image' replaced

        # EnsureChannelFirstd should also have both image keys
        ensure_keys = first_transform[1]['EnsureChannelFirstd']['keys']
        assert 'image_adc' in ensure_keys
        assert 'image_dwi' in ensure_keys

        # ConcatItemsd should be inserted at index 2
        assert 'ConcatItemsd' in first_transform[2]
        assert first_transform[2]['ConcatItemsd']['name'] == 'image'
        assert set(first_transform[2]['ConcatItemsd']['keys']) == {'image_adc', 'image_dwi'}

        # NormalizeIntensityd should still use 'image' (concatenated result)
        assert first_transform[3]['NormalizeIntensityd']['keys'] == ['image']

        # Step 4: Extract model config
        model_config = extract_model_config(split_lists)

        # Verify model config
        assert model_config['in_channels'] == 2  # 2 modalities
        assert model_config['out_channels'] == 1  # 1 label

    def test_backward_compatibility_single_modality(self, temp_single_modality_dataset):
        """Test that single-modality workflow remains unchanged (backward compatibility)."""
        # Step 1: Convert folders to split_lists (single modality)
        split_lists = folder_mode_to_split_lists(
            image_folders={'t1': temp_single_modality_dataset['img_folder']},
            label_folder=temp_single_modality_dataset['label_folder'],
            n_folds=2,
            subject_pattern=r'(sub-\d+)',
            random_seed=42
        )

        # Verify split_lists structure
        assert len(split_lists) == 2  # 2 folds
        assert sum(len(fold) for fold in split_lists) == 4  # 4 subjects

        # Verify subjects have single modality with identifier
        for fold in split_lists:
            for subject in fold:
                assert 'image_t1' in subject  # Single modality with identifier
                assert 'label' in subject
                assert len(subject) == 2

        # Step 2: Transform dict
        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}},
            ]
        }

        # Step 3: Adapt transforms (should handle single modality)
        adapted_dict = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # Verify LoadImaged uses image_t1
        load_keys = adapted_dict['first_transform'][0]['LoadImaged']['keys']
        assert 'image_t1' in load_keys
        assert 'label' in load_keys

        # Verify NO ConcatItemsd is added (single modality doesn't need concat)
        transform_names = [list(t.keys())[0] for t in adapted_dict['first_transform']]
        assert 'ConcatItemsd' not in transform_names

        # Step 4: Extract model config
        model_config = extract_model_config(split_lists)

        # Verify single-modality config
        assert model_config['in_channels'] == 1
        assert model_config['out_channels'] == 1

    def test_three_modalities_integration(self, tmp_path):
        """Test pipeline with three modalities (FLAIR, DWI, ADC)."""
        # Create 3-modality dataset
        flair_folder = tmp_path / "flair"
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        label_folder = tmp_path / "labels"

        for folder in [flair_folder, dwi_folder, adc_folder, label_folder]:
            folder.mkdir()

        # Create 3 subjects with all 3 modalities
        for i in range(1, 4):
            subject_id = f'sub-{i:03d}'
            img_shape = (10, 10, 10)

            for modality, folder in [('flair', flair_folder), ('dwi', dwi_folder), ('adc', adc_folder)]:
                data = np.random.rand(*img_shape).astype(np.float32)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, folder / f"{subject_id}_{modality}.nii.gz")

            # Label
            label_data = np.random.randint(0, 2, img_shape).astype(np.uint8)
            label_img = nib.Nifti1Image(label_data, affine=np.eye(4))
            nib.save(label_img, label_folder / f"{subject_id}_mask.nii.gz")

        # Full pipeline test
        split_lists = folder_mode_to_split_lists(
            image_folders={'flair': flair_folder, 'dwi': dwi_folder, 'adc': adc_folder},
            label_folder=label_folder,
            n_folds=2,
            subject_pattern=r'(sub-\d+)'
        )

        # Verify 3 image keys
        first_subject = split_lists[0][0]
        assert 'image_adc' in first_subject
        assert 'image_dwi' in first_subject
        assert 'image_flair' in first_subject
        assert 'label' in first_subject

        # Verify model config
        model_config = extract_model_config(split_lists)
        assert model_config['in_channels'] == 3  # 3 modalities
        assert model_config['out_channels'] == 1

        # Verify transform adaptation with 3 modalities
        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
            ]
        }

        adapted_dict = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # ConcatItemsd should merge all 3 modalities
        concat_keys = adapted_dict['first_transform'][2]['ConcatItemsd']['keys']
        assert len(concat_keys) == 3
        assert set(concat_keys) == {'image_adc', 'image_dwi', 'image_flair'}

        # Verify alphabetical ordering
        assert concat_keys == ['image_adc', 'image_dwi', 'image_flair']

    def test_file_existence_verification(self, temp_multimodal_dataset):
        """Verify that all files in split_lists actually exist."""
        split_lists = folder_mode_to_split_lists(
            image_folders={
                'dwi': temp_multimodal_dataset['dwi_folder'],
                'adc': temp_multimodal_dataset['adc_folder']
            },
            label_folder=temp_multimodal_dataset['label_folder'],
            n_folds=3,
            subject_pattern=r'(sub-\d+)'
        )

        # Verify all file paths exist
        for fold in split_lists:
            for subject in fold:
                for key, file_path in subject.items():
                    assert Path(file_path).exists(), f"File not found: {file_path}"
                    assert Path(file_path).suffix == '.gz', f"Not a .nii.gz file: {file_path}"

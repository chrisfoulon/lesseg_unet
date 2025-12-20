"""Unit tests for folder-per-modality converter.

Tests for converting folder-based multi-modal data structure to SplitLists format.
"""

import re
import pytest
from pathlib import Path
from lesseg_unet.data_utils import (
    folder_mode_to_split_lists,
    build_schema,
    validate_against_schema,
)


@pytest.fixture
def temp_data_structure(tmp_path):
    """Create temporary folder structure with mock NIfTI files.

    Structure:
        dwi/
            sub-001_dwi.nii.gz
            sub-002_dwi.nii.gz
            sub-003_dwi.nii.gz
        adc/
            sub-001_adc.nii.gz
            sub-002_adc.nii.gz
            sub-003_adc.nii.gz
        labels/
            sub-001_lesion.nii.gz
            sub-002_lesion.nii.gz
            sub-003_lesion.nii.gz
    """
    # Create folders
    dwi_folder = tmp_path / "dwi"
    adc_folder = tmp_path / "adc"
    label_folder = tmp_path / "labels"

    dwi_folder.mkdir()
    adc_folder.mkdir()
    label_folder.mkdir()

    # Create mock files (empty files for testing)
    subjects = ['sub-001', 'sub-002', 'sub-003']
    for subject in subjects:
        (dwi_folder / f"{subject}_dwi.nii.gz").touch()
        (adc_folder / f"{subject}_adc.nii.gz").touch()
        (label_folder / f"{subject}_lesion.nii.gz").touch()

    return {
        'dwi_folder': dwi_folder,
        'adc_folder': adc_folder,
        'label_folder': label_folder,
        'subjects': subjects
    }


@pytest.fixture
def temp_single_modality(tmp_path):
    """Create temporary structure with single modality (backward compatible)."""
    dwi_folder = tmp_path / "dwi"
    label_folder = tmp_path / "labels"

    dwi_folder.mkdir()
    label_folder.mkdir()

    subjects = ['sub-001', 'sub-002']
    for subject in subjects:
        (dwi_folder / f"{subject}_dwi.nii.gz").touch()
        (label_folder / f"{subject}_lesion.nii.gz").touch()

    return {
        'dwi_folder': dwi_folder,
        'label_folder': label_folder,
        'subjects': subjects
    }


@pytest.fixture
def temp_three_modalities(tmp_path):
    """Create temporary structure with three modalities."""
    dwi_folder = tmp_path / "dwi"
    adc_folder = tmp_path / "adc"
    flair_folder = tmp_path / "flair"
    label_folder = tmp_path / "labels"

    for folder in [dwi_folder, adc_folder, flair_folder, label_folder]:
        folder.mkdir()

    subjects = ['sub-001', 'sub-002']
    for subject in subjects:
        (dwi_folder / f"{subject}_dwi.nii.gz").touch()
        (adc_folder / f"{subject}_adc.nii.gz").touch()
        (flair_folder / f"{subject}_flair.nii.gz").touch()
        (label_folder / f"{subject}_lesion.nii.gz").touch()

    return {
        'dwi_folder': dwi_folder,
        'adc_folder': adc_folder,
        'flair_folder': flair_folder,
        'label_folder': label_folder,
        'subjects': subjects
    }


class TestFolderModeConverter:
    """Test folder_mode_to_split_lists() function."""

    def test_basic_two_modalities(self, temp_data_structure):
        """Test basic conversion with DWI + ADC modalities."""
        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=3,
            random_seed=42
        )

        # Check structure
        assert len(split_lists) == 3, "Should have 3 folds"

        # Check all subjects are present
        all_subjects = [subj for fold in split_lists for subj in fold]
        assert len(all_subjects) == 3, "Should have 3 subjects total"

        # Check first subject has correct keys
        first_subject = split_lists[0][0]
        assert 'image_dwi' in first_subject
        assert 'image_adc' in first_subject
        assert 'label' in first_subject
        assert len(first_subject) == 3  # Exactly these 3 keys

        # Build schema and validate all subjects
        schema = build_schema(split_lists[0][0])
        assert schema['has_multi_modal_images'] is True
        assert len(schema['image_keys']) == 2

        # Validate all subjects match schema
        for fold_idx, fold in enumerate(split_lists):
            for subj_idx, subject in enumerate(fold):
                validate_against_schema(
                    subject, schema, f"fold{fold_idx}_subj{subj_idx}"
                )

    def test_single_modality_backward_compatible(self, temp_single_modality):
        """Test single modality (backward compatible with existing workflows)."""
        image_folders = {
            'dwi': temp_single_modality['dwi_folder']
        }
        label_folder = temp_single_modality['label_folder']

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=2,
            random_seed=42
        )

        # Check structure
        assert len(split_lists) == 2

        # Check first subject
        first_subject = split_lists[0][0]
        assert 'image_dwi' in first_subject
        assert 'label' in first_subject

        # Schema should show single modality
        schema = build_schema(first_subject)
        assert schema['has_multi_modal_images'] is False  # Only one image
        assert len(schema['image_keys']) == 1

    def test_three_modalities(self, temp_three_modalities):
        """Test with three modalities (DWI + ADC + FLAIR)."""
        image_folders = {
            'dwi': temp_three_modalities['dwi_folder'],
            'adc': temp_three_modalities['adc_folder'],
            'flair': temp_three_modalities['flair_folder']
        }
        label_folder = temp_three_modalities['label_folder']

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=2,
            random_seed=42
        )

        # Check first subject has all three modalities
        first_subject = split_lists[0][0]
        assert 'image_dwi' in first_subject
        assert 'image_adc' in first_subject
        assert 'image_flair' in first_subject
        assert 'label' in first_subject

        # Schema should show multi-modal
        schema = build_schema(first_subject)
        assert schema['has_multi_modal_images'] is True
        assert len(schema['image_keys']) == 3

    def test_missing_modality_file(self, temp_data_structure):
        """Test error when subject is missing a modality file."""
        # Remove one ADC file
        adc_file = temp_data_structure['adc_folder'] / "sub-002_adc.nii.gz"
        adc_file.unlink()

        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        with pytest.raises(ValueError) as exc_info:
            folder_mode_to_split_lists(
                image_folders=image_folders,
                label_folder=label_folder,
                n_folds=2
            )

        error_msg = str(exc_info.value)
        assert 'sub-002' in error_msg
        assert 'adc' in error_msg.lower() or 'modality' in error_msg.lower()

    def test_missing_label_file(self, temp_data_structure):
        """Test error when subject is missing label file."""
        # Remove one label file
        label_file = temp_data_structure['label_folder'] / "sub-001_lesion.nii.gz"
        label_file.unlink()

        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        with pytest.raises(ValueError) as exc_info:
            folder_mode_to_split_lists(
                image_folders=image_folders,
                label_folder=label_folder,
                n_folds=2
            )

        error_msg = str(exc_info.value)
        assert 'sub-001' in error_msg
        assert 'label' in error_msg.lower()

    def test_custom_subject_pattern(self, tmp_path):
        """Test custom regex pattern for subject ID extraction."""
        # Create files with different naming: patient_001_dwi.nii.gz
        dwi_folder = tmp_path / "dwi"
        label_folder = tmp_path / "labels"
        dwi_folder.mkdir()
        label_folder.mkdir()

        subjects = ['patient_001', 'patient_002']
        for subject in subjects:
            (dwi_folder / f"{subject}_dwi.nii.gz").touch()
            (label_folder / f"{subject}_lesion.nii.gz").touch()

        image_folders = {'dwi': dwi_folder}

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=2,
            subject_pattern=r'(patient_\d+)',
            random_seed=42
        )

        # Should successfully match both subjects
        all_subjects = [subj for fold in split_lists for subj in fold]
        assert len(all_subjects) == 2

    def test_fold_distribution(self, tmp_path):
        """Test subjects are evenly distributed across folds."""
        # Create 23 subjects
        dwi_folder = tmp_path / "dwi"
        label_folder = tmp_path / "labels"
        dwi_folder.mkdir()
        label_folder.mkdir()

        n_subjects = 23
        for i in range(1, n_subjects + 1):
            subject_id = f"sub-{i:03d}"
            (dwi_folder / f"{subject_id}_dwi.nii.gz").touch()
            (label_folder / f"{subject_id}_lesion.nii.gz").touch()

        image_folders = {'dwi': dwi_folder}

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=5,
            random_seed=42
        )

        # Check fold sizes
        fold_sizes = [len(fold) for fold in split_lists]
        assert len(fold_sizes) == 5
        assert sum(fold_sizes) == n_subjects

        # Distribution should be roughly even (some folds might have +1)
        min_size = min(fold_sizes)
        max_size = max(fold_sizes)
        assert max_size - min_size <= 1  # At most 1 subject difference

    def test_reproducibility(self, temp_data_structure):
        """Test same random_seed produces identical fold assignments."""
        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        # Run twice with same seed
        split_lists_1 = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=3,
            random_seed=12345
        )

        split_lists_2 = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=3,
            random_seed=12345
        )

        # Compare fold by fold
        assert len(split_lists_1) == len(split_lists_2)
        for fold_idx in range(len(split_lists_1)):
            assert len(split_lists_1[fold_idx]) == len(split_lists_2[fold_idx])
            # Same file paths in same order
            for subj_idx in range(len(split_lists_1[fold_idx])):
                assert (split_lists_1[fold_idx][subj_idx] ==
                        split_lists_2[fold_idx][subj_idx])

    def test_empty_folder(self, tmp_path):
        """Test error when modality folder is empty."""
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"  # Will be empty
        label_folder = tmp_path / "labels"

        dwi_folder.mkdir()
        adc_folder.mkdir()
        label_folder.mkdir()

        # Only create DWI and label files (ADC folder empty)
        (dwi_folder / "sub-001_dwi.nii.gz").touch()
        (label_folder / "sub-001_lesion.nii.gz").touch()

        image_folders = {
            'dwi': dwi_folder,
            'adc': adc_folder  # Empty
        }

        with pytest.raises(ValueError) as exc_info:
            folder_mode_to_split_lists(
                image_folders=image_folders,
                label_folder=label_folder,
                n_folds=2
            )

        error_msg = str(exc_info.value)
        assert 'empty' in error_msg.lower() or 'no files' in error_msg.lower()

    def test_no_valid_subjects(self, tmp_path):
        """Test error when no subjects have complete data."""
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        label_folder = tmp_path / "labels"

        for folder in [dwi_folder, adc_folder, label_folder]:
            folder.mkdir()

        # Create mismatched subjects (no overlap)
        (dwi_folder / "sub-001_dwi.nii.gz").touch()
        (adc_folder / "sub-002_adc.nii.gz").touch()
        (label_folder / "sub-003_lesion.nii.gz").touch()

        image_folders = {
            'dwi': dwi_folder,
            'adc': adc_folder
        }

        with pytest.raises(ValueError) as exc_info:
            folder_mode_to_split_lists(
                image_folders=image_folders,
                label_folder=label_folder,
                n_folds=2
            )

        error_msg = str(exc_info.value)
        assert 'no valid subjects' in error_msg.lower() or 'no subjects found' in error_msg.lower()

    def test_file_paths_are_strings(self, temp_data_structure):
        """Test that returned file paths are strings (not Path objects)."""
        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=2,
            random_seed=42
        )

        # Check that all paths are strings
        for fold in split_lists:
            for subject in fold:
                for key, path in subject.items():
                    assert isinstance(path, str), f"Path should be string, got {type(path)}"

    def test_paths_are_absolute(self, temp_data_structure):
        """Test that returned file paths are absolute."""
        image_folders = {
            'dwi': temp_data_structure['dwi_folder'],
            'adc': temp_data_structure['adc_folder']
        }
        label_folder = temp_data_structure['label_folder']

        split_lists = folder_mode_to_split_lists(
            image_folders=image_folders,
            label_folder=label_folder,
            n_folds=2,
            random_seed=42
        )

        # Check that all paths are absolute
        for fold in split_lists:
            for subject in fold:
                for key, path in subject.items():
                    assert Path(path).is_absolute(), f"Path should be absolute: {path}"

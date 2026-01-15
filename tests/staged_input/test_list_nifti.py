"""
Tests for list_nifti_from_folders() function.

This module tests the Stage 0→1 transformation: listing NIfTI files from
folders and organizing them by modality and subject ID.
"""
import pytest
from pathlib import Path
from lesseg_unet.data_utils import list_nifti_from_folders


class TestListNifti:
    """Test NIfTI file listing functionality."""

    def test_list_from_single_folder(self, tmp_path):
        """Test listing files from a single modality folder.

        Scenario: One 'dwi' folder with 3 .nii.gz files
        Expected: {'dwi': {'sub-001': path, 'sub-002': path, 'sub-003': path}}
        """
        # Arrange - create test folder structure
        dwi_folder = tmp_path / "dwi"
        dwi_folder.mkdir()

        for i in range(1, 4):
            (dwi_folder / f"sub-{i:03d}_dwi.nii.gz").touch()

        folders = {'dwi': dwi_folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(
            folders=folders,
            pattern=pattern
        )

        # Assert
        assert 'dwi' in result
        assert len(result['dwi']) == 3
        assert 'sub-001' in result['dwi']
        assert 'sub-002' in result['dwi']
        assert 'sub-003' in result['dwi']

    def test_list_from_multiple_folders(self, tmp_path):
        """Test listing files from multiple modality folders.

        Scenario: 'dwi' and 'adc' folders each with 2 subjects
        Expected: {'dwi': {2 subjects}, 'adc': {2 subjects}}
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "sub-001_dwi.nii.gz").touch()
        (dwi_folder / "sub-002_dwi.nii.gz").touch()
        (adc_folder / "sub-001_adc.nii.gz").touch()
        (adc_folder / "sub-002_adc.nii.gz").touch()

        folders = {'dwi': dwi_folder, 'adc': adc_folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        assert len(result) == 2
        assert 'dwi' in result
        assert 'adc' in result
        assert len(result['dwi']) == 2
        assert len(result['adc']) == 2

    def test_list_empty_folder_returns_empty_dict(self, tmp_path):
        """Test that empty folder returns empty subject mapping.

        Scenario: Folder exists but contains no .nii.gz files
        Expected: {'dwi': {}}
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        dwi_folder.mkdir()

        folders = {'dwi': dwi_folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        assert 'dwi' in result
        assert len(result['dwi']) == 0

    def test_list_missing_folder_raises_error(self, tmp_path):
        """Test that missing folder raises ValueError.

        Scenario: Specified folder does not exist
        Expected: ValueError with clear message
        """
        # Arrange
        missing_folder = tmp_path / "does_not_exist"
        folders = {'dwi': missing_folder}
        pattern = r'(sub-\d+)'

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            list_nifti_from_folders(folders, pattern)

        assert 'does not exist' in str(exc_info.value).lower()
        assert 'dwi' in str(exc_info.value)

    def test_list_with_pattern_filtering(self, tmp_path):
        """Test that pattern correctly filters matching files.

        Scenario: Folder has .nii.gz files, but only some match pattern
        Expected: Only pattern-matching files returned
        """
        # Arrange
        folder = tmp_path / "data"
        folder.mkdir()

        (folder / "sub-001_dwi.nii.gz").touch()
        (folder / "sub-002_dwi.nii.gz").touch()
        (folder / "other_file.nii.gz").touch()  # Doesn't match pattern

        folders = {'data': folder}
        pattern = r'(sub-\d+)'  # Should only match sub-XXX

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        assert 'data' in result
        assert len(result['data']) == 2  # Only the 2 matching files
        assert 'sub-001' in result['data']
        assert 'sub-002' in result['data']

    def test_list_supports_nii_and_nii_gz(self, tmp_path):
        """Test that both .nii and .nii.gz files are found.

        Scenario: Folder contains mix of .nii and .nii.gz
        Expected: Both file types included
        """
        # Arrange
        folder = tmp_path / "data"
        folder.mkdir()

        (folder / "sub-001.nii").touch()
        (folder / "sub-002.nii.gz").touch()

        folders = {'data': folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        assert 'data' in result
        assert len(result['data']) == 2

    def test_list_returns_absolute_paths(self, tmp_path):
        """Test that returned paths are absolute.

        Scenario: Provide relative folder paths
        Expected: Paths in result are absolute
        """
        # Arrange
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "sub-001.nii.gz").touch()

        folders = {'data': folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        path = result['data']['sub-001']
        assert Path(path).is_absolute()

    def test_list_accepts_str_and_path_folders(self, tmp_path):
        """Test that folders can be str or Path objects.

        Scenario: Mix of str and Path in folders dict
        Expected: Both work correctly
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "sub-001_dwi.nii.gz").touch()
        (adc_folder / "sub-001_adc.nii.gz").touch()

        # Mix str and Path
        folders = {'dwi': str(dwi_folder), 'adc': adc_folder}
        pattern = r'(sub-\d+)'

        # Act
        result = list_nifti_from_folders(folders, pattern)

        # Assert
        assert 'dwi' in result
        assert 'adc' in result
        assert 'sub-001' in result['dwi']
        assert 'sub-001' in result['adc']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

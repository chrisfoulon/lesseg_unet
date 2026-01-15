"""
Tests for validate_subject_dicts() function.

This module tests validation of subject dictionaries to ensure
paths exist and files are loadable.
"""
import pytest
import nibabel as nib
import numpy as np
from pathlib import Path
from lesseg_unet.data_utils import validate_subject_dicts


class TestValidateBasic:
    """Test basic validation functionality."""

    def test_validate_all_paths_exist(self, tmp_path):
        """Test validation passes when all paths exist.

        Scenario: Subject dicts with valid file paths
        Expected: No exception raised
        """
        # Arrange - create test files
        file1 = tmp_path / "sub-001.nii.gz"
        file2 = tmp_path / "sub-002.nii.gz"
        file1.touch()
        file2.touch()

        subject_dicts = [
            {'image_dwi': str(file1)},
            {'image_dwi': str(file2)}
        ]

        # Act & Assert - should not raise
        validate_subject_dicts(
            subject_dicts=subject_dicts,
            check_loadable=False,
            min_size=0
        )

    def test_validate_missing_file_raises_error(self, tmp_path):
        """Test that missing file raises ValueError.

        Scenario: One subject dict has non-existent file path
        Expected: ValueError with clear message about which file is missing
        """
        # Arrange
        existing_file = tmp_path / "sub-001.nii.gz"
        existing_file.touch()
        missing_file = tmp_path / "does_not_exist.nii.gz"

        subject_dicts = [
            {'image_dwi': str(existing_file)},
            {'image_adc': str(missing_file)}  # This file doesn't exist!
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_subject_dicts(subject_dicts, check_loadable=False, min_size=0)

        error_msg = str(exc_info.value)
        assert 'does not exist' in error_msg.lower()
        assert 'does_not_exist.nii.gz' in error_msg

    def test_validate_file_too_small_raises_error(self, tmp_path):
        """Test that file smaller than min_size raises ValueError.

        Scenario: File exists but is smaller than min_size threshold
        Expected: ValueError indicating file might be corrupt
        """
        # Arrange - create tiny file (1 byte)
        tiny_file = tmp_path / "tiny.nii.gz"
        tiny_file.write_bytes(b'x')

        subject_dicts = [{'image_dwi': str(tiny_file)}]

        # Act & Assert - require at least 100 bytes
        with pytest.raises(ValueError) as exc_info:
            validate_subject_dicts(
                subject_dicts=subject_dicts,
                check_loadable=False,
                min_size=100
            )

        error_msg = str(exc_info.value)
        assert 'too small' in error_msg.lower()

    def test_validate_empty_list_succeeds(self):
        """Test that empty subject list is valid.

        Scenario: Empty subject_dicts list
        Expected: No exception (nothing to validate)
        """
        # Act & Assert - should not raise
        validate_subject_dicts(
            subject_dicts=[],
            check_loadable=False,
            min_size=0
        )

    def test_validate_multiple_keys_per_subject(self, tmp_path):
        """Test validation of multi-key subject dicts.

        Scenario: Each subject has image + label paths
        Expected: All paths validated
        """
        # Arrange
        dwi_file = tmp_path / "dwi.nii.gz"
        adc_file = tmp_path / "adc.nii.gz"
        label_file = tmp_path / "label.nii.gz"
        dwi_file.touch()
        adc_file.touch()
        label_file.touch()

        subject_dicts = [{
            'image_dwi': str(dwi_file),
            'image_adc': str(adc_file),
            'label_stroke': str(label_file)
        }]

        # Act & Assert
        validate_subject_dicts(subject_dicts, check_loadable=False, min_size=0)


class TestValidateLoadable:
    """Test loadability checking with nibabel."""

    def test_validate_loadable_valid_nifti(self, tmp_path):
        """Test that valid NIfTI files pass loadability check.

        Scenario: Create actual NIfTI file with nibabel
        Expected: Validation succeeds
        """
        # Arrange - create valid NIfTI file
        nifti_file = tmp_path / "test.nii.gz"
        data = np.random.rand(10, 10, 10)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, nifti_file)

        subject_dicts = [{'image_dwi': str(nifti_file)}]

        # Act & Assert
        validate_subject_dicts(
            subject_dicts=subject_dicts,
            check_loadable=True,
            min_size=0
        )

    def test_validate_loadable_invalid_file_raises_error(self, tmp_path):
        """Test that non-NIfTI file fails loadability check.

        Scenario: File exists but is not a valid NIfTI
        Expected: ValueError with nibabel error message
        """
        # Arrange - create non-NIfTI file
        invalid_file = tmp_path / "not_nifti.nii.gz"
        invalid_file.write_bytes(b'not a nifti file')

        subject_dicts = [{'image_dwi': str(invalid_file)}]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_subject_dicts(
                subject_dicts=subject_dicts,
                check_loadable=True,
                min_size=0
            )

        error_msg = str(exc_info.value)
        assert 'cannot be loaded' in error_msg.lower() or 'not loadable' in error_msg.lower()

    def test_validate_skip_loadable_check(self, tmp_path):
        """Test that check_loadable=False skips nibabel loading.

        Scenario: Invalid NIfTI file but check_loadable=False
        Expected: Validation succeeds (only checks existence)
        """
        # Arrange
        invalid_file = tmp_path / "not_nifti.nii.gz"
        invalid_file.write_bytes(b'not a nifti file')

        subject_dicts = [{'image_dwi': str(invalid_file)}]

        # Act & Assert - should NOT raise
        validate_subject_dicts(
            subject_dicts=subject_dicts,
            check_loadable=False,  # Skip loadability check
            min_size=0
        )


class TestValidateEdgeCases:
    """Test edge cases and special scenarios."""

    def test_validate_reports_first_error_only(self, tmp_path):
        """Test that validation reports first error encountered.

        Scenario: Multiple subjects with missing files
        Expected: ValueError mentions first missing file
        """
        # Arrange
        missing1 = tmp_path / "missing1.nii.gz"
        missing2 = tmp_path / "missing2.nii.gz"

        subject_dicts = [
            {'image_dwi': str(missing1)},
            {'image_dwi': str(missing2)}
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_subject_dicts(subject_dicts, check_loadable=False, min_size=0)

        error_msg = str(exc_info.value)
        # Should mention at least one of the missing files
        assert 'missing1.nii.gz' in error_msg or 'missing2.nii.gz' in error_msg

    def test_validate_accepts_path_objects(self, tmp_path):
        """Test that Path objects work as well as strings.

        Scenario: Subject dicts contain Path objects instead of strings
        Expected: Validation handles both types
        """
        # Arrange
        file_path = tmp_path / "test.nii.gz"
        file_path.touch()

        subject_dicts = [{'image_dwi': file_path}]  # Path object, not str

        # Act & Assert
        validate_subject_dicts(subject_dicts, check_loadable=False, min_size=0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

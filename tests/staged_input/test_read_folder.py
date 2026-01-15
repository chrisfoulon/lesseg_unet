"""
Tests for read_folder() function.

This module tests the folder loading functionality that lists NIfTI files
with optional glob-style filtering.
"""
import pytest
from pathlib import Path

# Import will fail until we implement the function
# from lesseg_unet.data_utils import read_folder


class TestReadFolderBasic:
    """Test basic folder reading functionality."""

    def test_list_all_nifti_files(self, tmp_path):
        """No pattern: returns all NIfTI files sorted.

        Scenario: Folder contains multiple NIfTI files
        Expected: All files returned, sorted alphabetically
        """
        # Arrange - create test files
        (tmp_path / "patient001.nii.gz").touch()
        (tmp_path / "patient002.nii.gz").touch()
        (tmp_path / "patient003.nii").touch()
        (tmp_path / "readme.txt").touch()  # Non-NIfTI, should be excluded

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path)

        # Assert
        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)
        # Should be sorted
        assert result[0].name == "patient001.nii.gz"
        assert result[1].name == "patient002.nii.gz"
        assert result[2].name == "patient003.nii"

    def test_filter_by_glob_pattern(self, tmp_path):
        """Pattern filters files correctly.

        Scenario: Folder has files with different prefixes
        Expected: Only matching files returned
        """
        # Arrange
        (tmp_path / "dwi_patient001.nii.gz").touch()
        (tmp_path / "dwi_patient002.nii.gz").touch()
        (tmp_path / "adc_patient001.nii.gz").touch()
        (tmp_path / "adc_patient002.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path, pattern="dwi*")

        # Assert
        assert len(result) == 2
        assert all("dwi" in p.name for p in result)

    def test_pattern_with_wildcard_in_middle(self, tmp_path):
        """Pattern with wildcard in middle works.

        Scenario: Pattern like '*_session1*' to match session
        Expected: Files matching pattern returned
        """
        # Arrange
        (tmp_path / "patient001_session1_dwi.nii.gz").touch()
        (tmp_path / "patient001_session2_dwi.nii.gz").touch()
        (tmp_path / "patient002_session1_dwi.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path, pattern="*_session1_*")

        # Assert
        assert len(result) == 2
        assert all("session1" in p.name for p in result)


class TestReadFolderErrors:
    """Test error handling for read_folder."""

    def test_empty_folder_raises_error(self, tmp_path):
        """Empty folder raises ValueError.

        Scenario: Folder exists but contains no NIfTI files
        Expected: ValueError with informative message
        """
        # Arrange - create folder with only non-NIfTI files
        (tmp_path / "readme.txt").touch()

        # Act & Assert
        from lesseg_unet.data_utils import read_folder
        with pytest.raises(ValueError) as exc_info:
            read_folder(tmp_path)

        assert "no nifti" in str(exc_info.value).lower()

    def test_nonexistent_folder_raises_error(self, tmp_path):
        """Nonexistent folder raises ValueError.

        Scenario: Folder path doesn't exist
        Expected: ValueError with path in message
        """
        # Arrange
        nonexistent = tmp_path / "does_not_exist"

        # Act & Assert
        from lesseg_unet.data_utils import read_folder
        with pytest.raises(ValueError) as exc_info:
            read_folder(nonexistent)

        assert "does not exist" in str(exc_info.value).lower()

    def test_pattern_matches_nothing_raises_error(self, tmp_path):
        """Pattern that matches nothing raises ValueError.

        Scenario: Files exist but pattern matches none
        Expected: ValueError mentioning the pattern
        """
        # Arrange
        (tmp_path / "patient001.nii.gz").touch()
        (tmp_path / "patient002.nii.gz").touch()

        # Act & Assert
        from lesseg_unet.data_utils import read_folder
        with pytest.raises(ValueError) as exc_info:
            read_folder(tmp_path, pattern="nonexistent*")

        error_msg = str(exc_info.value).lower()
        assert "no nifti" in error_msg or "pattern" in error_msg


class TestReadFolderRecursive:
    """Test recursive search functionality."""

    def test_recursive_finds_nested_files(self, tmp_path):
        """Recursive search finds files in subdirectories.

        Scenario: NIfTI files in nested subdirectories
        Expected: All files found with recursive=True
        """
        # Arrange
        (tmp_path / "sub-001").mkdir()
        (tmp_path / "sub-002").mkdir()
        (tmp_path / "sub-001" / "dwi.nii.gz").touch()
        (tmp_path / "sub-002" / "dwi.nii.gz").touch()
        (tmp_path / "top_level.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path, recursive=True)

        # Assert
        assert len(result) == 3

    def test_non_recursive_skips_subdirectories(self, tmp_path):
        """Non-recursive search only finds top-level files.

        Scenario: Files in both root and subdirectories
        Expected: Only root files with recursive=False
        """
        # Arrange
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.nii.gz").touch()
        (tmp_path / "top_level.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path, recursive=False)

        # Assert
        assert len(result) == 1
        assert result[0].name == "top_level.nii.gz"


class TestReadFolderEdgeCases:
    """Test edge cases and special scenarios."""

    def test_accepts_string_path(self, tmp_path):
        """Function accepts string path as well as Path object.

        Scenario: Pass string instead of Path
        Expected: Works correctly
        """
        # Arrange
        (tmp_path / "test.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(str(tmp_path))

        # Assert
        assert len(result) == 1

    def test_handles_both_nii_extensions(self, tmp_path):
        """Both .nii and .nii.gz are recognized.

        Scenario: Mix of compressed and uncompressed NIfTI files
        Expected: Both types included
        """
        # Arrange
        (tmp_path / "uncompressed.nii").touch()
        (tmp_path / "compressed.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path)

        # Assert
        assert len(result) == 2

    def test_returns_sorted_paths(self, tmp_path):
        """Results are sorted alphabetically.

        Scenario: Files created in random order
        Expected: Returned in sorted order
        """
        # Arrange - create in non-alphabetical order
        (tmp_path / "zebra.nii.gz").touch()
        (tmp_path / "alpha.nii.gz").touch()
        (tmp_path / "middle.nii.gz").touch()

        # Act
        from lesseg_unet.data_utils import read_folder
        result = read_folder(tmp_path)

        # Assert
        names = [p.name for p in result]
        assert names == sorted(names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

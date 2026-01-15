"""
Tests for read_list_file() function.

This module tests the list file loading functionality that reads file paths
from a text file and validates their existence.
"""
import pytest
from pathlib import Path


class TestReadListFileBasic:
    """Test basic list file reading functionality."""

    def test_read_valid_list_file(self, tmp_path):
        """Reads paths from text file correctly.

        Scenario: Text file with valid NIfTI paths
        Expected: Returns list of Path objects
        """
        # Arrange - create actual files and list file
        file1 = tmp_path / "patient001.nii.gz"
        file2 = tmp_path / "patient002.nii.gz"
        file1.touch()
        file2.touch()

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{file1}\n{file2}\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert result[0] == file1
        assert result[1] == file2

    def test_skips_empty_lines(self, tmp_path):
        """Empty lines in file are skipped.

        Scenario: Text file with empty lines between paths
        Expected: Only non-empty lines returned
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file2 = tmp_path / "file2.nii.gz"
        file1.touch()
        file2.touch()

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{file1}\n\n\n{file2}\n\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert
        assert len(result) == 2

    def test_skips_comment_lines(self, tmp_path):
        """Lines starting with # are treated as comments.

        Scenario: Text file with comment lines
        Expected: Comments ignored
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"# This is a comment\n{file1}\n# Another comment\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert
        assert len(result) == 1
        assert result[0] == file1

    def test_strips_whitespace(self, tmp_path):
        """Leading/trailing whitespace is stripped from paths.

        Scenario: Paths with extra whitespace
        Expected: Whitespace removed, paths valid
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"  {file1}  \n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert
        assert len(result) == 1
        assert result[0] == file1


class TestReadListFileValidation:
    """Test path existence validation."""

    def test_check_exists_validates_paths(self, tmp_path):
        """check_exists=True raises error for missing files.

        Scenario: List file contains path to nonexistent file
        Expected: ValueError with missing path in message
        """
        # Arrange
        existing = tmp_path / "exists.nii.gz"
        existing.touch()
        missing = tmp_path / "missing.nii.gz"

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{existing}\n{missing}\n")

        # Act & Assert
        from lesseg_unet.data_utils import read_list_file
        with pytest.raises(ValueError) as exc_info:
            read_list_file(list_file, check_exists=True)

        assert "missing.nii.gz" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value).lower()

    def test_skip_check_exists(self, tmp_path):
        """check_exists=False skips validation.

        Scenario: List file with nonexistent paths, validation disabled
        Expected: Paths returned without validation
        """
        # Arrange
        missing = tmp_path / "missing.nii.gz"

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{missing}\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file, check_exists=False)

        # Assert
        assert len(result) == 1
        assert result[0] == missing


class TestReadListFileErrors:
    """Test error handling."""

    def test_nonexistent_list_file_raises_error(self, tmp_path):
        """Nonexistent list file raises FileNotFoundError.

        Scenario: List file path doesn't exist
        Expected: FileNotFoundError
        """
        # Arrange
        nonexistent = tmp_path / "nonexistent.txt"

        # Act & Assert
        from lesseg_unet.data_utils import read_list_file
        with pytest.raises(FileNotFoundError):
            read_list_file(nonexistent)

    def test_empty_list_file_raises_error(self, tmp_path):
        """Empty list file raises ValueError.

        Scenario: List file exists but is empty
        Expected: ValueError indicating no paths found
        """
        # Arrange
        list_file = tmp_path / "empty.txt"
        list_file.write_text("")

        # Act & Assert
        from lesseg_unet.data_utils import read_list_file
        with pytest.raises(ValueError) as exc_info:
            read_list_file(list_file)

        assert "no paths" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_only_comments_raises_error(self, tmp_path):
        """File with only comments raises ValueError.

        Scenario: List file contains only comment lines
        Expected: ValueError indicating no paths found
        """
        # Arrange
        list_file = tmp_path / "comments_only.txt"
        list_file.write_text("# Comment 1\n# Comment 2\n")

        # Act & Assert
        from lesseg_unet.data_utils import read_list_file
        with pytest.raises(ValueError):
            read_list_file(list_file)


class TestReadListFileEdgeCases:
    """Test edge cases and special scenarios."""

    def test_accepts_string_path(self, tmp_path):
        """Function accepts string path as well as Path object.

        Scenario: Pass string instead of Path for list file
        Expected: Works correctly
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{file1}\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(str(list_file))

        # Assert
        assert len(result) == 1

    def test_preserves_order(self, tmp_path):
        """Paths are returned in file order (not sorted).

        Scenario: Paths listed in specific order
        Expected: Order preserved
        """
        # Arrange
        file_z = tmp_path / "zebra.nii.gz"
        file_a = tmp_path / "alpha.nii.gz"
        file_m = tmp_path / "middle.nii.gz"
        for f in [file_z, file_a, file_m]:
            f.touch()

        # Write in specific order: z, a, m
        list_file = tmp_path / "paths.txt"
        list_file.write_text(f"{file_z}\n{file_a}\n{file_m}\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert - order should match file, not alphabetical
        assert result[0].name == "zebra.nii.gz"
        assert result[1].name == "alpha.nii.gz"
        assert result[2].name == "middle.nii.gz"

    def test_handles_absolute_and_relative_paths(self, tmp_path):
        """Both absolute and relative paths are handled.

        Scenario: Mix of absolute paths in list file
        Expected: All paths converted to Path objects
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        list_file = tmp_path / "paths.txt"
        # Use absolute path
        list_file.write_text(f"{file1.absolute()}\n")

        # Act
        from lesseg_unet.data_utils import read_list_file
        result = read_list_file(list_file)

        # Assert
        assert len(result) == 1
        assert result[0].is_absolute()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for read_list_dicts() and read_presplit_json() functions.

These functions load pre-matched subject dictionaries from JSON files,
used when users provide already-matched data.
"""
import json
import pytest
from pathlib import Path


class TestReadListDictsBasic:
    """Test basic read_list_dicts functionality."""

    def test_read_valid_list_dicts(self, tmp_path):
        """Reads pre-matched subject dicts from JSON.

        Scenario: JSON file with list of subject dicts
        Expected: Returns list of dicts with Path values
        """
        # Arrange - create files and JSON
        file1 = tmp_path / "dwi_001.nii.gz"
        file2 = tmp_path / "adc_001.nii.gz"
        file1.touch()
        file2.touch()

        json_data = [
            {"image_dwi": str(file1), "image_adc": str(file2)}
        ]
        json_file = tmp_path / "subjects.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_list_dicts
        result = read_list_dicts(json_file)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "image_dwi" in result[0]
        assert isinstance(result[0]["image_dwi"], Path)

    def test_read_multiple_subjects(self, tmp_path):
        """Reads multiple subjects from JSON.

        Scenario: JSON with multiple subject entries
        Expected: All subjects returned
        """
        # Arrange
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.nii.gz"
            f.touch()
            files.append(f)

        json_data = [
            {"image": str(files[0])},
            {"image": str(files[1])},
            {"image": str(files[2])}
        ]
        json_file = tmp_path / "subjects.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_list_dicts
        result = read_list_dicts(json_file)

        # Assert
        assert len(result) == 3


class TestReadListDictsValidation:
    """Test path validation for read_list_dicts."""

    def test_check_exists_validates_paths(self, tmp_path):
        """check_exists=True validates all paths in dicts.

        Scenario: JSON contains path to nonexistent file
        Expected: ValueError with missing path
        """
        # Arrange
        existing = tmp_path / "exists.nii.gz"
        existing.touch()
        missing = tmp_path / "missing.nii.gz"

        json_data = [
            {"image": str(existing), "label": str(missing)}
        ]
        json_file = tmp_path / "subjects.json"
        json_file.write_text(json.dumps(json_data))

        # Act & Assert
        from lesseg_unet.data_utils import read_list_dicts
        with pytest.raises(ValueError) as exc_info:
            read_list_dicts(json_file, check_exists=True)

        assert "missing.nii.gz" in str(exc_info.value)

    def test_skip_check_exists(self, tmp_path):
        """check_exists=False skips validation.

        Scenario: JSON with nonexistent paths, validation disabled
        Expected: Data returned without validation
        """
        # Arrange
        json_data = [
            {"image": "/nonexistent/path.nii.gz"}
        ]
        json_file = tmp_path / "subjects.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_list_dicts
        result = read_list_dicts(json_file, check_exists=False)

        # Assert
        assert len(result) == 1


class TestReadListDictsErrors:
    """Test error handling for read_list_dicts."""

    def test_nonexistent_json_file(self, tmp_path):
        """Nonexistent JSON file raises FileNotFoundError."""
        from lesseg_unet.data_utils import read_list_dicts
        with pytest.raises(FileNotFoundError):
            read_list_dicts(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_error(self, tmp_path):
        """Invalid JSON raises ValueError.

        Scenario: File contains malformed JSON
        Expected: ValueError with parse error info
        """
        # Arrange
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {")

        # Act & Assert
        from lesseg_unet.data_utils import read_list_dicts
        with pytest.raises(ValueError) as exc_info:
            read_list_dicts(json_file)

        assert "json" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()

    def test_not_a_list_raises_error(self, tmp_path):
        """JSON that's not a list raises ValueError.

        Scenario: JSON is a dict instead of list
        Expected: ValueError about expected format
        """
        # Arrange
        json_file = tmp_path / "dict.json"
        json_file.write_text('{"key": "value"}')

        # Act & Assert
        from lesseg_unet.data_utils import read_list_dicts
        with pytest.raises(ValueError) as exc_info:
            read_list_dicts(json_file)

        assert "list" in str(exc_info.value).lower()

    def test_empty_list_raises_error(self, tmp_path):
        """Empty JSON list raises ValueError."""
        # Arrange
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")

        # Act & Assert
        from lesseg_unet.data_utils import read_list_dicts
        with pytest.raises(ValueError) as exc_info:
            read_list_dicts(json_file)

        assert "empty" in str(exc_info.value).lower()


# ============================================================================
# Tests for read_presplit_json
# ============================================================================

class TestReadPresplitJsonBasic:
    """Test basic read_presplit_json functionality."""

    def test_read_valid_presplit(self, tmp_path):
        """Reads pre-split fold structure from JSON.

        Scenario: JSON with nested list structure (folds)
        Expected: Returns list[list[dict]]
        """
        # Arrange - create files
        files = []
        for i in range(4):
            f = tmp_path / f"file_{i}.nii.gz"
            f.touch()
            files.append(f)

        # Create 2 folds with 2 subjects each
        json_data = [
            [  # Fold 0
                {"image": str(files[0])},
                {"image": str(files[1])}
            ],
            [  # Fold 1
                {"image": str(files[2])},
                {"image": str(files[3])}
            ]
        ]
        json_file = tmp_path / "split_lists.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_presplit_json
        result = read_presplit_json(json_file)

        # Assert
        assert len(result) == 2  # 2 folds
        assert len(result[0]) == 2  # 2 subjects in fold 0
        assert len(result[1]) == 2  # 2 subjects in fold 1
        assert isinstance(result[0][0], dict)
        assert isinstance(result[0][0]["image"], Path)

    def test_preserves_fold_structure(self, tmp_path):
        """Fold structure is preserved correctly.

        Scenario: Unequal fold sizes
        Expected: Structure maintained
        """
        # Arrange
        files = []
        for i in range(5):
            f = tmp_path / f"file_{i}.nii.gz"
            f.touch()
            files.append(f)

        # 3 folds: 2, 2, 1 subjects
        json_data = [
            [{"image": str(files[0])}, {"image": str(files[1])}],
            [{"image": str(files[2])}, {"image": str(files[3])}],
            [{"image": str(files[4])}]
        ]
        json_file = tmp_path / "split_lists.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_presplit_json
        result = read_presplit_json(json_file)

        # Assert
        assert len(result) == 3
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert len(result[2]) == 1


class TestReadPresplitJsonValidation:
    """Test validation for read_presplit_json."""

    def test_check_exists_validates_all_folds(self, tmp_path):
        """check_exists validates paths in all folds.

        Scenario: Missing file in second fold
        Expected: ValueError with missing path
        """
        # Arrange
        existing = tmp_path / "exists.nii.gz"
        existing.touch()

        json_data = [
            [{"image": str(existing)}],
            [{"image": "/missing/file.nii.gz"}]
        ]
        json_file = tmp_path / "split_lists.json"
        json_file.write_text(json.dumps(json_data))

        # Act & Assert
        from lesseg_unet.data_utils import read_presplit_json
        with pytest.raises(ValueError) as exc_info:
            read_presplit_json(json_file, check_exists=True)

        assert "missing" in str(exc_info.value).lower() or "not exist" in str(exc_info.value).lower()


class TestReadPresplitJsonErrors:
    """Test error handling for read_presplit_json."""

    def test_not_nested_list_raises_error(self, tmp_path):
        """JSON without nested structure raises ValueError.

        Scenario: Flat list instead of list[list[dict]]
        Expected: ValueError about expected format
        """
        # Arrange - flat list (should be nested)
        json_data = [{"image": "/path/to/file.nii.gz"}]
        json_file = tmp_path / "flat.json"
        json_file.write_text(json.dumps(json_data))

        # Act & Assert
        from lesseg_unet.data_utils import read_presplit_json
        with pytest.raises(ValueError) as exc_info:
            read_presplit_json(json_file, check_exists=False)

        error_msg = str(exc_info.value).lower()
        assert "fold" in error_msg or "nested" in error_msg or "list" in error_msg

    def test_empty_fold_raises_error(self, tmp_path):
        """Fold with empty list raises ValueError.

        Scenario: One fold has no subjects
        Expected: ValueError about empty fold
        """
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        json_data = [
            [{"image": str(file1)}],
            []  # Empty fold
        ]
        json_file = tmp_path / "empty_fold.json"
        json_file.write_text(json.dumps(json_data))

        # Act & Assert
        from lesseg_unet.data_utils import read_presplit_json
        with pytest.raises(ValueError) as exc_info:
            read_presplit_json(json_file, check_exists=False)

        assert "empty" in str(exc_info.value).lower()


class TestJsonLoadersEdgeCases:
    """Test edge cases for JSON loaders."""

    def test_accepts_string_path(self, tmp_path):
        """Both functions accept string paths."""
        # Arrange
        file1 = tmp_path / "file1.nii.gz"
        file1.touch()

        # List dicts format
        json_data = [{"image": str(file1)}]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(json_data))

        # Act & Assert
        from lesseg_unet.data_utils import read_list_dicts
        result = read_list_dicts(str(json_file))
        assert len(result) == 1

    def test_multikey_subject_dicts(self, tmp_path):
        """Subject dicts with multiple keys are handled.

        Scenario: Subject has image + label + control
        Expected: All keys preserved
        """
        # Arrange
        files = {}
        for key in ['image_dwi', 'image_adc', 'label_lesion']:
            f = tmp_path / f"{key}.nii.gz"
            f.touch()
            files[key] = f

        json_data = [{k: str(v) for k, v in files.items()}]
        json_file = tmp_path / "multikey.json"
        json_file.write_text(json.dumps(json_data))

        # Act
        from lesseg_unet.data_utils import read_list_dicts
        result = read_list_dicts(json_file)

        # Assert
        assert len(result[0]) == 3
        assert "image_dwi" in result[0]
        assert "image_adc" in result[0]
        assert "label_lesion" in result[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

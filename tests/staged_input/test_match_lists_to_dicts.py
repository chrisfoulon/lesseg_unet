"""
Tests for match_lists_to_dicts() function.

This module tests the core matching function that matches files across
modalities using either exact filename matching, strip patterns (residual
matching), or extract patterns (key extraction).
"""
import pytest
from pathlib import Path


class TestMatchListsToDictsDefault:
    """Tests for default exact filename matching."""

    def test_default_exact_match_identical_filenames(self, tmp_path):
        """No pattern: filenames must be identical across folders.

        Scenario: Two modalities with identical filenames in separate folders
        Expected: Files matched by filename
        """
        # Arrange - create files with IDENTICAL names in different folders
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "patient001.nii.gz").touch()
        (dwi_folder / "patient002.nii.gz").touch()
        (adc_folder / "patient001.nii.gz").touch()
        (adc_folder / "patient002.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "patient001.nii.gz", dwi_folder / "patient002.nii.gz"],
            'adc': [adc_folder / "patient001.nii.gz", adc_folder / "patient002.nii.gz"]
        }

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, control_dicts = match_lists_to_dicts(image_lists)

        # Assert
        assert len(subject_dicts) == 2
        assert len(control_dicts) == 0
        # Check structure
        assert 'image_dwi' in subject_dicts[0]
        assert 'image_adc' in subject_dicts[0]

    def test_default_exact_match_different_filenames_no_match(self, tmp_path):
        """No pattern + different filenames -> no matches -> error.

        Scenario: Files have modality-specific prefixes, no pattern provided
        Expected: ValueError (no matches found)
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        # Different filenames - won't match without strip pattern
        (dwi_folder / "dwi_patient001.nii.gz").touch()
        (adc_folder / "adc_patient001.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "dwi_patient001.nii.gz"],
            'adc': [adc_folder / "adc_patient001.nii.gz"]
        }

        # Act & Assert
        from lesseg_unet.data_utils import match_lists_to_dicts
        with pytest.raises(ValueError) as exc_info:
            match_lists_to_dicts(image_lists)

        # Should mention first unmatched file
        assert "dwi_patient001" in str(exc_info.value) or "no match" in str(exc_info.value).lower()


class TestMatchListsToDictsStripMechanism:
    """Tests for STRIP & match residual mechanism."""

    def test_strip_pattern_literal_string(self, tmp_path):
        """strip_pattern removes literal string, match residuals.

        Scenario: Files have modality prefixes
        Expected: After stripping modality, files match by residual
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "dwi_patient001.nii.gz").touch()
        (dwi_folder / "dwi_patient002.nii.gz").touch()
        (adc_folder / "adc_patient001.nii.gz").touch()
        (adc_folder / "adc_patient002.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "dwi_patient001.nii.gz", dwi_folder / "dwi_patient002.nii.gz"],
            'adc': [adc_folder / "adc_patient001.nii.gz", adc_folder / "adc_patient002.nii.gz"]
        }

        # Act - use modality names as strip patterns (one per modality)
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            strip_pattern={'dwi': 'dwi', 'adc': 'adc'}
        )

        # Assert
        assert len(subject_dicts) == 2

    def test_strip_single_pattern_for_all(self, tmp_path):
        """Single strip_pattern applied to all modalities.

        Scenario: Common prefix/suffix across all modalities
        Expected: Same pattern stripped from all
        """
        # Arrange
        folder1 = tmp_path / "mod1"
        folder2 = tmp_path / "mod2"
        folder1.mkdir()
        folder2.mkdir()

        # Both have "ses-01_" prefix
        (folder1 / "ses-01_patient001.nii.gz").touch()
        (folder2 / "ses-01_patient001.nii.gz").touch()

        image_lists = {
            'mod1': [folder1 / "ses-01_patient001.nii.gz"],
            'mod2': [folder2 / "ses-01_patient001.nii.gz"]
        }

        # Act - single pattern for all
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            strip_pattern="ses-01_"
        )

        # Assert
        assert len(subject_dicts) == 1

    def test_strip_pattern_regex(self, tmp_path):
        """strip_pattern with regex removes dynamic parts.

        Scenario: Files have run numbers like _run-01, _run-02
        Expected: Regex strips run numbers, files match
        """
        # Arrange
        folder1 = tmp_path / "mod1"
        folder2 = tmp_path / "mod2"
        folder1.mkdir()
        folder2.mkdir()

        (folder1 / "patient001_run-01.nii.gz").touch()
        (folder2 / "patient001_run-02.nii.gz").touch()

        image_lists = {
            'mod1': [folder1 / "patient001_run-01.nii.gz"],
            'mod2': [folder2 / "patient001_run-02.nii.gz"]
        }

        # Act - regex pattern
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            strip_pattern=r"_run-\d+"
        )

        # Assert
        assert len(subject_dicts) == 1

    def test_strip_modality_in_middle_of_filename(self, tmp_path):
        """Strip pattern works when modality is in middle of filename.

        Scenario: Format like "patient001_dwi_session1.nii.gz"
        Expected: Strips modality correctly
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "patient001_dwi_session1.nii.gz").touch()
        (adc_folder / "patient001_adc_session1.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "patient001_dwi_session1.nii.gz"],
            'adc': [adc_folder / "patient001_adc_session1.nii.gz"]
        }

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            strip_pattern={'dwi': '_dwi', 'adc': '_adc'}
        )

        # Assert - residuals should be "patient001_session1.nii.gz"
        assert len(subject_dicts) == 1


class TestMatchListsToDictsExtractMechanism:
    """Tests for EXTRACT & match key mechanism."""

    def test_extract_subject_id_pattern(self, tmp_path):
        """extract_pattern extracts subject ID as key.

        Scenario: Files have subject IDs like subj123
        Expected: Extract IDs, match by extracted key
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "scan_subj123_dwi_b1000.nii.gz").touch()
        (adc_folder / "processed_subj123_adc.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "scan_subj123_dwi_b1000.nii.gz"],
            'adc': [adc_folder / "processed_subj123_adc.nii.gz"]
        }

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            extract_pattern=r"subj\d+"
        )

        # Assert
        assert len(subject_dicts) == 1

    def test_extract_bids_style_pattern(self, tmp_path):
        """extract_pattern works with BIDS-style naming.

        Scenario: BIDS format sub-001, sub-002, etc.
        Expected: Extracts and matches by subject ID
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "sub-001_ses-01_dwi.nii.gz").touch()
        (dwi_folder / "sub-002_ses-01_dwi.nii.gz").touch()
        (adc_folder / "sub-001_ses-02_adc.nii.gz").touch()
        (adc_folder / "sub-002_ses-02_adc.nii.gz").touch()

        image_lists = {
            'dwi': [dwi_folder / "sub-001_ses-01_dwi.nii.gz",
                    dwi_folder / "sub-002_ses-01_dwi.nii.gz"],
            'adc': [adc_folder / "sub-001_ses-02_adc.nii.gz",
                    adc_folder / "sub-002_ses-02_adc.nii.gz"]
        }

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(
            image_lists,
            extract_pattern=r"sub-\d+"
        )

        # Assert
        assert len(subject_dicts) == 2

    def test_extract_pattern_not_found_raises_error(self, tmp_path):
        """File without extractable pattern raises ValueError.

        Scenario: Pattern doesn't match any file
        Expected: ValueError with helpful message
        """
        # Arrange
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "patient001.nii.gz").touch()

        image_lists = {
            'mod': [folder / "patient001.nii.gz"]
        }

        # Act & Assert
        from lesseg_unet.data_utils import match_lists_to_dicts
        with pytest.raises(ValueError) as exc_info:
            match_lists_to_dicts(image_lists, extract_pattern=r"subj\d+")

        assert "pattern" in str(exc_info.value).lower()


class TestMatchListsToDictsValidation:
    """Tests for validation and error handling."""

    def test_both_strip_and_extract_raises_error(self, tmp_path):
        """Cannot use both strip_pattern and extract_pattern.

        Scenario: Both patterns provided
        Expected: ValueError
        """
        # Arrange
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "test.nii.gz").touch()

        image_lists = {'mod': [folder / "test.nii.gz"]}

        # Act & Assert
        from lesseg_unet.data_utils import match_lists_to_dicts
        with pytest.raises(ValueError) as exc_info:
            match_lists_to_dicts(
                image_lists,
                strip_pattern="test",
                extract_pattern=r"\d+"
            )

        assert "both" in str(exc_info.value).lower() or "cannot" in str(exc_info.value).lower()

    def test_ambiguous_match_raises_error(self, tmp_path):
        """Multiple files producing same key raises error.

        Scenario: Two files match to same residual/key
        Expected: ValueError with details
        """
        # Arrange
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "patient001_v1.nii.gz").touch()
        (folder / "patient001_v2.nii.gz").touch()

        image_lists = {
            'mod': [folder / "patient001_v1.nii.gz", folder / "patient001_v2.nii.gz"]
        }

        # Act & Assert - both strip to "patient001.nii.gz"
        from lesseg_unet.data_utils import match_lists_to_dicts
        with pytest.raises(ValueError) as exc_info:
            match_lists_to_dicts(image_lists, strip_pattern=r"_v\d")

        assert "duplicate" in str(exc_info.value).lower() or "ambiguous" in str(exc_info.value).lower()

    def test_incomplete_match_warns_or_skips(self, tmp_path):
        """Missing modality for some subjects handled correctly.

        Scenario: Subject in one modality missing from another
        Expected: Only complete matches returned (no error by default)
        """
        # Arrange
        dwi_folder = tmp_path / "dwi"
        adc_folder = tmp_path / "adc"
        dwi_folder.mkdir()
        adc_folder.mkdir()

        (dwi_folder / "patient001.nii.gz").touch()
        (dwi_folder / "patient002.nii.gz").touch()
        (adc_folder / "patient001.nii.gz").touch()
        # patient002 missing from adc!

        image_lists = {
            'dwi': [dwi_folder / "patient001.nii.gz", dwi_folder / "patient002.nii.gz"],
            'adc': [adc_folder / "patient001.nii.gz"]
        }

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(image_lists)

        # Assert - only patient001 matched (patient002 incomplete)
        assert len(subject_dicts) == 1

    def test_no_matches_error_shows_first_file(self, tmp_path):
        """Error message includes first unmatched file and pattern.

        Scenario: No files match
        Expected: Helpful error message
        """
        # Arrange
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "file1.nii.gz").touch()
        (folder / "file2.nii.gz").touch()

        image_lists = {
            'mod1': [folder / "file1.nii.gz"],
            'mod2': [folder / "file2.nii.gz"]
        }

        # Act & Assert
        from lesseg_unet.data_utils import match_lists_to_dicts
        with pytest.raises(ValueError) as exc_info:
            match_lists_to_dicts(image_lists)

        error_msg = str(exc_info.value)
        # Should include file info
        assert "file1" in error_msg or "file2" in error_msg


class TestMatchListsToDictsWithLabelsAndControls:
    """Tests for multi-type matching."""

    def test_match_images_and_labels(self, tmp_path):
        """Images + labels matched correctly.

        Scenario: Image and label folders with matching files
        Expected: Subject dicts contain both
        """
        # Arrange
        img_folder = tmp_path / "images"
        lbl_folder = tmp_path / "labels"
        img_folder.mkdir()
        lbl_folder.mkdir()

        (img_folder / "patient001.nii.gz").touch()
        (lbl_folder / "patient001.nii.gz").touch()

        image_lists = {'dwi': [img_folder / "patient001.nii.gz"]}
        label_lists = {'lesion': [lbl_folder / "patient001.nii.gz"]}

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, _ = match_lists_to_dicts(image_lists, label_lists=label_lists)

        # Assert
        assert len(subject_dicts) == 1
        assert 'image_dwi' in subject_dicts[0]
        assert 'label_lesion' in subject_dicts[0]

    def test_match_with_controls(self, tmp_path):
        """Controls matched separately from subjects.

        Scenario: Subject images + control images
        Expected: Separate subject_dicts and control_dicts
        """
        # Arrange
        subj_folder = tmp_path / "subjects"
        ctrl_folder = tmp_path / "controls"
        subj_folder.mkdir()
        ctrl_folder.mkdir()

        (subj_folder / "patient001.nii.gz").touch()
        (ctrl_folder / "control001.nii.gz").touch()

        image_lists = {'dwi': [subj_folder / "patient001.nii.gz"]}
        control_lists = {'dwi': [ctrl_folder / "control001.nii.gz"]}

        # Act
        from lesseg_unet.data_utils import match_lists_to_dicts
        subject_dicts, control_dicts = match_lists_to_dicts(
            image_lists,
            control_lists=control_lists
        )

        # Assert
        assert len(subject_dicts) == 1
        assert len(control_dicts) == 1
        assert 'image_dwi' in subject_dicts[0]
        assert 'control_dwi' in control_dicts[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

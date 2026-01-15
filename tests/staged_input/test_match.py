"""
Tests for match_modalities_by_subject() function.

This module tests the Stage 1→2 transformation: matching modalities and labels
by subject ID to create subject dictionaries.
"""
import pytest
from pathlib import Path
from lesseg_unet.data_utils import match_modalities_by_subject


class TestMatchBasic:
    """Test basic matching functionality."""

    def test_match_two_modalities_with_label(self):
        """Test matching DWI + ADC + label for complete subjects.

        Scenario: 3 subjects with complete data (dwi, adc, label)
        Expected: 3 subject dicts with image_dwi, image_adc, label_stroke keys
        """
        # Arrange
        image_modalities = {
            'dwi': {
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz',
                'sub-003': '/data/dwi/sub-003.nii.gz'
            },
            'adc': {
                'sub-001': '/data/adc/sub-001.nii.gz',
                'sub-002': '/data/adc/sub-002.nii.gz',
                'sub-003': '/data/adc/sub-003.nii.gz'
            }
        }
        label_classes = {
            'stroke': {
                'sub-001': '/data/labels/sub-001.nii.gz',
                'sub-002': '/data/labels/sub-002.nii.gz',
                'sub-003': '/data/labels/sub-003.nii.gz'
            }
        }

        # Act
        subject_dicts, control_dicts = match_modalities_by_subject(
            image_modalities=image_modalities,
            label_classes=label_classes,
            control_modalities=None,
            require_all=True
        )

        # Assert
        assert len(subject_dicts) == 3
        assert len(control_dicts) == 0

        # Check first subject dict structure
        assert 'image_adc' in subject_dicts[0]
        assert 'image_dwi' in subject_dicts[0]
        assert 'label_stroke' in subject_dicts[0]

        # Check sorting (adc before dwi alphabetically)
        keys = list(subject_dicts[0].keys())
        assert keys == ['image_adc', 'image_dwi', 'label_stroke']

        # Check paths
        assert subject_dicts[0]['image_dwi'] == '/data/dwi/sub-001.nii.gz'
        assert subject_dicts[0]['image_adc'] == '/data/adc/sub-001.nii.gz'
        assert subject_dicts[0]['label_stroke'] == '/data/labels/sub-001.nii.gz'

    def test_match_incomplete_subject_raises_error(self):
        """Test that incomplete subject (missing modality) raises ValueError.

        Scenario: sub-002 missing ADC modality
        Expected: ValueError with clear message about missing data
        """
        # Arrange
        image_modalities = {
            'dwi': {
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz'
            },
            'adc': {
                'sub-001': '/data/adc/sub-001.nii.gz'
                # sub-002 missing ADC!
            }
        }
        label_classes = {
            'stroke': {
                'sub-001': '/data/labels/sub-001.nii.gz',
                'sub-002': '/data/labels/sub-002.nii.gz'
            }
        }

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            match_modalities_by_subject(
                image_modalities=image_modalities,
                label_classes=label_classes,
                control_modalities=None,
                require_all=True
            )

        # Check error message mentions the problem
        assert 'sub-002' in str(exc_info.value)
        assert 'incomplete' in str(exc_info.value).lower()

    def test_match_no_labels_for_segmentation(self):
        """Test matching without labels (segmentation mode).

        Scenario: 2 subjects with images only, no labels
        Expected: 2 subject dicts with only image keys, no label keys
        """
        # Arrange
        image_modalities = {
            'dwi': {
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz'
            },
            'adc': {
                'sub-001': '/data/adc/sub-001.nii.gz',
                'sub-002': '/data/adc/sub-002.nii.gz'
            }
        }

        # Act
        subject_dicts, control_dicts = match_modalities_by_subject(
            image_modalities=image_modalities,
            label_classes=None,  # No labels for segmentation
            control_modalities=None,
            require_all=True
        )

        # Assert
        assert len(subject_dicts) == 2
        assert len(control_dicts) == 0

        # Check no label keys present
        for subj_dict in subject_dicts:
            assert all(not k.startswith('label_') for k in subj_dict.keys())
            assert 'image_adc' in subj_dict
            assert 'image_dwi' in subj_dict

    def test_match_subject_ids_sorted_reproducibly(self):
        """Test that subject IDs are sorted for reproducible output.

        Scenario: Subjects provided in random order
        Expected: Output sorted by subject ID (sub-001, sub-002, sub-003)
        """
        # Arrange - deliberately out of order
        image_modalities = {
            'dwi': {
                'sub-003': '/data/dwi/sub-003.nii.gz',
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz'
            }
        }

        # Act
        subject_dicts, _ = match_modalities_by_subject(
            image_modalities=image_modalities,
            label_classes=None,
            control_modalities=None,
            require_all=True
        )

        # Assert - should be sorted
        paths = [d['image_dwi'] for d in subject_dicts]
        assert paths == [
            '/data/dwi/sub-001.nii.gz',
            '/data/dwi/sub-002.nii.gz',
            '/data/dwi/sub-003.nii.gz'
        ]


    def test_match_no_valid_subjects_raises_error(self):
        """Test that having zero valid subjects raises ValueError.

        Scenario: All subjects missing required modalities
        Expected: ValueError with clear message about no valid subjects
        """
        # Arrange - No overlap in subject IDs between modalities
        image_modalities = {
            'dwi': {
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz'
            },
            'adc': {
                'sub-003': '/data/adc/sub-003.nii.gz',  # Different subjects!
                'sub-004': '/data/adc/sub-004.nii.gz'
            }
        }
        label_classes = {
            'stroke': {
                'sub-005': '/data/labels/sub-005.nii.gz'  # Yet another subject!
            }
        }

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            match_modalities_by_subject(
                image_modalities=image_modalities,
                label_classes=label_classes,
                control_modalities=None,
                require_all=True
            )

        # Check error message content
        error_msg = str(exc_info.value)
        assert 'No valid subjects found' in error_msg
        assert 'Image modalities required' in error_msg


class TestMatchWithControls:
    """Test matching functionality with control subjects."""

    def test_match_with_controls_separate_output(self):
        """Test that controls are returned separately from subjects.

        Scenario: 2 subjects + 2 controls
        Expected: subject_dicts has 2 entries, control_dicts has 2 entries
        """
        # Arrange
        image_modalities = {
            'dwi': {
                'sub-001': '/data/dwi/sub-001.nii.gz',
                'sub-002': '/data/dwi/sub-002.nii.gz'
            }
        }
        label_classes = {
            'stroke': {
                'sub-001': '/data/labels/sub-001.nii.gz',
                'sub-002': '/data/labels/sub-002.nii.gz'
            }
        }
        control_modalities = {
            'dwi': {
                'ctr-001': '/data/controls/ctr-001.nii.gz',
                'ctr-002': '/data/controls/ctr-002.nii.gz'
            }
        }

        # Act
        subject_dicts, control_dicts = match_modalities_by_subject(
            image_modalities=image_modalities,
            label_classes=label_classes,
            control_modalities=control_modalities,
            require_all=True
        )

        # Assert
        assert len(subject_dicts) == 2
        assert len(control_dicts) == 2

        # Check subject dicts have both images and labels
        assert 'image_dwi' in subject_dicts[0]
        assert 'label_stroke' in subject_dicts[0]

        # Check control dicts have control_ prefix and no labels
        assert 'control_dwi' in control_dicts[0]
        assert all(not k.startswith('label_') for k in control_dicts[0].keys())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for shuffle_and_split_subjects() function.

This module tests the Stage 2→3 transformation: shuffling and splitting
subject dictionaries into cross-validation folds.
"""
import pytest
from lesseg_unet.data_utils import shuffle_and_split_subjects


class TestShuffleSplit:
    """Test shuffle and split functionality."""

    def test_split_reproducible_with_same_seed(self):
        """Test that same random seed produces same splits.

        Scenario: Split 10 subjects into 3 folds twice with seed=42
        Expected: Both runs produce identical fold assignments
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 11)
        ]

        # Act - split twice with same seed
        splits1 = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=3,
            shuffle=True,
            random_seed=42
        )
        splits2 = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=3,
            shuffle=True,
            random_seed=42
        )

        # Assert - both splits identical
        assert len(splits1) == 3
        assert len(splits2) == 3

        for fold1, fold2 in zip(splits1, splits2):
            assert fold1 == fold2

    def test_split_different_with_different_seed(self):
        """Test that different seeds produce different splits.

        Scenario: Split 10 subjects with seed=42 vs seed=123
        Expected: Different fold assignments
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 11)
        ]

        # Act
        splits1 = shuffle_and_split_subjects(
            subject_dicts, n_folds=3, shuffle=True, random_seed=42
        )
        splits2 = shuffle_and_split_subjects(
            subject_dicts, n_folds=3, shuffle=True, random_seed=123
        )

        # Assert - at least one fold differs
        assert splits1 != splits2

    def test_split_no_shuffle_preserves_order(self):
        """Test that shuffle=False maintains original order.

        Scenario: Split 9 subjects into 3 folds without shuffling
        Expected: Folds = [sub-001,002,003], [004,005,006], [007,008,009]
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 10)
        ]

        # Act
        splits = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=3,
            shuffle=False,
            random_seed=42
        )

        # Assert
        assert len(splits) == 3
        assert len(splits[0]) == 3
        assert len(splits[1]) == 3
        assert len(splits[2]) == 3

        # Check order preserved
        assert splits[0][0]['image_dwi'] == '/data/sub-001.nii.gz'
        assert splits[1][0]['image_dwi'] == '/data/sub-004.nii.gz'
        assert splits[2][0]['image_dwi'] == '/data/sub-007.nii.gz'

    def test_split_even_distribution(self):
        """Test that folds have balanced sizes.

        Scenario: 10 subjects into 3 folds
        Expected: Folds of size [4, 3, 3] or similar even distribution
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 11)
        ]

        # Act
        splits = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=3,
            shuffle=False,
            random_seed=42
        )

        # Assert - sizes should differ by at most 1
        sizes = [len(fold) for fold in splits]
        assert sum(sizes) == 10
        assert max(sizes) - min(sizes) <= 1

    def test_split_single_fold_returns_all_subjects(self):
        """Test that n_folds=1 returns all subjects in single fold.

        Scenario: Split 5 subjects into 1 fold
        Expected: [[all 5 subjects]]
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 6)
        ]

        # Act
        splits = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=1,
            shuffle=False,
            random_seed=42
        )

        # Assert
        assert len(splits) == 1
        assert len(splits[0]) == 5
        assert splits[0] == subject_dicts

    def test_split_more_folds_than_subjects(self):
        """Test behavior when n_folds > number of subjects.

        Scenario: 3 subjects, 5 folds requested
        Expected: 3 folds with 1 subject each, 2 empty folds
        """
        # Arrange
        subject_dicts = [
            {'image_dwi': f'/data/sub-{i:03d}.nii.gz'} for i in range(1, 4)
        ]

        # Act
        splits = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=5,
            shuffle=False,
            random_seed=42
        )

        # Assert - numpy.array_split creates 5 folds (some empty)
        assert len(splits) == 5
        non_empty_folds = [fold for fold in splits if len(fold) > 0]
        assert len(non_empty_folds) == 3

    def test_split_empty_list_raises_error(self):
        """Test that empty subject list raises ValueError.

        Scenario: Empty subject_dicts list
        Expected: ValueError with clear message
        """
        # Arrange
        subject_dicts = []

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            shuffle_and_split_subjects(
                subject_dicts=subject_dicts,
                n_folds=3,
                shuffle=True,
                random_seed=42
            )

        assert 'empty' in str(exc_info.value).lower()

    def test_split_maintains_dict_structure(self):
        """Test that subject dict structure is preserved.

        Scenario: Multi-key subject dicts (image + label)
        Expected: All keys present in output folds
        """
        # Arrange
        subject_dicts = [
            {
                'image_dwi': f'/data/dwi/sub-{i:03d}.nii.gz',
                'image_adc': f'/data/adc/sub-{i:03d}.nii.gz',
                'label_stroke': f'/data/labels/sub-{i:03d}.nii.gz'
            }
            for i in range(1, 6)
        ]

        # Act
        splits = shuffle_and_split_subjects(
            subject_dicts=subject_dicts,
            n_folds=2,
            shuffle=False,
            random_seed=42
        )

        # Assert - all subjects have all keys
        for fold in splits:
            for subject_dict in fold:
                assert 'image_dwi' in subject_dict
                assert 'image_adc' in subject_dict
                assert 'label_stroke' in subject_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Unit tests for lesseg_unet.data_utils module.

Tests for multi-modal data structure handling, key parsing, schema building,
and validation functions.
"""

import pytest
from lesseg_unet.data_utils import (
    parse_key,
    get_category_keys,
    build_schema,
    validate_against_schema,
    extract_model_config,
    SubjectDict,
)


class TestParseKey:
    """Test parse_key() function."""

    def test_single_word_key(self):
        """Test key without underscore (backward compatible)."""
        category, identifier = parse_key('image')
        assert category == 'image'
        assert identifier is None

    def test_key_with_identifier(self):
        """Test key with single underscore."""
        category, identifier = parse_key('image_dwi')
        assert category == 'image'
        assert identifier == 'dwi'

    def test_key_with_multiple_underscores(self):
        """Test key with multiple underscores (split on first only)."""
        category, identifier = parse_key('label_class0_subtype')
        assert category == 'label'
        assert identifier == 'class0_subtype'

    def test_control_key(self):
        """Test control category."""
        category, identifier = parse_key('control_dwi')
        assert category == 'control'
        assert identifier == 'dwi'

    def test_label_key(self):
        """Test label category."""
        category, identifier = parse_key('label')
        assert category == 'label'
        assert identifier is None

    def test_label_with_class(self):
        """Test multi-class label."""
        category, identifier = parse_key('label_class0')
        assert category == 'label'
        assert identifier == 'class0'


class TestGetCategoryKeys:
    """Test get_category_keys() function."""

    def test_single_modality_backward_compatible(self):
        """Test backward compatibility with single modality."""
        subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
        image_keys = get_category_keys(subject, 'image')
        assert image_keys == ['image']

    def test_multi_modal_images(self):
        """Test extracting multiple image modalities."""
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'image_adc': '/path/adc.nii.gz',
            'image_flair': '/path/flair.nii.gz',
            'label': '/path/mask.nii.gz'
        }
        image_keys = get_category_keys(subject, 'image')
        # Order should be preserved from dict
        assert set(image_keys) == {'image_dwi', 'image_adc', 'image_flair'}
        assert len(image_keys) == 3

    def test_multi_class_labels(self):
        """Test extracting multiple label classes."""
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'label_class0': '/path/lesion1.nii.gz',
            'label_class1': '/path/lesion2.nii.gz',
        }
        label_keys = get_category_keys(subject, 'label')
        assert set(label_keys) == {'label_class0', 'label_class1'}
        assert len(label_keys) == 2

    def test_no_matching_keys(self):
        """Test when category has no matching keys."""
        subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
        control_keys = get_category_keys(subject, 'control')
        assert control_keys == []

    def test_mixed_keys(self):
        """Test mixed keys (some with identifiers, some without)."""
        # This might not be a valid use case, but should handle it
        subject = {
            'image': '/path/t1.nii.gz',
            'image_dwi': '/path/dwi.nii.gz',
            'label': '/path/mask.nii.gz'
        }
        image_keys = get_category_keys(subject, 'image')
        assert set(image_keys) == {'image', 'image_dwi'}
        assert len(image_keys) == 2

    def test_controls(self):
        """Test extracting control keys."""
        subject = {
            'image_dwi': '/path/patient_dwi.nii.gz',
            'control_dwi': '/path/healthy_dwi.nii.gz',
            'label': '/path/mask.nii.gz'
        }
        control_keys = get_category_keys(subject, 'control')
        assert control_keys == ['control_dwi']


class TestBuildSchema:
    """Test build_schema() function."""

    def test_single_modality_schema(self):
        """Test schema for single modality (backward compatible)."""
        subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
        schema = build_schema(subject)

        assert schema['image_keys'] == ['image']
        assert schema['label_keys'] == ['label']
        assert schema['control_keys'] == []
        assert schema['has_multi_modal_images'] is False
        assert schema['has_multi_class_labels'] is False
        assert schema['has_controls'] is False

    def test_multi_modal_images_schema(self):
        """Test schema with multi-modal images."""
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'image_adc': '/path/adc.nii.gz',
            'label': '/path/mask.nii.gz'
        }
        schema = build_schema(subject)

        assert set(schema['image_keys']) == {'image_dwi', 'image_adc'}
        assert schema['label_keys'] == ['label']
        assert schema['has_multi_modal_images'] is True
        assert schema['has_multi_class_labels'] is False

    def test_multi_class_labels_schema(self):
        """Test schema with multi-class labels."""
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'label_class0': '/path/lesion1.nii.gz',
            'label_class1': '/path/lesion2.nii.gz',
        }
        schema = build_schema(subject)

        assert schema['image_keys'] == ['image_dwi']
        assert set(schema['label_keys']) == {'label_class0', 'label_class1'}
        assert schema['has_multi_modal_images'] is False
        assert schema['has_multi_class_labels'] is True

    def test_controls_schema(self):
        """Test schema with control subjects."""
        subject = {
            'image_dwi': '/path/patient_dwi.nii.gz',
            'control_dwi': '/path/healthy_dwi.nii.gz',
            'label': '/path/mask.nii.gz'
        }
        schema = build_schema(subject)

        assert schema['image_keys'] == ['image_dwi']
        assert schema['control_keys'] == ['control_dwi']
        assert schema['has_controls'] is True

    def test_full_schema(self):
        """Test schema with all features enabled."""
        subject = {
            'image_dwi': '/path/patient_dwi.nii.gz',
            'image_adc': '/path/patient_adc.nii.gz',
            'control_dwi': '/path/healthy_dwi.nii.gz',
            'control_adc': '/path/healthy_adc.nii.gz',
            'label_class0': '/path/lesion1.nii.gz',
            'label_class1': '/path/lesion2.nii.gz',
        }
        schema = build_schema(subject)

        assert set(schema['image_keys']) == {'image_dwi', 'image_adc'}
        assert set(schema['label_keys']) == {'label_class0', 'label_class1'}
        assert set(schema['control_keys']) == {'control_dwi', 'control_adc'}
        assert schema['has_multi_modal_images'] is True
        assert schema['has_multi_class_labels'] is True
        assert schema['has_controls'] is True


class TestValidateAgainstSchema:
    """Test validate_against_schema() function."""

    def test_valid_single_modality(self):
        """Test validation passes for single modality."""
        schema = {
            'image_keys': ['image'],
            'label_keys': ['label'],
            'control_keys': [],
        }
        subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}

        # Should not raise
        validate_against_schema(subject, schema, 'subject_001')

    def test_valid_multi_modal(self):
        """Test validation passes for multi-modal."""
        schema = {
            'image_keys': ['image_dwi', 'image_adc'],
            'label_keys': ['label'],
            'control_keys': [],
        }
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'image_adc': '/path/adc.nii.gz',
            'label': '/path/mask.nii.gz'
        }

        # Should not raise
        validate_against_schema(subject, schema, 'subject_002')

    def test_missing_image_key(self):
        """Test validation fails when image key is missing."""
        schema = {
            'image_keys': ['image_dwi', 'image_adc'],
            'label_keys': ['label'],
            'control_keys': [],
        }
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            # Missing 'image_adc'
            'label': '/path/mask.nii.gz'
        }

        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(subject, schema, 'subject_003')

        # Check error message is descriptive
        error_msg = str(exc_info.value)
        assert 'subject_003' in error_msg
        assert 'image_adc' in error_msg
        assert 'missing' in error_msg.lower()

    def test_missing_label_key(self):
        """Test validation fails when label key is missing."""
        schema = {
            'image_keys': ['image_dwi'],
            'label_keys': ['label_class0', 'label_class1'],
            'control_keys': [],
        }
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'label_class0': '/path/lesion1.nii.gz',
            # Missing 'label_class1'
        }

        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(subject, schema, 'subject_004')

        error_msg = str(exc_info.value)
        assert 'subject_004' in error_msg
        assert 'label_class1' in error_msg

    def test_missing_control_key(self):
        """Test validation fails when control key is missing."""
        schema = {
            'image_keys': ['image_dwi'],
            'label_keys': ['label'],
            'control_keys': ['control_dwi'],
        }
        subject = {
            'image_dwi': '/path/patient_dwi.nii.gz',
            'label': '/path/mask.nii.gz',
            # Missing 'control_dwi'
        }

        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(subject, schema, 'subject_005')

        error_msg = str(exc_info.value)
        assert 'subject_005' in error_msg
        assert 'control_dwi' in error_msg

    def test_extra_keys_allowed(self):
        """Test validation allows extra keys (for extensibility)."""
        schema = {
            'image_keys': ['image'],
            'label_keys': ['label'],
            'control_keys': [],
        }
        subject = {
            'image': '/path/img.nii.gz',
            'label': '/path/mask.nii.gz',
            'extra_metadata': '/path/meta.json',  # Extra key
        }

        # Should not raise - extra keys are allowed
        validate_against_schema(subject, schema, 'subject_006')

    def test_error_message_shows_expected_and_found(self):
        """Test error message shows both expected and found keys."""
        schema = {
            'image_keys': ['image_dwi', 'image_adc'],
            'label_keys': ['label'],
            'control_keys': [],
        }
        subject = {
            'image_dwi': '/path/dwi.nii.gz',
            'label': '/path/mask.nii.gz',
        }

        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(subject, schema, 'subject_007')

        error_msg = str(exc_info.value)
        # Should show what was expected
        assert 'image_dwi' in error_msg or 'image_adc' in error_msg
        # Should show what was found
        assert 'image_dwi' in error_msg
        assert 'label' in error_msg


class TestExtractModelConfig:
    """Test extract_model_config() function."""

    def test_single_modality_default_config(self):
        """Single modality should return default config with in_channels=1, out_channels=1."""
        split_lists = [[
            {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
        ]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 1
        assert config['out_channels'] == 1

    def test_multi_modal_images(self):
        """Multi-modal images should return in_channels = number of modalities."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 2
        assert config['out_channels'] == 1

    def test_multi_class_labels(self):
        """Multi-class labels should return out_channels = number of classes."""
        split_lists = [[
            {
                'image': '/path/img.nii.gz',
                'label_class0': '/path/mask0.nii.gz',
                'label_class1': '/path/mask1.nii.gz',
                'label_class2': '/path/mask2.nii.gz'
            }
        ]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 1
        assert config['out_channels'] == 3

    def test_multi_modal_and_multi_class(self):
        """Both multi-modal images and multi-class labels."""
        split_lists = [[
            {
                'image_flair': '/path/flair.nii.gz',
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label_class0': '/path/mask0.nii.gz',
                'label_class1': '/path/mask1.nii.gz'
            }
        ]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 3
        assert config['out_channels'] == 2

    def test_empty_split_lists(self):
        """Empty split_lists should return default config."""
        split_lists = []

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 1
        assert config['out_channels'] == 1

    def test_empty_fold(self):
        """Empty fold should return default config."""
        split_lists = [[]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 1
        assert config['out_channels'] == 1

    def test_single_modality_with_identifier(self):
        """Single modality with identifier (e.g., image_dwi only) should return in_channels=1."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        config = extract_model_config(split_lists)

        assert config['in_channels'] == 1
        assert config['out_channels'] == 1

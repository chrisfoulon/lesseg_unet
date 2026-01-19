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


class TestAdaptTransformsForResolution:
    """Test adapt_transforms_for_resolution() function."""

    def test_no_change_when_same_resolution(self):
        """No scaling when base and target resolution match."""
        from lesseg_unet.data_utils import adapt_transforms_for_resolution

        transform_dict = {
            'monai_transform': [
                {'Rand3DElasticd': {
                    'keys': ['image', 'label'],
                    'sigma_range': (3, 15),
                    'magnitude_range': (3, 10),
                    'translate_range': (0.5, 3),
                }}
            ]
        }

        result = adapt_transforms_for_resolution(transform_dict, base_resolution=2, target_resolution=2)

        # Should be unchanged (but a copy)
        assert result['monai_transform'][0]['Rand3DElasticd']['sigma_range'] == (3, 15)
        assert result['monai_transform'][0]['Rand3DElasticd']['magnitude_range'] == (3, 10)

    def test_scale_up_for_higher_resolution(self):
        """Parameters should scale UP for 1mm (higher resolution = more voxels needed)."""
        from lesseg_unet.data_utils import adapt_transforms_for_resolution

        transform_dict = {
            'monai_transform': [
                {'Rand3DElasticd': {
                    'keys': ['image', 'label'],
                    'sigma_range': (3, 15),
                    'magnitude_range': (3, 10),
                    'translate_range': (0.5, 3),
                }}
            ]
        }

        result = adapt_transforms_for_resolution(transform_dict, base_resolution=2, target_resolution=1)

        # Scale factor = 2/1 = 2 (need MORE voxels for same physical deformation)
        assert result['monai_transform'][0]['Rand3DElasticd']['sigma_range'] == (6, 30)
        assert result['monai_transform'][0]['Rand3DElasticd']['magnitude_range'] == (6, 20)
        assert result['monai_transform'][0]['Rand3DElasticd']['translate_range'] == (1, 6)

    def test_scale_down_for_lower_resolution(self):
        """Parameters should scale DOWN for 3mm (lower resolution = fewer voxels needed)."""
        from lesseg_unet.data_utils import adapt_transforms_for_resolution

        transform_dict = {
            'monai_transform': [
                {'Rand3DElasticd': {
                    'keys': ['image', 'label'],
                    'sigma_range': (3, 15),
                    'magnitude_range': (3, 10),
                    'translate_range': (0.5, 3),
                }}
            ]
        }

        result = adapt_transforms_for_resolution(transform_dict, base_resolution=2, target_resolution=3)

        # Scale factor = 2/3 ≈ 0.667 (need FEWER voxels for same physical deformation)
        import pytest
        sigma = result['monai_transform'][0]['Rand3DElasticd']['sigma_range']
        magnitude = result['monai_transform'][0]['Rand3DElasticd']['magnitude_range']
        translate = result['monai_transform'][0]['Rand3DElasticd']['translate_range']

        assert sigma == pytest.approx((2, 10), rel=1e-6)
        assert magnitude == pytest.approx((2, 20/3), rel=1e-6)
        assert translate == pytest.approx((1/3, 2), rel=1e-6)

    def test_original_not_modified(self):
        """Original transform dict should not be modified."""
        from lesseg_unet.data_utils import adapt_transforms_for_resolution

        transform_dict = {
            'monai_transform': [
                {'Rand3DElasticd': {
                    'keys': ['image', 'label'],
                    'sigma_range': (3, 15),
                }}
            ]
        }

        adapt_transforms_for_resolution(transform_dict, base_resolution=2, target_resolution=1)

        # Original should be unchanged
        assert transform_dict['monai_transform'][0]['Rand3DElasticd']['sigma_range'] == (3, 15)

    def test_other_transforms_unchanged(self):
        """Non-Rand3DElasticd transforms should be unchanged."""
        from lesseg_unet.data_utils import adapt_transforms_for_resolution

        transform_dict = {
            'monai_transform': [
                {'RandHistogramShiftd': {'keys': ['image'], 'prob': 0.1}},
                {'Rand3DElasticd': {'sigma_range': (3, 15)}},
            ]
        }

        result = adapt_transforms_for_resolution(transform_dict, base_resolution=2, target_resolution=1)

        # RandHistogramShiftd should be unchanged
        assert result['monai_transform'][0]['RandHistogramShiftd']['prob'] == 0.1


class TestExpandPerModalityTransforms:
    """Test expand_per_modality_transforms() function."""

    def test_no_modality_intensity_section(self):
        """Return unchanged if no modality_intensity section."""
        from lesseg_unet.data_utils import expand_per_modality_transforms

        transform_dict = {
            'first_transform': [{'LoadImaged': {'keys': ['image', 'label']}}],
            'monai_transform': [{'RandFlipd': {'keys': ['image', 'label']}}],
        }

        result = expand_per_modality_transforms(transform_dict, ['image_dwi', 'image_adc'])

        # Should be unchanged (same structure)
        assert 'first_transform' in result
        assert 'monai_transform' in result
        assert 'expanded_modality_intensity' not in result

    def test_expand_for_two_modalities(self):
        """Expand transforms for DWI and ADC modalities."""
        from lesseg_unet.data_utils import expand_per_modality_transforms

        transform_dict = {
            'modality_intensity': [
                {'RandHistogramShiftd': {'keys': ['image'], 'prob': 0.1}},
            ]
        }

        result = expand_per_modality_transforms(transform_dict, ['image_dwi', 'image_adc'])

        # Should have expanded transforms
        assert 'expanded_modality_intensity' in result
        assert 'modality_intensity' not in result

        expanded = result['expanded_modality_intensity']
        assert len(expanded) == 2  # One per modality

        # Check keys are replaced
        assert expanded[0]['RandHistogramShiftd']['keys'] == ['image_dwi']
        assert expanded[1]['RandHistogramShiftd']['keys'] == ['image_adc']

    def test_modality_specific_params_applied(self):
        """Modality-specific parameters should be applied."""
        from lesseg_unet.data_utils import expand_per_modality_transforms, MODALITY_PARAMS

        transform_dict = {
            'modality_intensity': [
                {'RandKSpaceSpikeNoised': {'keys': ['image'], 'prob': 0.2}},
            ]
        }

        result = expand_per_modality_transforms(transform_dict, ['image_dwi', 'image_adc'])

        expanded = result['expanded_modality_intensity']

        # DWI should have its specific prob
        dwi_prob = expanded[0]['RandKSpaceSpikeNoised']['prob']
        adc_prob = expanded[1]['RandKSpaceSpikeNoised']['prob']

        # ADC should have lower prob (from MODALITY_PARAMS)
        assert adc_prob == MODALITY_PARAMS['adc']['RandKSpaceSpikeNoised']['prob']
        assert dwi_prob == MODALITY_PARAMS['dwi']['RandKSpaceSpikeNoised']['prob']

    def test_single_modality_uses_default(self):
        """Single modality with no specific params uses defaults."""
        from lesseg_unet.data_utils import expand_per_modality_transforms

        transform_dict = {
            'modality_intensity': [
                {'RandHistogramShiftd': {'keys': ['image'], 'prob': 0.1}},
            ]
        }

        # Single modality case
        result = expand_per_modality_transforms(transform_dict, ['image_flair'])

        expanded = result['expanded_modality_intensity']
        assert len(expanded) == 1
        assert expanded[0]['RandHistogramShiftd']['keys'] == ['image_flair']


class TestCreateMultimodalTransformDict:
    """Test create_multimodal_transform_dict() function."""

    def test_default_parameters(self):
        """Default function call should return valid dict."""
        from lesseg_unet.data.transform_dicts import create_multimodal_transform_dict

        result = create_multimodal_transform_dict()

        assert 'first_transform' in result
        assert 'modality_intensity' in result
        assert 'monai_transform' in result
        assert 'last_transform' in result
        assert 'patches' in result

    def test_resolution_scaling(self):
        """Resolution parameter should scale elastic params correctly.

        Higher resolution (smaller mm) means more voxels per physical distance,
        so we need MORE voxels (larger params) to achieve the same physical effect.
        """
        from lesseg_unet.data.transform_dicts import create_multimodal_transform_dict

        # 2mm base
        result_2mm = create_multimodal_transform_dict(resolution_mm=2)
        # 1mm should have LARGER params (more voxels needed for same physical deformation)
        result_1mm = create_multimodal_transform_dict(resolution_mm=1)

        elastic_2mm = result_2mm['monai_transform'][0]['Rand3DElasticd']
        elastic_1mm = result_1mm['monai_transform'][0]['Rand3DElasticd']

        # 1mm should have DOUBLE the voxel params (scale = 2mm/1mm = 2)
        assert elastic_1mm['sigma_range'][0] > elastic_2mm['sigma_range'][0]
        assert elastic_1mm['magnitude_range'][0] > elastic_2mm['magnitude_range'][0]
        # Verify exact scaling: 1mm params should be 2x 2mm params
        assert elastic_1mm['sigma_range'][0] == elastic_2mm['sigma_range'][0] * 2
        assert elastic_1mm['magnitude_range'][0] == elastic_2mm['magnitude_range'][0] * 2

    def test_patch_size_parameter(self):
        """Patch size parameter should be reflected in patches."""
        from lesseg_unet.data.transform_dicts import create_multimodal_transform_dict

        result_96 = create_multimodal_transform_dict(patch_size=96)
        result_64 = create_multimodal_transform_dict(patch_size=64)

        crop_96 = result_96['patches'][0]['RandCropByPosNegLabeld']
        crop_64 = result_64['patches'][0]['RandCropByPosNegLabeld']

        assert crop_96['spatial_size'] == [96, 96, 96]
        assert crop_64['spatial_size'] == [64, 64, 64]

    def test_denoised_data_reduces_noise(self):
        """denoised_data=True should reduce noise parameters."""
        from lesseg_unet.data.transform_dicts import create_multimodal_transform_dict

        result_normal = create_multimodal_transform_dict(denoised_data=False)
        result_denoised = create_multimodal_transform_dict(denoised_data=True)

        # Find RandGibbsNoised in modality_intensity
        gibbs_normal = None
        gibbs_denoised = None
        for t in result_normal['modality_intensity']:
            if 'RandGibbsNoised' in t:
                gibbs_normal = t['RandGibbsNoised']
        for t in result_denoised['modality_intensity']:
            if 'RandGibbsNoised' in t:
                gibbs_denoised = t['RandGibbsNoised']

        # Denoised should have lower probability
        assert gibbs_denoised['prob'] < gibbs_normal['prob']

    def test_convenience_functions(self):
        """Test mm1_p96, mm1_p64, mm2_p96 convenience functions."""
        from lesseg_unet.data.transform_dicts import mm1_p96, mm1_p64, mm2_p96

        result_mm1_p96 = mm1_p96()
        result_mm1_p64 = mm1_p64()
        result_mm2_p96 = mm2_p96()

        # Check patch sizes
        assert result_mm1_p96['patches'][0]['RandCropByPosNegLabeld']['spatial_size'] == [96, 96, 96]
        assert result_mm1_p64['patches'][0]['RandCropByPosNegLabeld']['spatial_size'] == [64, 64, 64]
        assert result_mm2_p96['patches'][0]['RandCropByPosNegLabeld']['spatial_size'] == [96, 96, 96]

        # Check resolution scaling (mm1 should have LARGER elastic params than mm2)
        # Higher resolution = more voxels needed for same physical deformation
        elastic_mm1 = result_mm1_p96['monai_transform'][0]['Rand3DElasticd']
        elastic_mm2 = result_mm2_p96['monai_transform'][0]['Rand3DElasticd']

        assert elastic_mm1['sigma_range'][0] > elastic_mm2['sigma_range'][0]
        # Verify exact ratio: mm1 params should be 2x mm2 params
        assert elastic_mm1['sigma_range'][0] == elastic_mm2['sigma_range'][0] * 2

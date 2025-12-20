"""Tests for multi-modal transform adaptation."""

import pytest
from lesseg_unet.data_utils import adapt_transforms_for_multimodal


class TestAdaptTransformsForMultimodal:
    """Test adapt_transforms_for_multimodal() function."""

    def test_single_modality_unchanged(self):
        """Single modality transform dict should remain unchanged."""
        split_lists = [[
            {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # Should be unchanged (same reference)
        assert adapted == transform_dict
        assert adapted['first_transform'][0]['LoadImaged']['keys'] == ['image', 'label']

    def test_multi_modal_keys_replaced(self):
        """Multi-modal split_lists should replace 'image' with image_* keys."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # LoadImaged should have both image_* keys
        load_keys = adapted['first_transform'][0]['LoadImaged']['keys']
        assert 'image_adc' in load_keys
        assert 'image_dwi' in load_keys
        assert 'label' in load_keys
        assert 'image' not in load_keys  # Original 'image' replaced

    def test_concat_inserted_after_load(self):
        """ConcatItemsd should be inserted after EnsureChannelFirstd."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # ConcatItemsd should be at index 2 (after EnsureChannelFirstd at index 1)
        concat_dict = adapted['first_transform'][2]
        assert 'ConcatItemsd' in concat_dict
        assert concat_dict['ConcatItemsd']['name'] == 'image'
        assert concat_dict['ConcatItemsd']['dim'] == 0
        assert set(concat_dict['ConcatItemsd']['keys']) == {'image_adc', 'image_dwi'}

    def test_keys_sorted_alphabetically(self):
        """Image keys should be sorted alphabetically for consistency."""
        split_lists = [[
            {
                'image_flair': '/path/flair.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'image_dwi': '/path/dwi.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        load_keys = adapted['first_transform'][0]['LoadImaged']['keys']
        # Image keys should be sorted: adc, dwi, flair
        image_keys_in_order = [k for k in load_keys if k.startswith('image_')]
        assert image_keys_in_order == ['image_adc', 'image_dwi', 'image_flair']

    def test_label_key_preserved(self):
        """Label keys should not be modified."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'Binarized': {'keys': ['label']}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # Label should remain unchanged
        assert adapted['first_transform'][2]['Binarized']['keys'] == ['label']

    @pytest.mark.skip(reason="Torchio transforms come after concat, don't need multi-key support")
    def test_torchio_include_parameter(self):
        """Torchio transforms use 'include' instead of 'keys'."""
        # Torchio transforms typically come after ConcatItemsd in the pipeline
        # so they work on the concatenated 'image' key, not individual image_* keys
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'torchio_transform': [
                {'RandomNoise': {'include': ['image'], 'p': 0.5}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # Torchio transforms should be unchanged (they use concatenated 'image')
        include_list = adapted['torchio_transform'][0]['RandomNoise']['include']
        assert include_list == ['image']

    def test_empty_split_lists(self):
        """Empty split_lists should return unchanged dict."""
        split_lists = []

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)
        assert adapted == transform_dict

    def test_multiple_transform_stages(self):
        """Only early transforms (LoadImaged, EnsureChannelFirstd) use image_* keys."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
                {'NormalizeIntensityd': {'keys': ['image']}}
            ],
            'monai_transform': [
                {'RandAffined': {'keys': ['image', 'label']}}
            ],
            'last_transform': [
                {'NormalizeIntensityd': {'keys': ['image']}}
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # LoadImaged should use image_* keys
        load_keys = adapted['first_transform'][0]['LoadImaged']['keys']
        assert 'image_adc' in load_keys
        assert 'image_dwi' in load_keys
        assert 'label' in load_keys

        # EnsureChannelFirstd should also use image_* keys
        ensure_keys = adapted['first_transform'][1]['EnsureChannelFirstd']['keys']
        assert 'image_adc' in ensure_keys
        assert 'image_dwi' in ensure_keys

        # ConcatItemsd should be at index 2 (after EnsureChannelFirstd)
        assert 'ConcatItemsd' in adapted['first_transform'][2]

        # Transforms after concat should use 'image' (unchanged)
        norm_keys = adapted['first_transform'][3]['NormalizeIntensityd']['keys']
        assert norm_keys == ['image']

        # Other transform stages should be unchanged
        assert adapted['monai_transform'][0]['RandAffined']['keys'] == ['image', 'label']
        assert adapted['last_transform'][0]['NormalizeIntensityd']['keys'] == ['image']

    def test_original_dict_not_modified(self):
        """Original transform dict should not be modified (deep copy)."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'image_adc': '/path/adc.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        original_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
            ]
        }

        # Make a copy to compare later
        import copy
        original_copy = copy.deepcopy(original_dict)

        adapted = adapt_transforms_for_multimodal(original_dict, split_lists)

        # Original should be unchanged
        assert original_dict == original_copy
        assert adapted != original_dict

    def test_single_modality_with_identifier(self):
        """Single modality with identifier (e.g., image_dwi only) replaces key but no concat."""
        split_lists = [[
            {
                'image_dwi': '/path/dwi.nii.gz',
                'label': '/path/mask.nii.gz'
            }
        ]]

        transform_dict = {
            'first_transform': [
                {'LoadImaged': {'keys': ['image', 'label']}},
                {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
            ]
        }

        adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)

        # LoadImaged should use image_dwi
        load_keys = adapted['first_transform'][0]['LoadImaged']['keys']
        assert 'image_dwi' in load_keys
        assert 'image' not in load_keys

        # len(image_keys) == 1, so no concat added
        assert len(adapted['first_transform']) == 2  # No concat added

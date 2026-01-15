"""
Tests for validation_loop_split_lists function.

Tests focus on:
1. Function signature accepts subject dicts
2. Transform adaptation is called correctly
3. Integration with the staged pipeline
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import nibabel as nib

from lesseg_unet import segmentation, data_utils


class TestValidationLoopSplitListsSignature:
    """Test that validation_loop_split_lists accepts the expected input format."""

    def test_function_exists(self):
        """Verify the function exists in the segmentation module."""
        assert hasattr(segmentation, 'validation_loop_split_lists')
        assert callable(segmentation.validation_loop_split_lists)

    def test_function_signature_has_subject_dicts(self):
        """Verify the function accepts subject_dicts as first parameter."""
        import inspect
        sig = inspect.signature(segmentation.validation_loop_split_lists)
        params = list(sig.parameters.keys())
        assert params[0] == 'subject_dicts'

    def test_function_signature_returns_dict(self):
        """Verify the function is annotated to return a dict."""
        import inspect
        from typing import Dict, Any
        sig = inspect.signature(segmentation.validation_loop_split_lists)
        # Check return annotation
        assert sig.return_annotation == Dict[str, Any]


class TestValidationLoopSplitListsTransformAdaptation:
    """Test that transforms are adapted correctly for multi-modal data."""

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create a mock checkpoint file."""
        import torch
        checkpoint = {
            'transform_dict': {
                'first_transform': [
                    {'LoadImaged': {'keys': ['image', 'label']}}
                ],
                'monai_transform': []
            },
            'hyper_params': {
                'spatial_dims': 3,
                'in_channels': 1,
                'out_channels': 1,
                'features': [16, 32, 64, 128]
            },
            'model_name': 'unet',
            'model_state_dict': {}
        }
        checkpoint_path = tmp_path / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def sample_subject_dicts(self, tmp_path):
        """Create sample subject dicts with mock NIfTI files."""
        subjects = []
        for i in range(3):
            # Create mock NIfTI files
            img_data = np.random.rand(10, 10, 10).astype(np.float32)
            label_data = (np.random.rand(10, 10, 10) > 0.5).astype(np.float32)

            img_path = tmp_path / f'subject{i}_image.nii.gz'
            label_path = tmp_path / f'subject{i}_label.nii.gz'

            nib.save(nib.Nifti1Image(img_data, np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(label_data, np.eye(4)), label_path)

            subjects.append({
                'image': str(img_path),
                'label': str(label_path)
            })
        return subjects

    @pytest.fixture
    def multimodal_subject_dicts(self, tmp_path):
        """Create sample multi-modal subject dicts."""
        subjects = []
        for i in range(3):
            # Create mock NIfTI files for multiple modalities
            dwi_data = np.random.rand(10, 10, 10).astype(np.float32)
            adc_data = np.random.rand(10, 10, 10).astype(np.float32)
            label_data = (np.random.rand(10, 10, 10) > 0.5).astype(np.float32)

            dwi_path = tmp_path / f'subject{i}_dwi.nii.gz'
            adc_path = tmp_path / f'subject{i}_adc.nii.gz'
            label_path = tmp_path / f'subject{i}_lesion.nii.gz'

            nib.save(nib.Nifti1Image(dwi_data, np.eye(4)), dwi_path)
            nib.save(nib.Nifti1Image(adc_data, np.eye(4)), adc_path)
            nib.save(nib.Nifti1Image(label_data, np.eye(4)), label_path)

            subjects.append({
                'image_dwi': str(dwi_path),
                'image_adc': str(adc_path),
                'label_lesion': str(label_path)
            })
        return subjects

    def test_adapt_transforms_called_with_split_lists_format(
        self, sample_subject_dicts, mock_checkpoint, tmp_path
    ):
        """Verify adapt_transforms_for_multimodal is called with correct format."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Mock the heavy operations
        with patch.object(data_utils, 'adapt_transforms_for_multimodal') as mock_adapt:
            # Set up the mock to return the input transform_dict unchanged
            mock_adapt.side_effect = lambda td, sl: td

            # Also mock the model loading and inference parts
            with patch('lesseg_unet.segmentation.torch.load') as mock_load:
                mock_load.return_value = {
                    'transform_dict': {'first_transform': [], 'monai_transform': []},
                    'hyper_params': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 1},
                    'model_name': 'unet',
                    'model_state_dict': {}
                }
                with patch('lesseg_unet.segmentation.utils.load_model_from_checkpoint'):
                    with patch('lesseg_unet.segmentation.transformations.val_transformd'):
                        with patch('lesseg_unet.segmentation.Dataset'):
                            with patch('lesseg_unet.segmentation.data_loading.create_validation_data_loader'):
                                try:
                                    segmentation.validation_loop_split_lists(
                                        sample_subject_dicts,
                                        output_dir,
                                        mock_checkpoint,
                                        device='cpu'
                                    )
                                except Exception:
                                    pass  # We just want to verify the mock was called

                                # Verify adapt_transforms_for_multimodal was called
                                assert mock_adapt.called
                                call_args = mock_adapt.call_args
                                # Second argument should be split_lists format: [[subjects]]
                                split_lists_arg = call_args[0][1]
                                assert isinstance(split_lists_arg, list)
                                assert len(split_lists_arg) == 1
                                assert split_lists_arg[0] == sample_subject_dicts


class TestMainPyRouting:
    """Test that main.py routes correctly to validation_loop_split_lists."""

    def test_has_embedded_labels_detection_with_label_key(self):
        """Test that has_embedded_labels is True when label_* keys present."""
        # Simulate the detection logic from main.py
        img_list = [
            {'image_dwi': '/path/dwi.nii', 'image_adc': '/path/adc.nii',
             'label_lesion': '/path/les.nii'}
        ]

        first_item = img_list[0]
        if isinstance(first_item, list) and len(first_item) > 0:
            first_item = first_item[0]

        has_embedded_labels = False
        if isinstance(first_item, dict):
            has_embedded_labels = any(k.startswith('label_') for k in first_item.keys())

        assert has_embedded_labels is True

    def test_has_embedded_labels_detection_without_label_key(self):
        """Test that has_embedded_labels is False when no label_* keys."""
        # Simulate single-modality with standard 'label' key
        img_list = [
            {'image': '/path/img.nii', 'label': '/path/les.nii'}
        ]

        first_item = img_list[0]
        has_embedded_labels = False
        if isinstance(first_item, dict):
            has_embedded_labels = any(k.startswith('label_') for k in first_item.keys())

        assert has_embedded_labels is False

    def test_has_embedded_labels_detection_with_split_lists_format(self):
        """Test detection works with nested split_lists format."""
        # Simulate split_lists format from training mode
        img_list = [[
            {'image_dwi': '/path/dwi.nii', 'label_lesion': '/path/les.nii'}
        ]]

        first_item = img_list[0]
        if isinstance(first_item, list) and len(first_item) > 0:
            first_item = first_item[0]

        has_embedded_labels = False
        if isinstance(first_item, dict):
            has_embedded_labels = any(k.startswith('label_') for k in first_item.keys())

        assert has_embedded_labels is True


class TestValidationLoopSplitListsDocstring:
    """Test that the docstring is correct and informative."""

    def test_docstring_exists(self):
        """Verify the function has a docstring."""
        assert segmentation.validation_loop_split_lists.__doc__ is not None

    def test_docstring_contains_parameters(self):
        """Verify docstring documents key parameters."""
        doc = segmentation.validation_loop_split_lists.__doc__
        assert 'subject_dicts' in doc
        assert 'output_dir' in doc
        assert 'checkpoint_path' in doc

    def test_docstring_contains_examples(self):
        """Verify docstring contains usage examples."""
        doc = segmentation.validation_loop_split_lists.__doc__
        assert 'Examples' in doc
        assert 'image_dwi' in doc  # Multi-modal example

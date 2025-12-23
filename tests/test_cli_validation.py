"""Tests for CLI argument validation (multi-modal support)."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from lesseg_unet.main import validate_modality_names, validate_multimodal_arguments, build_modality_dict


class TestValidateModalityNames:
    """Test validate_modality_names() function."""

    def test_valid_names(self):
        """Valid modality names should pass without error."""
        validate_modality_names(['dwi', 'adc', 'flair'], reserved={'label', 'control'})
        # Should not raise

    def test_reserved_name(self):
        """Reserved names should raise ValueError."""
        with pytest.raises(ValueError, match="reserved"):
            validate_modality_names(['label'], reserved={'label', 'control'})

    def test_duplicate_names(self):
        """Duplicate names should raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate"):
            validate_modality_names(['dwi', 'adc', 'dwi'], reserved={'label'})

    def test_invalid_identifier(self):
        """Invalid Python identifiers should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid modality name"):
            validate_modality_names(['123invalid'], reserved={'label'})

        with pytest.raises(ValueError, match="Invalid modality name"):
            validate_modality_names(['my-name'], reserved={'label'})


class TestBuildModalityDict:
    """Test build_modality_dict() function."""

    def test_with_explicit_names(self):
        """Build dict with explicit modality names."""
        result = build_modality_dict(
            ['/data/dwi', '/data/adc'],
            ['dwi', 'adc'],
            'image'
        )
        assert result == {'dwi': Path('/data/dwi'), 'adc': Path('/data/adc')}

    def test_with_auto_alphabetic_names(self):
        """Build dict with auto-generated alphabetic names."""
        result = build_modality_dict(
            ['/data/mod1', '/data/mod2', '/data/mod3'],
            None,
            'image'
        )
        assert result == {
            'a': Path('/data/mod1'),
            'b': Path('/data/mod2'),
            'c': Path('/data/mod3')
        }

    def test_single_path(self):
        """Single path should get 'a' as name."""
        result = build_modality_dict(['/data/images'], None, 'image')
        assert result == {'a': Path('/data/images')}

    def test_none_paths(self):
        """None paths should return None."""
        result = build_modality_dict(None, None, 'image')
        assert result is None

    def test_empty_list(self):
        """Empty list should return None."""
        result = build_modality_dict([], None, 'image')
        assert result is None

    def test_many_paths(self):
        """Test alphabetic naming beyond 26 paths."""
        paths = [f'/data/mod{i}' for i in range(30)]
        result = build_modality_dict(paths, None, 'image')

        # First 26 should be a-z
        assert result['a'] == Path('/data/mod0')
        assert result['z'] == Path('/data/mod25')

        # After 26, should be aa, ab, ac, ad
        assert result['aa'] == Path('/data/mod26')
        assert result['ab'] == Path('/data/mod27')
        assert result['ac'] == Path('/data/mod28')
        assert result['ad'] == Path('/data/mod29')


class TestValidateMultimodalArguments:
    """Test validate_multimodal_arguments() function."""

    class MockArgs:
        """Mock argparse.Namespace for testing."""

        def __init__(self):
            self.input_path = None
            self.image_modality_names = None
            self.lesion_input_path = None
            self.label_modality_names = None
            self.controls_path = None
            self.control_modality_names = None

    def test_single_modality_no_validation_needed(self):
        """Single modality should pass without validation."""
        args = self.MockArgs()
        args.input_path = ['/data/images']
        args.image_modality_names = None

        result = validate_multimodal_arguments(args)
        assert result is args  # Should return same object

    def test_multi_modal_with_matching_names(self):
        """Multi-modal with matching name count should pass."""
        args = self.MockArgs()
        args.input_path = ['/data/dwi', '/data/adc']
        args.image_modality_names = ['dwi', 'adc']

        result = validate_multimodal_arguments(args)
        assert result is args

    def test_multi_modal_without_names(self):
        """Multi-modal without names should pass (will auto-generate)."""
        args = self.MockArgs()
        args.input_path = ['/data/mod1', '/data/mod2']
        args.image_modality_names = None

        result = validate_multimodal_arguments(args)
        assert result is args

    def test_mismatched_image_name_count(self):
        """Mismatched image modality name count should raise ValueError."""
        args = self.MockArgs()
        args.input_path = ['/data/dwi', '/data/adc']
        args.image_modality_names = ['dwi']  # Only 1 name for 2 paths

        with pytest.raises(ValueError, match="Image modality count mismatch"):
            validate_multimodal_arguments(args)

    def test_mismatched_label_name_count(self):
        """Mismatched label name count should raise ValueError."""
        args = self.MockArgs()
        args.lesion_input_path = ['/labels/l1', '/labels/l2']
        args.label_modality_names = ['lesion']  # Only 1 name for 2 paths

        with pytest.raises(ValueError, match="Label class count mismatch"):
            validate_multimodal_arguments(args)

    def test_mismatched_control_name_count(self):
        """Mismatched control name count should raise ValueError."""
        args = self.MockArgs()
        args.controls_path = ['/controls/c1', '/controls/c2']
        args.control_modality_names = ['dwi']  # Only 1 name for 2 paths

        with pytest.raises(ValueError, match="Control modality count mismatch"):
            validate_multimodal_arguments(args)

    def test_reserved_name_in_image_modalities(self):
        """Reserved name in image modalities should raise ValueError."""
        args = self.MockArgs()
        args.input_path = ['/data/mod1', '/data/mod2']
        args.image_modality_names = ['dwi', 'label']  # 'label' is reserved

        with pytest.raises(ValueError, match="reserved"):
            validate_multimodal_arguments(args)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

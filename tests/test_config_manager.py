"""Unit tests for config_manager module."""

import json
import pytest
from pathlib import Path
import tempfile

from lesseg_unet.auto_config import TrainingConfig, AutoConfigResult


@pytest.fixture
def minimal_config():
    """Minimal TrainingConfig for testing."""
    return TrainingConfig(
        batch_size=4,
        val_batch_size=8,
        patch_size=(96, 96, 96),
        num_workers=8,
        network_depth=5,
        feature_size=48,
        use_amp=True,
        num_gpus=1,
        vram_safety_margin=0.95
    )


@pytest.fixture
def full_config():
    """Full TrainingConfig with all optional fields."""
    return TrainingConfig(
        batch_size=4,
        val_batch_size=8,
        patch_size=(96, 96, 96),
        num_workers=8,
        network_depth=5,
        feature_size=48,
        use_amp=True,
        num_gpus=1,
        vram_safety_margin=0.95,
        learning_rate=1e-4,
        num_epochs=100,
        model_type='swinunetr',
        hardware_profile={
            'device_type': 'cuda',
            'gpus': [{'name': 'NVIDIA RTX 3090', 'total_memory_mb': 24576}],
            'cpu': {'available_cores': 16}
        },
        command='python -m lesseg_unet.main --auto_config',
        timestamp='2025-01-01T12:00:00',
        reasoning={'batch_size': 'Binary search found max batch=4'},
        memory_estimate={'total_gb': 20.5},
        user_overrides={'patch_size': (96, 96, 96)}
    )


@pytest.fixture
def auto_config_result():
    """Sample AutoConfigResult for testing."""
    return AutoConfigResult(
        batch_size=4,
        val_batch_size=8,
        patch_size=(96, 96, 96),
        num_workers=8,
        network_depth=5,
        feature_size=48,
        use_amp=True,
        use_checkpoint=False,
        num_gpus=1,
        vram_safety_margin=0.95,
        reasoning={'batch_size': 'Binary search found max batch=4'},
        memory_estimate={'total_gb': 20.5}
    )


class TestTrainingConfigCreation:
    """Test TrainingConfig creation."""

    def test_minimal_config_creation(self, minimal_config):
        """Test creating minimal config."""
        assert minimal_config.batch_size == 4
        assert minimal_config.patch_size == (96, 96, 96)
        assert minimal_config.num_workers == 8
        assert minimal_config.network_depth == 5
        assert minimal_config.feature_size == 48
        assert minimal_config.use_amp is True
        assert minimal_config.num_gpus == 1
        assert minimal_config.vram_safety_margin == 0.95

    def test_full_config_creation(self, full_config):
        """Test creating full config with all fields."""
        assert full_config.learning_rate == 1e-4
        assert full_config.num_epochs == 100
        assert full_config.model_type == 'swinunetr'
        assert full_config.hardware_profile is not None
        assert full_config.command is not None
        assert full_config.timestamp is not None
        assert len(full_config.reasoning) > 0
        assert len(full_config.memory_estimate) > 0
        assert len(full_config.user_overrides) > 0


class TestFromAutoConfig:
    """Test creating TrainingConfig from AutoConfigResult."""

    def test_from_auto_config_minimal(self, auto_config_result):
        """Test conversion from AutoConfigResult."""
        config = TrainingConfig.from_auto_config(auto_config_result)

        assert config.batch_size == 4
        assert config.patch_size == (96, 96, 96)
        assert config.num_workers == 8
        assert config.network_depth == 5
        assert config.feature_size == 48
        assert config.use_amp is True
        assert config.num_gpus == 1
        assert config.vram_safety_margin == 0.95
        assert config.reasoning == auto_config_result.reasoning
        assert config.memory_estimate == auto_config_result.memory_estimate
        assert config.timestamp is not None  # Should be auto-generated

    def test_from_auto_config_with_metadata(self, auto_config_result):
        """Test conversion with additional metadata."""
        hw_profile = {
            'device_type': 'cuda',
            'gpus': [{'name': 'NVIDIA RTX 3090'}]
        }

        config = TrainingConfig.from_auto_config(
            auto_config_result,
            hardware_profile=hw_profile,
            command='python -m lesseg_unet.main --auto_config',
            learning_rate=1e-4,
            num_epochs=100,
            model_type='swinunetr',
            user_overrides={'batch_size': 4}
        )

        assert config.hardware_profile == hw_profile
        assert config.command == 'python -m lesseg_unet.main --auto_config'
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 100
        assert config.model_type == 'swinunetr'
        assert config.user_overrides == {'batch_size': 4}


class TestSerialization:
    """Test to_dict and from_dict."""

    def test_to_dict_minimal(self, minimal_config):
        """Test converting minimal config to dict."""
        data = minimal_config.to_dict()

        assert isinstance(data, dict)
        assert data['batch_size'] == 4
        assert data['patch_size'] == [96, 96, 96]  # Converted to list for YAML/JSON
        assert data['num_workers'] == 8
        assert data['network_depth'] == 5

    def test_to_dict_full(self, full_config):
        """Test converting full config to dict."""
        data = full_config.to_dict()

        assert data['learning_rate'] == 1e-4
        assert data['num_epochs'] == 100
        assert data['model_type'] == 'swinunetr'
        assert 'hardware_profile' in data
        assert 'command' in data
        assert 'timestamp' in data

    def test_from_dict_minimal(self):
        """Test creating config from dict."""
        data = {
            'batch_size': 4,
            'val_batch_size': 8,
            'patch_size': [96, 96, 96],  # List (will be converted to tuple)
            'num_workers': 8,
            'network_depth': 5,
            'feature_size': 48,
            'use_amp': True,
            'num_gpus': 1,
            'vram_safety_margin': 0.95
        }

        config = TrainingConfig.from_dict(data)

        assert config.batch_size == 4
        assert config.val_batch_size == 8
        assert config.patch_size == (96, 96, 96)  # Should be tuple
        assert config.num_workers == 8

    def test_roundtrip_to_dict_from_dict(self, full_config):
        """Test roundtrip: config -> dict -> config."""
        data = full_config.to_dict()
        config2 = TrainingConfig.from_dict(data)

        assert config2.batch_size == full_config.batch_size
        assert config2.patch_size == full_config.patch_size
        assert config2.learning_rate == full_config.learning_rate
        assert config2.hardware_profile == full_config.hardware_profile


class TestSaveLoad:
    """Test save and load methods."""

    def test_save_yaml(self, minimal_config):
        """Test saving config as YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            minimal_config.save(output_path, format='yaml')

            assert output_path.exists()

            # Verify content
            import yaml
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)

            assert data['batch_size'] == 4
            assert data['patch_size'] == [96, 96, 96]

    def test_save_json(self, minimal_config):
        """Test saving config as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.json'
            minimal_config.save(output_path, format='json')

            assert output_path.exists()

            # Verify content
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data['batch_size'] == 4
            assert data['patch_size'] == [96, 96, 96]

    def test_save_creates_parent_directory(self, minimal_config):
        """Test that save creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'subdir' / 'config.yaml'
            minimal_config.save(output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_fails_without_overwrite(self, minimal_config):
        """Test that save fails if file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            minimal_config.save(output_path)

            with pytest.raises(FileExistsError):
                minimal_config.save(output_path, overwrite=False)

    def test_save_succeeds_with_overwrite(self, minimal_config):
        """Test that save succeeds if overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            minimal_config.save(output_path)

            # Modify config
            minimal_config.batch_size = 8

            # Should succeed with overwrite=True
            minimal_config.save(output_path, overwrite=True)

            # Verify updated content
            loaded = TrainingConfig.load(output_path)
            assert loaded.batch_size == 8

    def test_save_invalid_format(self, minimal_config):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.txt'
            with pytest.raises(ValueError):
                minimal_config.save(output_path, format='txt')

    def test_load_yaml(self, minimal_config):
        """Test loading config from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            minimal_config.save(output_path)

            loaded = TrainingConfig.load(output_path)

            assert loaded.batch_size == minimal_config.batch_size
            assert loaded.patch_size == minimal_config.patch_size
            assert loaded.num_workers == minimal_config.num_workers

    def test_load_json(self, minimal_config):
        """Test loading config from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.json'
            minimal_config.save(output_path, format='json')

            loaded = TrainingConfig.load(output_path)

            assert loaded.batch_size == minimal_config.batch_size
            assert loaded.patch_size == minimal_config.patch_size

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load('/nonexistent/config.yaml')

    def test_load_invalid_format(self):
        """Test that loading unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.txt'
            output_path.write_text('invalid')

            with pytest.raises(ValueError):
                TrainingConfig.load(output_path)

    def test_roundtrip_save_load_yaml(self, full_config):
        """Test roundtrip: save -> load for YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.yaml'
            full_config.save(output_path)

            loaded = TrainingConfig.load(output_path)

            assert loaded.batch_size == full_config.batch_size
            assert loaded.learning_rate == full_config.learning_rate
            assert loaded.hardware_profile == full_config.hardware_profile
            assert loaded.reasoning == full_config.reasoning

    def test_roundtrip_save_load_json(self, full_config):
        """Test roundtrip: save -> load for JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'config.json'
            full_config.save(output_path, format='json')

            loaded = TrainingConfig.load(output_path)

            assert loaded.batch_size == full_config.batch_size
            assert loaded.learning_rate == full_config.learning_rate


class TestApplyToNamespace:
    """Test apply_to_namespace method."""

    def test_apply_to_namespace_basic(self, minimal_config):
        """Test applying config to namespace."""
        import argparse
        args = argparse.Namespace(
            batch_size=1,
            patch_size=(64, 64, 64),
            num_workers=1,
            network_depth=4,
            feature_size=16,
            use_amp=False,
            num_gpus=1
        )

        minimal_config.apply_to_namespace(args)

        assert args.batch_size == 4
        assert args.patch_size == (96, 96, 96)
        assert args.num_workers == 8
        assert args.network_depth == 5
        assert args.feature_size == 48
        assert args.use_amp is True

    def test_apply_to_namespace_with_alternatives(self, full_config):
        """Test applying with alternative arg names (amp, lr)."""
        import argparse
        args = argparse.Namespace(
            batch_size=1,
            amp=False,  # Alternative name for use_amp
            lr=1e-3  # Alternative name for learning_rate
        )

        full_config.apply_to_namespace(args)

        assert args.batch_size == 4
        assert args.amp is True  # Should be updated
        assert args.lr == 1e-4  # Should be updated

    def test_apply_to_namespace_missing_attrs(self, minimal_config):
        """Test that missing attributes are skipped."""
        import argparse
        args = argparse.Namespace(batch_size=1)  # Only one attribute

        # Should not raise error
        minimal_config.apply_to_namespace(args)

        assert args.batch_size == 4
        # Other attributes not added since they don't exist in namespace

    def test_apply_to_namespace_none_values(self):
        """Test that None values are not applied."""
        import argparse
        config = TrainingConfig(
            batch_size=4,
            val_batch_size=8,
            patch_size=(96, 96, 96),
            num_workers=8,
            network_depth=5,
            feature_size=48,
            use_amp=True,
            num_gpus=1,
            vram_safety_margin=0.95,
            learning_rate=None  # None value
        )

        args = argparse.Namespace(lr=1e-3)

        config.apply_to_namespace(args)

        # lr should not be updated since learning_rate is None
        assert args.lr == 1e-3

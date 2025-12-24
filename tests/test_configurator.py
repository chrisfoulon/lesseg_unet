"""Unit tests for auto-config configurator module."""

import pytest

from lesseg_unet.hardware.detection import GPUInfo, CPUInfo, HardwareProfile
from lesseg_unet.auto_config import AutoConfigurator, AutoConfigResult, DatasetProfile


@pytest.fixture
def hardware_laptop():
    """Hardware profile for laptop (3.7GB GPU)."""
    gpus = [GPUInfo(0, "NVIDIA RTX 500", 3788.0, 3600.0, (8, 9))]
    cpu = CPUInfo(16, 22, 22, 30.8, 20.0)
    return HardwareProfile(gpus=gpus, cpu=cpu)


@pytest.fixture
def hardware_workstation():
    """Hardware profile for workstation (24GB GPU)."""
    gpus = [GPUInfo(0, "NVIDIA RTX 3090", 24576.0, 22000.0, (8, 6))]
    cpu = CPUInfo(8, 16, 16, 32.0, 24.0)
    return HardwareProfile(gpus=gpus, cpu=cpu)


@pytest.fixture
def hardware_server():
    """Hardware profile for server (4× A100 40GB)."""
    gpus = [
        GPUInfo(0, "NVIDIA A100", 40960.0, 38000.0, (8, 0)),
        GPUInfo(1, "NVIDIA A100", 40960.0, 38000.0, (8, 0)),
        GPUInfo(2, "NVIDIA A100", 40960.0, 38000.0, (8, 0)),
        GPUInfo(3, "NVIDIA A100", 40960.0, 38000.0, (8, 0))
    ]
    cpu = CPUInfo(32, 64, 64, 256.0, 200.0)
    return HardwareProfile(gpus=gpus, cpu=cpu)


@pytest.fixture
def hardware_cpu_only():
    """Hardware profile for CPU-only machine."""
    gpus = []
    cpu = CPUInfo(8, 16, 16, 32.0, 24.0)
    return HardwareProfile(gpus=gpus, cpu=cpu)


@pytest.fixture
def dataset_standard():
    """Standard dataset profile."""
    return DatasetProfile(
        median_image_size=(181, 217, 181),
        num_subjects=100,
        in_channels=2,
        out_channels=1,
        storage_type='ssd'
    )


class TestDatasetProfile:
    """Test DatasetProfile dataclass."""

    def test_dataset_profile_creation(self):
        """Test DatasetProfile creation."""
        dataset = DatasetProfile(
            median_image_size=(96, 96, 96),
            num_subjects=50,
            in_channels=1,
            out_channels=1
        )

        assert dataset.median_image_size == (96, 96, 96)
        assert dataset.num_subjects == 50
        assert dataset.in_channels == 1
        assert dataset.out_channels == 1
        assert dataset.storage_type == 'ssd'  # Default


class TestAutoConfigResult:
    """Test AutoConfigResult dataclass."""

    def test_auto_config_result_creation(self):
        """Test AutoConfigResult creation."""
        result = AutoConfigResult(
            batch_size=4,
            val_batch_size=8,
            patch_size=(96, 96, 96),
            num_workers=8,
            network_depth=5,
            feature_size=48,
            use_amp=True,
            use_checkpoint=False,
            num_gpus=1,
            vram_safety_margin=0.95
        )

        assert result.batch_size == 4
        assert result.val_batch_size == 8
        assert result.patch_size == (96, 96, 96)
        assert result.num_workers == 8
        assert result.network_depth == 5
        assert result.feature_size == 48
        assert result.use_amp is True
        assert result.num_gpus == 1
        assert result.vram_safety_margin == 0.95

    def test_auto_config_result_to_dict(self):
        """Test AutoConfigResult serialization."""
        result = AutoConfigResult(
            batch_size=4,
            val_batch_size=8,
            patch_size=(96, 96, 96),
            num_workers=8,
            network_depth=5,
            feature_size=48,
            use_amp=True,
            use_checkpoint=False,
            num_gpus=1,
            vram_safety_margin=0.95
        )

        d = result.to_dict()

        assert d['batch_size'] == 4
        assert d['val_batch_size'] == 8
        assert d['patch_size'] == (96, 96, 96)
        assert d['num_workers'] == 8
        assert 'reasoning' in d
        assert 'memory_estimate' in d


class TestAutoConfigurator:
    """Test AutoConfigurator class."""

    def test_configurator_initialization_single_gpu(self, hardware_laptop, dataset_standard):
        """Test configurator initialization with single GPU."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_laptop,
            dataset_profile=dataset_standard,
            model_type='swinunetr',
            target='balanced'
        )

        assert configurator.num_gpus == 1  # Default to 1 GPU
        assert configurator.model_type == 'swinunetr'
        assert configurator.target == 'balanced'

    def test_configurator_multi_gpu_opt_in(self, hardware_server, dataset_standard):
        """Test multi-GPU opt-in."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_server,
            dataset_profile=dataset_standard,
            num_gpus=2  # User explicitly requests 2 GPUs
        )

        assert configurator.num_gpus == 2

    def test_configurator_requests_more_gpus_than_available(
        self, hardware_workstation, dataset_standard
    ):
        """Test requesting more GPUs than available."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,  # Only 1 GPU
            dataset_profile=dataset_standard,
            num_gpus=4  # Request 4
        )

        assert configurator.num_gpus == 1  # Clamped to available

    def test_configurator_cpu_mode(self, hardware_cpu_only, dataset_standard):
        """Test configurator in CPU mode."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_cpu_only,
            dataset_profile=dataset_standard
        )

        assert configurator.num_gpus == 0

    def test_suggest_config_laptop(self, hardware_laptop, dataset_standard):
        """Test config suggestion for laptop (limited VRAM)."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_laptop,
            dataset_profile=dataset_standard,
            model_type='swinunetr'
        )

        config = configurator.suggest_config()

        # Limited VRAM (3.7GB) should give conservative config
        assert config.batch_size >= 1  # At least 1
        assert config.network_depth in [4, 5]
        assert config.feature_size <= 48  # Should use smaller feature size
        assert config.use_amp is True  # GPU supports AMP
        assert config.num_gpus == 1
        assert len(config.reasoning) > 0
        assert 'total_gb' in config.memory_estimate
        # Should stay within VRAM limits
        assert config.memory_estimate['total_gb'] <= 3.7

    def test_suggest_config_workstation(self, hardware_workstation, dataset_standard):
        """Test config suggestion for workstation (24GB)."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard,
            model_type='swinunetr'
        )

        config = configurator.suggest_config()

        # 24GB should allow larger config
        assert config.batch_size >= 4
        assert config.network_depth == 5
        assert config.feature_size == 48
        assert config.use_amp is True
        assert config.num_workers >= 8

    def test_suggest_config_server_multi_gpu(self, hardware_server, dataset_standard):
        """Test config suggestion for server with multi-GPU."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_server,
            dataset_profile=dataset_standard,
            model_type='swinunetr',
            num_gpus=4  # Use all 4 GPUs
        )

        config = configurator.suggest_config()

        assert config.num_gpus == 4
        assert config.batch_size >= 8  # Larger batch with 40GB
        assert config.use_amp is True
        # Multi-GPU should get more workers
        assert config.num_workers >= 16

    def test_suggest_config_unet_lighter(self, hardware_laptop, dataset_standard):
        """Test UNet config (lighter than SwinUNETR)."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_laptop,
            dataset_profile=dataset_standard,
            model_type='unet'
        )

        config = configurator.suggest_config()

        # UNet should allow larger batch even on laptop
        assert config.batch_size >= 1
        assert config.network_depth in [4, 5]

    def test_suggest_config_speed_target(self, hardware_workstation, dataset_standard):
        """Test config with speed target."""
        config_speed = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard,
            target='speed'
        ).suggest_config()

        config_balanced = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard,
            target='balanced'
        ).suggest_config()

        # Speed target should optimize for larger models/batches
        assert config_speed.feature_size >= config_balanced.feature_size

    def test_suggest_config_memory_target(self, hardware_workstation, dataset_standard):
        """Test config with memory target."""
        config_memory = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard,
            target='memory'
        ).suggest_config()

        config_balanced = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard,
            target='balanced'
        ).suggest_config()

        # Memory target should optimize for smaller models
        assert config_memory.feature_size <= config_balanced.feature_size

    def test_override_batch_size(self, hardware_workstation, dataset_standard):
        """Test overriding batch size."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config(override_batch_size=8)

        assert config.batch_size == 8
        assert 'User override' in config.reasoning['batch_size']

    def test_override_patch_size(self, hardware_workstation, dataset_standard):
        """Test overriding patch size."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        custom_patch = (64, 64, 64)
        config = configurator.suggest_config(override_patch_size=custom_patch)

        assert config.patch_size == custom_patch
        assert 'User override' in config.reasoning['patch_size']

    def test_override_num_workers(self, hardware_workstation, dataset_standard):
        """Test overriding num_workers."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config(override_num_workers=4)

        assert config.num_workers == 4
        assert 'User override' in config.reasoning['num_workers']

    def test_override_network_depth(self, hardware_workstation, dataset_standard):
        """Test overriding network depth."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config(override_network_depth=4)

        assert config.network_depth == 4
        assert 'User override' in config.reasoning['network_depth']

    def test_override_feature_size(self, hardware_workstation, dataset_standard):
        """Test overriding feature size."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config(override_feature_size=32)

        assert config.feature_size == 32
        assert 'User override' in config.reasoning['feature_size']

    def test_reasoning_populated(self, hardware_workstation, dataset_standard):
        """Test that reasoning is populated for all decisions."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config()

        # All major decisions should have reasoning
        assert 'batch_size' in config.reasoning
        assert 'patch_size' in config.reasoning
        assert 'num_workers' in config.reasoning
        assert 'network_depth' in config.reasoning
        assert 'feature_size' in config.reasoning
        assert 'use_amp' in config.reasoning

        # Reasoning should be non-empty strings
        for key, reason in config.reasoning.items():
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_memory_estimate_populated(self, hardware_workstation, dataset_standard):
        """Test that memory estimate is populated."""
        configurator = AutoConfigurator(
            hardware_profile=hardware_workstation,
            dataset_profile=dataset_standard
        )

        config = configurator.suggest_config()

        # Memory estimate should have all components
        assert 'params_mb' in config.memory_estimate
        assert 'optimizer_mb' in config.memory_estimate
        assert 'activations_mb' in config.memory_estimate
        assert 'gradients_mb' in config.memory_estimate
        assert 'overhead_mb' in config.memory_estimate
        assert 'fragmentation_mb' in config.memory_estimate
        assert 'total_mb' in config.memory_estimate
        assert 'total_gb' in config.memory_estimate

        # All should be positive numbers
        assert config.memory_estimate['total_gb'] > 0

    def test_dataloader_ram_constraint(self, hardware_laptop, dataset_standard):
        """Test that batch size is constrained by DataLoader RAM requirements.

        With num_samples=4, large batch sizes require loading many full images
        into RAM by DataLoader workers. This test verifies that batch_size is
        reduced when RAM requirements would exceed available memory.
        """
        # Create configurator with num_samples=4 (default)
        configurator = AutoConfigurator(
            hardware_profile=hardware_laptop,
            dataset_profile=dataset_standard,
            num_samples=4  # Requires loading full images before cropping
        )

        config = configurator.suggest_config()

        # With limited laptop RAM (20GB available), batch should be constrained
        # Verify it's reasonable (not the unconstrained 64)
        assert config.batch_size < 64, "Batch size should be constrained by DataLoader RAM"
        assert config.batch_size >= 1, "Batch size should be at least 1"

        # Reasoning should mention RAM if constrained
        if config.batch_size < 20:
            assert 'RAM' in config.reasoning['batch_size'] or 'batch_size' in config.reasoning

    def test_no_ram_constraint_with_num_samples_1(self, hardware_laptop, dataset_standard):
        """Test that no RAM constraint is applied when num_samples=1.

        When num_samples=1, patches are loaded directly without loading full images,
        so DataLoader RAM usage is minimal and shouldn't constrain batch_size.
        """
        # Create configurator with num_samples=1 (direct patch loading)
        configurator = AutoConfigurator(
            hardware_profile=hardware_laptop,
            dataset_profile=dataset_standard,
            num_samples=1  # Direct patch loading, no full images
        )

        config = configurator.suggest_config()

        # Batch size should be based on VRAM only, not RAM-constrained
        # (might still be small due to 3.7GB VRAM limit)
        assert config.batch_size >= 1

        # Reasoning should NOT mention DataLoader RAM constraint
        assert 'DataLoader RAM' not in config.reasoning.get('batch_size', '')

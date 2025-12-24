"""Unit tests for auto-config heuristics module."""

import pytest

from lesseg_unet.hardware.detection import GPUInfo, CPUInfo
from lesseg_unet.hardware.memory_model import SwinUNETRMemoryCalculator, UNetMemoryCalculator
from lesseg_unet.auto_config import heuristics


class TestSuggestBatchSize:
    """Test suggest_batch_size heuristic."""

    def test_suggest_batch_size_24gb(self):
        """Test batch size suggestion for 24GB VRAM."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        batch = heuristics.suggest_batch_size(calc, vram_gb=24.0, target_usage=0.95)

        assert batch >= 1
        assert batch <= 64
        # Should suggest reasonable batch for 24GB
        assert batch >= 4

    def test_suggest_batch_size_4gb(self):
        """Test batch size suggestion for 4GB VRAM (limited)."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1
        )

        batch = heuristics.suggest_batch_size(calc, vram_gb=4.0, target_usage=0.95)

        assert batch >= 1
        # Limited VRAM should give small batch
        assert batch <= 2

    def test_suggest_batch_size_respects_min(self):
        """Test that min_batch_size is respected."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1
        )

        batch = heuristics.suggest_batch_size(
            calc,
            vram_gb=1.0,  # Very small
            min_batch_size=2
        )

        assert batch >= 2


class TestSuggestPatchSize:
    """Test suggest_patch_size heuristic."""

    def test_patch_size_divisibility_depth_5(self):
        """Test patch size divisibility for depth 5 (divisible by 16)."""
        patch = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=5,
            vram_gb=24.0
        )

        h, w, d = patch
        assert h % 16 == 0
        assert w % 16 == 0
        assert d % 16 == 0

    def test_patch_size_divisibility_depth_4(self):
        """Test patch size divisibility for depth 4 (divisible by 8)."""
        patch = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=4,
            vram_gb=24.0
        )

        h, w, d = patch
        assert h % 8 == 0
        assert w % 8 == 0
        assert d % 8 == 0

    def test_patch_size_larger_for_speed(self):
        """Test that speed target gives larger patches."""
        patch_speed = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=5,
            vram_gb=24.0,
            target='speed'
        )

        patch_memory = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=5,
            vram_gb=24.0,
            target='memory'
        )

        assert patch_speed[0] >= patch_memory[0]

    def test_patch_size_scales_with_vram(self):
        """Test that patch size scales with VRAM."""
        patch_small = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=5,
            vram_gb=4.0
        )

        patch_large = heuristics.suggest_patch_size(
            median_image_size=(181, 217, 181),
            network_depth=5,
            vram_gb=40.0
        )

        assert patch_large[0] >= patch_small[0]


class TestSuggestNumWorkers:
    """Test suggest_num_workers heuristic."""

    def test_num_workers_single_gpu(self):
        """Test num_workers for single GPU."""
        workers = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=1,
            storage_type='ssd'
        )

        assert workers >= 1
        assert workers <= 64
        # Should use most cores for single GPU
        assert workers >= 30

    def test_num_workers_multi_gpu(self):
        """Test num_workers for multi-GPU."""
        workers = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=4,
            storage_type='ssd'
        )

        assert workers >= 4
        # Should scale with num_gpus
        assert workers <= 64

    def test_num_workers_network_storage(self):
        """Test num_workers with network storage (I/O bound)."""
        workers_network = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=1,
            storage_type='network'
        )

        workers_ssd = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=1,
            storage_type='ssd'
        )

        # Network storage should use fewer workers
        assert workers_network < workers_ssd

    def test_num_workers_memory_target(self):
        """Test num_workers with memory target."""
        workers_memory = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=1,
            target='memory'
        )

        workers_speed = heuristics.suggest_num_workers(
            available_cores=64,
            num_gpus=1,
            target='speed'
        )

        # Memory target should use fewer workers
        assert workers_memory < workers_speed


class TestSuggestNetworkDepth:
    """Test suggest_network_depth heuristic."""

    def test_network_depth_swinunetr_sufficient_vram(self):
        """Test depth 5 for SwinUNETR with sufficient VRAM."""
        depth = heuristics.suggest_network_depth(
            vram_gb=24.0,
            model_type='swinunetr'
        )

        assert depth == 5

    def test_network_depth_swinunetr_limited_vram(self):
        """Test depth 4 for SwinUNETR with limited VRAM."""
        depth = heuristics.suggest_network_depth(
            vram_gb=4.0,
            model_type='swinunetr'
        )

        assert depth == 4

    def test_network_depth_unet(self):
        """Test depth for UNet (lighter model)."""
        depth = heuristics.suggest_network_depth(
            vram_gb=4.0,
            model_type='unet'
        )

        assert depth == 5  # UNet lighter, can use depth 5 even with 4GB

    def test_network_depth_memory_target(self):
        """Test depth with memory target prefers depth 4."""
        depth = heuristics.suggest_network_depth(
            vram_gb=10.0,
            model_type='swinunetr',
            target='memory'
        )

        assert depth == 4


class TestSuggestFeatureSize:
    """Test suggest_feature_size heuristic."""

    def test_feature_size_swinunetr_large_vram(self):
        """Test feature size for SwinUNETR with large VRAM."""
        feature_size = heuristics.suggest_feature_size(
            vram_gb=24.0,
            model_type='swinunetr',
            network_depth=5
        )

        assert feature_size >= 48

    def test_feature_size_swinunetr_small_vram(self):
        """Test feature size for SwinUNETR with small VRAM."""
        feature_size = heuristics.suggest_feature_size(
            vram_gb=4.0,
            model_type='swinunetr',
            network_depth=4
        )

        assert feature_size <= 32

    def test_feature_size_unet(self):
        """Test feature size for UNet."""
        feature_size = heuristics.suggest_feature_size(
            vram_gb=8.0,
            model_type='unet',
            network_depth=5
        )

        assert feature_size in [16, 32, 48, 64]

    def test_feature_size_speed_target(self):
        """Test feature size with speed target (larger model)."""
        feature_speed = heuristics.suggest_feature_size(
            vram_gb=24.0,
            model_type='swinunetr',
            network_depth=5,
            target='speed'
        )

        feature_balanced = heuristics.suggest_feature_size(
            vram_gb=24.0,
            model_type='swinunetr',
            network_depth=5,
            target='balanced'
        )

        assert feature_speed >= feature_balanced


class TestAllocateResourcesProportionally:
    """Test allocate_resources_proportionally heuristic."""

    def test_proportional_allocation_half_gpus(self):
        """Test using half of GPUs gets half of cores."""
        cores = heuristics.allocate_resources_proportionally(
            total_gpus_on_system=8,
            num_gpus_requested=4,
            total_cpu_cores=64
        )

        # 4/8 GPUs = 50% of cores
        assert cores == 32

    def test_proportional_allocation_quarter_gpus(self):
        """Test using quarter of GPUs gets quarter of cores."""
        cores = heuristics.allocate_resources_proportionally(
            total_gpus_on_system=8,
            num_gpus_requested=2,
            total_cpu_cores=64
        )

        # 2/8 GPUs = 25% of cores
        assert cores == 16

    def test_proportional_allocation_all_gpus(self):
        """Test using all GPUs gets all cores."""
        cores = heuristics.allocate_resources_proportionally(
            total_gpus_on_system=4,
            num_gpus_requested=4,
            total_cpu_cores=64
        )

        assert cores == 64

    def test_proportional_allocation_minimum(self):
        """Test minimum allocation (at least num_gpus cores)."""
        cores = heuristics.allocate_resources_proportionally(
            total_gpus_on_system=100,
            num_gpus_requested=2,
            total_cpu_cores=8
        )

        # Would be 0.16 cores, but minimum is num_gpus
        assert cores >= 2


class TestSuggestUseAmp:
    """Test suggest_use_amp heuristic."""

    def test_use_amp_modern_gpu(self):
        """Test AMP enabled for modern GPU (compute >= 7.0)."""
        gpus = [
            GPUInfo(0, "NVIDIA A100", 40960.0, 38000.0, (8, 0))  # Compute 8.0
        ]

        use_amp = heuristics.suggest_use_amp(gpus, model_type='swinunetr')

        assert use_amp is True

    def test_use_amp_old_gpu(self):
        """Test AMP disabled for old GPU (compute < 7.0)."""
        gpus = [
            GPUInfo(0, "Old GPU", 4096.0, 3000.0, (6, 1))  # Compute 6.1
        ]

        use_amp = heuristics.suggest_use_amp(gpus, model_type='swinunetr')

        assert use_amp is False

    def test_use_amp_mixed_gpus(self):
        """Test AMP with mixed GPU capabilities (all must support)."""
        gpus = [
            GPUInfo(0, "NVIDIA A100", 40960.0, 38000.0, (8, 0)),  # Supports
            GPUInfo(1, "Old GPU", 4096.0, 3000.0, (6, 1))  # Doesn't support
        ]

        use_amp = heuristics.suggest_use_amp(gpus, model_type='swinunetr')

        # All must support, so False
        assert use_amp is False

    def test_use_amp_cpu_mode(self):
        """Test AMP disabled for CPU mode."""
        gpus = []

        use_amp = heuristics.suggest_use_amp(gpus, model_type='swinunetr')

        assert use_amp is False


class TestSuggestUseCheckpoint:
    """Test suggest_use_checkpoint heuristic."""

    def test_use_checkpoint_tiny_gpu_swinunetr(self):
        """Test checkpointing enabled for tiny GPU (3.65 GB) with SwinUNETR."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=3.65,
            model_type='swinunetr'
        )

        assert use_checkpoint is True

    def test_use_checkpoint_small_gpu_swinunetr(self):
        """Test checkpointing enabled for small GPU (4 GB) with SwinUNETR."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=4.0,
            model_type='swinunetr'
        )

        assert use_checkpoint is True

    def test_use_checkpoint_medium_gpu_balanced(self):
        """Test checkpointing disabled for medium GPU (7 GB) with balanced target."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=7.0,
            model_type='swinunetr',
            target='balanced'
        )

        assert use_checkpoint is False

    def test_use_checkpoint_medium_gpu_memory_target(self):
        """Test checkpointing enabled for medium GPU (7 GB) with memory target."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=7.0,
            model_type='swinunetr',
            target='memory'
        )

        assert use_checkpoint is True

    def test_use_checkpoint_large_gpu_swinunetr(self):
        """Test checkpointing disabled for large GPU (11 GB) with SwinUNETR."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=11.0,
            model_type='swinunetr'
        )

        assert use_checkpoint is False

    def test_use_checkpoint_unet_always_false(self):
        """Test checkpointing disabled for UNet (already memory-efficient)."""
        # Small GPU
        use_checkpoint_small = heuristics.suggest_use_checkpoint(
            vram_gb=3.65,
            model_type='unet'
        )
        assert use_checkpoint_small is False

        # Large GPU
        use_checkpoint_large = heuristics.suggest_use_checkpoint(
            vram_gb=24.0,
            model_type='unet'
        )
        assert use_checkpoint_large is False

    def test_use_checkpoint_speed_target(self):
        """Test checkpointing disabled for speed target (even medium GPU)."""
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=7.0,
            model_type='swinunetr',
            target='speed'
        )

        assert use_checkpoint is False

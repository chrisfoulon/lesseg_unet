"""Unit tests for memory model module."""

import pytest

from lesseg_unet.hardware.memory_model import (
    MemoryBreakdown,
    SwinUNETRMemoryCalculator,
    UNetMemoryCalculator
)


class TestMemoryBreakdown:
    """Test MemoryBreakdown dataclass."""

    def test_memory_breakdown_creation(self):
        """Test MemoryBreakdown creation."""
        breakdown = MemoryBreakdown(
            params_mb=520.0,
            optimizer_mb=1040.0,
            activations_mb=8000.0,
            gradients_mb=8000.0,
            overhead_mb=400.0,
            fragmentation_mb=1500.0,
            total_mb=19460.0
        )

        assert breakdown.params_mb == 520.0
        assert breakdown.optimizer_mb == 1040.0
        assert breakdown.activations_mb == 8000.0
        assert breakdown.gradients_mb == 8000.0
        assert breakdown.overhead_mb == 400.0
        assert breakdown.fragmentation_mb == 1500.0
        assert breakdown.total_mb == 19460.0

    def test_total_gb_property(self):
        """Test total_gb property calculation."""
        breakdown = MemoryBreakdown(
            params_mb=512.0,
            optimizer_mb=1024.0,
            activations_mb=2048.0,
            gradients_mb=2048.0,
            overhead_mb=400.0,
            fragmentation_mb=100.0,
            total_mb=6132.0
        )

        assert breakdown.total_gb == pytest.approx(6132.0 / 1024, rel=0.01)

    def test_to_dict(self):
        """Test to_dict serialization."""
        breakdown = MemoryBreakdown(
            params_mb=100.0,
            optimizer_mb=200.0,
            activations_mb=300.0,
            gradients_mb=300.0,
            overhead_mb=400.0,
            fragmentation_mb=50.0,
            total_mb=1350.0
        )

        result = breakdown.to_dict()

        assert result['params_mb'] == 100.0
        assert result['optimizer_mb'] == 200.0
        assert result['activations_mb'] == 300.0
        assert result['gradients_mb'] == 300.0
        assert result['overhead_mb'] == 400.0
        assert result['fragmentation_mb'] == 50.0
        assert result['total_mb'] == 1350.0
        assert 'total_gb' in result


class TestSwinUNETRMemoryCalculator:
    """Test SwinUNETRMemoryCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2],
            use_mixed_precision=True
        )

        assert calc.img_size == (96, 96, 96)
        assert calc.in_channels == 2
        assert calc.out_channels == 1
        assert calc.feature_size == 48
        assert calc.depths == (2, 2, 2, 2, 2)
        assert calc.use_mixed_precision is True
        assert calc.bytes_per_element == 3  # Mixed precision

    def test_calculator_no_mixed_precision(self):
        """Test calculator without mixed precision."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            use_mixed_precision=False
        )

        assert calc.bytes_per_element == 4  # FP32

    def test_invalid_depths(self):
        """Test calculator with invalid depths."""
        with pytest.raises(ValueError, match="depths must have 4 or 5 stages"):
            SwinUNETRMemoryCalculator(
                img_size=(96, 96, 96),
                in_channels=2,
                out_channels=1,
                depths=[2, 2, 2]  # Only 3 stages, invalid
            )

    def test_calculate_model_parameters(self):
        """Test model parameter calculation."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        params = calc.calculate_model_parameters()

        # SwinUNETR with these settings should have ~20M parameters
        assert params > 15.0  # At least 15M
        assert params < 30.0  # Less than 30M

    def test_calculate_model_parameters_depth_4(self):
        """Test model parameter calculation with depth 4."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2]  # 4 stages
        )

        params = calc.calculate_model_parameters()

        # Depth 4 should have fewer parameters than depth 5
        assert params > 3.0
        assert params < 10.0

    def test_calculate_activation_memory(self):
        """Test activation memory calculation."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        activations_mb = calc.calculate_activation_memory(batch_size=4)

        # Should be several GB for batch=4
        assert activations_mb > 1000.0  # At least 1GB
        assert activations_mb < 20000.0  # Less than 20GB

    def test_activation_memory_scales_with_batch(self):
        """Test that activation memory scales linearly with batch size."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        activations_batch_2 = calc.calculate_activation_memory(batch_size=2)
        activations_batch_4 = calc.calculate_activation_memory(batch_size=4)

        # Batch 4 should be approximately 2x batch 2
        ratio = activations_batch_4 / activations_batch_2
        assert ratio == pytest.approx(2.0, rel=0.1)

    def test_calculate_gradient_memory(self):
        """Test gradient memory calculation."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1
        )

        # Gradients should equal activations
        activations = calc.calculate_activation_memory(batch_size=4)
        gradients = calc.calculate_gradient_memory(batch_size=4)

        assert gradients == activations

    def test_calculate_optimizer_memory(self):
        """Test optimizer memory calculation."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        optimizer_mb = calc.calculate_optimizer_memory()

        # Optimizer should be 2x parameters (Adam state)
        params = calc.calculate_model_parameters() * 1e6
        expected_optimizer_mb = (params * 2 * 4) / (1024 ** 2)

        assert optimizer_mb == pytest.approx(expected_optimizer_mb, rel=0.01)

    def test_estimate_total_memory(self):
        """Test total memory estimation."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        memory = calc.estimate_total_memory(batch_size=4)

        # Check all components are present
        assert memory.params_mb > 0
        assert memory.optimizer_mb > 0
        assert memory.activations_mb > 0
        assert memory.gradients_mb > 0
        assert memory.overhead_mb == 400.0  # Constant
        assert memory.fragmentation_mb > 0

        # Total should be sum of all components
        expected_total = (
            memory.params_mb +
            memory.optimizer_mb +
            memory.activations_mb +
            memory.gradients_mb +
            memory.overhead_mb +
            memory.fragmentation_mb
        )
        assert memory.total_mb == pytest.approx(expected_total, rel=0.01)

    def test_find_max_batch_size(self):
        """Test finding maximum batch size."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        # For 24GB VRAM
        max_batch = calc.find_max_batch_size(vram_gb=24.0, target_usage=0.95)

        # Should find a reasonable batch size
        assert max_batch >= 1
        assert max_batch <= 64

        # Verify batch size actually fits
        memory = calc.estimate_total_memory(batch_size=max_batch)
        assert memory.total_mb <= 24.0 * 1024 * 0.95

    def test_find_max_batch_size_small_vram(self):
        """Test finding max batch size with limited VRAM."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        # For 4GB VRAM (very limited)
        max_batch = calc.find_max_batch_size(vram_gb=4.0, target_usage=0.95)

        # Should still find batch size of at least 1
        assert max_batch >= 1

    def test_find_max_batch_size_large_vram(self):
        """Test finding max batch size with large VRAM."""
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        # For 80GB VRAM (A100)
        max_batch = calc.find_max_batch_size(vram_gb=80.0, target_usage=0.95)

        # Should find larger batch size
        assert max_batch > 4


class TestUNetMemoryCalculator:
    """Test UNetMemoryCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = UNetMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            use_mixed_precision=True
        )

        assert calc.img_size == (96, 96, 96)
        assert calc.in_channels == 2
        assert calc.out_channels == 1
        assert calc.channels == (16, 32, 64, 128, 256)
        assert calc.use_mixed_precision is True

    def test_calculate_model_parameters(self):
        """Test model parameter calculation for UNet."""
        calc = UNetMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256)
        )

        params = calc.calculate_model_parameters()

        # UNet should have fewer parameters than SwinUNETR
        assert params > 1.0  # At least 1M
        assert params < 20.0  # Less than 20M

    def test_calculate_activation_memory(self):
        """Test activation memory calculation for UNet."""
        calc = UNetMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256)
        )

        activations_mb = calc.calculate_activation_memory(batch_size=4)

        assert activations_mb > 100.0  # At least 100MB
        assert activations_mb < 10000.0  # Less than 10GB

    def test_estimate_total_memory(self):
        """Test total memory estimation for UNet."""
        calc = UNetMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256)
        )

        memory = calc.estimate_total_memory(batch_size=4)

        # UNet should use less memory than SwinUNETR
        assert memory.total_gb > 0.5  # At least 0.5GB
        assert memory.total_gb < 20.0  # Less than 20GB

    def test_find_max_batch_size(self):
        """Test finding maximum batch size for UNet."""
        calc = UNetMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256)
        )

        # For 24GB VRAM
        max_batch = calc.find_max_batch_size(vram_gb=24.0, target_usage=0.95)

        # UNet should support larger batches than SwinUNETR
        assert max_batch >= 1
        assert max_batch <= 64

    def test_unet_uses_less_memory_than_swinunetr(self):
        """Test that UNet uses less memory than SwinUNETR."""
        # Same input configuration
        img_size = (96, 96, 96)
        in_channels = 2
        out_channels = 1
        batch_size = 4

        swin_calc = SwinUNETRMemoryCalculator(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            depths=[2, 2, 2, 2, 2]
        )

        unet_calc = UNetMemoryCalculator(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256)
        )

        swin_memory = swin_calc.estimate_total_memory(batch_size=batch_size)
        unet_memory = unet_calc.estimate_total_memory(batch_size=batch_size)

        # UNet should use significantly less memory
        assert unet_memory.total_mb < swin_memory.total_mb

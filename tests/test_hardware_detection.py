"""Unit tests for hardware detection module."""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from lesseg_unet.hardware.detection import (
    GPUInfo,
    CPUInfo,
    HardwareProfile,
    detect_gpus,
    detect_cpu,
    get_hardware_profile
)


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test GPUInfo creation and derived fields."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA A100",
            total_memory_mb=40960.0,
            free_memory_mb=38000.0,
            compute_capability=(8, 0)
        )

        assert gpu.index == 0
        assert gpu.name == "NVIDIA A100"
        assert gpu.total_memory_mb == 40960.0
        assert gpu.free_memory_mb == 38000.0
        assert gpu.compute_capability == (8, 0)
        assert gpu.supports_mixed_precision is True  # Compute >= 7.0

    def test_gpu_info_old_compute_capability(self):
        """Test GPUInfo with old compute capability (no mixed precision)."""
        gpu = GPUInfo(
            index=0,
            name="Old GPU",
            total_memory_mb=4096.0,
            free_memory_mb=3000.0,
            compute_capability=(6, 1)
        )

        assert gpu.supports_mixed_precision is False  # Compute < 7.0

    def test_gpu_info_to_dict(self):
        """Test GPUInfo serialization to dict."""
        gpu = GPUInfo(
            index=1,
            name="NVIDIA RTX 3090",
            total_memory_mb=24576.0,
            free_memory_mb=20000.0,
            compute_capability=(8, 6)
        )

        result = gpu.to_dict()

        assert result['index'] == 1
        assert result['name'] == "NVIDIA RTX 3090"
        assert result['total_memory_mb'] == 24576.0
        assert result['free_memory_mb'] == 20000.0
        assert result['compute_capability'] == (8, 6)
        assert result['supports_mixed_precision'] is True


class TestCPUInfo:
    """Test CPUInfo dataclass."""

    def test_cpu_info_creation(self):
        """Test CPUInfo creation."""
        cpu = CPUInfo(
            physical_cores=32,
            logical_cores=64,
            available_cores=64,
            total_ram_gb=256.0,
            available_ram_gb=200.0
        )

        assert cpu.physical_cores == 32
        assert cpu.logical_cores == 64
        assert cpu.available_cores == 64
        assert cpu.total_ram_gb == 256.0
        assert cpu.available_ram_gb == 200.0

    def test_cpu_info_to_dict(self):
        """Test CPUInfo serialization to dict."""
        cpu = CPUInfo(
            physical_cores=8,
            logical_cores=16,
            available_cores=12,
            total_ram_gb=32.0,
            available_ram_gb=24.0
        )

        result = cpu.to_dict()

        assert result['physical_cores'] == 8
        assert result['logical_cores'] == 16
        assert result['available_cores'] == 12
        assert result['total_ram_gb'] == 32.0
        assert result['available_ram_gb'] == 24.0


class TestHardwareProfile:
    """Test HardwareProfile dataclass."""

    def test_hardware_profile_with_gpus(self):
        """Test HardwareProfile with GPUs."""
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_mb=8192.0,
            free_memory_mb=6000.0,
            compute_capability=(7, 5)
        )

        cpu = CPUInfo(
            physical_cores=8,
            logical_cores=16,
            available_cores=16,
            total_ram_gb=32.0,
            available_ram_gb=24.0
        )

        profile = HardwareProfile(gpus=[gpu], cpu=cpu)

        assert len(profile.gpus) == 1
        assert profile.device_type == 'cuda'
        assert profile.cpu == cpu

    def test_hardware_profile_cpu_only(self):
        """Test HardwareProfile without GPUs (CPU mode)."""
        cpu = CPUInfo(
            physical_cores=4,
            logical_cores=8,
            available_cores=8,
            total_ram_gb=16.0,
            available_ram_gb=12.0
        )

        profile = HardwareProfile(gpus=[], cpu=cpu)

        assert len(profile.gpus) == 0
        assert profile.device_type == 'cpu'

    def test_get_gpu_valid_index(self):
        """Test get_gpu with valid index."""
        gpus = [
            GPUInfo(0, "GPU 0", 8192.0, 6000.0, (7, 5)),
            GPUInfo(1, "GPU 1", 8192.0, 6000.0, (7, 5))
        ]

        cpu = CPUInfo(8, 16, 16, 32.0, 24.0)
        profile = HardwareProfile(gpus=gpus, cpu=cpu)

        gpu0 = profile.get_gpu(0)
        gpu1 = profile.get_gpu(1)

        assert gpu0 is not None
        assert gpu0.index == 0
        assert gpu1 is not None
        assert gpu1.index == 1

    def test_get_gpu_invalid_index(self):
        """Test get_gpu with invalid index."""
        gpus = [GPUInfo(0, "GPU 0", 8192.0, 6000.0, (7, 5))]
        cpu = CPUInfo(8, 16, 16, 32.0, 24.0)
        profile = HardwareProfile(gpus=gpus, cpu=cpu)

        assert profile.get_gpu(5) is None
        assert profile.get_gpu(-1) is None

    def test_hardware_profile_to_dict(self):
        """Test HardwareProfile serialization to dict."""
        gpu = GPUInfo(0, "Test GPU", 8192.0, 6000.0, (7, 5))
        cpu = CPUInfo(8, 16, 16, 32.0, 24.0)
        profile = HardwareProfile(gpus=[gpu], cpu=cpu)

        result = profile.to_dict()

        assert 'gpus' in result
        assert len(result['gpus']) == 1
        assert 'cpu' in result
        assert result['device_type'] == 'cuda'


class TestDetectGPUs:
    """Test detect_gpus function."""

    @patch('torch.cuda.is_available')
    def test_detect_gpus_no_cuda(self, mock_cuda_available):
        """Test detect_gpus when CUDA not available."""
        mock_cuda_available.return_value = False

        gpus = detect_gpus()

        assert gpus == []

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.mem_get_info')
    @patch('torch.cuda.set_device')
    def test_detect_gpus_single_gpu(
        self,
        mock_set_device,
        mock_mem_info,
        mock_props,
        mock_device_count,
        mock_cuda_available
    ):
        """Test detect_gpus with single GPU."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1

        # Mock device properties
        mock_device_props = Mock()
        mock_device_props.name = "NVIDIA RTX 3090"
        mock_device_props.total_memory = 24 * 1024 ** 3  # 24 GB
        mock_device_props.major = 8
        mock_device_props.minor = 6

        mock_props.return_value = mock_device_props
        mock_mem_info.return_value = (20 * 1024 ** 3, 24 * 1024 ** 3)  # (free, total)

        gpus = detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA RTX 3090"
        assert gpus[0].compute_capability == (8, 6)
        assert gpus[0].supports_mixed_precision is True

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.mem_get_info')
    @patch('torch.cuda.set_device')
    def test_detect_gpus_multiple_gpus(
        self,
        mock_set_device,
        mock_mem_info,
        mock_props,
        mock_device_count,
        mock_cuda_available
    ):
        """Test detect_gpus with multiple GPUs."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        # Mock device properties for each GPU
        def mock_props_func(i):
            props = Mock()
            props.name = f"GPU {i}"
            props.total_memory = 40 * 1024 ** 3
            props.major = 8
            props.minor = 0
            return props

        mock_props.side_effect = mock_props_func
        mock_mem_info.return_value = (35 * 1024 ** 3, 40 * 1024 ** 3)

        gpus = detect_gpus()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[1].index == 1


class TestDetectCPU:
    """Test detect_cpu function."""

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_cpu_no_slurm(self, mock_vmem, mock_cpu_count):
        """Test detect_cpu without SLURM environment."""
        mock_cpu_count.side_effect = [8, 16]  # physical, logical

        mock_mem = Mock()
        mock_mem.total = 32 * 1024 ** 3
        mock_mem.available = 24 * 1024 ** 3
        mock_vmem.return_value = mock_mem

        # Ensure no SLURM env var
        with patch.dict(os.environ, {}, clear=True):
            cpu = detect_cpu()

        assert cpu.physical_cores == 8
        assert cpu.logical_cores == 16
        assert cpu.available_cores == 16  # No SLURM, use all logical
        assert cpu.total_ram_gb == pytest.approx(32.0, rel=0.01)
        assert cpu.available_ram_gb == pytest.approx(24.0, rel=0.01)

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_cpu_with_slurm(self, mock_vmem, mock_cpu_count):
        """Test detect_cpu with SLURM_CPUS_PER_TASK."""
        mock_cpu_count.side_effect = [16, 32]  # physical, logical

        mock_mem = Mock()
        mock_mem.total = 128 * 1024 ** 3
        mock_mem.available = 100 * 1024 ** 3
        mock_vmem.return_value = mock_mem

        # Set SLURM env var
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8'}):
            cpu = detect_cpu()

        assert cpu.available_cores == 8  # SLURM limit

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_cpu_invalid_slurm(self, mock_vmem, mock_cpu_count):
        """Test detect_cpu with invalid SLURM value."""
        mock_cpu_count.side_effect = [8, 16]

        mock_mem = Mock()
        mock_mem.total = 32 * 1024 ** 3
        mock_mem.available = 24 * 1024 ** 3
        mock_vmem.return_value = mock_mem

        # Set invalid SLURM value
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': 'invalid'}):
            cpu = detect_cpu()

        assert cpu.available_cores == 16  # Fall back to logical cores


class TestGetHardwareProfile:
    """Test get_hardware_profile function."""

    @patch('lesseg_unet.hardware.detection.detect_gpus')
    @patch('lesseg_unet.hardware.detection.detect_cpu')
    def test_get_hardware_profile_with_gpu(self, mock_detect_cpu, mock_detect_gpus):
        """Test get_hardware_profile with GPU."""
        mock_gpus = [GPUInfo(0, "Test GPU", 8192.0, 6000.0, (7, 5))]
        mock_cpu = CPUInfo(8, 16, 16, 32.0, 24.0)

        mock_detect_gpus.return_value = mock_gpus
        mock_detect_cpu.return_value = mock_cpu

        profile = get_hardware_profile()

        assert len(profile.gpus) == 1
        assert profile.device_type == 'cuda'
        assert profile.cpu == mock_cpu

    @patch('lesseg_unet.hardware.detection.detect_gpus')
    @patch('lesseg_unet.hardware.detection.detect_cpu')
    def test_get_hardware_profile_cpu_only(self, mock_detect_cpu, mock_detect_gpus):
        """Test get_hardware_profile without GPU."""
        mock_detect_gpus.return_value = []
        mock_cpu = CPUInfo(4, 8, 8, 16.0, 12.0)
        mock_detect_cpu.return_value = mock_cpu

        profile = get_hardware_profile()

        assert len(profile.gpus) == 0
        assert profile.device_type == 'cpu'
        assert profile.cpu == mock_cpu

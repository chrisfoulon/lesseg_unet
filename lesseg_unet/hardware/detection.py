"""Hardware detection for auto-configuration.

Detects available GPUs, CPU cores, and RAM. Respects SLURM environment limits.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch


logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU.

    Attributes
    ----------
    index : int
        GPU index (0-based).
    name : str
        GPU name/model (e.g., "NVIDIA A100").
    total_memory_mb : float
        Total VRAM in MB.
    free_memory_mb : float
        Currently free VRAM in MB.
    compute_capability : tuple[int, int]
        Compute capability version (major, minor).
    supports_mixed_precision : bool
        Whether GPU supports mixed precision (compute >= 7.0).
    """

    index: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    compute_capability: tuple[int, int]
    supports_mixed_precision: bool = field(init=False)

    def __post_init__(self):
        """Calculate derived fields."""
        major, _ = self.compute_capability
        self.supports_mixed_precision = major >= 7

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            GPU info as dictionary.
        """
        return {
            'index': self.index,
            'name': self.name,
            'total_memory_mb': self.total_memory_mb,
            'free_memory_mb': self.free_memory_mb,
            'compute_capability': self.compute_capability,
            'supports_mixed_precision': self.supports_mixed_precision
        }


@dataclass
class CPUInfo:
    """Information about CPU resources.

    Attributes
    ----------
    physical_cores : int
        Number of physical CPU cores.
    logical_cores : int
        Number of logical CPU cores (with hyperthreading).
    available_cores : int
        Cores available to this process (respects SLURM_CPUS_PER_TASK).
    total_ram_gb : float
        Total system RAM in GB.
    available_ram_gb : float
        Currently available RAM in GB.
    """

    physical_cores: int
    logical_cores: int
    available_cores: int
    total_ram_gb: float
    available_ram_gb: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            CPU info as dictionary.
        """
        return {
            'physical_cores': self.physical_cores,
            'logical_cores': self.logical_cores,
            'available_cores': self.available_cores,
            'total_ram_gb': self.total_ram_gb,
            'available_ram_gb': self.available_ram_gb
        }


@dataclass
class HardwareProfile:
    """Complete hardware profile for auto-configuration.

    Attributes
    ----------
    gpus : list[GPUInfo]
        List of available GPUs (empty if CPU-only).
    cpu : CPUInfo
        CPU information.
    device_type : str
        'cuda' or 'cpu'.
    """

    gpus: list[GPUInfo]
    cpu: CPUInfo
    device_type: str = field(init=False)

    def __post_init__(self):
        """Calculate derived fields."""
        self.device_type = 'cuda' if self.gpus else 'cpu'

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Hardware profile as dictionary.
        """
        return {
            'gpus': [gpu.to_dict() for gpu in self.gpus],
            'cpu': self.cpu.to_dict(),
            'device_type': self.device_type
        }

    def get_gpu(self, index: int) -> Optional[GPUInfo]:
        """Get GPU by index.

        Parameters
        ----------
        index : int
            GPU index.

        Returns
        -------
        GPUInfo or None
            GPU info if index valid, None otherwise.
        """
        if 0 <= index < len(self.gpus):
            return self.gpus[index]
        return None


def detect_gpus() -> list[GPUInfo]:
    """Detect available GPUs and their capabilities.

    Returns
    -------
    list[GPUInfo]
        List of detected GPUs. Empty list if no CUDA available.

    Examples
    --------
    >>> gpus = detect_gpus()
    >>> if gpus:
    ...     print(f"Found {len(gpus)} GPU(s)")
    ...     for gpu in gpus:
    ...         print(f"  GPU {gpu.index}: {gpu.name} ({gpu.total_memory_mb / 1024:.1f} GB)")
    """
    gpus = []

    if not torch.cuda.is_available():
        logger.info("CUDA not available. Running in CPU mode.")
        return gpus

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s)")

    for i in range(num_gpus):
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(i)

            # Get memory info
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_memory, _ = torch.cuda.mem_get_info(i)

            gpu = GPUInfo(
                index=i,
                name=props.name,
                total_memory_mb=total_memory / (1024 ** 2),
                free_memory_mb=free_memory / (1024 ** 2),
                compute_capability=(props.major, props.minor)
            )

            gpus.append(gpu)

            logger.debug(
                f"GPU {i}: {gpu.name}, "
                f"VRAM: {gpu.total_memory_mb / 1024:.1f} GB, "
                f"Free: {gpu.free_memory_mb / 1024:.1f} GB, "
                f"Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}, "
                f"Mixed Precision: {gpu.supports_mixed_precision}"
            )

        except Exception as e:
            logger.warning(f"Error detecting GPU {i}: {e}")

    return gpus


def detect_cpu() -> CPUInfo:
    """Detect CPU resources, respecting SLURM limits.

    Returns
    -------
    CPUInfo
        CPU information including available cores and RAM.

    Notes
    -----
    Respects SLURM_CPUS_PER_TASK environment variable for HPC environments.

    Examples
    --------
    >>> cpu = detect_cpu()
    >>> print(f"Available cores: {cpu.available_cores}/{cpu.logical_cores}")
    >>> print(f"Available RAM: {cpu.available_ram_gb:.1f} GB")
    """
    # Get CPU core counts
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    # Respect SLURM limits if present
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            available_cores = int(slurm_cpus)
            logger.info(f"Using SLURM_CPUS_PER_TASK: {available_cores} cores")
        except ValueError:
            logger.warning(f"Invalid SLURM_CPUS_PER_TASK value: {slurm_cpus}")
            available_cores = logical_cores
    else:
        available_cores = logical_cores

    # Get RAM info
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    available_ram_gb = mem.available / (1024 ** 3)

    cpu = CPUInfo(
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        available_cores=available_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb
    )

    logger.debug(
        f"CPU: {physical_cores} physical / {logical_cores} logical cores, "
        f"Available: {available_cores}, "
        f"RAM: {total_ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available"
    )

    return cpu


def get_hardware_profile() -> HardwareProfile:
    """Get complete hardware profile.

    Main entry point for hardware detection. Combines GPU and CPU detection.

    Returns
    -------
    HardwareProfile
        Complete hardware profile including GPUs and CPU.

    Examples
    --------
    >>> hw = get_hardware_profile()
    >>> print(f"Device: {hw.device_type}")
    >>> print(f"GPUs: {len(hw.gpus)}")
    >>> print(f"CPU cores: {hw.cpu.available_cores}")
    """
    logger.info("Detecting hardware...")

    gpus = detect_gpus()
    cpu = detect_cpu()

    profile = HardwareProfile(gpus=gpus, cpu=cpu)

    logger.info(
        f"Hardware profile: {len(profile.gpus)} GPU(s), "
        f"{profile.cpu.available_cores} CPU cores, "
        f"{profile.cpu.total_ram_gb:.1f} GB RAM"
    )

    return profile

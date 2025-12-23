"""Hardware detection and resource management for lesseg_unet."""

from lesseg_unet.hardware.detection import (
    GPUInfo,
    CPUInfo,
    HardwareProfile,
    detect_gpus,
    detect_cpu,
    get_hardware_profile
)

from lesseg_unet.hardware.memory_model import (
    MemoryBreakdown,
    SwinUNETRMemoryCalculator,
    UNetMemoryCalculator
)

from lesseg_unet.hardware.validator import (
    ValidationResult,
    validate_config,
    find_safe_config
)

__all__ = [
    'GPUInfo',
    'CPUInfo',
    'HardwareProfile',
    'detect_gpus',
    'detect_cpu',
    'get_hardware_profile',
    'MemoryBreakdown',
    'SwinUNETRMemoryCalculator',
    'UNetMemoryCalculator',
    'ValidationResult',
    'validate_config',
    'find_safe_config'
]

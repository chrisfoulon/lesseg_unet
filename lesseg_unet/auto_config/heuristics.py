"""Heuristics for auto-configuration.

Individual heuristic functions for suggesting optimal training parameters
based on hardware capabilities and dataset characteristics.
"""

import logging
import math
from typing import Optional, Literal

from lesseg_unet.hardware.detection import HardwareProfile
from lesseg_unet.hardware.memory_model import SwinUNETRMemoryCalculator, UNetMemoryCalculator


logger = logging.getLogger(__name__)


def suggest_batch_size(
    memory_calculator: SwinUNETRMemoryCalculator | UNetMemoryCalculator,
    vram_gb: float,
    target_usage: float = 0.95,
    min_batch_size: int = 1,
    max_batch_size: int = 64
) -> int:
    """Suggest optimal batch size based on available VRAM.

    Uses binary search on theoretical memory model to find maximum
    batch size that fits within target VRAM usage.

    Parameters
    ----------
    memory_calculator : SwinUNETRMemoryCalculator or UNetMemoryCalculator
        Memory calculator for the model architecture.
    vram_gb : float
        Available VRAM in GB.
    target_usage : float
        Target VRAM usage (0.0-1.0). Default: 0.95 (95%).
    min_batch_size : int
        Minimum batch size. Default: 1.
    max_batch_size : int
        Maximum batch size to consider. Default: 64.

    Returns
    -------
    int
        Suggested batch size.

    Examples
    --------
    >>> from lesseg_unet.hardware.memory_model import SwinUNETRMemoryCalculator
    >>> calc = SwinUNETRMemoryCalculator((96,96,96), 2, 1)
    >>> batch = suggest_batch_size(calc, vram_gb=24.0)
    >>> print(f"Suggested batch size: {batch}")
    """
    batch_size = memory_calculator.find_max_batch_size(
        vram_gb=vram_gb,
        target_usage=target_usage,
        max_batch=max_batch_size
    )

    # Ensure within bounds
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))

    logger.info(
        f"Suggested batch_size={batch_size} for {vram_gb:.1f}GB VRAM "
        f"({target_usage*100:.0f}% target)"
    )

    return batch_size


def suggest_patch_size(
    median_image_size: tuple[int, int, int],
    network_depth: int,
    vram_gb: float,
    target: Literal['speed', 'memory', 'balanced'] = 'balanced'
) -> tuple[int, int, int]:
    """Suggest optimal patch size based on image size and network depth.

    Parameters
    ----------
    median_image_size : tuple[int, int, int]
        Median image size in dataset (H, W, D).
    network_depth : int
        Network depth (4 or 5).
    vram_gb : float
        Available VRAM in GB.
    target : {'speed', 'memory', 'balanced'}
        Optimization target. Default: 'balanced'.
        - 'speed': Larger patches (fewer per image, faster)
        - 'memory': Smaller patches (more per image, memory efficient)
        - 'balanced': Middle ground

    Returns
    -------
    tuple[int, int, int]
        Suggested patch size (H, W, D).

    Notes
    -----
    Patch sizes must be divisible by 2^(network_depth-1):
    - Depth 4: divisible by 8 (2^3)
    - Depth 5: divisible by 16 (2^4)

    Examples
    --------
    >>> patch = suggest_patch_size((181, 217, 181), network_depth=5, vram_gb=24.0)
    >>> print(f"Suggested patch: {patch}")
    """
    # Divisibility requirement
    divisor = 2 ** (network_depth - 1)  # Depth 4→8, Depth 5→16

    # Base patch sizes by target
    if target == 'speed':
        # Larger patches (fewer per image)
        if vram_gb >= 24.0:
            base_size = 128
        elif vram_gb >= 16.0:
            base_size = 112
        elif vram_gb >= 8.0:
            base_size = 96
        else:
            base_size = 80
    elif target == 'memory':
        # Smaller patches (more per image, memory efficient)
        if vram_gb >= 24.0:
            base_size = 96
        elif vram_gb >= 16.0:
            base_size = 80
        elif vram_gb >= 8.0:
            base_size = 64
        else:
            base_size = 48
    else:  # balanced
        if vram_gb >= 24.0:
            base_size = 96
        elif vram_gb >= 16.0:
            base_size = 96
        elif vram_gb >= 8.0:
            base_size = 80
        else:
            base_size = 64

    # Ensure divisibility
    base_size = (base_size // divisor) * divisor

    # Clamp to reasonable range
    min_size = divisor * 2  # At least 2× divisor
    max_size = divisor * 16  # At most 16× divisor
    base_size = max(min_size, min(base_size, max_size))

    # For now, use isotropic patches
    # Future: could adapt to image aspect ratio
    patch_size = (base_size, base_size, base_size)

    logger.info(
        f"Suggested patch_size={patch_size} for median_image={median_image_size}, "
        f"depth={network_depth}, target={target}"
    )

    return patch_size


def suggest_num_workers(
    available_cores: int,
    num_gpus: int,
    storage_type: Literal['ssd', 'hdd', 'network'] = 'ssd',
    target: Literal['speed', 'memory', 'balanced'] = 'balanced'
) -> int:
    """Suggest optimal number of dataloader workers.

    Parameters
    ----------
    available_cores : int
        Available CPU cores.
    num_gpus : int
        Number of GPUs being used.
    storage_type : {'ssd', 'hdd', 'network'}
        Storage type. Default: 'ssd'.
        - 'ssd': Fast local SSD (can use more workers)
        - 'hdd': Slower HDD (fewer workers)
        - 'network': Network storage (even fewer workers)
    target : {'speed', 'memory', 'balanced'}
        Optimization target. Default: 'balanced'.

    Returns
    -------
    int
        Suggested number of workers.

    Notes
    -----
    Allocates CPU resources proportionally to GPU usage:
    - Y GPUs out of X total → use Y/X of available cores
    - Accounts for storage bottlenecks
    - Leaves some cores for system operations

    Examples
    --------
    >>> workers = suggest_num_workers(available_cores=64, num_gpus=2)
    >>> print(f"Suggested workers: {workers}")
    """
    # Reserve some cores for system (at least 2, or 10%)
    reserved = max(2, int(available_cores * 0.1))
    usable_cores = available_cores - reserved

    # Proportional allocation for multi-GPU
    # If using Y out of X GPUs, use Y/X of cores (user might share HPC)
    # For single GPU, assume X=total available GPUs on system
    # For simplicity, use num_gpus directly (user controls this via --num_gpus)
    cores_per_gpu = max(1, usable_cores // max(1, num_gpus))

    # Adjust for storage type
    if storage_type == 'network':
        # Network storage: I/O bound, fewer workers better
        workers_per_gpu = min(cores_per_gpu, 4)
    elif storage_type == 'hdd':
        # HDD: Sequential reads better, moderate workers
        workers_per_gpu = min(cores_per_gpu, 8)
    else:  # ssd
        # SSD: Can handle more parallel I/O
        workers_per_gpu = cores_per_gpu

    # Adjust for target
    if target == 'speed':
        # Use more workers for faster data loading
        workers_per_gpu = min(int(workers_per_gpu * 1.5), cores_per_gpu)
    elif target == 'memory':
        # Fewer workers to reduce memory overhead
        workers_per_gpu = max(1, workers_per_gpu // 2)

    # Total workers across all GPUs
    total_workers = workers_per_gpu * num_gpus

    # Clamp to reasonable range
    total_workers = max(1, min(total_workers, usable_cores))

    logger.info(
        f"Suggested num_workers={total_workers} "
        f"({workers_per_gpu}/GPU × {num_gpus} GPUs) "
        f"from {available_cores} cores, storage={storage_type}"
    )

    return total_workers


def suggest_network_depth(
    vram_gb: float,
    model_type: Literal['swinunetr', 'unet'],
    target: Literal['speed', 'memory', 'balanced'] = 'balanced'
) -> int:
    """Suggest network depth based on available VRAM.

    Parameters
    ----------
    vram_gb : float
        Available VRAM in GB.
    model_type : {'swinunetr', 'unet'}
        Model architecture type.
    target : {'speed', 'memory', 'balanced'}
        Optimization target. Default: 'balanced'.

    Returns
    -------
    int
        Suggested network depth (4 or 5).

    Notes
    -----
    Depth affects:
    - Model capacity (depth 5 has more parameters)
    - Memory usage (depth 5 uses more VRAM)
    - Patch size granularity (depth 5 requires divisible by 16, depth 4 by 8)

    For SwinUNETR:
    - Depth 5: Standard, ~20M params, requires ≥8GB VRAM
    - Depth 4: Lighter, ~5M params, works with ≥4GB VRAM

    Examples
    --------
    >>> depth = suggest_network_depth(vram_gb=16.0, model_type='swinunetr')
    >>> print(f"Suggested depth: {depth}")
    """
    if model_type == 'swinunetr':
        # SwinUNETR: depth 5 is standard but memory-hungry
        if target == 'memory':
            # Prefer depth 4 for memory efficiency
            depth = 4 if vram_gb < 12.0 else 5
        elif target == 'speed':
            # Prefer depth 5 for quality (if VRAM allows)
            depth = 5 if vram_gb >= 8.0 else 4
        else:  # balanced
            # Depth 5 if we have enough VRAM
            depth = 5 if vram_gb >= 8.0 else 4
    else:  # unet
        # UNet: depth 5 is standard and lighter than SwinUNETR
        depth = 5 if vram_gb >= 4.0 else 4

    logger.info(
        f"Suggested network_depth={depth} for {model_type} "
        f"with {vram_gb:.1f}GB VRAM, target={target}"
    )

    return depth


def suggest_feature_size(
    vram_gb: float,
    model_type: Literal['swinunetr', 'unet'],
    network_depth: int,
    target: Literal['speed', 'memory', 'balanced'] = 'balanced'
) -> int:
    """Suggest model feature size based on available VRAM.

    Parameters
    ----------
    vram_gb : float
        Available VRAM in GB.
    model_type : {'swinunetr', 'unet'}
        Model architecture type.
    network_depth : int
        Network depth (4 or 5).
    target : {'speed', 'memory', 'balanced'}
        Optimization target. Default: 'balanced'.

    Returns
    -------
    int
        Suggested feature size.

    Notes
    -----
    For SwinUNETR:
    - feature_size controls base embedding dimension
    - Common values: 24, 32, 48, 64, 96
    - Higher = more capacity but more memory

    For UNet:
    - Feature size controls initial channel count
    - Common values: 16, 32, 48, 64

    Examples
    --------
    >>> feature_size = suggest_feature_size(
    ...     vram_gb=24.0,
    ...     model_type='swinunetr',
    ...     network_depth=5
    ... )
    >>> print(f"Suggested feature size: {feature_size}")
    """
    if model_type == 'swinunetr':
        # SwinUNETR feature sizes
        if target == 'memory':
            if vram_gb >= 16.0:
                feature_size = 48
            elif vram_gb >= 8.0:
                feature_size = 32
            else:
                feature_size = 24
        elif target == 'speed':
            # Larger model for quality
            if vram_gb >= 24.0:
                feature_size = 64
            elif vram_gb >= 16.0:
                feature_size = 48
            elif vram_gb >= 8.0:
                feature_size = 32
            else:
                feature_size = 24
        else:  # balanced
            if vram_gb >= 16.0:
                feature_size = 48
            elif vram_gb >= 8.0:
                feature_size = 32
            else:
                feature_size = 24

        # Reduce for depth 4 (already lighter)
        if network_depth == 4 and feature_size > 32:
            feature_size = max(32, feature_size - 16)

    else:  # unet
        # UNet feature sizes (initial channels)
        if target == 'memory':
            feature_size = 16 if vram_gb < 8.0 else 32
        elif target == 'speed':
            feature_size = 32 if vram_gb >= 8.0 else 16
        else:  # balanced
            feature_size = 32

    logger.info(
        f"Suggested feature_size={feature_size} for {model_type}, "
        f"depth={network_depth}, {vram_gb:.1f}GB VRAM, target={target}"
    )

    return feature_size


def allocate_resources_proportionally(
    total_gpus_on_system: int,
    num_gpus_requested: int,
    total_cpu_cores: int
) -> int:
    """Allocate CPU resources proportionally to GPU usage.

    For multi-GPU opt-in: if using Y out of X GPUs, use Y/X of CPU cores.
    This is critical for shared HPC environments.

    Parameters
    ----------
    total_gpus_on_system : int
        Total GPUs available on the system.
    num_gpus_requested : int
        Number of GPUs user wants to use.
    total_cpu_cores : int
        Total CPU cores available.

    Returns
    -------
    int
        Allocated CPU cores.

    Examples
    --------
    >>> # Using 2 out of 8 GPUs → use 25% of CPU cores
    >>> cores = allocate_resources_proportionally(
    ...     total_gpus_on_system=8,
    ...     num_gpus_requested=2,
    ...     total_cpu_cores=64
    ... )
    >>> print(f"Allocated cores: {cores}")  # 16
    """
    if num_gpus_requested >= total_gpus_on_system or total_gpus_on_system == 0:
        # Using all GPUs or can't determine → use all cores
        return total_cpu_cores

    # Proportional allocation: Y/X GPUs → Y/X cores
    proportion = num_gpus_requested / total_gpus_on_system
    allocated_cores = int(total_cpu_cores * proportion)

    # Ensure at least some cores
    allocated_cores = max(num_gpus_requested, allocated_cores)

    logger.info(
        f"Proportional allocation: {num_gpus_requested}/{total_gpus_on_system} GPUs "
        f"→ {allocated_cores}/{total_cpu_cores} cores ({proportion*100:.0f}%)"
    )

    return allocated_cores


def suggest_use_amp(
    gpus: list,
    model_type: Literal['swinunetr', 'unet']
) -> bool:
    """Suggest whether to use automatic mixed precision (AMP).

    Parameters
    ----------
    gpus : list[GPUInfo]
        List of GPUs being used.
    model_type : {'swinunetr', 'unet'}
        Model architecture type.

    Returns
    -------
    bool
        True if AMP should be used, False otherwise.

    Notes
    -----
    AMP requirements:
    - GPU compute capability >= 7.0 (Volta and newer)
    - All GPUs must support mixed precision

    Benefits:
    - ~2× faster training
    - ~40% less VRAM usage
    - Negligible accuracy impact for medical imaging

    Examples
    --------
    >>> from lesseg_unet.hardware import get_hardware_profile
    >>> hw = get_hardware_profile()
    >>> use_amp = suggest_use_amp(hw.gpus, model_type='swinunetr')
    >>> print(f"Use AMP: {use_amp}")
    """
    if not gpus:
        # CPU mode: no AMP
        return False

    # Check if all GPUs support mixed precision
    all_support_amp = all(gpu.supports_mixed_precision for gpu in gpus)

    logger.info(
        f"Suggested use_amp={all_support_amp} "
        f"(all {len(gpus)} GPU(s) support: {all_support_amp})"
    )

    return all_support_amp

"""Auto-configurator for training parameters.

Orchestrates all heuristics to suggest optimal training configuration
based on hardware capabilities and dataset characteristics.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from lesseg_unet.hardware.detection import HardwareProfile
from lesseg_unet.hardware.memory_model import SwinUNETRMemoryCalculator, UNetMemoryCalculator
from lesseg_unet.auto_config import heuristics


logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    """Dataset characteristics for auto-configuration.

    Attributes
    ----------
    median_image_size : tuple[int, int, int]
        Median image size in dataset (H, W, D).
    num_subjects : int
        Total number of subjects in dataset.
    in_channels : int
        Number of input modalities/channels.
    out_channels : int
        Number of output classes.
    storage_type : {'ssd', 'hdd', 'network'}
        Storage type. Default: 'ssd'.
    """

    median_image_size: tuple[int, int, int]
    num_subjects: int
    in_channels: int
    out_channels: int
    storage_type: Literal['ssd', 'hdd', 'network'] = 'ssd'


@dataclass
class AutoConfigResult:
    """Result of auto-configuration.

    Attributes
    ----------
    batch_size : int
        Suggested batch size.
    patch_size : tuple[int, int, int]
        Suggested patch size (H, W, D).
    num_workers : int
        Suggested number of dataloader workers.
    network_depth : int
        Suggested network depth (4 or 5).
    feature_size : int
        Suggested model feature size.
    use_amp : bool
        Whether to use automatic mixed precision.
    use_checkpoint : bool
        Whether to use gradient checkpointing (SwinUNETR only).
    num_gpus : int
        Number of GPUs to use.
    vram_safety_margin : float
        VRAM safety margin used (0.0-1.0).
    reasoning : dict[str, str]
        Reasoning for each decision.
    memory_estimate : dict[str, float]
        Estimated memory usage breakdown.
    """

    batch_size: int
    val_batch_size: int
    patch_size: tuple[int, int, int]
    num_workers: int
    network_depth: int
    feature_size: int
    use_amp: bool
    use_checkpoint: bool
    num_gpus: int
    vram_safety_margin: float
    reasoning: dict[str, str] = field(default_factory=dict)
    memory_estimate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            'batch_size': self.batch_size,
            'val_batch_size': self.val_batch_size,
            'patch_size': self.patch_size,
            'num_workers': self.num_workers,
            'network_depth': self.network_depth,
            'feature_size': self.feature_size,
            'use_amp': self.use_amp,
            'use_checkpoint': self.use_checkpoint,
            'num_gpus': self.num_gpus,
            'vram_safety_margin': self.vram_safety_margin,
            'reasoning': self.reasoning,
            'memory_estimate': self.memory_estimate
        }


class AutoConfigurator:
    """Automatic configuration based on hardware and dataset.

    Orchestrates all heuristics to suggest optimal training parameters
    that prevent OOM while maximizing hardware utilization.

    Parameters
    ----------
    hardware_profile : HardwareProfile
        Hardware capabilities (GPUs, CPU, RAM).
    dataset_profile : DatasetProfile
        Dataset characteristics (image size, channels, etc.).
    model_type : {'swinunetr', 'unet'}
        Model architecture type. Default: 'swinunetr'.
    target : {'speed', 'memory', 'balanced'}
        Optimization target. Default: 'balanced'.
        - 'speed': Maximize training speed (larger batches, models)
        - 'memory': Minimize memory usage (smaller batches, models)
        - 'balanced': Balance between speed and memory
    vram_safety_margin : float
        Target VRAM usage (0.0-1.0). Default: 0.95 (95%).
    num_gpus : Optional[int]
        Number of GPUs to use. If None, uses all available.
        Multi-GPU is opt-in via this parameter.
    num_samples : int
        Number of patches cropped from each image by RandCropByPosNegLabeld.
        Used to calculate DataLoader RAM requirements. Default: 4.

    Examples
    --------
    >>> from lesseg_unet.hardware import get_hardware_profile
    >>> from lesseg_unet.auto_config import AutoConfigurator, DatasetProfile
    >>>
    >>> hw = get_hardware_profile()
    >>> dataset = DatasetProfile(
    ...     median_image_size=(181, 217, 181),
    ...     num_subjects=100,
    ...     in_channels=2,
    ...     out_channels=1
    ... )
    >>>
    >>> configurator = AutoConfigurator(
    ...     hardware_profile=hw,
    ...     dataset_profile=dataset,
    ...     target='balanced'
    ... )
    >>>
    >>> config = configurator.suggest_config()
    >>> print(f"Batch size: {config.batch_size}")
    >>> print(f"Patch size: {config.patch_size}")
    """

    def __init__(
        self,
        hardware_profile: HardwareProfile,
        dataset_profile: DatasetProfile,
        model_type: Literal['swinunetr', 'unet'] = 'swinunetr',
        target: Literal['speed', 'memory', 'balanced'] = 'balanced',
        vram_safety_margin: float = 0.95,
        num_gpus: Optional[int] = None,
        num_samples: int = 4
    ):
        """Initialize auto-configurator."""
        self.hardware = hardware_profile
        self.dataset = dataset_profile
        self.model_type = model_type
        self.target = target
        self.vram_safety_margin = vram_safety_margin
        self.num_samples = num_samples

        # Determine number of GPUs to use
        if num_gpus is None:
            # Default: single GPU if available, else CPU
            self.num_gpus = min(1, len(self.hardware.gpus))
        else:
            # User-specified (opt-in for multi-GPU)
            available_gpus = len(self.hardware.gpus)
            if num_gpus > available_gpus:
                logger.warning(
                    f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                    f"Using {available_gpus}."
                )
                self.num_gpus = available_gpus
            else:
                self.num_gpus = num_gpus

        # Track reasoning for decisions
        self.reasoning = {}

    def suggest_config(
        self,
        override_batch_size: Optional[int] = None,
        override_patch_size: Optional[tuple[int, int, int]] = None,
        override_num_workers: Optional[int] = None,
        override_network_depth: Optional[int] = None,
        override_feature_size: Optional[int] = None
    ) -> AutoConfigResult:
        """Suggest complete training configuration.

        Orchestrates all heuristics to determine optimal parameters.
        User can override any suggestion.

        Parameters
        ----------
        override_batch_size : Optional[int]
            Override suggested batch size.
        override_patch_size : Optional[tuple[int, int, int]]
            Override suggested patch size.
        override_num_workers : Optional[int]
            Override suggested num_workers.
        override_network_depth : Optional[int]
            Override suggested network depth.
        override_feature_size : Optional[int]
            Override suggested feature size.

        Returns
        -------
        AutoConfigResult
            Complete configuration with reasoning.
        """
        logger.info("=" * 60)
        logger.info("Auto-Configuration")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Target: {self.target}")
        logger.info(f"GPUs: {self.num_gpus} / {len(self.hardware.gpus)} available")
        logger.info(f"VRAM safety margin: {self.vram_safety_margin*100:.0f}%")
        logger.info("")

        # Step 1: Suggest mixed precision
        use_amp = heuristics.suggest_use_amp(
            gpus=self.hardware.gpus[:self.num_gpus],
            model_type=self.model_type
        )
        self.reasoning['use_amp'] = (
            f"GPU(s) support mixed precision (compute >= 7.0)" if use_amp
            else "GPU(s) do not support mixed precision or using CPU"
        )

        # Step 1b: Suggest gradient checkpointing (SwinUNETR memory optimization)
        vram_gb = self._get_vram_per_gpu()
        use_checkpoint = heuristics.suggest_use_checkpoint(
            vram_gb=vram_gb,
            model_type=self.model_type,
            target=self.target
        )
        if use_checkpoint:
            self.reasoning['use_checkpoint'] = (
                f"Enabled gradient checkpointing for {self.model_type} "
                f"({vram_gb:.1f}GB VRAM < 6GB threshold). "
                f"Reduces memory ~40-50% at cost of ~20% slower training."
            )
        else:
            self.reasoning['use_checkpoint'] = (
                f"Disabled gradient checkpointing "
                f"(sufficient VRAM: {vram_gb:.1f}GB or model is memory-efficient)"
            )

        # Step 2: Suggest network depth
        if override_network_depth is not None:
            network_depth = override_network_depth
            self.reasoning['network_depth'] = f"User override: {network_depth}"
        else:
            vram_gb = self._get_vram_per_gpu()
            network_depth = heuristics.suggest_network_depth(
                vram_gb=vram_gb,
                model_type=self.model_type,
                target=self.target
            )
            self.reasoning['network_depth'] = (
                f"Depth {network_depth} optimal for {vram_gb:.1f}GB VRAM "
                f"(target: {self.target})"
            )

        # Step 3: Suggest feature size
        if override_feature_size is not None:
            feature_size = override_feature_size
            self.reasoning['feature_size'] = f"User override: {feature_size}"
        else:
            vram_gb = self._get_vram_per_gpu()
            feature_size = heuristics.suggest_feature_size(
                vram_gb=vram_gb,
                model_type=self.model_type,
                network_depth=network_depth,
                target=self.target
            )
            self.reasoning['feature_size'] = (
                f"Feature size {feature_size} balances capacity and memory "
                f"for {vram_gb:.1f}GB VRAM"
            )

        # Step 4: Suggest patch size
        if override_patch_size is not None:
            patch_size = override_patch_size
            self.reasoning['patch_size'] = f"User override: {patch_size}"
        else:
            vram_gb = self._get_vram_per_gpu()
            patch_size = heuristics.suggest_patch_size(
                median_image_size=self.dataset.median_image_size,
                network_depth=network_depth,
                vram_gb=vram_gb,
                target=self.target,
                model_type=self.model_type
            )
            self.reasoning['patch_size'] = (
                f"Patch {patch_size} optimal for {self.model_type}, median image "
                f"{self.dataset.median_image_size}, depth {network_depth}"
            )

        # Step 5: Create memory calculator
        memory_calculator = self._create_memory_calculator(
            patch_size=patch_size,
            network_depth=network_depth,
            feature_size=feature_size,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint
        )

        # Step 6: Suggest batch size
        if override_batch_size is not None:
            batch_size = override_batch_size
            self.reasoning['batch_size'] = f"User override: {batch_size}"
        else:
            vram_gb = self._get_vram_per_gpu()
            batch_size = heuristics.suggest_batch_size(
                memory_calculator=memory_calculator,
                vram_gb=vram_gb,
                target_usage=self.vram_safety_margin
            )
            # Estimate memory usage
            memory_breakdown = memory_calculator.estimate_total_memory(batch_size, vram_gb=vram_gb)
            self.reasoning['batch_size'] = (
                f"Batch {batch_size} uses ~{memory_breakdown.total_gb:.2f}GB / "
                f"{vram_gb:.2f}GB ({memory_breakdown.total_gb/vram_gb*100:.0f}%)"
            )

        # Step 7: Suggest num_workers
        if override_num_workers is not None:
            num_workers = override_num_workers
            self.reasoning['num_workers'] = f"User override: {num_workers}"
        else:
            # Allocate CPU cores proportionally if multi-GPU
            if self.num_gpus > 1 and len(self.hardware.gpus) > 1:
                available_cores = heuristics.allocate_resources_proportionally(
                    total_gpus_on_system=len(self.hardware.gpus),
                    num_gpus_requested=self.num_gpus,
                    total_cpu_cores=self.hardware.cpu.available_cores
                )
            else:
                available_cores = self.hardware.cpu.available_cores

            num_workers = heuristics.suggest_num_workers(
                available_cores=available_cores,
                num_gpus=self.num_gpus,
                storage_type=self.dataset.storage_type,
                target=self.target
            )
            self.reasoning['num_workers'] = (
                f"{num_workers} workers optimal for {available_cores} cores, "
                f"{self.num_gpus} GPU(s), storage={self.dataset.storage_type}"
            )

        # Step 6.5: Constrain batch_size by DataLoader RAM requirements
        # This accounts for RAM needed to load full images when num_samples > 1
        if override_batch_size is None:
            constrained_batch, ram_warning = self._constrain_batch_by_dataloader_ram(
                batch_size=batch_size,
                num_workers=num_workers
            )
            if ram_warning is not None:
                logger.warning(ram_warning)
                batch_size = constrained_batch
                # Update memory estimate with constrained batch size
                memory_breakdown = memory_calculator.estimate_total_memory(batch_size, vram_gb=vram_gb)
                vram_gb = self._get_vram_per_gpu()
                self.reasoning['batch_size'] = (
                    f"Batch {batch_size} constrained by DataLoader RAM "
                    f"(uses ~{memory_breakdown.total_gb:.2f}GB / {vram_gb:.2f}GB model "
                    f"+ {num_workers}×{int(batch_size/self.num_samples)} images in RAM)"
                )

        # Get memory estimate
        memory_breakdown = memory_calculator.estimate_total_memory(batch_size)
        memory_estimate = memory_breakdown.to_dict()

        # Step 8: Suggest validation batch size
        # Validation can use larger batch (no gradients), but be conservative
        # CPU mode needs extra conservative multiplier due to RAM constraints
        if self.num_gpus == 0:
            # CPU mode: very conservative multiplier
            val_batch_multiplier = 1.2
            reason_suffix = "(CPU mode - very conservative)"
        else:
            vram_gb = self._get_vram_per_gpu()
            if vram_gb > 20:  # Large VRAM (e.g., A100, RTX 3090)
                val_batch_multiplier = 4
                reason_suffix = "(large VRAM)"
            elif vram_gb > 10:  # Medium VRAM (e.g., RTX 3080)
                val_batch_multiplier = 3
                reason_suffix = "(medium VRAM)"
            elif vram_gb > 4:  # Small VRAM (e.g., RTX 3060)
                val_batch_multiplier = 2
                reason_suffix = "(small VRAM)"
            else:  # Very small VRAM
                val_batch_multiplier = 1.5
                reason_suffix = "(very small VRAM)"

        val_batch_size = max(1, int(batch_size * val_batch_multiplier))
        self.reasoning['val_batch_size'] = (
            f"Validation batch {val_batch_size} = {val_batch_multiplier}× training batch "
            f"{reason_suffix}"
        )

        # Log configuration summary
        self._log_configuration_summary(
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            network_depth=network_depth,
            feature_size=feature_size,
            use_amp=use_amp,
            memory_estimate=memory_estimate
        )

        return AutoConfigResult(
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            network_depth=network_depth,
            feature_size=feature_size,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint,
            num_gpus=self.num_gpus,
            vram_safety_margin=self.vram_safety_margin,
            reasoning=self.reasoning,
            memory_estimate=memory_estimate
        )

    def _get_vram_per_gpu(self) -> float:
        """Get VRAM per GPU in GB (or RAM for CPU mode).

        Returns
        -------
        float
            VRAM per GPU in GB. If no GPU (CPU mode), returns 50% of available RAM.
        """
        if not self.hardware.gpus or self.num_gpus == 0:
            # CPU mode: use available RAM instead of placeholder
            # Be conservative - use 50% of available RAM to leave room for OS and other processes
            return self.hardware.cpu.available_ram_gb * 0.5

        # Use smallest GPU VRAM (conservative)
        gpus_to_use = self.hardware.gpus[:self.num_gpus]
        min_vram_mb = min(gpu.total_memory_mb for gpu in gpus_to_use)
        return min_vram_mb / 1024

    def _constrain_batch_by_dataloader_ram(
        self,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2
    ) -> tuple[int, str | None]:
        """Constrain batch size based on DataLoader RAM requirements.

        When using num_samples > 1 in RandCropByPosNegLabeld, the DataLoader
        needs to load full images before cropping patches. This calculates RAM
        needed by workers and reduces batch_size if it exceeds available RAM.

        Parameters
        ----------
        batch_size : int
            Suggested batch size from VRAM calculation.
        num_workers : int
            Number of DataLoader workers.
        prefetch_factor : int
            PyTorch DataLoader prefetch_factor. Default: 2.

        Returns
        -------
        tuple[int, str | None]
            (constrained_batch_size, warning_message)
            If no constraint needed, warning_message is None.
        """
        if self.num_samples == 1 or num_workers == 0:
            # No RAM issue if num_samples=1 (loads patches directly)
            # or num_workers=0 (main process only)
            return batch_size, None

        # Calculate bytes per full image (all channels)
        h, w, d = self.dataset.median_image_size
        channels = self.dataset.in_channels
        bytes_per_image = h * w * d * channels * 4  # float32
        gb_per_image = bytes_per_image / (1024 ** 3)

        # Calculate images per batch (batch_size / num_samples)
        # num_samples is how many patches we crop from each image
        images_per_batch = max(1, int(batch_size / self.num_samples))

        # Total RAM for DataLoader:
        # num_workers × prefetch_factor × images_per_batch × GB_per_image
        dataloader_ram_gb = num_workers * prefetch_factor * images_per_batch * gb_per_image

        # Available RAM (be conservative - use 25% for DataLoader to account for
        # Python interpreter overhead in each worker, loaded libraries, etc.)
        max_dataloader_ram_gb = self.hardware.cpu.available_ram_gb * 0.25

        if dataloader_ram_gb <= max_dataloader_ram_gb:
            # No constraint needed
            return batch_size, None

        # Need to reduce batch_size
        # Solve for batch_size: dataloader_ram = workers × prefetch × (batch/samples) × GB_per_img
        # batch_size = (max_ram / (workers × prefetch × GB_per_img)) × samples
        max_images_per_batch = int(max_dataloader_ram_gb / (num_workers * prefetch_factor * gb_per_image))
        constrained_batch = max(1, max_images_per_batch * self.num_samples)

        warning = (
            f"Batch reduced {batch_size}→{constrained_batch} due to DataLoader RAM constraint "
            f"(would need {dataloader_ram_gb:.1f}GB but {max_dataloader_ram_gb:.1f}GB available "
            f"for {num_workers} workers with num_samples={self.num_samples})"
        )

        return constrained_batch, warning

    def _create_memory_calculator(
        self,
        patch_size: tuple[int, int, int],
        network_depth: int,
        feature_size: int,
        use_amp: bool,
        use_checkpoint: bool = False
    ):
        """Create appropriate memory calculator.

        Parameters
        ----------
        patch_size : tuple[int, int, int]
            Patch size (H, W, D).
        network_depth : int
            Network depth (4 or 5).
        feature_size : int
            Model feature size.
        use_amp : bool
            Whether using mixed precision.
        use_checkpoint : bool
            Whether using gradient checkpointing (SwinUNETR only).

        Returns
        -------
        SwinUNETRMemoryCalculator or UNetMemoryCalculator
            Memory calculator for the model.
        """
        if self.model_type == 'swinunetr':
            depths = [2, 2, 2, 2] if network_depth == 4 else [2, 2, 2, 2, 2]
            return SwinUNETRMemoryCalculator(
                img_size=patch_size,
                in_channels=self.dataset.in_channels,
                out_channels=self.dataset.out_channels,
                feature_size=feature_size,
                depths=depths,
                use_mixed_precision=use_amp,
                use_checkpoint=use_checkpoint
            )
        else:  # unet
            # UNet channels based on feature_size
            base = feature_size
            channels = (base, base*2, base*4, base*8, base*16)
            if network_depth == 4:
                channels = channels[:4]

            return UNetMemoryCalculator(
                img_size=patch_size,
                in_channels=self.dataset.in_channels,
                out_channels=self.dataset.out_channels,
                channels=channels,
                use_mixed_precision=use_amp
            )

    def _log_configuration_summary(
        self,
        batch_size: int,
        patch_size: tuple[int, int, int],
        num_workers: int,
        network_depth: int,
        feature_size: int,
        use_amp: bool,
        memory_estimate: dict
    ):
        """Log configuration summary.

        Parameters
        ----------
        batch_size : int
            Batch size.
        patch_size : tuple[int, int, int]
            Patch size.
        num_workers : int
            Number of workers.
        network_depth : int
            Network depth.
        feature_size : int
            Feature size.
        use_amp : bool
            Use mixed precision.
        memory_estimate : dict
            Memory breakdown.
        """
        logger.info("Configuration Summary:")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"    → {self.reasoning['batch_size']}")
        logger.info(f"  patch_size: {patch_size}")
        logger.info(f"    → {self.reasoning['patch_size']}")
        logger.info(f"  num_workers: {num_workers}")
        logger.info(f"    → {self.reasoning['num_workers']}")
        logger.info(f"  network_depth: {network_depth}")
        logger.info(f"    → {self.reasoning['network_depth']}")
        logger.info(f"  feature_size: {feature_size}")
        logger.info(f"    → {self.reasoning['feature_size']}")
        logger.info(f"  use_amp: {use_amp}")
        logger.info(f"    → {self.reasoning['use_amp']}")
        logger.info("")
        logger.info("Memory Estimate:")
        logger.info(f"  Total: {memory_estimate['total_gb']:.2f} GB")
        logger.info(f"  - Params:      {memory_estimate['params_mb']:>8.1f} MB")
        logger.info(f"  - Optimizer:   {memory_estimate['optimizer_mb']:>8.1f} MB")
        logger.info(f"  - Activations: {memory_estimate['activations_mb']:>8.1f} MB")
        logger.info(f"  - Gradients:   {memory_estimate['gradients_mb']:>8.1f} MB")
        logger.info(f"  - Overhead:    {memory_estimate['overhead_mb']:>8.1f} MB")
        logger.info(f"  - Fragment:    {memory_estimate['fragmentation_mb']:>8.1f} MB")
        logger.info("=" * 60)

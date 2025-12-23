"""Dry-run validation for training configurations.

Tests configurations with actual forward/backward passes to validate
memory estimates before starting full training.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes
    ----------
    success : bool
        Whether validation succeeded (no OOM).
    peak_memory_mb : float
        Peak VRAM usage during validation in MB.
    free_memory_mb : float
        Free VRAM after validation in MB.
    error_message : Optional[str]
        Error message if validation failed.
    """

    success: bool
    peak_memory_mb: float
    free_memory_mb: float
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Validation result as dictionary.
        """
        return {
            'success': self.success,
            'peak_memory_mb': self.peak_memory_mb,
            'free_memory_mb': self.free_memory_mb,
            'error_message': self.error_message
        }


def validate_config(
    model: nn.Module,
    batch_size: int,
    patch_size: tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    device: str = 'cuda',
    num_iterations: int = 2
) -> ValidationResult:
    """Validate configuration with dry-run forward/backward passes.

    Runs actual forward and backward passes to test if configuration
    fits in memory before starting full training.

    Parameters
    ----------
    model : nn.Module
        Model to validate.
    batch_size : int
        Batch size to test.
    patch_size : tuple[int, int, int]
        Patch size (H, W, D).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    device : str
        Device to run on ('cuda' or 'cpu'). Default: 'cuda'.
    num_iterations : int
        Number of forward/backward iterations to test. Default: 2.

    Returns
    -------
    ValidationResult
        Validation result with success status and memory stats.

    Examples
    --------
    >>> from monai.networks.nets import SwinUNETR
    >>> model = SwinUNETR(img_size=(96,96,96), in_channels=2, out_channels=1)
    >>> result = validate_config(
    ...     model=model,
    ...     batch_size=4,
    ...     patch_size=(96, 96, 96),
    ...     in_channels=2,
    ...     out_channels=1,
    ...     device='cuda'
    ... )
    >>> if result.success:
    ...     print(f"Success! Peak VRAM: {result.peak_memory_mb / 1024:.2f} GB")
    ... else:
    ...     print(f"Failed: {result.error_message}")
    """
    logger.info(
        f"Validating config: batch_size={batch_size}, "
        f"patch_size={patch_size}, device={device}"
    )

    # Track peak memory
    peak_memory_mb = 0.0
    free_memory_mb = 0.0
    error_message = None

    try:
        # Move model to device
        model = model.to(device)
        model.train()

        # Create dummy optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Reset peak memory stats (CUDA only)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Run test iterations
        for i in range(num_iterations):
            logger.debug(f"Validation iteration {i+1}/{num_iterations}")

            # Create dummy input and target
            dummy_input = torch.randn(
                batch_size,
                in_channels,
                *patch_size,
                device=device
            )

            dummy_target = torch.randn(
                batch_size,
                out_channels,
                *patch_size,
                device=device
            )

            # Forward pass
            output = model(dummy_input)

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, dummy_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track memory (CUDA only)
            if device == 'cuda':
                current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_memory_mb = max(peak_memory_mb, current_peak)

            # Clean up
            del dummy_input, dummy_target, output, loss

        # Get final memory stats
        if device == 'cuda':
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            free_memory_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            torch.cuda.empty_cache()

        logger.info(
            f"Validation successful. Peak VRAM: {peak_memory_mb / 1024:.2f} GB, "
            f"Free: {free_memory_mb / 1024:.2f} GB"
        )

        return ValidationResult(
            success=True,
            peak_memory_mb=peak_memory_mb,
            free_memory_mb=free_memory_mb
        )

    except RuntimeError as e:
        # Check if OOM error
        if 'out of memory' in str(e).lower():
            error_message = f"OOM error: {str(e)}"
            logger.warning(f"Validation failed: {error_message}")
        else:
            error_message = f"Runtime error: {str(e)}"
            logger.error(f"Validation failed: {error_message}")

        # Clean up
        if device == 'cuda':
            torch.cuda.empty_cache()

        return ValidationResult(
            success=False,
            peak_memory_mb=peak_memory_mb,
            free_memory_mb=free_memory_mb,
            error_message=error_message
        )

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Validation failed: {error_message}")

        return ValidationResult(
            success=False,
            peak_memory_mb=peak_memory_mb,
            free_memory_mb=free_memory_mb,
            error_message=error_message
        )

    finally:
        # Clean up model
        if device == 'cuda':
            torch.cuda.empty_cache()


def find_safe_config(
    model_factory,
    initial_batch_size: int,
    patch_size: tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    device: str = 'cuda',
    min_batch_size: int = 1
) -> tuple[Optional[int], Optional[ValidationResult]]:
    """Find safe configuration by progressively reducing batch size on OOM.

    Attempts validation with initial batch size. If OOM occurs, reduces
    batch size by half and retries until success or minimum reached.

    Parameters
    ----------
    model_factory : callable
        Function that creates and returns a model instance.
        Must accept no arguments or use functools.partial.
    initial_batch_size : int
        Initial batch size to try.
    patch_size : tuple[int, int, int]
        Patch size (H, W, D).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    device : str
        Device to run on ('cuda' or 'cpu'). Default: 'cuda'.
    min_batch_size : int
        Minimum batch size to try. Default: 1.

    Returns
    -------
    tuple[Optional[int], Optional[ValidationResult]]
        (safe_batch_size, validation_result) if found, (None, None) otherwise.

    Examples
    --------
    >>> from functools import partial
    >>> from monai.networks.nets import SwinUNETR
    >>>
    >>> # Create model factory
    >>> model_factory = partial(
    ...     SwinUNETR,
    ...     img_size=(96, 96, 96),
    ...     in_channels=2,
    ...     out_channels=1,
    ...     feature_size=48
    ... )
    >>>
    >>> # Find safe config
    >>> safe_batch, result = find_safe_config(
    ...     model_factory=model_factory,
    ...     initial_batch_size=8,
    ...     patch_size=(96, 96, 96),
    ...     in_channels=2,
    ...     out_channels=1
    ... )
    >>>
    >>> if safe_batch:
    ...     print(f"Safe batch size: {safe_batch}")
    ... else:
    ...     print("Could not find safe configuration")
    """
    current_batch = initial_batch_size

    logger.info(
        f"Finding safe config starting with batch_size={initial_batch_size}, "
        f"min_batch_size={min_batch_size}"
    )

    while current_batch >= min_batch_size:
        logger.info(f"Testing batch_size={current_batch}")

        # Create fresh model
        model = model_factory()

        # Validate
        result = validate_config(
            model=model,
            batch_size=current_batch,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device
        )

        if result.success:
            logger.info(f"Found safe batch_size={current_batch}")
            return current_batch, result

        # OOM - reduce batch size by half
        logger.warning(
            f"batch_size={current_batch} failed, reducing batch size"
        )
        current_batch = max(current_batch // 2, min_batch_size)

        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    logger.error(
        f"Could not find safe configuration (tried down to batch_size={min_batch_size})"
    )
    return None, None

"""Theoretical memory estimation for neural network training.

Calculates VRAM/RAM usage from model architecture and input dimensions
without empirical profiling.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class MemoryBreakdown:
    """Detailed breakdown of memory usage.

    Attributes
    ----------
    params_mb : float
        Model parameters in MB.
    optimizer_mb : float
        Optimizer state in MB (e.g., Adam momentum/variance).
    activations_mb : float
        Forward pass activations in MB.
    gradients_mb : float
        Backward pass gradients in MB.
    cudnn_workspace_mb : float
        CuDNN workspace for convolutions in MB.
    overhead_mb : float
        PyTorch/CUDA overhead in MB.
    fragmentation_mb : float
        Memory fragmentation overhead in MB.
    total_mb : float
        Total memory in MB.
    """

    params_mb: float
    optimizer_mb: float
    activations_mb: float
    gradients_mb: float
    cudnn_workspace_mb: float
    overhead_mb: float
    fragmentation_mb: float
    total_mb: float

    @property
    def total_gb(self) -> float:
        """Total memory in GB.

        Returns
        -------
        float
            Total memory in GB.
        """
        return self.total_mb / 1024

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Memory breakdown as dictionary.
        """
        return {
            'params_mb': self.params_mb,
            'optimizer_mb': self.optimizer_mb,
            'activations_mb': self.activations_mb,
            'gradients_mb': self.gradients_mb,
            'cudnn_workspace_mb': self.cudnn_workspace_mb,
            'overhead_mb': self.overhead_mb,
            'fragmentation_mb': self.fragmentation_mb,
            'total_mb': self.total_mb,
            'total_gb': self.total_gb
        }


class SwinUNETRMemoryCalculator:
    """Theoretical memory calculator for SwinUNETR.

    Calculates VRAM usage from architecture parameters without
    instantiating the model or running forward/backward passes.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Input image size (H, W, D).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    feature_size : int
        Base feature size (default: 48).
    depths : tuple[int, ...] or list[int]
        Number of transformer blocks per stage (default: [2, 2, 2, 2, 2]).
    use_mixed_precision : bool
        Whether mixed precision (AMP) is used (default: True).
    use_checkpoint : bool
        Whether gradient checkpointing is enabled (default: False).
        Reduces activation memory by ~50% at cost of ~20% slower training.

    Notes
    -----
    Memory components:
    1. Model parameters: count(params) × 4 bytes (FP32)
    2. Optimizer state (Adam): params × 2 (momentum + variance, always FP32)
    3. Forward activations: batch_size × layer_outputs (mixed precision ~3 bytes/elem)
    4. Backward gradients: ≈ forward activations
    5. PyTorch overhead: ~400 MB constant
    6. Fragmentation: ~8% of allocated memory

    Examples
    --------
    >>> calc = SwinUNETRMemoryCalculator(
    ...     img_size=(96, 96, 96),
    ...     in_channels=2,
    ...     out_channels=1,
    ...     feature_size=48,
    ...     depths=[2, 2, 2, 2, 2]
    ... )
    >>> memory = calc.estimate_total_memory(batch_size=4)
    >>> print(f"Total: {memory.total_gb:.2f} GB")
    >>> max_batch = calc.find_max_batch_size(vram_gb=24.0)
    >>> print(f"Max batch size: {max_batch}")
    """

    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        depths: tuple[int, ...] | list[int] = (2, 2, 2, 2, 2),
        use_mixed_precision: bool = True,
        use_checkpoint: bool = False
    ):
        """Initialize memory calculator."""
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.depths = tuple(depths) if isinstance(depths, list) else depths
        self.use_mixed_precision = use_mixed_precision
        self.use_checkpoint = use_checkpoint

        # Bytes per element (mixed precision: avg of FP32 and FP16)
        self.bytes_per_element = 3 if use_mixed_precision else 4

        # Validate depths
        if len(self.depths) not in [4, 5]:
            raise ValueError(f"depths must have 4 or 5 stages, got {len(self.depths)}")

    def calculate_model_parameters(self) -> float:
        """Calculate total model parameters in millions.

        Returns
        -------
        float
            Number of parameters in millions.

        Notes
        -----
        SwinUNETR architecture:
        - Patch embedding: Conv3d with 4×4×4 kernel
        - Encoder: Swin Transformer blocks (LayerNorm + Attention + MLP)
        - Decoder: Upsampling convolutions
        - Output: Final convolution
        """
        params = 0

        # Patch embedding (Conv3d: 4×4×4 patches)
        params += self.in_channels * (4 ** 3) * self.feature_size

        # Encoder transformer blocks
        for i, depth in enumerate(self.depths):
            embed_dim = self.feature_size * (2 ** i)

            # Each transformer block contains:
            # - LayerNorm (2 × embed_dim)
            # - Attention (Q, K, V, proj: 4 × embed_dim²)
            # - MLP (2 linear layers with 4× expansion: 8 × embed_dim²)
            params_per_block = (
                2 * embed_dim +           # LayerNorm
                4 * embed_dim ** 2 +      # Attention
                8 * embed_dim ** 2        # MLP
            )
            params += depth * params_per_block

        # Decoder (upsampling convolutions)
        for i in reversed(range(len(self.depths) - 1)):
            embed_dim = self.feature_size * (2 ** i)
            # 2×2×2 upsampling kernel
            params += embed_dim ** 2 * 8

        # Output layer
        params += self.feature_size * self.out_channels

        return params / 1e6  # Return in millions

    def calculate_activation_memory(self, batch_size: int) -> float:
        """Calculate forward pass activation memory in MB.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        float
            Activation memory in MB.

        Notes
        -----
        Activations include:
        - Input tensor
        - Patch embedding output
        - All transformer block outputs
        - Attention maps (typically FP16)
        - Decoder outputs
        - Final output
        """
        H, W, D = self.img_size
        bytes_per_elem = self.bytes_per_element
        activations_mb = 0

        # Input
        activations_mb += (batch_size * self.in_channels * H * W * D * bytes_per_elem) / (1024 ** 2)

        # Patch embed output (spatial dims reduced by 4)
        h, w, d = H // 4, W // 4, D // 4
        activations_mb += (batch_size * self.feature_size * h * w * d * bytes_per_elem) / (1024 ** 2)

        # Encoder stages
        for i, depth in enumerate(self.depths):
            embed_dim = self.feature_size * (2 ** i)

            # Transformer block outputs
            activations_mb += (batch_size * depth * embed_dim * h * w * d * bytes_per_elem) / (1024 ** 2)

            # Attention maps (typically FP16, 2 bytes)
            seq_len = h * w * d
            num_heads = [3, 6, 12, 24, 48][i]  # Standard Swin num_heads
            activations_mb += (batch_size * num_heads * seq_len * seq_len * 2) / (1024 ** 2)

            # Downsample spatial dims (except last stage)
            if i < len(self.depths) - 1:
                h, w, d = h // 2, w // 2, d // 2

        # Decoder stages
        for i in reversed(range(len(self.depths) - 1)):
            h, w, d = h * 2, w * 2, d * 2
            embed_dim = self.feature_size * (2 ** i)
            activations_mb += (batch_size * embed_dim * h * w * d * bytes_per_elem) / (1024 ** 2)

        # Output
        activations_mb += (batch_size * self.out_channels * H * W * D * bytes_per_elem) / (1024 ** 2)

        # Apply gradient checkpointing reduction
        if self.use_checkpoint:
            # Gradient checkpointing reduces activation memory by ~50%
            # (recomputes activations during backward pass instead of storing them)
            activations_mb = activations_mb * 0.5

        return activations_mb

    def calculate_gradient_memory(self, batch_size: int) -> float:
        """Calculate backward pass gradient memory in MB.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        float
            Gradient memory in MB.

        Notes
        -----
        Gradients approximately match activations (need to store for backprop).
        Conservative: assume 1:1 ratio.
        """
        return self.calculate_activation_memory(batch_size)

    def calculate_cudnn_workspace(self, batch_size: int) -> float:
        """Calculate CuDNN workspace memory in MB.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        float
            CuDNN workspace memory in MB.

        Notes
        -----
        CuDNN allocates workspace for convolution operations.
        Size depends on algorithm choice and input dimensions.
        Conservative estimate: base + input-dependent component.
        """
        H, W, D = self.img_size
        voxels = H * W * D

        # Base workspace: ~500 MB for algorithm selection and caching
        base_workspace = 500

        # Input-dependent workspace scales with batch×voxels×features
        # Empirically, workspace ≈ 50% of (batch × voxels × feature_dim × 4 bytes)
        input_dependent = (batch_size * voxels * self.feature_size * 4) / (1024 ** 2) * 0.5

        # Cap total workspace at 2 GB (typical CuDNN limit)
        workspace_mb = min(base_workspace + input_dependent, 2000)

        return workspace_mb

    def calculate_optimizer_memory(self) -> float:
        """Calculate optimizer state memory in MB.

        Returns
        -------
        float
            Optimizer memory in MB.

        Notes
        -----
        Adam optimizer stores:
        - First moment (momentum): same size as parameters
        - Second moment (variance): same size as parameters
        Always FP32, even with mixed precision.
        """
        params = self.calculate_model_parameters() * 1e6  # Convert back to count
        # 2× parameters (momentum + variance), 4 bytes each (FP32)
        return (params * 2 * 4) / (1024 ** 2)

    def estimate_total_memory(self, batch_size: int, vram_gb: float = None) -> MemoryBreakdown:
        """Estimate total memory usage for given batch size.

        Parameters
        ----------
        batch_size : int
            Batch size.
        vram_gb : float, optional
            Available VRAM in GB. Used to apply safety multiplier for small GPUs.
            If None, no safety multiplier is applied.

        Returns
        -------
        MemoryBreakdown
            Detailed memory breakdown.

        Notes
        -----
        Applies conservative estimates including:
        - CuDNN workspace for convolutions
        - Peak memory factor (backward pass spikes)
        - Safety multiplier for small GPUs (< 6 GB VRAM)

        Examples
        --------
        >>> calc = SwinUNETRMemoryCalculator((96, 96, 96), 2, 1)
        >>> mem = calc.estimate_total_memory(batch_size=4, vram_gb=8.0)
        >>> print(f"Total: {mem.total_gb:.2f} GB")
        >>> print(f"  Params: {mem.params_mb:.0f} MB")
        >>> print(f"  Activations: {mem.activations_mb:.0f} MB")
        """
        # 1. Model parameters
        params_count = self.calculate_model_parameters() * 1e6
        params_mb = (params_count * 4) / (1024 ** 2)  # FP32

        # 2. Optimizer state (Adam)
        optimizer_mb = self.calculate_optimizer_memory()

        # 3. Forward activations
        activations_mb = self.calculate_activation_memory(batch_size)

        # 4. Backward gradients (≈ activations)
        gradients_mb = self.calculate_gradient_memory(batch_size)

        # 5. CuDNN workspace (CRITICAL - often 500 MB - 2 GB)
        cudnn_workspace_mb = self.calculate_cudnn_workspace(batch_size)

        # 6. PyTorch overhead (increased from 400 MB to 600 MB)
        # Includes: CUDA context, allocator, kernel cache, loss functions
        overhead_mb = 600

        # 7. Fragmentation
        allocated_mb = params_mb + optimizer_mb + activations_mb + gradients_mb + cudnn_workspace_mb
        fragmentation_mb = allocated_mb * 0.10  # 10% overhead (increased from 8%)

        # Subtotal before peak/safety factors
        subtotal_mb = params_mb + optimizer_mb + activations_mb + gradients_mb + cudnn_workspace_mb + overhead_mb + fragmentation_mb

        # 8. Peak memory factor (backward pass creates temporary tensors)
        # Peak can be 30% higher than steady-state during backprop
        peak_factor = 1.30
        total_mb = subtotal_mb * peak_factor

        # 9. Safety multiplier for small GPUs
        # Theoretical estimates often underestimate for small VRAM
        # due to less efficient memory allocation and higher fragmentation
        if vram_gb is not None and vram_gb < 6.0:
            # Apply 1.2x safety multiplier for GPUs < 6 GB
            safety_multiplier = 1.20
            total_mb = total_mb * safety_multiplier
            logger.debug(f"Applied {safety_multiplier}x safety multiplier for {vram_gb:.1f} GB VRAM")

        return MemoryBreakdown(
            params_mb=params_mb,
            optimizer_mb=optimizer_mb,
            activations_mb=activations_mb,
            gradients_mb=gradients_mb,
            cudnn_workspace_mb=cudnn_workspace_mb,
            overhead_mb=overhead_mb,
            fragmentation_mb=fragmentation_mb,
            total_mb=total_mb
        )

    def find_max_batch_size(
        self,
        vram_gb: float,
        target_usage: float = 0.95,
        max_batch: int = 64
    ) -> int:
        """Find maximum batch size that fits in VRAM.

        Uses binary search to find the largest batch size that fits within
        the target VRAM usage.

        Parameters
        ----------
        vram_gb : float
            Available VRAM in GB.
        target_usage : float
            Target VRAM usage (0.0-1.0). Default: 0.95 (95%).
        max_batch : int
            Maximum batch size to consider. Default: 64.

        Returns
        -------
        int
            Maximum batch size that fits in VRAM.

        Examples
        --------
        >>> calc = SwinUNETRMemoryCalculator((96, 96, 96), 2, 1)
        >>> max_batch = calc.find_max_batch_size(vram_gb=24.0)
        >>> print(f"Max batch size for 24GB: {max_batch}")
        """
        vram_mb = vram_gb * 1024 * target_usage

        low, high = 1, max_batch
        best_batch = 1

        while low <= high:
            mid = (low + high) // 2

            memory = self.estimate_total_memory(batch_size=mid, vram_gb=vram_gb)

            if memory.total_mb <= vram_mb:
                best_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        logger.debug(
            f"Max batch size for {vram_gb:.1f} GB VRAM ({target_usage*100:.0f}% target): {best_batch}"
        )

        return best_batch


class UNetMemoryCalculator:
    """Theoretical memory calculator for UNet.

    Simplified version of SwinUNETRMemoryCalculator for standard UNet architecture.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Input image size (H, W, D).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    channels : tuple[int, ...]
        Channel counts per level (default: (16, 32, 64, 128, 256)).
    use_mixed_precision : bool
        Whether mixed precision (AMP) is used (default: True).

    Examples
    --------
    >>> calc = UNetMemoryCalculator(
    ...     img_size=(96, 96, 96),
    ...     in_channels=2,
    ...     out_channels=1,
    ...     channels=(16, 32, 64, 128, 256)
    ... )
    >>> memory = calc.estimate_total_memory(batch_size=4)
    >>> print(f"Total: {memory.total_gb:.2f} GB")
    """

    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        use_mixed_precision: bool = True
    ):
        """Initialize memory calculator."""
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.use_mixed_precision = use_mixed_precision

        self.bytes_per_element = 3 if use_mixed_precision else 4

    def calculate_model_parameters(self) -> float:
        """Calculate total model parameters in millions.

        Returns
        -------
        float
            Number of parameters in millions.
        """
        params = 0

        # Encoder
        prev_channels = self.in_channels
        for ch in self.channels:
            # Two conv blocks per level
            params += prev_channels * ch * (3 ** 3) * 2  # 3×3×3 kernels
            prev_channels = ch

        # Decoder (symmetric)
        for ch in reversed(self.channels[:-1]):
            params += prev_channels * ch * (3 ** 3) * 2
            prev_channels = ch

        # Output layer
        params += prev_channels * self.out_channels * (1 ** 3)

        return params / 1e6

    def calculate_activation_memory(self, batch_size: int) -> float:
        """Calculate forward pass activation memory in MB.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        float
            Activation memory in MB.
        """
        H, W, D = self.img_size
        bytes_per_elem = self.bytes_per_element
        activations_mb = 0

        # Input
        activations_mb += (batch_size * self.in_channels * H * W * D * bytes_per_elem) / (1024 ** 2)

        # Encoder
        h, w, d = H, W, D
        for ch in self.channels:
            activations_mb += (batch_size * ch * h * w * d * bytes_per_elem) / (1024 ** 2)
            h, w, d = h // 2, w // 2, d // 2

        # Decoder
        for ch in reversed(self.channels[:-1]):
            h, w, d = h * 2, w * 2, d * 2
            activations_mb += (batch_size * ch * h * w * d * bytes_per_elem) / (1024 ** 2)

        # Output
        activations_mb += (batch_size * self.out_channels * H * W * D * bytes_per_elem) / (1024 ** 2)

        return activations_mb

    def calculate_gradient_memory(self, batch_size: int) -> float:
        """Calculate backward pass gradient memory in MB.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        float
            Gradient memory in MB.
        """
        return self.calculate_activation_memory(batch_size)

    def calculate_optimizer_memory(self) -> float:
        """Calculate optimizer state memory in MB.

        Returns
        -------
        float
            Optimizer memory in MB.
        """
        params = self.calculate_model_parameters() * 1e6
        return (params * 2 * 4) / (1024 ** 2)

    def estimate_total_memory(self, batch_size: int, vram_gb: float = None) -> MemoryBreakdown:
        """Estimate total memory usage for given batch size.

        Parameters
        ----------
        batch_size : int
            Batch size.
        vram_gb : float, optional
            Available VRAM in GB. Used to apply safety multiplier for small GPUs.

        Returns
        -------
        MemoryBreakdown
            Detailed memory breakdown.
        """
        params_count = self.calculate_model_parameters() * 1e6
        params_mb = (params_count * 4) / (1024 ** 2)

        optimizer_mb = self.calculate_optimizer_memory()
        activations_mb = self.calculate_activation_memory(batch_size)
        gradients_mb = self.calculate_gradient_memory(batch_size)

        # CuDNN workspace (simpler for UNet than SwinUNETR)
        H, W, D = self.img_size
        voxels = H * W * D
        cudnn_workspace_mb = min(300 + (batch_size * voxels * 32 * 4) / (1024 ** 2) * 0.3, 1500)

        # Increased overhead from 400 to 600 MB
        overhead_mb = 600

        allocated_mb = params_mb + optimizer_mb + activations_mb + gradients_mb + cudnn_workspace_mb
        fragmentation_mb = allocated_mb * 0.10

        # Peak memory factor
        subtotal_mb = params_mb + optimizer_mb + activations_mb + gradients_mb + cudnn_workspace_mb + overhead_mb + fragmentation_mb
        total_mb = subtotal_mb * 1.30  # 30% peak factor

        # Safety multiplier for small GPUs
        if vram_gb is not None and vram_gb < 6.0:
            total_mb = total_mb * 1.20

        return MemoryBreakdown(
            params_mb=params_mb,
            optimizer_mb=optimizer_mb,
            activations_mb=activations_mb,
            gradients_mb=gradients_mb,
            cudnn_workspace_mb=cudnn_workspace_mb,
            overhead_mb=overhead_mb,
            fragmentation_mb=fragmentation_mb,
            total_mb=total_mb
        )

    def find_max_batch_size(
        self,
        vram_gb: float,
        target_usage: float = 0.95,
        max_batch: int = 64
    ) -> int:
        """Find maximum batch size that fits in VRAM.

        Parameters
        ----------
        vram_gb : float
            Available VRAM in GB.
        target_usage : float
            Target VRAM usage (0.0-1.0). Default: 0.95.
        max_batch : int
            Maximum batch size to consider. Default: 64.

        Returns
        -------
        int
            Maximum batch size that fits in VRAM.
        """
        vram_mb = vram_gb * 1024 * target_usage

        low, high = 1, max_batch
        best_batch = 1

        while low <= high:
            mid = (low + high) // 2

            memory = self.estimate_total_memory(batch_size=mid, vram_gb=vram_gb)

            if memory.total_mb <= vram_mb:
                best_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_batch

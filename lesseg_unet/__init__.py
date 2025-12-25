"""
Lesion Segmentation U-Net (lesseg_unet)

Deep learning models for automated segmentation of acute ischemic stroke lesions
from diffusion-weighted MRI (DWI) scans.
"""

import logging
import warnings

__version__ = "2.0.12"

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def _check_cuda_availability():
    """Check CUDA availability and warn if GPU is detected but not accessible."""
    try:
        import torch
        import subprocess

        # Check if nvidia-smi detects GPUs
        has_nvidia_gpu = False
        try:
            result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True,
                text=True,
                timeout=2
            )
            has_nvidia_gpu = result.returncode == 0 and 'GPU' in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check if PyTorch can access CUDA
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda

        # Warn if GPU exists but PyTorch can't use it
        if has_nvidia_gpu and not cuda_available:
            warnings.warn(
                "\n" + "="*80 + "\n"
                "⚠️  GPU DETECTED BUT NOT ACCESSIBLE\n"
                "="*80 + "\n"
                "NVIDIA GPU(s) detected but PyTorch cannot access them.\n"
                "You have CPU-only PyTorch installed.\n\n"
                "To enable GPU acceleration:\n"
                "  1. Run: python install_pytorch_cuda.py\n"
                "  2. Or reinstall PyTorch with CUDA:\n"
                "     pip uninstall torch torchvision torchaudio\n"
                "     pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \\\n"
                "       --index-url https://download.pytorch.org/whl/cu128\n\n"
                "Training will use CPU (much slower than GPU).\n"
                "="*80,
                UserWarning,
                stacklevel=2
            )
        # Only log on first import (debug mode only)
        elif cuda_available:
            logging.debug(
                f"GPU acceleration enabled: {torch.cuda.device_count()} GPU(s) detected "
                f"(CUDA {cuda_version})"
            )
        elif cuda_version is None:
            logging.debug("CPU-only mode: PyTorch compiled without CUDA support")

    except ImportError:
        # PyTorch not installed yet (during setup)
        pass
    except Exception as e:
        # Don't crash on import if check fails
        logging.debug(f"CUDA availability check failed: {e}")


# Run CUDA check on import (but don't crash if it fails)
_check_cuda_availability()


__all__ = ['__version__']

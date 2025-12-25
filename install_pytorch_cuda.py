#!/usr/bin/env python
"""
Helper script to install PyTorch with CUDA support for lesseg_unet.

This script should be run BEFORE pip install to ensure GPU support.

Usage:
    python install_pytorch_cuda.py [--cuda-version VERSION]

Examples:
    python install_pytorch_cuda.py                    # Auto-detect CUDA
    python install_pytorch_cuda.py --cuda-version 12.8
"""

import argparse
import subprocess
import sys
import re


def get_cuda_version_from_nvidia_smi() -> str | None:
    """Detect CUDA version from nvidia-smi output.

    Returns
    -------
    str or None
        CUDA version (e.g., "12.8") or None if detection fails.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        # Extract CUDA version from nvidia-smi output
        match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    return None


def map_cuda_to_pytorch_index(cuda_version: str) -> tuple[str, str]:
    """Map CUDA version to PyTorch index URL.

    Parameters
    ----------
    cuda_version : str
        CUDA version (e.g., "12.8", "11.8")

    Returns
    -------
    tuple[str, str]
        (torch_index, description)
        e.g., ("cu128", "CUDA 12.8")
    """
    major, minor = map(int, cuda_version.split('.'))

    if major == 12 and minor >= 8:
        return "cu128", "CUDA 12.8"
    elif major == 12 and minor >= 6:
        return "cu126", "CUDA 12.6"
    elif major == 11 and minor >= 8:
        return "cu118", "CUDA 11.8"
    else:
        print(f"Warning: Unusual CUDA version {cuda_version}")
        print("Defaulting to CUDA 12.8 support")
        return "cu128", "CUDA 12.8"


def check_current_pytorch() -> tuple[bool, str | None]:
    """Check if PyTorch is installed and if it has CUDA support.

    Returns
    -------
    tuple[bool, str | None]
        (is_installed, version_string)
    """
    try:
        import torch
        version = torch.__version__
        has_cuda = torch.cuda.is_available()
        cuda_version = torch.version.cuda

        return True, f"{version} (CUDA: {cuda_version if cuda_version else 'None'})"
    except ImportError:
        return False, None


def install_pytorch(torch_index: str, pytorch_version: str = "2.7.0",
                   torchvision_version: str = "0.22.0",
                   torchaudio_version: str = "2.7.0") -> bool:
    """Install PyTorch with CUDA support.

    Parameters
    ----------
    torch_index : str
        PyTorch index (e.g., "cu128", "cu126", "cu118")
    pytorch_version : str
        PyTorch version to install
    torchvision_version : str
        torchvision version to install
    torchaudio_version : str
        torchaudio version to install

    Returns
    -------
    bool
        True if installation succeeded, False otherwise
    """
    index_url = f"https://download.pytorch.org/whl/{torch_index}"

    print(f"\n{'='*80}")
    print("INSTALLING PYTORCH WITH CUDA SUPPORT")
    print(f"{'='*80}")
    print(f"Index URL: {index_url}")
    print(f"PyTorch: {pytorch_version}")
    print(f"torchvision: {torchvision_version}")
    print(f"torchaudio: {torchaudio_version}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable, "-m", "pip", "install",
        f"torch=={pytorch_version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
        "--index-url", index_url
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed with error code {e.returncode}")
        return False


def verify_cuda() -> bool:
    """Verify that PyTorch can access CUDA.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise
    """
    try:
        import torch

        print(f"\n{'='*80}")
        print("VERIFICATION")
        print(f"{'='*80}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"{'='*80}\n")
            print("⚠️  CUDA is not available. GPU training will not work.")
            return False

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Install PyTorch with CUDA support for lesseg_unet"
    )
    parser.add_argument(
        '--cuda-version',
        type=str,
        help='CUDA version (e.g., "12.8", "11.8"). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Install CPU-only version (for testing without GPU)'
    )
    parser.add_argument(
        '--skip-uninstall',
        action='store_true',
        help='Skip uninstalling existing PyTorch'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("LESSEG_UNET PYTORCH INSTALLER")
    print(f"{'='*80}\n")

    # Check current PyTorch installation
    is_installed, current_version = check_current_pytorch()
    if is_installed:
        print(f"Current PyTorch: {current_version}")

        if not args.skip_uninstall:
            print("\nUninstalling existing PyTorch...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y",
                 "torch", "torchvision", "torchaudio"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓ Uninstalled")
    else:
        print("No existing PyTorch installation detected")

    # CPU-only installation
    if args.cpu_only:
        print("\nInstalling CPU-only PyTorch...")
        torch_index = "cpu"
        print("⚠️  This will NOT support GPU training!")
    else:
        # Detect or use specified CUDA version
        if args.cuda_version:
            cuda_version = args.cuda_version
            print(f"Using specified CUDA version: {cuda_version}")
        else:
            print("Auto-detecting CUDA version from nvidia-smi...")
            cuda_version = get_cuda_version_from_nvidia_smi()

            if cuda_version is None:
                print("\n❌ Could not detect CUDA version.")
                print("Options:")
                print("  1. Specify manually: python install_pytorch_cuda.py --cuda-version 12.8")
                print("  2. Install CPU-only: python install_pytorch_cuda.py --cpu-only")
                sys.exit(1)

            print(f"✓ Detected CUDA {cuda_version}")

        # Map to PyTorch index
        torch_index, description = map_cuda_to_pytorch_index(cuda_version)
        print(f"Using PyTorch build: {description} ({torch_index})")

    # Install PyTorch
    success = install_pytorch(torch_index)

    if not success:
        print("\n❌ Installation failed!")
        sys.exit(1)

    # Verify installation
    if not args.cpu_only:
        cuda_available = verify_cuda()

        if cuda_available:
            print("✅ SUCCESS! PyTorch with CUDA support is ready.")
            print("\nNext step: Install lesseg_unet")
            print("  cd lesseg_unet")
            print("  pip install -e .")
        else:
            print("⚠️  PyTorch installed but CUDA is not available.")
            print("This may indicate a driver/toolkit version mismatch.")
            sys.exit(1)
    else:
        print("✅ CPU-only PyTorch installed.")
        print("\nNext step: Install lesseg_unet")
        print("  cd lesseg_unet")
        print("  pip install -e .")


if __name__ == "__main__":
    main()

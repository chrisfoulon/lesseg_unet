# Lesion Segmentation U-Net (lesseg\_unet)

This project provides automated segmentation of acute ischemic stroke lesions from diffusion-weighted MRI (DWI) scans using deep learning models. It supports various models (UNet, UNETR, SWIN-UNETR) and includes distributed training, advanced data transformations, and segmentation output options.

## Features

- Supports UNet and transformer-based UNETR/SWIN-UNETR architectures.
- Distributed training with PyTorch (DDP).
- Flexible input/output options: paths, lists, or segmentation dictionaries.
- Customizable transformation and augmentation settings.
- Comprehensive segmentation evaluation and output options.

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-capable GPU (optional, for GPU training)
- CUDA Toolkit 11.8, 12.6, or 12.8 (if using GPU)

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd lesseg_unet
```

### Step 2: Create a virtual environment

```bash
# Using conda (recommended)
conda create -n lesseg_unet_env python=3.11
conda activate lesseg_unet_env

# OR using venv
python -m venv lesseg_unet_env
source lesseg_unet_env/bin/activate  # On Windows: lesseg_unet_env\Scripts\activate
```

### Step 3: Install PyTorch with CUDA support

**IMPORTANT**: Install PyTorch BEFORE installing the package to ensure CUDA support.

#### Option A: Automated Installation (Recommended)

Use the provided helper script that auto-detects your CUDA version:

```bash
python install_pytorch_cuda.py
```

This will:
- Auto-detect your CUDA version from nvidia-smi
- Install the correct PyTorch build
- Verify GPU accessibility

#### Option B: Manual Installation

First, check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```

Then install PyTorch with the matching CUDA version:

```bash
# For CUDA 12.8 (most recent)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (no GPU)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch can see your GPU:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

### Step 4: Install the package

```bash
pip install -e .
```

**Note**: If you install the package without GPU-enabled PyTorch, you'll see a warning on first import with instructions to fix it.

### Troubleshooting

If you get CUDA errors despite nvidia-smi working:
1. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - If it shows `2.7.0+cpu`, you have the CPU-only version
   - Reinstall with the correct `--index-url` (see Step 3)
2. See `/tmp/cuda_diagnostics.md` for detailed troubleshooting

## Usage

The script provides multiple entry points for training, segmentation, and validation. Below are common usage examples:

### General Syntax

```bash
python -m lesseg_unet.main --output <output_directory> <additional_flags>
```

### Training Example

```bash
python -m lesseg_unet.main --output ./training_output \
                            --input_path ./data/train_images \
                            --lesion_input_path ./data/train_labels \
                            --model_type SWINUNETR \
                            --num_epochs 100 \
                            --batch_size 8
```

### Segmentation Example

```bash
torchrun --nproc_per_node=1 ./lesseg_unet/main.py \
    -o "./data/output_folder" \
    -p "./data/b1000_folder" \
    -trs unetr_cc \
    -nw 2 \
    -mt swin-unetr \
    -overlap \
    -sa \
    -pt "./data/checkpoint.pth"
```

### Validation Example

```bash
python -m lesseg_unet.main --output ./validation_output \
                            --checkpoint ./checkpoints/best_model.pth \
                            --input_path ./data/validation_images \
                            --lesion_input_path ./data/validation_labels
```

### Key Arguments

- `--output`: Directory to save output results.
- `--input_path`, `--input_list`: Path or list of input MRI images.
- `--lesion_input_path`, `--lesion_input_list`: Path or list of corresponding lesion labels.
- `--checkpoint`: Path to a model checkpoint file.
- `--model_type`: Model architecture (`UNet`, `UNETR`, or `SWINUNETR`).
- `--num_epochs`: Number of training epochs.
- `--batch_size`: Training batch size.
- `--output_mode`: Output format (`segmentation`, `sigmoid`, `logits`).

## Checkpoints

- **Best Performing Model:** [Best Model Checkpoint](https://www.dropbox.com/scl/fi/v57g6gg0skd7z7zakhha2/fold_1_best_model_segmentation3d_epo_219.pth?rlkey=6h92ywvoo6bu5f9nqeyz0jxva\&st=7gvpmpxh\&dl=0)
- **All Fold Checkpoints:** [All Folds Checkpoints Folder](https://www.dropbox.com/scl/fo/65s2mzocvg3olws4td8pm/AIfc7aw_WooRjjOcFIjqXAM?rlkey=sr4b67llz7yjx547wfqnjs2g9\&st=gpeyn27o\&dl=0)

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Contact

For further inquiries or support, please contact the project maintainers or open an issue on the repository.


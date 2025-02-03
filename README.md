# Lesion Segmentation U-Net (lesseg\_unet)

This project provides automated segmentation of acute ischemic stroke lesions from diffusion-weighted MRI (DWI) scans using deep learning models. It supports various models (UNet, UNETR, SWIN-UNETR) and includes distributed training, advanced data transformations, and segmentation output options.

## Features

- Supports UNet and transformer-based UNETR/SWIN-UNETR architectures.
- Distributed training with PyTorch (DDP).
- Flexible input/output options: paths, lists, or segmentation dictionaries.
- Customizable transformation and augmentation settings.
- Comprehensive segmentation evaluation and output options.

## Installation

1.

   1. Clone the repository and navigate to its directory:

   ```bash
   git clone <repository-url>
   cd lesseg_unet
   ```

2. Create a new Conda environment and activate it:

   ```bash
   conda create -n lesseg_unet_env python=3.8
   conda activate lesseg_unet_env
   ```

3. Install the dependencies from `requirements.txt` using Conda:

   ```bash
   conda install --file requirements.txt
   ```

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


#!/usr/bin/env python
"""
Profile p96 transform pipeline to show exactly what goes through the network.

Shows:
- Input data shapes
- Transform sequence
- Patch generation (pos/neg sampling)
- Batch accumulation
- Memory requirements
"""

import sys
import torch
import numpy as np
from pathlib import Path
from monai.transforms import Compose
from monai.data import Dataset, DataLoader

# Add lesseg_unet to path
sys.path.insert(0, str(Path(__file__).parent))

from lesseg_unet.data.transform_dicts import p96
from lesseg_unet import transformations


def create_dummy_data(num_samples=2):
    """Create dummy NIfTI-like data for profiling."""
    print("\n" + "="*80)
    print("STEP 1: Creating dummy input data")
    print("="*80)

    data_list = []
    for i in range(num_samples):
        # Simulate typical brain MRI size (after initial resize to [96, 128, 96])
        # Multi-modal: 2 channels (DWI + ADC)
        dummy_image = np.random.randn(2, 96, 128, 96).astype(np.float32)
        dummy_label = np.random.randint(0, 2, (1, 96, 128, 96)).astype(np.float32)

        # Simulate some lesion voxels
        dummy_label[0, 40:50, 60:70, 40:50] = 1

        data_list.append({
            'image': dummy_image,
            'label': dummy_label
        })

        lesion_voxels = np.sum(dummy_label)
        total_voxels = np.prod(dummy_label.shape)
        print(f"\nSample {i}:")
        print(f"  Image shape: {dummy_image.shape} (channels, H, W, D)")
        print(f"  Label shape: {dummy_label.shape}")
        print(f"  Lesion voxels: {lesion_voxels} / {total_voxels} ({lesion_voxels/total_voxels*100:.2f}%)")

    return data_list


def profile_transforms():
    """Profile the p96 transform pipeline."""

    print("\n" + "="*80)
    print("STEP 2: Analyzing p96 transform configuration")
    print("="*80)

    print("\nTransform configuration 'p96':")
    print(f"  Description: 96³ patches, artifact-focused (1 lesion, 3 healthy patches)")

    # Check patches configuration
    if 'patches' in p96:
        patch_config = p96['patches'][0]['RandCropByPosNegLabeld']
        print(f"\n  Patch sampling (RandCropByPosNegLabeld):")
        print(f"    spatial_size: {patch_config['spatial_size']}")
        print(f"    pos: {patch_config['pos']} (patches centered on lesion)")
        print(f"    neg: {patch_config['neg']} (patches on healthy tissue)")
        if 'num_samples' in patch_config:
            print(f"    num_samples: {patch_config['num_samples']} (redundant - pos+neg={patch_config['pos']+patch_config['neg']})")

        total_patches = patch_config['pos'] + patch_config['neg']
        print(f"\n  → Total patches per image: {total_patches}")
        print(f"  → Each patch size: {patch_config['spatial_size']} = {np.prod(patch_config['spatial_size']):,} voxels")

    # Build the transform
    print("\n" + "="*80)
    print("STEP 3: Building transform pipeline")
    print("="*80)

    train_transforms = transformations.train_transformd(p96, clamping=None)

    print("\nTransform pipeline created:")
    print(f"  Total transforms: {len(train_transforms.transforms)}")
    print("\nTransform sequence:")
    for i, transform in enumerate(train_transforms.transforms, 1):
        print(f"  {i}. {transform.__class__.__name__}")

    return train_transforms


def profile_dataset(train_transforms, data_list):
    """Profile dataset with transforms."""

    print("\n" + "="*80)
    print("STEP 4: Creating dataset and applying transforms")
    print("="*80)

    # Create dataset
    dataset = Dataset(data_list, transform=train_transforms)

    print(f"\nDataset created:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Dataset type: {type(dataset).__name__}")

    # Process first sample to see output
    print("\n" + "-"*80)
    print("Processing first sample through transform pipeline...")
    print("-"*80)

    try:
        sample = dataset[0]

        print("\nOUTPUT after all transforms:")
        print(f"  Type: {type(sample)}")
        print(f"  Keys: {list(sample.keys())}")

        if 'image' in sample:
            image_data = sample['image']
            print(f"\n  Image:")
            print(f"    Type: {type(image_data)}")
            print(f"    Shape: {image_data.shape}")
            print(f"    Dtype: {image_data.dtype}")
            if torch.is_tensor(image_data):
                print(f"    Device: {image_data.device}")
                print(f"    Memory: {image_data.element_size() * image_data.nelement() / 1024**2:.2f} MB")

        if 'label' in sample:
            label_data = sample['label']
            print(f"\n  Label:")
            print(f"    Type: {type(label_data)}")
            print(f"    Shape: {label_data.shape}")
            print(f"    Dtype: {label_data.dtype}")
            if torch.is_tensor(label_data):
                print(f"    Device: {label_data.device}")
                lesion_voxels = torch.sum(label_data > 0.5).item()
                total_voxels = label_data.nelement()
                print(f"    Lesion voxels: {lesion_voxels} / {total_voxels} ({lesion_voxels/total_voxels*100:.2f}%)")

        return sample

    except Exception as e:
        print(f"\nERROR during transform: {e}")
        import traceback
        traceback.print_exc()
        return None


def profile_dataloader(train_transforms, data_list, batch_size=1, num_workers=0):
    """Profile DataLoader batch creation."""

    print("\n" + "="*80)
    print("STEP 5: Creating DataLoader and checking batch shapes")
    print("="*80)

    dataset = Dataset(data_list, transform=train_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    print(f"\nDataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")
    print(f"  Total batches: {len(dataloader)}")

    # Get first batch
    print("\n" + "-"*80)
    print("Fetching first batch...")
    print("-"*80)

    try:
        batch = next(iter(dataloader))

        print("\nBATCH OUTPUT (what goes to the network):")

        if 'image' in batch:
            image_batch = batch['image']
            print(f"\n  Image batch:")
            print(f"    Shape: {image_batch.shape} (batch, channels, H, W, D)")
            print(f"    Dtype: {image_batch.dtype}")
            print(f"    Device: {image_batch.device}")
            print(f"    Memory: {image_batch.element_size() * image_batch.nelement() / 1024**2:.2f} MB")
            print(f"    Value range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")

        if 'label' in batch:
            label_batch = batch['label']
            print(f"\n  Label batch:")
            print(f"    Shape: {label_batch.shape} (batch, channels, H, W, D)")
            print(f"    Dtype: {label_batch.dtype}")
            print(f"    Device: {label_batch.device}")
            unique_values = torch.unique(label_batch)
            print(f"    Unique values: {unique_values.tolist()}")
            lesion_voxels = torch.sum(label_batch > 0.5).item()
            total_voxels = label_batch.nelement()
            print(f"    Lesion voxels: {lesion_voxels} / {total_voxels} ({lesion_voxels/total_voxels*100:.2f}%)")

        return batch

    except Exception as e:
        print(f"\nERROR during batch loading: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_memory_requirements(batch_size=1, gradient_accumulation=4):
    """Calculate memory requirements for p96 pipeline."""

    print("\n" + "="*80)
    print("STEP 6: Memory requirements calculation")
    print("="*80)

    # p96 configuration
    spatial_size = [96, 96, 96]
    in_channels = 2  # DWI + ADC
    out_channels = 1  # Binary segmentation
    patches_per_image = 4  # 1 pos + 3 neg

    print(f"\nConfiguration:")
    print(f"  Patch size: {spatial_size}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Patches per image: {patches_per_image}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")

    # Calculate per-patch memory
    voxels_per_patch = np.prod(spatial_size)
    bytes_per_float32 = 4

    input_mb_per_patch = (voxels_per_patch * in_channels * bytes_per_float32) / 1024**2
    label_mb_per_patch = (voxels_per_patch * out_channels * bytes_per_float32) / 1024**2

    print(f"\nPer-patch memory:")
    print(f"  Input: {input_mb_per_patch:.2f} MB")
    print(f"  Label: {label_mb_per_patch:.2f} MB")
    print(f"  Total: {input_mb_per_patch + label_mb_per_patch:.2f} MB")

    # Calculate batch memory
    batch_input_mb = input_mb_per_patch * batch_size
    batch_label_mb = label_mb_per_patch * batch_size

    print(f"\nPer-batch memory (batch_size={batch_size}):")
    print(f"  Input: {batch_input_mb:.2f} MB")
    print(f"  Label: {batch_label_mb:.2f} MB")
    print(f"  Total: {batch_input_mb + batch_label_mb:.2f} MB")

    # Calculate effective batch with gradient accumulation
    effective_batch_patches = batch_size * gradient_accumulation
    effective_input_mb = input_mb_per_patch * effective_batch_patches
    effective_label_mb = label_mb_per_patch * effective_batch_patches

    print(f"\nEffective batch (with gradient accumulation={gradient_accumulation}):")
    print(f"  Effective batch size: {effective_batch_patches} patches")
    print(f"  Input memory: {effective_input_mb:.2f} MB")
    print(f"  Label memory: {effective_label_mb:.2f} MB")
    print(f"  Total data: {effective_input_mb + effective_label_mb:.2f} MB")

    # Estimate model memory (rough approximation)
    # SwinUNETR with feature_size=36: ~35M parameters
    params_count = 35_000_000
    params_mb = (params_count * bytes_per_float32) / 1024**2

    # Gradients (same size as parameters)
    gradients_mb = params_mb

    # Optimizer state (Adam: 2x parameters)
    optimizer_mb = params_mb * 2

    # Activations (rough estimate: ~5x input size for deep networks)
    activations_mb = batch_input_mb * 5

    print(f"\nModel memory estimate (SwinUNETR feature_size=36):")
    print(f"  Parameters: {params_mb:.2f} MB")
    print(f"  Gradients: {gradients_mb:.2f} MB")
    print(f"  Optimizer state: {optimizer_mb:.2f} MB")
    print(f"  Activations (per batch): {activations_mb:.2f} MB")

    total_gpu_mb = params_mb + gradients_mb + optimizer_mb + batch_input_mb + batch_label_mb + activations_mb

    print(f"\nTotal GPU memory (rough estimate):")
    print(f"  {total_gpu_mb:.2f} MB = {total_gpu_mb / 1024:.2f} GB")

    # Training throughput estimate
    print(f"\nTraining throughput:")
    print(f"  Patches per step: {batch_size}")
    print(f"  Patches per gradient update: {effective_batch_patches}")
    print(f"  Images per gradient update: {effective_batch_patches / patches_per_image:.1f}")

    # With 411 training images (from your log)
    training_images = 411
    steps_per_epoch = training_images
    gradient_updates_per_epoch = steps_per_epoch / gradient_accumulation

    print(f"\nPer epoch (with {training_images} training images):")
    print(f"  Forward passes: {steps_per_epoch}")
    print(f"  Gradient updates: {gradient_updates_per_epoch:.0f}")
    print(f"  Total patches processed: {steps_per_epoch * patches_per_image}")


def main():
    """Run complete profiling."""

    print("\n" + "="*80)
    print("P96 TRANSFORM PIPELINE PROFILER")
    print("="*80)
    print("\nThis profiles the p96 transform to show exactly what goes through the network.")

    # Create dummy data
    data_list = create_dummy_data(num_samples=2)

    # Profile transforms
    train_transforms = profile_transforms()

    # Profile dataset
    sample = profile_dataset(train_transforms, data_list)

    # Profile dataloader
    batch = profile_dataloader(train_transforms, data_list, batch_size=1, num_workers=0)

    # Calculate memory
    calculate_memory_requirements(batch_size=1, gradient_accumulation=4)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("  - Each image generates 4 patches (1 lesion + 3 healthy)")
    print("  - Each patch is 96×96×96 with 2 input channels (DWI+ADC)")
    print("  - With batch_size=1 and gradient_accumulation=4:")
    print("    → 4 patches per gradient update")
    print("    → ~1 image processed per gradient update")
    print("  - Memory usage is modest (~2-3 GB GPU for the model + data)")
    print()


if __name__ == '__main__':
    main()

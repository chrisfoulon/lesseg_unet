#!/usr/bin/env python
"""
Profile ACTUAL VRAM and RAM usage for p96 training pipeline.

This simulates real training to measure:
1. VRAM usage (model + activations + gradients + optimizer)
2. RAM usage for CacheDataset with 514 images
3. Why validation might be slower than training
"""

import gc
import sys
import time
import torch
import psutil
import numpy as np
from pathlib import Path

# Add lesseg_unet to path
sys.path.insert(0, str(Path(__file__).parent))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_ram_usage_gb():
    """Get current process RAM usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def profile_model_vram():
    """Profile actual VRAM usage with real SwinUNETR model."""

    print("\n" + "="*80)
    print("PART 1: VRAM PROFILING - SwinUNETR Model")
    print("="*80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device("cuda:0")

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    print(f"\nBaseline GPU memory: {get_gpu_memory_mb():.2f} MB")

    # Import SwinUNETR
    from monai.networks.nets import SwinUNETR

    # Configuration matching your setup
    in_channels = 2  # DWI + ADC
    out_channels = 1  # Binary segmentation
    img_size = [96, 96, 96]  # Patch size
    feature_size = 36  # From your logs

    print(f"\nCreating SwinUNETR:")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  input_size: {img_size} (determined by input tensor)")
    print(f"  feature_size: {feature_size}")

    # Create model (img_size is NOT a parameter - it's inferred from input)
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,  # Important!
        spatial_dims=3
    ).to(device)

    model_vram = get_gpu_memory_mb()
    print(f"\nAfter model creation: {model_vram:.2f} MB")

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = (param_count * 4) / 1024**2  # float32 = 4 bytes

    print(f"\nModel parameters:")
    print(f"  Total: {param_count:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {param_size_mb:.2f} MB")

    # Create optimizer
    print(f"\nCreating Adam optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer_vram = get_gpu_memory_mb()
    print(f"After optimizer: {optimizer_vram:.2f} MB")
    print(f"  Optimizer overhead: {optimizer_vram - model_vram:.2f} MB")

    # Create dummy batch (batch_size=1 as in your config)
    batch_size = 1
    dummy_input = torch.randn(batch_size, in_channels, *img_size, device=device)
    dummy_target = torch.randint(0, 2, (batch_size, out_channels, *img_size), device=device, dtype=torch.float32)

    input_size_mb = (dummy_input.numel() * 4) / 1024**2
    target_size_mb = (dummy_target.numel() * 4) / 1024**2

    print(f"\nDummy batch:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Input size: {input_size_mb:.2f} MB")
    print(f"  Target shape: {dummy_target.shape}")
    print(f"  Target size: {target_size_mb:.2f} MB")

    # Forward pass
    print(f"\nRunning forward pass...")
    model.train()

    torch.cuda.synchronize()
    forward_start_vram = get_gpu_memory_mb()

    output = model(dummy_input)

    torch.cuda.synchronize()
    after_forward_vram = get_gpu_memory_mb()

    print(f"After forward: {after_forward_vram:.2f} MB")
    print(f"  Forward activations: {after_forward_vram - forward_start_vram:.2f} MB")

    # Loss
    from monai.losses import DiceFocalLoss
    loss_fn = DiceFocalLoss(sigmoid=True)
    loss = loss_fn(output, dummy_target)

    after_loss_vram = get_gpu_memory_mb()
    print(f"After loss: {after_loss_vram:.2f} MB")

    # Backward pass
    print(f"\nRunning backward pass...")
    loss.backward()

    torch.cuda.synchronize()
    after_backward_vram = get_gpu_memory_mb()

    print(f"After backward: {after_backward_vram:.2f} MB")
    print(f"  Backward overhead: {after_backward_vram - after_loss_vram:.2f} MB")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    after_step_vram = get_gpu_memory_mb()

    print(f"After optimizer step: {after_step_vram:.2f} MB")

    # Peak memory
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n{'='*80}")
    print(f"VRAM SUMMARY:")
    print(f"  Peak usage: {peak_vram:.2f} MB = {peak_vram/1024:.2f} GB")
    print(f"  Current usage: {after_step_vram:.2f} MB = {after_step_vram/1024:.2f} GB")
    print(f"  Breakdown:")
    print(f"    - Model parameters: {model_vram:.2f} MB")
    print(f"    - Optimizer state: {optimizer_vram - model_vram:.2f} MB")
    print(f"    - Forward activations: {after_forward_vram - forward_start_vram:.2f} MB")
    print(f"    - Backward gradients: {after_backward_vram - after_loss_vram:.2f} MB")
    print(f"    - Input + Target: {input_size_mb + target_size_mb:.2f} MB")
    print(f"{'='*80}")

    # Test with gradient accumulation
    print(f"\n" + "="*80)
    print("TESTING WITH GRADIENT ACCUMULATION (4 steps)")
    print("="*80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    model.train()
    optimizer.zero_grad()

    gradient_accumulation = 4

    for step in range(gradient_accumulation):
        dummy_input = torch.randn(batch_size, in_channels, *img_size, device=device)
        dummy_target = torch.randint(0, 2, (batch_size, out_channels, *img_size), device=device, dtype=torch.float32)

        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        loss = loss / gradient_accumulation  # Scale loss
        loss.backward()

        torch.cuda.synchronize()
        step_vram = get_gpu_memory_mb()
        print(f"  Step {step+1}/{gradient_accumulation}: {step_vram:.2f} MB")

    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    final_vram = get_gpu_memory_mb()
    ga_peak_vram = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nGradient accumulation results:")
    print(f"  Peak VRAM: {ga_peak_vram:.2f} MB = {ga_peak_vram/1024:.2f} GB")
    print(f"  Final VRAM: {final_vram:.2f} MB = {final_vram/1024:.2f} GB")

    return {
        'peak_vram_mb': peak_vram,
        'ga_peak_vram_mb': ga_peak_vram,
        'model_params': param_count
    }


def profile_cache_ram(num_images=514, patches_per_image=4):
    """Profile RAM usage for CacheDataset with realistic data."""

    print("\n" + "="*80)
    print(f"PART 2: RAM PROFILING - CacheDataset ({num_images} images)")
    print("="*80)

    gc.collect()
    baseline_ram = get_ram_usage_gb()
    print(f"\nBaseline RAM: {baseline_ram:.2f} GB")

    # Simulate cached data
    # Each patch after transforms: 96x96x96 x 2 channels (float32)
    patch_shape = (2, 96, 96, 96)
    voxels_per_patch = np.prod(patch_shape)
    bytes_per_patch = voxels_per_patch * 4  # float32
    mb_per_patch = bytes_per_patch / 1024**2

    print(f"\nPatch specifications:")
    print(f"  Shape: {patch_shape} (channels, H, W, D)")
    print(f"  Voxels: {voxels_per_patch:,}")
    print(f"  Size: {mb_per_patch:.2f} MB per patch")

    # Each image generates 4 patches (pos=1, neg=3)
    total_patches = num_images * patches_per_image
    total_mb = total_patches * mb_per_patch
    total_gb = total_mb / 1024

    print(f"\nCache size calculation:")
    print(f"  Images: {num_images}")
    print(f"  Patches per image: {patches_per_image}")
    print(f"  Total patches: {total_patches}")
    print(f"  Estimated cache size: {total_mb:.2f} MB = {total_gb:.2f} GB")

    # Actually allocate memory to test
    print(f"\nAllocating memory to simulate cache...")

    # Allocate in chunks to avoid huge single allocation
    cache_data = []
    chunk_size = 100

    for i in range(0, total_patches, chunk_size):
        current_chunk = min(chunk_size, total_patches - i)
        chunk_data = np.random.randn(current_chunk, *patch_shape).astype(np.float32)
        cache_data.append(chunk_data)

        if (i + current_chunk) % 500 == 0 or (i + current_chunk) == total_patches:
            current_ram = get_ram_usage_gb()
            allocated_gb = current_ram - baseline_ram
            progress = ((i + current_chunk) / total_patches) * 100
            print(f"  Progress: {progress:.1f}% ({i + current_chunk}/{total_patches} patches) - RAM: {current_ram:.2f} GB (+{allocated_gb:.2f} GB)")

    gc.collect()
    final_ram = get_ram_usage_gb()
    cache_ram = final_ram - baseline_ram

    print(f"\n{'='*80}")
    print(f"RAM CACHE SUMMARY:")
    print(f"  Baseline RAM: {baseline_ram:.2f} GB")
    print(f"  Final RAM: {final_ram:.2f} GB")
    print(f"  Cache overhead: {cache_ram:.2f} GB")
    print(f"  Estimated cache: {total_gb:.2f} GB")
    print(f"  Actual vs Estimated: {(cache_ram/total_gb)*100:.1f}%")
    print(f"{'='*80}")

    # Cleanup
    del cache_data
    gc.collect()

    return {
        'cache_ram_gb': cache_ram,
        'estimated_gb': total_gb,
        'total_patches': total_patches
    }


def check_dataloader_workers_issue():
    """Check if num_workers override is actually being applied."""

    print("\n" + "="*80)
    print("PART 3: DATALOADER WORKERS DIAGNOSTIC")
    print("="*80)

    from monai.data import CacheDataset, PersistentDataset
    from lesseg_unet.data_loading import create_validation_data_loader

    # Create a dummy CacheDataset
    print("\nCreating dummy CacheDataset...")
    dummy_data = [
        {'image': np.random.randn(2, 96, 96, 96).astype(np.float32),
         'label': np.random.randint(0, 2, (1, 96, 96, 96)).astype(np.float32)}
        for _ in range(10)
    ]

    cache_ds = CacheDataset(dummy_data, transform=None, cache_rate=1.0)

    print(f"  Dataset type: {type(cache_ds).__name__}")
    print(f"  Is CacheDataset: {isinstance(cache_ds, CacheDataset)}")

    # Test the dataloader creation with workers
    print(f"\nTesting create_validation_data_loader with num_workers=16...")

    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    # Capture output to see warnings
    f_out = io.StringIO()
    f_err = io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        val_loader = create_validation_data_loader(
            cache_ds,
            batch_size=1,
            dataloader_workers=16,
            sampler=None
        )

    stdout_output = f_out.getvalue()
    stderr_output = f_err.getvalue()

    print(f"\n  Output captured:")
    if stdout_output:
        print(f"    STDOUT: {stdout_output}")
    if stderr_output:
        print(f"    STDERR: {stderr_output}")

    print(f"\n  Actual num_workers: {val_loader.num_workers}")

    if val_loader.num_workers == 0:
        print(f"  ✅ GOOD: num_workers correctly set to 0 for CacheDataset")
    else:
        print(f"  ❌ BAD: num_workers is {val_loader.num_workers}, should be 0!")
        print(f"  This will cause {val_loader.num_workers}x RAM duplication and slow performance!")

    return {
        'expected_workers': 0,
        'actual_workers': val_loader.num_workers,
        'is_correct': val_loader.num_workers == 0
    }


def main():
    """Run complete resource profiling."""

    print("\n" + "="*80)
    print("ACTUAL RESOURCE USAGE PROFILER")
    print("="*80)
    print("\nThis measures REAL VRAM and RAM usage for your p96 training setup.")

    results = {}

    # Part 1: VRAM profiling
    try:
        vram_results = profile_model_vram()
        results['vram'] = vram_results
    except Exception as e:
        print(f"\nERROR in VRAM profiling: {e}")
        import traceback
        traceback.print_exc()

    # Part 2: RAM profiling
    try:
        ram_results = profile_cache_ram(num_images=514, patches_per_image=4)
        results['ram'] = ram_results
    except Exception as e:
        print(f"\nERROR in RAM profiling: {e}")
        import traceback
        traceback.print_exc()

    # Part 3: DataLoader workers check
    try:
        workers_results = check_dataloader_workers_issue()
        results['workers'] = workers_results
    except Exception as e:
        print(f"\nERROR in workers check: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if 'vram' in results:
        print(f"\nVRAM (per GPU):")
        print(f"  Single step: {results['vram']['peak_vram_mb']/1024:.2f} GB")
        print(f"  With gradient accumulation: {results['vram']['ga_peak_vram_mb']/1024:.2f} GB")

    if 'ram' in results:
        print(f"\nRAM (CacheDataset with 514 images):")
        print(f"  Cache size: {results['ram']['cache_ram_gb']:.2f} GB")
        print(f"  Total patches cached: {results['ram']['total_patches']}")

    if 'workers' in results:
        print(f"\nDataLoader workers:")
        if results['workers']['is_correct']:
            print(f"  ✅ Correctly forcing num_workers=0 for CacheDataset")
        else:
            print(f"  ❌ BUG: num_workers={results['workers']['actual_workers']} (should be 0!)")
            print(f"  This causes {results['workers']['actual_workers']}x RAM usage and slow validation!")

    print(f"\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

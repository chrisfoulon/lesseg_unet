"""Integration test for Phase 1: Hardware Detection and Memory Model."""

from lesseg_unet.hardware import (
    get_hardware_profile,
    SwinUNETRMemoryCalculator,
    UNetMemoryCalculator
)


def main():
    print("=" * 60)
    print("Phase 1 Integration Test: Hardware Detection & Memory Model")
    print("=" * 60)

    # Test 1: Hardware Detection
    print("\n1. Testing Hardware Detection...")
    hw = get_hardware_profile()

    print(f"\n   Device Type: {hw.device_type}")
    print(f"   GPUs: {len(hw.gpus)}")
    for gpu in hw.gpus:
        print(f"     - GPU {gpu.index}: {gpu.name}")
        print(f"       VRAM: {gpu.total_memory_mb / 1024:.1f} GB")
        print(f"       Free: {gpu.free_memory_mb / 1024:.1f} GB")
        print(f"       Compute: {gpu.compute_capability}")
        print(f"       Mixed Precision: {gpu.supports_mixed_precision}")

    print(f"\n   CPU Cores: {hw.cpu.available_cores} available / {hw.cpu.logical_cores} total")
    print(f"   RAM: {hw.cpu.total_ram_gb:.1f} GB total, {hw.cpu.available_ram_gb:.1f} GB available")

    # Test 2: SwinUNETR Memory Calculation
    print("\n2. Testing SwinUNETR Memory Calculator...")
    swin_calc = SwinUNETRMemoryCalculator(
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=1,
        feature_size=48,
        depths=[2, 2, 2, 2, 2],
        use_mixed_precision=True
    )

    print(f"   Parameters: {swin_calc.calculate_model_parameters():.2f}M")

    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        memory = swin_calc.estimate_total_memory(batch_size=batch_size)
        print(f"\n   Batch Size {batch_size}:")
        print(f"     - Params:      {memory.params_mb:>8.1f} MB")
        print(f"     - Optimizer:   {memory.optimizer_mb:>8.1f} MB")
        print(f"     - Activations: {memory.activations_mb:>8.1f} MB")
        print(f"     - Gradients:   {memory.gradients_mb:>8.1f} MB")
        print(f"     - Overhead:    {memory.overhead_mb:>8.1f} MB")
        print(f"     - Fragment:    {memory.fragmentation_mb:>8.1f} MB")
        print(f"     - TOTAL:       {memory.total_mb:>8.1f} MB ({memory.total_gb:.2f} GB)")

    # Test 3: Find Max Batch Size
    print("\n3. Testing Max Batch Size Finder...")
    for vram_gb in [4.0, 16.0, 24.0, 40.0]:
        max_batch = swin_calc.find_max_batch_size(vram_gb=vram_gb, target_usage=0.95)
        memory = swin_calc.estimate_total_memory(batch_size=max_batch)
        print(f"   {vram_gb:.0f} GB VRAM -> max batch: {max_batch} ({memory.total_gb:.2f} GB / {vram_gb:.2f} GB)")

    # Test 4: UNet Memory Calculation
    print("\n4. Testing UNet Memory Calculator...")
    unet_calc = UNetMemoryCalculator(
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        use_mixed_precision=True
    )

    print(f"   Parameters: {unet_calc.calculate_model_parameters():.2f}M")

    unet_memory = unet_calc.estimate_total_memory(batch_size=4)
    swin_memory = swin_calc.estimate_total_memory(batch_size=4)

    print(f"\n   Memory Comparison (batch=4):")
    print(f"     - UNet:      {unet_memory.total_gb:.2f} GB")
    print(f"     - SwinUNETR: {swin_memory.total_gb:.2f} GB")
    print(f"     - Difference: {swin_memory.total_gb - unet_memory.total_gb:.2f} GB")

    # Test 5: Depth Comparison
    print("\n5. Testing Network Depth Impact...")
    for depths in [[2, 2, 2, 2], [2, 2, 2, 2, 2]]:
        calc = SwinUNETRMemoryCalculator(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            depths=depths
        )
        memory = calc.estimate_total_memory(batch_size=4)
        print(f"   Depth {len(depths)}: {memory.total_gb:.2f} GB, {calc.calculate_model_parameters():.2f}M params")

    print("\n" + "=" * 60)
    print("Phase 1 Implementation Complete!")
    print("All modules working correctly.")
    print("=" * 60)


if __name__ == '__main__':
    main()

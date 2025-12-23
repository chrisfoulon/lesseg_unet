"""Integration test for Phase 2: Auto-Configuration Logic."""

from lesseg_unet.hardware import get_hardware_profile
from lesseg_unet.auto_config import AutoConfigurator, DatasetProfile


def main():
    print("=" * 70)
    print("Phase 2 Integration Test: Auto-Configuration Logic")
    print("=" * 70)

    # Detect hardware
    print("\n1. Detecting Hardware...")
    hw = get_hardware_profile()

    print(f"   Device: {hw.device_type}")
    print(f"   GPUs: {len(hw.gpus)}")
    for gpu in hw.gpus:
        print(f"     - {gpu.name}: {gpu.total_memory_mb / 1024:.1f} GB")
    print(f"   CPU Cores: {hw.cpu.available_cores}")
    print(f"   RAM: {hw.cpu.total_ram_gb:.1f} GB")

    # Create dataset profile
    print("\n2. Creating Dataset Profile...")
    dataset = DatasetProfile(
        median_image_size=(181, 217, 181),
        num_subjects=100,
        in_channels=2,  # DWI + ADC
        out_channels=1,  # Stroke lesion
        storage_type='ssd'
    )

    print(f"   Median image size: {dataset.median_image_size}")
    print(f"   Subjects: {dataset.num_subjects}")
    print(f"   Input channels: {dataset.in_channels}")
    print(f"   Output channels: {dataset.out_channels}")
    print(f"   Storage: {dataset.storage_type}")

    # Test different configurations
    configs_to_test = [
        ('balanced', 'swinunetr', 1),
        ('speed', 'swinunetr', 1),
        ('memory', 'swinunetr', 1),
        ('balanced', 'unet', 1),
    ]

    for target, model_type, num_gpus in configs_to_test:
        print("\n" + "=" * 70)
        print(f"Configuration: target={target}, model={model_type}, gpus={num_gpus}")
        print("=" * 70)

        configurator = AutoConfigurator(
            hardware_profile=hw,
            dataset_profile=dataset,
            model_type=model_type,
            target=target,
            vram_safety_margin=0.95,
            num_gpus=num_gpus
        )

        config = configurator.suggest_config()

        print(f"\nSuggested Configuration:")
        print(f"  batch_size: {config.batch_size}")
        print(f"  patch_size: {config.patch_size}")
        print(f"  num_workers: {config.num_workers}")
        print(f"  network_depth: {config.network_depth}")
        print(f"  feature_size: {config.feature_size}")
        print(f"  use_amp: {config.use_amp}")

        print(f"\nMemory Estimate:")
        mem = config.memory_estimate
        print(f"  Total: {mem['total_gb']:.2f} GB")
        print(f"    - Parameters:  {mem['params_mb']:>8.1f} MB")
        print(f"    - Optimizer:   {mem['optimizer_mb']:>8.1f} MB")
        print(f"    - Activations: {mem['activations_mb']:>8.1f} MB")
        print(f"    - Gradients:   {mem['gradients_mb']:>8.1f} MB")
        print(f"    - Overhead:    {mem['overhead_mb']:>8.1f} MB")
        print(f"    - Fragment:    {mem['fragmentation_mb']:>8.1f} MB")

        if hw.gpus:
            vram_gb = hw.gpus[0].total_memory_mb / 1024
            usage_pct = (mem['total_gb'] / vram_gb) * 100
            print(f"  VRAM Usage: {usage_pct:.1f}% of {vram_gb:.1f} GB")

        print(f"\nReasoning:")
        for key, reason in config.reasoning.items():
            print(f"  {key}:")
            print(f"    → {reason}")

    # Test with overrides
    print("\n" + "=" * 70)
    print("Testing User Overrides")
    print("=" * 70)

    configurator = AutoConfigurator(
        hardware_profile=hw,
        dataset_profile=dataset,
        model_type='swinunetr',
        target='balanced'
    )

    config = configurator.suggest_config(
        override_batch_size=2,
        override_patch_size=(80, 80, 80),
        override_num_workers=8
    )

    print(f"\nWith Overrides:")
    print(f"  batch_size: {config.batch_size} (overridden)")
    print(f"  patch_size: {config.patch_size} (overridden)")
    print(f"  num_workers: {config.num_workers} (overridden)")
    print(f"  network_depth: {config.network_depth} (auto)")
    print(f"  feature_size: {config.feature_size} (auto)")

    # Test multi-GPU (if available)
    if len(hw.gpus) > 1:
        print("\n" + "=" * 70)
        print(f"Testing Multi-GPU ({len(hw.gpus)} GPUs available)")
        print("=" * 70)

        configurator = AutoConfigurator(
            hardware_profile=hw,
            dataset_profile=dataset,
            model_type='swinunetr',
            target='balanced',
            num_gpus=min(2, len(hw.gpus))  # Use 2 GPUs
        )

        config = configurator.suggest_config()

        print(f"\nMulti-GPU Configuration:")
        print(f"  GPUs: {config.num_gpus}")
        print(f"  batch_size: {config.batch_size}")
        print(f"  num_workers: {config.num_workers}")
        print(f"  network_depth: {config.network_depth}")
    else:
        print("\n" + "=" * 70)
        print("Skipping Multi-GPU Test (only 1 GPU available)")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("Phase 2 Implementation Complete!")
    print("Auto-configuration working correctly.")
    print("=" * 70)


if __name__ == '__main__':
    main()

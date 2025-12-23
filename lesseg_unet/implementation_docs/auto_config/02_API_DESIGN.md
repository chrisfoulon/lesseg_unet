# API Design

## CLI Arguments

### Auto-Configuration Control
```bash
--auto_config                 # Enable auto-configuration (opt-in)
--auto_config_target {speed|memory|balanced}
                              # Optimization target (default: balanced)
--no_dryrun                   # Skip dry-run validation (faster but risky)
```

### Hardware Configuration
```bash
--num_gpus INT                # Number of GPUs to use (default: 1, opt-in for multi-GPU)
--gpu_ids STR                 # Specific GPU IDs (e.g., "0,2,5")
--vram_safety_margin FLOAT    # Target VRAM usage 0.0-1.0 (default: 0.95)
--ram_safety_margin FLOAT     # Target RAM usage (default: 0.90)
```

### Model Architecture
```bash
--network_depth {4|5}         # Network depth (default: auto-configured)
--feature_size INT            # SwinUNETR feature size (default: auto-configured)
```

### Manual Overrides (Expert Mode)
```bash
--override_batch_size INT     # Override auto-configured batch size
--override_patch_size INT INT INT  # Override patch size
--override_num_workers INT    # Override dataloader workers
```

## Python API

### Hardware Detection
```python
from lesseg_unet.hardware.detection import get_hardware_profile

hw = get_hardware_profile()
# Returns: HardwareProfile(gpus=[...], cpu={...}, ram_gb=...)
```

### Memory Estimation
```python
from lesseg_unet.hardware.memory_model import SwinUNETRMemoryCalculator

calc = SwinUNETRMemoryCalculator(
    img_size=(96, 96, 96),
    in_channels=2,
    out_channels=1,
    feature_size=48,
    depths=[2, 2, 2, 2, 2],
    use_mixed_precision=True
)

# Estimate memory for batch_size=4
memory = calc.estimate_total_memory(batch_size=4)
print(f"Total: {memory.total_mb / 1024:.2f} GB")

# Find max batch size
max_batch = calc.find_max_batch_size(vram_gb=24, target_usage=0.95)
```

### Auto-Configuration
```python
from lesseg_unet.auto_config.configurator import AutoConfigurator

config = AutoConfigurator(
    hardware_profile=hw,
    dataset_profile=dataset_info,
    model_type='swinunetr',
    target='balanced',
    vram_safety_margin=0.95
)

suggested = config.suggest_config()
# Returns: {
#   'batch_size': 4,
#   'patch_size': (96, 96, 96),
#   'num_workers': 8,
#   'network_depth': 5,
#   'use_amp': True,
#   ...
# }
```

### Config Management
```python
from lesseg_unet.auto_config.config_manager import TrainingConfig

config = TrainingConfig.from_args(
    args=parsed_args,
    auto_config=suggested_config,
    hardware_profile=hw.to_dict(),
    command=' '.join(sys.argv)
)

# Save to disk
config.save(output_dir)

# Load from disk
loaded = TrainingConfig.load(output_dir / 'config.yaml')
```

### Dry-Run Validation
```python
from lesseg_unet.hardware.validator import validate_config

success, stats = validate_config(
    config=suggested,
    model=model,
    sample_batch=sample,
    device='cuda'
)

if not success:
    # Reduce config and retry
    config['batch_size'] //= 2
```

## Example Usage

### Simple Auto-Config
```bash
# Automatic configuration with defaults
python -m lesseg_unet.main \
  --auto_config \
  -p /data/dwi /data/adc \
  -imn dwi adc \
  -lp /data/stroke \
  -lmn stroke \
  -o output/
```

### Multi-GPU with Custom Settings
```bash
# Use 2 GPUs with 98% VRAM target
python -m lesseg_unet.main \
  --auto_config \
  --num_gpus 2 \
  --vram_safety_margin 0.98 \
  -p /data/dwi /data/adc \
  -imn dwi adc \
  -lp /data/stroke \
  -lmn stroke \
  -o output/
```

### Manual Override
```bash
# Auto-config but force batch_size=8
python -m lesseg_unet.main \
  --auto_config \
  --override_batch_size 8 \
  -p /data/dwi /data/adc \
  -imn dwi adc \
  -lp /data/stroke \
  -lmn stroke \
  -o output/
```

### Resume from Config
```bash
# Resume with exact config from previous run
python -m lesseg_unet.main \
  --resume_from_config output/config.yaml \
  -p /data/dwi /data/adc \
  -lp /data/stroke \
  -o output_resumed/
```

## Output Files

After training starts:
```
output/
├── config.yaml              # Human-readable full config
├── config.json              # Machine-readable (same data)
├── hardware_profile.json    # Hardware info at training time
├── training_command.txt     # Original command
└── [existing training outputs]
```

## Logging Output

```
=== Hardware Detection ===
GPU 0: NVIDIA A100 40GB (40.0 GB VRAM, Compute 8.0)
GPU 1: NVIDIA A100 40GB (40.0 GB VRAM, Compute 8.0)
CPU: 64 cores (32 physical)
RAM: 256.0 GB

=== Auto-Configuration ===
Target: balanced
VRAM safety margin: 95%

Decisions:
  batch_size: 4 (per GPU: 2)
    → Max batch that fits in 40GB VRAM (95% target)
    → Estimated: 36.2 GB / 40.0 GB (90.5%)

  patch_size: (96, 96, 96)
    → Optimal for median image size (181, 217, 181)
    → Allows batch_size >= 2

  num_workers: 16 (per GPU: 8)
    → Proportional allocation: 2/8 GPUs = 25% of 64 cores

  network_depth: 5
    → Sufficient VRAM for 5 layers (standard)

  use_amp: True
    → GPU supports mixed precision (compute 8.0+)

=== Dry-Run Validation ===
Testing configuration...
✓ Success. Peak VRAM: 36.8 GB / 40.0 GB (92%)
Configuration validated and ready for training.

Configuration saved to output/config.yaml
```

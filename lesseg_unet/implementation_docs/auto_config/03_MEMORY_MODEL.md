# Theoretical Memory Model

## Overview
Calculate VRAM usage from model architecture + input dimensions without empirical profiling.

## Memory Components

### 1. Model Parameters
```python
params_bytes = count_parameters(model) * bytes_per_param

# FP32: 4 bytes per param
# FP16: 2 bytes per param
# Mixed precision: 4 bytes (optimizer keeps FP32 copy)
```

### 2. Forward Pass Activations
Intermediate tensors stored for backpropagation.

**For each layer**:
```python
activation_bytes = batch_size * output_channels * H * W * D * bytes_per_element

# bytes_per_element depends on dtype:
# - FP32: 4 bytes
# - FP16: 2 bytes (with AMP, most activations)
# - Mixed: average ~3 bytes (conservative)
```

**SwinUNETR specifics**:
- Patch embedding: `batch * feature_size * (H/4) * (W/4) * (D/4)`
- Each encoder stage: `batch * embed_dim * h * w * d` (embed_dim doubles, spatial halves)
- Attention maps: `batch * num_heads * seq_len * seq_len` (FP16 typically)
- Decoder: symmetric to encoder

### 3. Backward Pass Gradients
```python
gradient_bytes ≈ activation_bytes

# Gradients have same shape as activations
# Conservative: assume 1:1 ratio
# In practice, slightly less (some ops don't store grads)
```

### 4. Optimizer State (Adam)
```python
optimizer_bytes = params_bytes * 2

# Adam stores:
# - First moment (momentum): same size as params
# - Second moment (variance): same size as params
# Always FP32, even with AMP
```

### 5. PyTorch Overhead
```python
pytorch_overhead_mb = 400  # Constant

# Includes:
# - CUDA context (~200 MB)
# - Memory allocator metadata
# - Kernel cache
# - Temporary buffers
```

### 6. Memory Fragmentation
```python
fragmentation_mb = allocated_mb * 0.08  # 8% overhead

# PyTorch's caching allocator can fragment memory
# Conservative 5-10% overhead
```

## SwinUNETR Memory Formula

```python
def estimate_swinunetr_memory(
    batch_size: int,
    img_size: tuple,
    in_channels: int,
    feature_size: int,
    depths: list,
    use_amp: bool
) -> float:
    """
    Estimate total VRAM in MB.

    Returns breakdown and total.
    """

    # 1. Model parameters
    params = 0

    # Patch embedding (Conv3d: 4x4x4 patches)
    params += in_channels * 4**3 * feature_size

    # Encoder transformer blocks
    for i, depth in enumerate(depths):
        embed_dim = feature_size * (2 ** i)
        # Each block: LayerNorm + Attention + MLP
        params_per_block = (
            2 * embed_dim +           # LayerNorm
            4 * embed_dim**2 +        # Attention (Q, K, V, proj)
            8 * embed_dim**2          # MLP (2 layers, 4x expansion)
        )
        params += depth * params_per_block

    # Decoder (upsampling convs)
    for i in reversed(range(len(depths) - 1)):
        embed_dim = feature_size * (2 ** i)
        params += embed_dim**2 * 8  # 2x2x2 upsampling kernel

    # Output layer
    params += feature_size * out_channels

    params_mb = (params * 4) / 1024**2  # FP32

    # 2. Forward activations
    H, W, D = img_size
    bytes_per_elem = 3 if use_amp else 4  # Mixed precision avg

    activations_mb = 0

    # Input
    activations_mb += (batch_size * in_channels * H * W * D * bytes_per_elem) / 1024**2

    # Patch embed output
    h, w, d = H // 4, W // 4, D // 4
    activations_mb += (batch_size * feature_size * h * w * d * bytes_per_elem) / 1024**2

    # Encoder stages
    for i, depth in enumerate(depths):
        embed_dim = feature_size * (2 ** i)

        # Transformer blocks
        activations_mb += (batch_size * depth * embed_dim * h * w * d * bytes_per_elem) / 1024**2

        # Attention maps (typically FP16)
        seq_len = h * w * d
        num_heads = [3, 6, 12, 24, 48][i]
        activations_mb += (batch_size * num_heads * seq_len * seq_len * 2) / 1024**2

        # Downsample
        if i < len(depths) - 1:
            h, w, d = h // 2, w // 2, d // 2

    # Decoder stages
    for i in reversed(range(len(depths) - 1)):
        h, w, d = h * 2, w * 2, d * 2
        embed_dim = feature_size * (2 ** i)
        activations_mb += (batch_size * embed_dim * h * w * d * bytes_per_elem) / 1024**2

    # Output
    activations_mb += (batch_size * out_channels * H * W * D * bytes_per_elem) / 1024**2

    # 3. Gradients (≈ activations)
    gradients_mb = activations_mb

    # 4. Optimizer state (Adam, always FP32)
    optimizer_mb = params_mb * 2

    # 5. PyTorch overhead
    overhead_mb = 400

    # 6. Fragmentation
    allocated_mb = params_mb + activations_mb + gradients_mb + optimizer_mb
    fragmentation_mb = allocated_mb * 0.08

    # Total
    total_mb = params_mb + activations_mb + gradients_mb + optimizer_mb + overhead_mb + fragmentation_mb

    return {
        'params_mb': params_mb,
        'activations_mb': activations_mb,
        'gradients_mb': gradients_mb,
        'optimizer_mb': optimizer_mb,
        'overhead_mb': overhead_mb,
        'fragmentation_mb': fragmentation_mb,
        'total_mb': total_mb,
        'total_gb': total_mb / 1024
    }
```

## Finding Maximum Batch Size

Binary search for largest batch that fits:

```python
def find_max_batch_size(vram_gb: float, target_usage: float = 0.95) -> int:
    """
    Binary search for max batch size.

    target_usage: 0.95 = use 95% of VRAM (5% safety margin)
    """
    vram_mb = vram_gb * 1024 * target_usage

    low, high = 1, 64
    max_batch = 1

    while low <= high:
        mid = (low + high) // 2

        memory = estimate_swinunetr_memory(
            batch_size=mid,
            img_size=patch_size,
            ...
        )

        if memory['total_mb'] <= vram_mb:
            max_batch = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_batch
```

## Validation Strategy

1. **Theoretical calculation** (this model)
2. **Dry-run test** (actual forward + backward)
3. **Compare**: theoretical vs actual
4. **Adjust safety margin** if needed

Expected accuracy: ±5-10% (good enough for OOM prevention)

## Example Calculations

### Case 1: SwinUNETR (96³, batch=2, 24GB VRAM)
```
Parameters: 130M × 4 bytes = 520 MB
Optimizer: 520 MB × 2 = 1040 MB
Activations: ~8000 MB (batch=2, 96³)
Gradients: ~8000 MB
Overhead: 400 MB
Fragmentation: ~1500 MB
━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~19.5 GB (81% of 24GB) ✓
```

### Case 2: SwinUNETR (96³, batch=4, 24GB VRAM)
```
Parameters: 520 MB (same)
Optimizer: 1040 MB (same)
Activations: ~16000 MB (batch=4, doubles)
Gradients: ~16000 MB (doubles)
Overhead: 400 MB (same)
Fragmentation: ~2700 MB
━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~36.6 GB (153% of 24GB) ✗ OOM
```

### Case 3: Reducing to depth=4
```
Parameters: ~100M × 4 = 400 MB (fewer layers)
Optimizer: 800 MB
Activations: ~14000 MB (batch=4, depth=4)
Gradients: ~14000 MB
Overhead: 400 MB
Fragmentation: ~2300 MB
━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~31.9 GB (133% of 24GB) ✗ Still OOM

Reduce batch to 3:
Total: ~25.5 GB (106% of 24GB) ✗ Still tight

Reduce batch to 2:
Total: ~18.8 GB (78% of 24GB) ✓
```

## Key Insights

1. **Activations scale linearly with batch size** (biggest component)
2. **Parameters + optimizer are constant** (independent of batch)
3. **Gradients double memory** (need to store activations for backprop)
4. **Network depth affects activations** (more layers = more memory)
5. **Safety margin critical** (fragmentation is real, ~5-10%)

## Implementation Priority

Start with SwinUNETR (most complex, most resource-hungry).
UNet is simpler - same principles apply.

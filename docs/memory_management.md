# Memory Management

## Overview

lesseg_unet automatically manages CUDA memory to prevent out-of-memory (OOM) errors during long training runs. This is especially important when training with large models like SwinUNETR on multiple GPUs.

## Automatic Memory Management

### Expandable Segments (Default: Enabled)

**What it does:**
Enables PyTorch's `expandable_segments` feature, which allows CUDA memory allocations to grow dynamically, reducing memory fragmentation.

**Why it helps:**
During long training runs (100+ epochs), GPU memory becomes fragmented (like disk fragmentation). Even with plenty of "free" memory, PyTorch may fail to allocate large contiguous blocks, causing OOM errors. Expandable segments solves this by allowing allocations to expand into fragmented space.

**When to disable:**
Some GPU architectures (H100, A100) may have compatibility issues with this experimental feature. If you encounter errors like "expandable_segments not supported on this platform", disable it with:

```bash
--disable-expandable-segments
```

### Periodic Garbage Collection

**What it does:**
Runs Python's garbage collector and clears CUDA cache every 10 epochs.

**Why it helps:**
- Frees memory from deleted objects with circular references
- Returns unused GPU memory to the allocator
- Minimal overhead (< 1 second every 10 epochs)

**Implementation:**
Automatically enabled, cannot be disabled. Runs after validation completes.

## Usage Examples

### Normal Training (Memory Management Enabled)
```bash
torchrun --nproc_per_node=2 -m lesseg_unet.main \
  -p /path/to/dwi /path/to/adc \
  -imn dwi adc \
  -lp /path/to/labels \
  -o /path/to/output \
  -nw 8 \
  -loof 16384 \
  -bs 1 \
  -ne 1500
```

### Disable for H100/A100 Compatibility
```bash
torchrun --nproc_per_node=2 -m lesseg_unet.main \
  --disable-expandable-segments \
  [other arguments...]
```

## Troubleshooting

### Still Getting OOM Errors?

1. **Increase file descriptor limit** (for DataLoader workers):
   ```bash
   -loof 16384  # or higher
   ```

2. **Reduce batch size**:
   ```bash
   -bs 1  # minimum
   ```

3. **Enable gradient checkpointing** (for VRAM < 6GB):
   ```bash
   # Automatically enabled by auto_config for small GPUs
   --auto_config
   ```

4. **Reduce number of workers**:
   ```bash
   -nw 4  # or lower
   ```

5. **Check actual memory usage**:
   ```bash
   nvidia-smi dmon -s u
   ```

### Platform-Specific Issues

**H100/A100 Warning:**
If you see "expandable_segments not supported on this platform", use `--disable-expandable-segments`.

**Multi-GPU Issues:**
Ensure you're using the correct DDP setup with torchrun and that each GPU has enough VRAM for your batch size.

## Technical Details

### How Expandable Segments Works

PyTorch's default caching allocator pre-allocates large blocks of GPU memory and splits them as needed. Over time, these splits create fragmentation. Expandable segments allows blocks to dynamically grow into fragmented space, reducing allocation failures.

**Performance Impact:** Minimal (< 1% overhead in most cases)

### Garbage Collection Frequency

**Why every 10 epochs?**
- Frequent enough to prevent accumulation
- Infrequent enough to avoid overhead
- Balanced for typical training runs (100-1500 epochs)

**Overhead:** < 1 second per collection on modern systems

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Memory Semantics](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [lesseg_unet Issue #973](https://github.com/pytorch/pytorch/issues/973) - Original expandable_segments proposal

## Version History

- **v2.0.13**: Added automatic expandable_segments + periodic garbage collection

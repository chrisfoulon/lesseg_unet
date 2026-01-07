# Empirical Profiling Research

**Date:** 2026-01-07 (Recreated from lost /tmp file)
**Status:** Research Complete - Implementation Deferred (Task 4)
**Related:** Task 4 (Auto-Optimization with Profiling)

---

## Problem Statement

### Current Issue
The `auto_config` system uses **theoretical memory calculation** to estimate GPU memory usage and configure batch size/patch size. However:

**Problem:** Theoretical calculator **underestimates actual GPU memory by ~2x**, causing OOM errors even after enabling gradient checkpointing.

**Example:**
```
Theoretical estimate: 2.5GB VRAM needed
Actual usage:         5.0GB VRAM (2x higher!)
Result:               OOM error on 4GB GPU
```

### Why Theoretical Calculations Fail

1. **Framework Overhead Not Accounted For:**
   - PyTorch memory allocator overhead
   - CUDA kernel workspace memory
   - Temporary buffers during operations
   - cuDNN workspace

2. **Dynamic Memory Patterns:**
   - Operations create temporary tensors
   - Peak memory ≠ steady-state memory
   - Gradient accumulation creates additional buffers

3. **Model-Specific Factors:**
   - Attention mechanisms (transformer models)
   - Skip connections hold intermediate activations
   - Custom layers with unexpected memory behavior

4. **Hardware Variations:**
   - Different GPUs have different memory allocation strategies
   - Driver versions affect memory usage
   - Multi-GPU vs single-GPU has different overhead

**Conclusion:** Theoretical estimation is fundamentally unreliable for production use.

---

## Proposed Solution: Empirical Profiling

### Core Idea
**Run actual test batches** to measure real memory consumption, rather than estimating theoretically.

### Approach
Similar to **PyTorch Lightning** and **nnUNet**:

1. **Run test forward pass** with candidate configuration
2. **Measure actual GPU memory** used
3. **Catch OOM errors** safely (no crash)
4. **Binary search** to find optimal batch_size/patch_size that fits

### Advantages
✅ **Accurate:** Measures real memory usage on actual hardware
✅ **Safe:** Can catch and handle OOM errors
✅ **Adaptive:** Automatically adjusts to hardware differences
✅ **Fast:** Only needs 2-3 test runs (binary search)

### Disadvantages
⚠️ **Adds startup time:** ~30-60 seconds for profiling
⚠️ **Requires sample data:** Needs at least one data sample
⚠️ **Not perfect:** First epoch might still OOM if profiling sample is smaller than training data

---

## Implementation Design

### Phase 1: GPU VRAM Profiling (SAFE)

**Why Safe:** `torch.cuda.OutOfMemoryError` is catchable, process won't crash.

```python
def find_max_batch_size(model, spatial_size, device, max_tries=10):
    """
    Find maximum batch size that fits in GPU memory via binary search.

    Args:
        model: The neural network model
        spatial_size: (D, H, W) tuple for 3D patches
        device: CUDA device
        max_tries: Maximum search iterations

    Returns:
        int: Maximum safe batch size
    """
    import torch
    import gc

    model.eval()  # Eval mode (no gradients needed for sizing)

    low, high = 1, 32  # Search range
    best_batch_size = 1

    for iteration in range(max_tries):
        if low > high:
            break

        mid = (low + high + 1) // 2

        try:
            # Clear cache before test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            gc.collect()

            # Create test batch
            test_batch = torch.randn(
                mid, 1, *spatial_size,  # (B, C, D, H, W)
                device=device,
                dtype=torch.float32
            )

            # Test forward pass
            with torch.no_grad():
                output = model(test_batch)

            # Measure peak memory
            peak_memory = torch.cuda.max_memory_allocated(device)

            # Cleanup
            del test_batch, output
            torch.cuda.empty_cache()
            gc.collect()

            # Success - try larger batch
            best_batch_size = mid
            low = mid + 1

            print(f"  Batch size {mid}: OK ({peak_memory / 1e9:.2f}GB peak)")

        except torch.cuda.OutOfMemoryError:
            # Failed - try smaller batch
            torch.cuda.empty_cache()
            gc.collect()
            high = mid - 1

            print(f"  Batch size {mid}: OOM")

        except Exception as e:
            # Unexpected error - bail out
            print(f"  Batch size {mid}: Error ({e})")
            break

    return best_batch_size
```

**Usage:**
```python
# During training startup (before creating dataloaders)
if args.auto_optimize:
    print("Running GPU VRAM profiling...")
    optimal_batch_size = find_max_batch_size(
        model, spatial_size=(96, 128, 96), device=device
    )
    print(f"Optimal batch size: {optimal_batch_size}")

    # Override user's batch_size if too large
    if batch_size > optimal_batch_size:
        print(f"Reducing batch_size from {batch_size} to {optimal_batch_size}")
        batch_size = optimal_batch_size
```

### Phase 2: Training Memory Profiling (More Complex)

**Challenge:** Training mode uses more memory than inference (gradients, optimizer state).

```python
def find_max_batch_size_training(model, optimizer, spatial_size, device,
                                  gradient_accumulation_steps=1):
    """
    Find max batch size for TRAINING (with gradients and optimizer).

    More accurate than inference-only profiling.
    """
    model.train()  # Training mode

    low, high = 1, 16  # Training typically uses 2-3x more memory
    best_batch_size = 1

    for _ in range(10):
        if low > high:
            break

        mid = (low + high + 1) // 2

        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Create test batch
            test_input = torch.randn(mid, 1, *spatial_size, device=device)
            test_label = torch.randn(mid, 1, *spatial_size, device=device)

            # Test forward + backward pass
            optimizer.zero_grad()
            output = model(test_input)

            # Dummy loss
            loss = torch.nn.functional.mse_loss(output, test_label)

            # Backward pass (creates gradients)
            loss.backward()

            # Optimizer step (would update weights)
            # optimizer.step()  # Don't actually update

            # Cleanup
            del test_input, test_label, output, loss
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

            # Success
            best_batch_size = mid
            low = mid + 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            high = mid - 1

    return best_batch_size
```

### Phase 3: RAM Cache Size Estimation (UNSAFE)

**Why Unsafe:** Cannot safely test RAM limits - OS will kill process if exceeded.

**Solution:** Estimate only, don't test.

```python
def estimate_cache_size(sample_path, transform):
    """
    Estimate RAM cache size by transforming one sample.

    Returns estimated bytes for full dataset cache.
    """
    import psutil

    # Transform one sample
    sample = {'image': sample_path['image'], 'label': sample_path['label']}
    transformed = transform(sample)

    # Calculate size
    sample_size = 0
    for key, value in transformed.items():
        if hasattr(value, 'nbytes'):  # Numpy/torch tensor
            sample_size += value.nbytes

    return sample_size

# Usage:
num_samples = len(train_data_list)
sample_size = estimate_cache_size(train_data_list[0], train_transform)
total_cache_size = sample_size * num_samples
available_ram = psutil.virtual_memory().available

if total_cache_size > available_ram * 0.5:
    print(f"WARNING: Cache size (~{total_cache_size/1e9:.1f}GB) may exceed "
          f"available RAM ({available_ram/1e9:.1f}GB)")
    print("Recommendation: Use --cache-mode disk instead of --cache-mode ram")
```

---

## Integration Points

### 1. Auto-Config System
**File:** `lesseg_unet/auto_config/profiler.py` (new)

Integrate empirical profiling into auto-config:
```python
# When user runs with --auto-config
if args.auto_config:
    # Existing: Theoretical calculation
    theoretical_config = auto_config.calculate_config(...)

    # NEW: Empirical profiling (optional)
    if args.empirical_profiling:
        empirical_config = profiler.find_optimal_config(
            model, sample_data, device
        )
        # Override theoretical with empirical
        config = empirical_config
    else:
        config = theoretical_config
```

### 2. Training Startup
**File:** `lesseg_unet/training.py` (around line 630)

Run profiling before creating dataloaders:
```python
# After model creation, before dataloader creation
if args.profile or args.auto_optimize:
    from lesseg_unet.profiling import run_profiling_pass

    # Transform profiling (Task 2.1)
    transform_results = run_profiling_pass(...)

    # GPU VRAM profiling (Task 4)
    if args.auto_optimize and torch.cuda.is_available():
        optimal_batch_size = find_max_batch_size(model, spatial_size, device)
        if optimal_batch_size < batch_size:
            print(f"Auto-optimization: Reducing batch_size to {optimal_batch_size}")
            batch_size = optimal_batch_size
```

---

## References and Prior Art

### PyTorch Lightning Auto Batch Size Finding
**Source:** https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html

**Approach:**
```python
# PyTorch Lightning example
trainer = Trainer(auto_scale_batch_size=True)
trainer.tune(model)  # Runs batch size finder
```

**Implementation:**
- Binary search algorithm
- Catches OOM errors
- Tests with dummy batches
- Automatically scales batch size up/down

**Lessons:**
- ✅ Works well in practice
- ✅ Adds ~30 seconds startup time
- ⚠️ Sometimes finds batch size slightly too large (still OOMs on real data)
- ✅ Safety margin helps (find max, then use 80% of max)

### nnUNet Batch Size Selection
**Source:** https://github.com/MIC-DKFZ/nnUNet

**Approach:**
```python
# nnUNet approach (simplified)
def determine_batch_size(gpu_memory_target=0.85):
    """
    Find batch size that uses ~85% of available GPU memory.
    Leaves 15% margin for safety.
    """
    max_bs = run_binary_search(...)
    safe_bs = int(max_bs * 0.85)  # 15% safety margin
    return safe_bs
```

**Lessons:**
- ✅ Safety margin critical (nnUNet uses 85% of max)
- ✅ Different architectures need different margins
- ✅ Validation batch size can be larger (no gradients)
- ✅ Cache size estimation is essential

### TensorFlow Auto-tuning
**Source:** TensorFlow autotune API

**Approach:**
- Profile multiple configurations in parallel
- Use heuristics to prune search space
- Adaptive buffer sizes
- Cache profiling results

**Lessons:**
- ⚠️ Complex implementation
- ✅ Works across diverse hardware
- ⚠️ Overhead can be significant (minutes)

---

## Safety Considerations

### GPU VRAM: SAFE ✅
- `torch.cuda.OutOfMemoryError` is catchable
- Process continues after OOM
- Can safely binary search
- No data corruption risk

### System RAM: UNSAFE ⚠️
- OS OOM killer terminates process
- Cannot catch or recover
- Must estimate, not test
- Conservative estimates required

### Disk Space: CHECK ONLY ✓
- Can check available space before caching
- No risk of crash from testing
- Can warn if insufficient space

---

## Performance Overhead

### Empirical Profiling Cost
**Estimated time:**
- GPU VRAM profiling: 10-30 seconds (5-10 test runs)
- Training profiling: 20-40 seconds (includes backward pass)
- Transform profiling: 10-20 seconds (already done in Task 2.1)
- **Total:** ~40-90 seconds added to startup

**Is it worth it?**
- ✅ YES if prevents OOM crashes during long training runs
- ✅ YES if user doesn't know optimal batch size
- ⚠️ MAYBE if user is experienced and knows their hardware
- ❌ NO if running many short experiments (overhead dominates)

**Solution:** Make it optional
```bash
# Quick start (no profiling)
python -m lesseg_unet.main ...

# With profiling (recommended for first run)
python -m lesseg_unet.main ... --profile --auto-optimize

# Force reprofile (if data/model changes)
python -m lesseg_unet.main ... --profile --force-reprofile
```

---

## Implementation Plan (Task 4)

### Subtask 4.1: GPU VRAM Profiling
**Priority:** HIGH (safe, high value)
**Files:**
- `lesseg_unet/auto_config/profiler.py` (new)
- `lesseg_unet/training.py` (integration)
- `lesseg_unet/main.py` (--auto-optimize flag)

**Testing:**
- Test on 4GB, 8GB, 12GB GPUs
- Verify OOM is caught safely
- Measure overhead (<30 seconds)
- Validate batch size is safe (no OOM in actual training)

### Subtask 4.2: Training Mode Profiling
**Priority:** MEDIUM (more accurate but slower)
**Extends:** Subtask 4.1

**Testing:**
- Compare inference vs training batch sizes
- Verify gradient accumulation is accounted for
- Test with different optimizers (AdamW, SGD)

### Subtask 4.3: RAM Cache Estimation
**Priority:** MEDIUM (safety feature)
**Files:**
- `lesseg_unet/data_loading.py` (estimate_cache_size)
- Integration with cache_mode='ram'

**Testing:**
- Compare estimated vs actual cache size
- Verify warnings appear when appropriate
- Test with different transform dicts

### Subtask 4.4: Heuristics Database
**Priority:** LOW (future enhancement)
**Files:**
- `lesseg_unet/auto_config/heuristics.json`
- `lesseg_unet/auto_config/optimizer.py`

**Content:**
```json
{
  "gpu_vram_heuristics": {
    "safety_margin": 0.85,
    "validation_multiplier": 1.5,
    "gradient_checkpoint_reduction": 0.5
  },
  "transform_heuristics": {
    "cacheable_threshold": 0.4,
    "ram_vs_disk_threshold_gb": 8.0
  }
}
```

---

## Open Questions

1. **Safety Margin:** What % of max batch size should we use? (Lightning uses 100%, nnUNet uses 85%)
   - Recommendation: Start with 90%, make configurable

2. **Validation Batch Size:** Should we profile separately for validation? (Can be larger since no gradients)
   - Recommendation: Yes, but only if --auto-optimize is set

3. **Multi-GPU:** How to handle batch size with DDP? (Effective batch = batch_size * num_gpus)
   - Recommendation: Profile per-GPU, then scale accordingly

4. **Caching Overhead:** Does cache creation affect batch size calculation?
   - Recommendation: Profile after cache is created (first epoch)

---

## Related Work

- Task 2.0: Transform profiling (completed)
- Task 2.1: Profiling integration (planned)
- Task 3: Caching system (in progress)
- Task 4: Auto-optimization (this research)

---

## Conclusion

**Empirical profiling is feasible and valuable:**
- ✅ Safe for GPU VRAM (catchable errors)
- ✅ Proven approach (Lightning, nnUNet)
- ✅ Reasonable overhead (30-60s)
- ✅ Prevents OOM crashes in production

**Recommended implementation order:**
1. Task 3: Caching (enables faster profiling)
2. Task 2.1: Transform profiling integration
3. Task 4.1: GPU VRAM profiling (safe, high value)
4. Task 4.3: RAM estimation (safety)
5. Task 4.2: Training profiling (accuracy improvement)
6. Task 4.4: Heuristics database (nice-to-have)

**Status:** Research complete, ready for Task 4 implementation after Task 3 is done.

---

Created: 2026-01-07 (Recreated from lost research)
Last Updated: 2026-01-07

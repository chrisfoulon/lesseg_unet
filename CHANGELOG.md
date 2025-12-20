# lesseg_unet - Changelog

This file tracks all bug fixes and improvements made to lesseg_unet.

---

## [2025-12-20] PyTorch 2.7 Migration & Bug Fixes

### 1. Memory Cleanup on Ctrl+C

**Issue**: Memory (RAM) was not being released when stopping training with Ctrl+C, requiring manual process termination.

**Root Cause**:
- No signal handler for SIGINT (Ctrl+C)
- DataLoader worker processes kept running
- Model, optimizer, and scaler objects remained in memory
- DDP process group not destroyed

**Solution** (training.py):
- Added signal handler for SIGINT that triggers cleanup
- Implemented `cleanup_on_exit()` function that:
  - Deletes model, optimizer, scaler, and dataloaders
  - Clears CUDA cache (GPU) and forces garbage collection (CPU)
  - Destroys DDP process group
- Registered objects in `_cleanup_context` global for cleanup tracking
- Added cleanup at normal fold completion

**Files Modified**: `lesseg_unet/training.py`

**Impact**: Users can now safely interrupt training with Ctrl+C without memory leaks.

---

### 2. Sliding Window Inference Size Mismatch

**Issue**: Validation/segmentation failed with `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 24 but got size 23` when using models trained with patches.

**Root Cause**:
- `segmentation.py` only searched for `spatial_size` parameter in transform_dict
- Patch transforms like `RandSpatialCropSamplesd` use `roi_size` instead
- Code fell back to full image size (e.g., 181×217×181) instead of patch size (96×96×96)
- `sliding_window_inference` used wrong size, causing skip connection dimension mismatches in SwinUNETR

**Transform Parameter Names**:
| Transform | Parameter | Use Case |
|-----------|-----------|----------|
| `ResizeWithPadOrCropd` | `spatial_size` | Resize to fixed size |
| `RandCropByPosNegLabeld` | `spatial_size` | Random crop patches |
| `RandSpatialCropSamplesd` | `roi_size` | Multiple patches per image |
| `CropForegroundd` | `spatial_size` | Crop to foreground |

**Solution**:

**segmentation.py** (lines 116-125):
```python
# Try spatial_size first (for resize transforms)
training_img_size = transformations.find_param_from_hyper_dict(
    transform_dict, 'spatial_size', find_last=True)
# Try roi_size if spatial_size not found (for patch transforms)
if training_img_size is None:
    training_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'roi_size', find_last=True)
# Fall back to full image size as last resort
if training_img_size is None:
    training_img_size = utils.get_img_size(img_path_list[0])
```

**training.py** (lines 441-453):
```python
# Try roi_size first (for patches), then spatial_size (for resize transforms)
model_img_size = transformations.find_param_from_hyper_dict(
    transform_dict, 'roi_size', find_last=True)
if model_img_size is None:
    model_img_size = transformations.find_param_from_hyper_dict(
        transform_dict, 'spatial_size', find_last=True)
if model_img_size is not None:
    model_img_size = model_img_size[-3:]
else:
    raise ValueError(
        "Could not find 'roi_size' or 'spatial_size' in transform_dict. "
        "Please ensure your transforms include a cropping or resizing operation."
    )
```

**Files Modified**: `lesseg_unet/segmentation.py`, `lesseg_unet/training.py`

**Impact**:
- Validation and segmentation now work correctly with patch-trained models
- Better error messages when size parameters are missing
- Fix is modality-agnostic (will work with future multi-modal support)

**Tested With**:
- Checkpoint: `best_dice_model_segmentation3d_epo_612.pth`
- Before: Falls back to (181,217,181) → Error
- After: Correctly extracts [96,96,96] → Success

---

## Previous Fixes (PyTorch 2.7 Migration)

### CPU and Distributed Training Support
- Added full CPU training support (single-process and multi-process DDP)
- Implemented automatic AMP disabling on CPU (AMP only works with CUDA)
- Fixed device placement for both CPU and GPU modes
- Added proper DDP wrapping for CPU multi-process training (gloo backend)
- Guarded `torch.cuda.set_device()` to prevent crashes on CPU

**Files Modified**: `lesseg_unet/training.py`, `lesseg_unet/segmentation.py`

**Compatibility Matrix**:
| Mode | DDP Wrapper | AMP | Status |
|------|-------------|-----|--------|
| CPU Single | No | No | ✅ Works |
| CPU Multi | Yes | No | ✅ Works |
| GPU Single | Yes | Yes | ✅ Works |
| GPU Multi | Yes | Yes | ✅ Works |

### Deprecation Fixes
- Replaced deprecated `torch.cuda.amp.autocast()` with `torch.autocast(device_type='cuda')`
- Fixed checkpoint saving to handle DDP-wrapped models: `model.module.state_dict()` vs `model.state_dict()`

**Files Modified**: `lesseg_unet/training.py`, `lesseg_unet/segmentation.py`, `lesseg_unet/utils.py`

---

## Testing

### Example Training Command
```bash
# Quick test with sample data (5 epochs)
lesseg_unet \
  -o /tmp/lesseg_test/training \
  -p /path/to/data/dwi \
  -lp /path/to/data/stroke \
  -ics _dwi \
  -nw 4 \
  -trs p64 \
  -lfct dicefocal \
  -vlfct dice_dist \
  -bs 1 \
  -vbs 1 \
  -nf 5 \
  -ne 5 \
  -sbe 2 \
  -mt swinunetr \
  --local_rank 0
```

### Example Inference Command
```bash
# Run segmentation with trained model
lesseg_unet \
  -o /tmp/lesseg_test/segmentation \
  -li <image_list.csv> \
  -trs p64 \
  -nw 4 \
  -mt swinunetr \
  -overlap \
  -sa \
  -pt <path/to/model.pth> \
  --local_rank 0
```

### CPU Training
```bash
# Single-process CPU training
lesseg_unet -d cpu -ne 5 -bs 1 -nw 4 -o /tmp/test_cpu ...

# Multi-process CPU DDP (4 workers)
torchrun --nproc_per_node=4 \
  $(which lesseg_unet) \
  -d cpu --world_size 4 -ne 5 -bs 1 ...
```

---

## Notes

- All fixes maintain backward compatibility
- No breaking changes to existing training workflows
- Tested with PyTorch 2.7 + MONAI 1.5.1
- Python >= 3.11 required (due to scipy 1.16.3)

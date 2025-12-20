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

### Deprecation Fixes
- Replaced deprecated `torch.cuda.amp.autocast()` with `torch.autocast(device_type='cuda')`
- Fixed checkpoint saving to handle DDP-wrapped models: `model.module.state_dict()` vs `model.state_dict()`

### Files Modified
- `lesseg_unet/training.py`
- `lesseg_unet/utils.py`

**See**: `DEPRECATION_AND_CHECKPOINT_FIXES.md` for details

---

## Notes

- All fixes maintain backward compatibility
- No breaking changes to existing training workflows
- Tested with PyTorch 2.7 + MONAI 1.5.1

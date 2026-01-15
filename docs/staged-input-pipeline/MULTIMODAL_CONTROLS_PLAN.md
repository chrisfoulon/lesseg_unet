# Multi-Modal Controls Implementation Plan

**Status**: 🔄 NOT STARTED - Ready for implementation
**Complexity**: Medium (~100-150 lines across 2-3 files)
**Prerequisites**: Phase 3 complete, validation_loop_split_lists implemented
**Tests to enable**: 2 skipped tests in `tests/test_folder_converter.py`

---

## Quick Start for Claude

**To continue this implementation, say:**
> "Let's implement multi-modal controls. Read the plan at `docs/staged-input-pipeline/MULTIMODAL_CONTROLS_PLAN.md`"

---

## What Are Multi-Modal Controls?

Controls are **healthy subjects** (no lesions) used during training to help the model learn what "normal" brain tissue looks like. Currently, controls support only a single modality (e.g., just DWI). Multi-modal controls would allow controls to have multiple modalities (e.g., DWI + ADC), matching the image modalities of abnormal subjects.

### Use Case Example
```bash
# Current (single-modality controls) - WORKS
lesseg_unet -p dwi_folder adc_folder -m dwi adc \
            -lp labels -lm lesion \
            -cp controls_dwi  # Only one control folder

# Multi-modal controls - NOT YET IMPLEMENTED
lesseg_unet -p dwi_folder adc_folder -m dwi adc \
            -lp labels -lm lesion \
            -cp controls_dwi controls_adc -cm dwi adc  # Multiple control folders
```

---

## Current Architecture

### How Controls Work (training.py:726-734)

1. **Separate lists**: `split_lists` = abnormal subjects, `ctr_split_lists` = controls
2. **At runtime**: Controls are MERGED into subject dicts:
   ```python
   for img_dict in split_lists_with_ctr[i]:
       img_dict.update(ctr_fold_list.pop())
   ```
3. **Result**: `{'image': ..., 'label': ..., 'control': ...}` (single 'control' key)
4. **Transforms**: `add_control_key()` adds 'control' wherever 'image' is used

### Current Data Flow

```
Single-modality controls:
  {'control': '/path/ctr1.nii'}
     ↓
  Merged into subject dict at runtime
     ↓
  {'image': ..., 'label': ..., 'control': '/path/ctr1.nii'}
```

### What Multi-Modal Controls Need

```
Multi-modal controls:
  {'control_dwi': '/path/ctr1_dwi.nii', 'control_adc': '/path/ctr1_adc.nii'}
     ↓
  Concatenate to single 'control' key (like images → 'image')
     ↓
  {'control': <concatenated tensor>}
     ↓
  Merged into subject dict at runtime
     ↓
  {'image': ..., 'label': ..., 'control': <concatenated tensor>}
```

---

## Implementation Steps

### Step 1: Update `adapt_transforms_for_multimodal()` in data_utils.py

**Location**: `lesseg_unet/data_utils.py:1832-1971`

**Current behavior**: Only handles `image_*` and `label_*` keys, explicitly ignores control keys (line 1882).

**Required changes**:

```python
# Around line 1890, after detecting image_keys and label_keys:
control_keys = get_category_keys(first_subject, 'control')

# Around line 1894-1900, update the check:
has_multi_modal_controls = len(control_keys) > 1 or (
    len(control_keys) == 1 and control_keys[0] != 'control'
)

# If multi-modal controls detected, need adaptation
if not has_multi_modal and not has_named_labels and not has_multi_modal_controls:
    return transform_dict

# Sort control keys
control_keys_sorted = sorted(control_keys) if control_keys else []

# In the transform update loop (around line 1919-1937):
# Add control keys to LoadImaged and EnsureChannelFirstd
for key in params['keys']:
    if key == 'control' and control_keys_sorted:
        new_keys.extend(control_keys_sorted)
    # ... existing image/label handling

# After image ConcatItemsd (around line 1940-1949), add control ConcatItemsd:
if len(control_keys) > 1 and insertion_index is not None:
    control_concat_transform = {
        'ConcatItemsd': {
            'keys': control_keys_sorted,
            'name': 'control',  # Output key
            'dim': 0  # Concatenate along channel dimension
        }
    }
    # Insert after image ConcatItemsd
    insert_position = insertion_index + 1
    if len(image_keys) > 1:
        insert_position += 1  # Account for image ConcatItemsd
    adapted_dict['first_transform'].insert(insert_position, control_concat_transform)
```

### Step 2: Update `add_control_key()` in transformations.py (Optional)

**Location**: `lesseg_unet/transformations.py:939-963`

**Note**: This may not need changes if `adapt_transforms_for_multimodal()` handles everything. The function adds 'control' wherever 'image' is found, which should still work after concatenation.

### Step 3: Remove NotImplementedError in main.py

**Location**: `lesseg_unet/main.py:657-664`

```python
# DELETE this block:
if control_folders_dict and len(control_folders_dict) > 1:
    raise NotImplementedError(
        f"\nMulti-modal control subjects are not yet implemented.\n"
        ...
    )
```

### Step 4: Remove NotImplementedError in data_utils.py (deprecated function)

**Location**: `lesseg_unet/data_utils.py:1780-1784`

```python
# DELETE this block (in deprecated folder_mode_to_split_lists):
if control_folders is not None and len(control_folders) > 1:
    raise NotImplementedError(
        f"Multi-modal controls not yet implemented. "
        ...
    )
```

### Step 5: Enable skipped tests

**Location**: `tests/test_folder_converter.py`

Remove `@pytest.mark.skip` from:
- `test_two_control_modalities` (line 744)
- `test_controls_with_multimodal_images` (line 763)

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `lesseg_unet/data_utils.py` | 1832-1971 | Update `adapt_transforms_for_multimodal()` to handle control_* keys |
| `lesseg_unet/main.py` | 657-664 | Remove NotImplementedError |
| `lesseg_unet/data_utils.py` | 1780-1784 | Remove NotImplementedError (deprecated function) |
| `tests/test_folder_converter.py` | 744, 763 | Remove @pytest.mark.skip decorators |

---

## Key Functions Reference

### `get_category_keys()` - data_utils.py:615-645
```python
def get_category_keys(subject_dict: dict, category: str) -> list[str]:
    """Get all keys for a category (image, label, control) from a subject dict."""
```

### `adapt_transforms_for_multimodal()` - data_utils.py:1832-1971
```python
def adapt_transforms_for_multimodal(transform_dict: dict, split_lists: SplitLists) -> dict:
    """Adapt transform dictionary for multi-modal data."""
```

### `add_control_key()` - transformations.py:939-963
```python
def add_control_key(transform_dict, add_allow_missing_keys=True):
    """Add the control key to transforms that have 'image' key."""
```

---

## Testing Strategy

1. **Unit test**: Verify `adapt_transforms_for_multimodal()` detects and concatenates control_* keys
2. **Integration test**: Enable the 2 skipped tests in test_folder_converter.py
3. **Manual test**: Run training with multi-modal controls on test data

### Test Commands
```bash
# Run specific tests
pytest tests/test_folder_converter.py::TestMultiModalControls -v

# Run all staged_input tests
pytest tests/staged_input/ tests/test_folder_converter.py -v
```

---

## Acceptance Criteria

1. ✅ `adapt_transforms_for_multimodal()` handles `control_*` keys like `image_*` keys
2. ✅ `ConcatItemsd` is inserted for multi-modal controls
3. ✅ NotImplementedError removed from main.py and data_utils.py
4. ✅ `test_two_control_modalities` passes
5. ✅ `test_controls_with_multimodal_images` passes
6. ✅ All existing tests still pass (108 tests)

---

## Related Documentation

- **Staged pipeline overview**: `docs/staged-input-pipeline/COMPACT_CONTEXT.md`
- **Original design**: `docs/staged-input-pipeline/phase3_design_plan.md`
- **Test files**: `tests/staged_input/`, `tests/test_folder_converter.py`

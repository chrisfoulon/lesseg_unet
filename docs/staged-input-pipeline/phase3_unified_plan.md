# Phase 3: Unified Input Pipeline - Detailed Plan

## Problem Statement

Currently (after my hasty Phase 3 attempt):
- **Training path**: Calls `folder_mode_to_split_lists()` (which internally uses stage functions)
- **Inference path**: Duplicates Stage 0→1→2 logic inline
- **Issue**: Code duplication, not following the unified pipeline vision

## Goal

Create ONE unified pipeline where:
```
ALL modes:   Stage 0→1 (list files) → Stage 1→2 (match by ID)
Training:    + Stage 2→3 (shuffle + split)
Inference:   (stop at Stage 2, flat list)
```

## Current Code Flow Analysis

### Before Phase 3 (original):
```python
# Line 634-642 in main.py (BEFORE my changes)
img_list = folder_mode_to_split_lists(
    image_folders=image_folders_dict,
    label_folders=label_folders_dict,
    control_folders=control_folders_dict,
    n_folds=args.folds_number,
    subject_pattern=args.subject_pattern,
    control_pattern=args.control_pattern,
    random_seed=42
)
```
- Called for BOTH training and inference
- Always shuffled (THE BUG!)
- Always split into folds

### After Phase 3 (my hasty changes):
- Added `is_training = args.checkpoint is None` detection
- Training: calls old function
- Inference: duplicates Stage 0→1→2 inline
- **Problem**: Code duplication, training still goes through old path

### Desired State (unified):
```python
# Stage 0→1: ALL modes
image_modalities = list_nifti_from_folders(...)
label_classes = list_nifti_from_folders(...) if labels else None
control_modalities = list_nifti_from_folders(...) if controls else None

# Stage 1→2: ALL modes
subject_dicts, control_dicts = match_modalities_by_subject(...)
all_subjects = subject_dicts + control_dicts

# Stage 2→3: TRAINING ONLY
if is_training:
    img_list = shuffle_and_split_subjects(all_subjects, ...)  # [[fold0], [fold1], ...]
else:
    img_list = all_subjects  # [dict1, dict2, ...]
```

## Risk Assessment

### High Risk Changes:
1. **Removing `folder_mode_to_split_lists()` call for training** - Could break existing workflows
2. **Changing split_lists format** - Downstream code expects `[[fold0], [fold1]]`

### Medium Risk:
3. **Training must produce identical splits** - Reproducibility requirement
4. **All 13 existing tests must still pass**

### Low Risk:
5. **Inference path already changed** - No backward compat needed there

## Testing Strategy

### Pre-change Verification:
1. ✅ Run all existing tests to establish baseline
2. ✅ Document current folder_mode_to_split_lists() behavior
3. ⬜ Create test to verify shuffle reproducibility (seed=42)
4. ⬜ Create test to verify split distribution matches numpy.array_split

### Post-change Verification:
5. ⬜ All existing tests pass (13 from test_folder_converter.py)
6. ⬜ New unified pipeline produces identical splits for training
7. ⬜ Inference path produces flat list (no shuffle)

## Implementation Plan

### Option A: Minimal Change (RECOMMENDED)
**Keep existing `folder_mode_to_split_lists()` wrapper, just make it smarter**

```python
def folder_mode_to_split_lists(..., shuffle=True):
    # Stage 0→1
    image_mods = list_nifti_from_folders(...)
    # Stage 1→2
    subjects, controls = match_modalities_by_subject(...)
    all_subjects = subjects + controls

    # Stage 2→3 (conditional)
    if shuffle:
        return shuffle_and_split_subjects(all_subjects, n_folds, ...)
    else:
        return all_subjects  # Flat list for inference
```

**Changes needed**:
- Add `shuffle` parameter to `folder_mode_to_split_lists()` (default=True for backward compat)
- main.py: Pass `shuffle=is_training` when calling
- Remove duplicated inline Stage 0→1→2 code

**Pros**: Minimal changes, backward compatible, single call site
**Cons**: Still a wrapper function (but now it's truly unified)

### Option B: Remove Wrapper (HIGHER RISK)
**Inline Stages 0→1→2 in main.py, remove wrapper entirely**

**Changes needed**:
- Remove `folder_mode_to_split_lists()` from codebase
- Update main.py to call stage functions directly
- Update all imports

**Pros**: True unified pipeline visible in main.py
**Cons**: Larger diff, more places to introduce bugs

## Recommendation

**Use Option A (Minimal Change)**:
1. It achieves the unified pipeline goal (all modes use same stages)
2. Minimal code changes = lower regression risk
3. Backward compatible (existing tests pass)
4. Single call site in main.py = easier to maintain
5. Can still do Option B later if needed (incremental improvement)

## Next Steps

1. ⬜ Get user approval on Option A vs Option B
2. ⬜ If Option A: Write test for shuffle parameter
3. ⬜ Implement `shuffle` parameter in `folder_mode_to_split_lists()`
4. ⬜ Update main.py to use single call with `shuffle=is_training`
5. ⬜ Verify all tests pass
6. ⬜ Document the unified pipeline behavior

## Questions for User

1. **Is Option A (add shuffle parameter) acceptable**, or do you want the stages fully visible in main.py (Option B)?
2. **Are there any existing production workflows** we need to verify don't break?
3. **Do you want to see the test for reproducibility first** before implementation?

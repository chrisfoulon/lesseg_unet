# Staged Input Pipeline - Implementation Complete

**Date**: 2026-01-15
**Status**: ✅ PHASE 3 + VALIDATION LOOP COMPLETE - All tests passing

---

## Summary

The staged input pipeline has been completely implemented. The pipeline now uses:
1. **New loading functions**: `read_folder()`, `read_list_file()`, `read_list_dicts()`, `read_presplit_json()`
2. **New matching function**: `match_lists_to_dicts()` with strip/extract pattern support
3. **Explicit pipeline in main.py** (not hidden in wrapper functions)
4. **Deprecation warnings** on old functions

---

## New Functions (data_utils.py)

### Loading Functions (Stage 0)

| Function | Returns | Purpose |
|----------|---------|---------|
| `read_folder(folder, pattern?, recursive?)` | `list[Path]` | List NIfTI files from folder |
| `read_list_file(filepath, check_exists?)` | `list[Path]` | Read paths from text file |
| `read_list_dicts(filepath, check_exists?)` | `list[dict[str, Path]]` | Read pre-matched subject dicts |
| `read_presplit_json(filepath, check_exists?)` | `list[list[dict[str, Path]]]` | Read pre-split fold structure |

### Matching Function (Stage 1)

**`match_lists_to_dicts(image_lists, label_lists?, control_lists?, strip_pattern?, extract_pattern?, ...)`**

Two matching mechanisms:
1. **Default (no patterns)**: Exact filename match across folders
2. **STRIP**: Remove pattern from filename, match by residual
3. **EXTRACT**: Extract pattern from filename as matching key

Example:
```python
# Strip: "dwi_patient001.nii" → strip "dwi" → "_patient001.nii"
match_lists_to_dicts(image_lists, strip_pattern={'dwi': 'dwi', 'adc': 'adc'})

# Extract: "scan_sub-001_dwi.nii" → extract "sub-001"
match_lists_to_dicts(image_lists, extract_pattern=r"sub-\d+")
```

---

## CLI Argument Changes

### Removed (deprecated):
- `--subject-pattern` (default `r'(sub-\d+)'`)
- `--control-pattern` (default `r'(ctr-\d+)'`)

### Added:
- `--strip-pattern`: Pattern to REMOVE from filenames (residual matching)
- `--extract-pattern`: Pattern to EXTRACT from filenames (key matching)
- `--control-strip-pattern`: Strip pattern for controls
- `--control-extract-pattern`: Extract pattern for controls
- `--image-filter`: Glob pattern(s) to filter image files
- `--label-filter`: Glob pattern(s) to filter label files
- `--control-filter`: Glob pattern(s) to filter control files

---

## Main.py Pipeline (Explicit)

The pipeline is now explicit in main.py (lines 666-793):

```python
# Stage 0: Load files from folders
image_lists = load_modality_files(image_folders_dict, args.image_filter)
label_lists = load_modality_files(label_folders_dict, args.label_filter)
control_lists = load_modality_files(control_folders_dict, args.control_filter)

# Stage 1: Match files across modalities
subject_dicts, control_dicts = match_lists_to_dicts(
    image_lists=image_lists,
    label_lists=label_lists,
    control_lists=control_lists,
    strip_pattern=args.strip_pattern,
    extract_pattern=args.extract_pattern
)

# Stage 2: Split (training) or flat list (inference)
if is_training:
    img_list = shuffle_and_split_subjects(subject_dicts + control_dicts, n_folds)
else:
    img_list = subject_dicts + control_dicts
```

---

## Deprecated Functions

The following functions now emit `DeprecationWarning` and will be removed in v3.0:

1. **`list_nifti_from_folders()`** → Use `read_folder()` + `match_lists_to_dicts()`
2. **`match_modalities_by_subject()`** → Use `match_lists_to_dicts()`
3. **`folder_mode_to_split_lists()`** → Use explicit pipeline in main.py

---

## Test Status

| Test Suite | Status | Count |
|------------|--------|-------|
| staged_input tests | ✅ PASS | 95 |
| folder_converter tests | ✅ PASS | 13 (7 skipped for unimplemented features) |
| **Total** | ✅ | 108 passing |

---

## Validation Loop with Subject Dicts

**NEW**: `validation_loop_split_lists()` in `segmentation.py` now accepts pre-matched subject dictionaries:

```python
# Inference with embedded labels now works!
segmentation.validation_loop_split_lists(
    subject_dicts=[
        {'image_dwi': '/data/s1_dwi.nii', 'image_adc': '/data/s1_adc.nii',
         'label_lesion': '/data/s1_lesion.nii'},
        ...
    ],
    output_dir='/output',
    checkpoint_path='/model.pth'
)
```

**Routing in main.py**: When `has_embedded_labels` is detected, main.py routes to `validation_loop_split_lists()` instead of `validation_loop()`.

---

## Remaining Limitations (Phase 4 Work)

1. **Multi-class labels**: Not yet implemented (NotImplementedError) - Complex, requires training loop changes
2. **Multi-modal controls**: Ready for implementation - See `MULTIMODAL_CONTROLS_PLAN.md`
   - To implement: "Let's implement multi-modal controls. Read the plan at `docs/staged-input-pipeline/MULTIMODAL_CONTROLS_PLAN.md`"

---

## Files Modified

| File | Changes |
|------|---------|
| `lesseg_unet/data_utils.py` | Added loading functions (lines 70-398), matching function (lines 405-592), deprecation warnings |
| `lesseg_unet/main.py` | Updated imports, replaced CLI args (lines 222-262), new pipeline (lines 666-780), routing for embedded labels (lines 1428-1458) |
| `lesseg_unet/segmentation.py` | Added `validation_loop_split_lists()` function (lines 731-1048) |
| `pytest.ini` | Added deprecation warning filters |
| `tests/staged_input/test_validation_loop.py` | New test file (10 tests) |

---

## Reference

- Design plan: `docs/staged-input-pipeline/phase3_design_plan.md`
- Original vision: `docs/staged-input-pipeline/unified_pipeline_vision.md`
- Test files: `tests/staged_input/`

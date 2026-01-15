# Context: Staged Input Pipeline Refactoring

## Level 1: Plain English Summary

### Current Architecture

The input handling pipeline in lesseg_unet currently uses a monolithic function `folder_mode_to_split_lists()` that combines multiple responsibilities:

1. **Listing files** from folders using regex patterns
2. **Matching** modalities and labels by subject ID
3. **Validating** completeness (all modalities present)
4. **Shuffling** subjects with seed=42
5. **Splitting** into cross-validation folds

This function is called for **both training and inference**, which causes problems:
- Validation/segmentation data gets shuffled (non-deterministic results)
- Test data is split into unnecessary folds (creates nested [[dict], [dict]])
- Cannot reuse matching logic without also shuffling/splitting

### Key Components

**Current Flow (main.py:634-649)**:
```
User provides -p folders → folder_mode_to_split_lists() → split_lists (nested)
                                     ↓
                            ALWAYS shuffles + splits
                                     ↓
                     Used for training AND validation/segmentation
```

**Problem**: The shuffle/split operations (lines 735-743 in data_utils.py) execute regardless of whether the data is for training (needs shuffle/split) or inference (should NOT shuffle/split).

**Impact**:
- Test validation results are non-deterministic
- Unnecessary complexity in validation/segmentation routing

### Related Components

**1. data_utils.py** - Input processing and matching logic
- `folder_mode_to_split_lists()` (lines 443-744) - Monolithic function to refactor
- `_build_subject_to_file_mapping()` (lines 358-430) - Helper to reuse
- `_list_nifti_files()` (lines 331-355) - Helper to reuse
- `adapt_transforms_for_multimodal()` (lines 747-920) - Depends on split_lists format

**2. main.py** - Pipeline orchestration and routing
- Lines 634-642: Calls `folder_mode_to_split_lists()` with `n_folds=args.folds_number`
- Lines 1221-1232: Detection of embedded labels (incomplete implementation)
- Lines 1234-1279: Routing to segmentation_loop or validation_loop

**3. segmentation.py** - Validation and segmentation loops
- `validation_loop()` (lines 471-680) - Expects separate img_path_list and seg_path_list
- `segmentation_loop()` (lines 310-465) - Expects simple path list
- Neither function handles dictionaries with embedded labels

**4. training.py** - Training pipeline (will NOT be modified)
- Lines 365-367: Detects pre-split mode when `lbl_path_list is None`
- Already handles both split_lists and matched dicts correctly

**5. data_loading.py** - Dataset creation (will NOT be modified)
- `create_fold_dataloaders()` (lines 484-533) - Already handles dict format
- `create_dataset()` (lines 209-295) - MONAI Dataset accepts dicts directly

### Integration Strategy

**Approach**: Extract and compose
- Extract generic matching logic (Stage 1→2) from `folder_mode_to_split_lists()`
- Extract shuffle/split logic (Stage 2→3) from `folder_mode_to_split_lists()`
- Add wrapper functions for file listing (Stage 0→1)
- Refactor `folder_mode_to_split_lists()` to call extracted functions
- Update main.py routing to detect training vs inference

**Backward Compatibility**:
- Existing function signature preserved
- Same shuffle behavior (seed=42)
- Same split distribution (numpy.array_split)
- training.py unchanged (already handles both formats)

---

## Level 2: API Table

### Current Functions (To Extract From)

| Symbol | Purpose | Inputs | Outputs | Side-effects |
|--------|---------|--------|---------|--------------|
| `folder_mode_to_split_lists()` | List files, match by ID, validate, shuffle, split | image_folders: Dict[str, Path]<br>label_folders: Optional[Dict[str, Path]]<br>control_folders: Optional[Dict[str, Path]]<br>n_folds: int<br>subject_pattern: str<br>control_pattern: str<br>random_seed: int | SplitLists:<br>[[{image_mod: path, label: path}, ...], ...] | None |
| `_build_subject_to_file_mapping()` | List NIfTI files and extract subject IDs | folder: Path<br>pattern: str | Dict[str, str]:<br>{subject_id: absolute_path} | None |
| `_list_nifti_files()` | Find all .nii/.nii.gz files in folder | folder: Path | List[Path] | None |

### New Functions (To Create)

| Symbol | Purpose | Inputs | Outputs | Side-effects |
|--------|---------|--------|---------|--------------|
| `list_nifti_from_folders()` | Stage 0→1: List files from multiple folders | folders: Dict[str, Path]<br>pattern: str | Dict[str, Dict[str, str]]:<br>{modality: {subj_id: path}} | None |
| `match_modalities_by_subject()` | Stage 1→2: Match modalities and labels by ID | image_modalities: Dict[str, Dict[str, str]]<br>label_classes: Optional[...]<br>control_modalities: Optional[...]<br>require_all: bool | Tuple[List[Dict], List[Dict]]:<br>(subject_dicts, control_dicts) | None |
| `validate_subject_dicts()` | Validate paths exist and are loadable | subject_dicts: List[Dict]<br>check_loadable: bool<br>min_size: int | None | Raises ValueError if invalid |
| `shuffle_and_split_subjects()` | Stage 2→3: Shuffle and split into folds | subject_dicts: List[Dict]<br>n_folds: int<br>shuffle: bool<br>random_seed: int | List[List[Dict]]:<br>[[fold0], [fold1], ...] | None |
| `load_path_lists_from_files()` | Stage 0→1: Load paths from CSV/text files | list_files: Dict[str, Path] | Dict[str, List[str]]:<br>{modality: [path1, ...]} | None |

### Routing Functions (To Modify in main.py)

| Symbol | Current Behavior | New Behavior |
|--------|------------------|--------------|
| Input handling (line 634) | Always calls `folder_mode_to_split_lists(n_folds=5)` | Detect `is_training`, call appropriate function:<br>- Training: `folder_mode_to_split_lists()`<br>- Inference: `list_nifti_from_folders()` → `match_modalities_by_subject()` |
| Label detection (line 1221) | Partially implemented | Check for `label_*` keys in first dict to route validation vs segmentation |

---

## Level 3: Code Snippets

### Current Implementation (data_utils.py:735-744)

**Problem code - Always executes regardless of use case:**

```python
# Step 8: Shuffle and split into folds
np.random.seed(random_seed)
shuffled_indices = np.random.permutation(len(subject_list))
shuffled_subjects = [subject_list[idx] for idx in shuffled_indices]

# Use numpy's array_split for even distribution (same as existing codebase)
split_arrays = np.array_split(np.array(shuffled_subjects, dtype=object), n_folds)
split_lists = [list(fold) for fold in split_arrays]

return split_lists
```

**This needs to become optional** - Only execute for training, not inference.

### Current Call Site (main.py:634-642)

**Problem - No distinction between training and inference:**

```python
# Call folder converter to get SplitLists
img_list = folder_mode_to_split_lists(
    image_folders=image_folders_dict,
    label_folders=label_folders_dict,
    control_folders=control_folders_dict,
    n_folds=args.folds_number,  # Always 5, even for validation/segmentation!
    subject_pattern=args.subject_pattern,
    control_pattern=args.control_pattern,
    random_seed=42
)
```

**Checkpoint flag is available but not used:**
- `args.checkpoint` is None for training
- `args.checkpoint` is set for validation/segmentation
- But this distinction isn't used when calling the function

### Existing Helper (data_utils.py:358-430)

**Will be reused by new functions:**

```python
def _build_subject_to_file_mapping(folder: Path, pattern: str) -> dict[str, str]:
    r"""Build mapping from subject IDs to file paths.

    Parameters
    ----------
    folder : Path
        Folder containing NIfTI files
    pattern : str
        Regex pattern to extract subject ID from filenames

    Returns
    -------
    dict[str, str]
        Dictionary mapping subject IDs to absolute file paths

    Raises
    ------
    ValueError
        If multiple files match the same subject ID
    """
    nifti_files = _list_nifti_files(folder)

    if not nifti_files:
        return {}

    regex = re.compile(pattern)
    subject_to_file = {}

    for file_path in nifti_files:
        match = regex.search(file_path.name)
        if not match:
            continue

        subject_id = match.group(1)

        if subject_id in subject_to_file:
            raise ValueError(
                f"Multiple files match subject ID '{subject_id}' in {folder}:\n"
                f"  - {subject_to_file[subject_id]}\n"
                f"  - {file_path}"
            )

        subject_to_file[subject_id] = str(file_path.resolve())

    return subject_to_file
```

### Integration with Training (training.py:365-367)

**Already supports both formats - No changes needed:**

```python
if lbl_path_list is None:
    # If no label list is provided, then it means img_path_list is a split list (split per fold)
    split_lists_to_share = [img_path_list]
else:
    # Old single-modality: match + split
    img_dict, controls = data_loading.match_img_seg_by_names(...)
    split_lists_to_share = [utils.split_lists_in_folds(img_dict, folds_number, ...)]
```

This code path already handles:
- Split lists (nested): `[[fold0_dicts], [fold1_dicts]]`
- Flat lists: `[{dict1}, {dict2}]` (would need wrapping in list)

---

## Maintenance Opportunities in Target Files

### High Priority (Address During Implementation)

#### segmentation.py:241
```python
F821 undefined name 'vol_output'
```
**Issue**: Variable used before definition
**Context**: Line 241 in a validation context
**Action**: Fix while implementing new validation_loop_split_lists()

### Medium Priority (Boy Scout Rule - Consider Fixing)

#### main.py:824
```python
F811 redefinition of unused 'nib' from line 17
```
**Issue**: `nib` imported at top (line 17), redefined later
**Action**: Remove redundant import

#### main.py:924, 987
```python
F841 local variable 'e' is assigned to but never used
F841 local variable 'min_fold_size' is assigned to but never used
```
**Issue**: Variables created but not used
**Action**: Remove or use in logging

#### segmentation.py Multiple unused variables
```python
F841 local variable 'ing_img__np' is assigned to but never used (line 62)
F841 local variable 'cpu_device' is assigned to but never used (line 102)
F841 local variable 'post_trans' is assigned to but never used (line 151)
F841 local variable 'dist_ratio' is assigned to but never used (line 530)
F841 local variable 'perf_measure_names' is assigned to but never used (line 564)
```
**Issue**: Dead code or incomplete features
**Action**: Clean up while working in these files

---

## Dependencies and Constraints

### External Dependencies
- `numpy` - Array operations, random seed
- `pathlib` - Path manipulation
- `re` - Regex pattern matching
- `nibabel` - NIfTI file validation (optional check)

### Internal Dependencies
- `_build_subject_to_file_mapping()` - Will be reused
- `_list_nifti_files()` - Will be reused
- `adapt_transforms_for_multimodal()` - Expects split_lists format (backward compatible)
- `create_fold_dataloaders()` - Already handles dicts (no changes needed)

### Constraints
1. **Backward compatibility**: Existing training runs must work unchanged
2. **No checkpoint format changes**: Resume training must work
3. **Preserve CLI**: All current flags continue to work
4. **Same shuffle behavior**: seed=42, numpy.array_split for distribution
5. **Controls separate**: Controls use same functions but separate pipeline

---

## Testing Strategy

### Unit Tests (Per Stage Function)

**Stage 0→1 Functions:**
- `test_list_nifti_from_folders_success()` - Basic listing
- `test_list_nifti_from_folders_empty()` - Empty folder handling
- `test_list_nifti_from_folders_missing()` - Missing folder error
- `test_list_nifti_from_folders_pattern_mismatch()` - No files match pattern
- `test_load_path_lists_from_files_csv()` - Load from CSV
- `test_load_path_lists_from_files_missing_path()` - Invalid path in file

**Stage 1→2 Functions:**
- `test_match_modalities_basic()` - Simple 2 modalities + label
- `test_match_modalities_multi_modal()` - DWI + ADC + label
- `test_match_modalities_incomplete_subject()` - Missing modality error
- `test_match_modalities_no_labels()` - Images only (segmentation mode)
- `test_match_modalities_controls()` - Separate control handling

**Stage 2→3 Functions:**
- `test_shuffle_and_split_reproducible()` - Same seed = same splits
- `test_shuffle_and_split_no_shuffle()` - shuffle=False preserves order
- `test_shuffle_and_split_distribution()` - Even distribution across folds
- `test_shuffle_and_split_single_fold()` - n_folds=1 returns [all_subjects]

**Validation Function:**
- `test_validate_subject_dicts_success()` - All paths valid
- `test_validate_subject_dicts_missing_file()` - Detects missing file
- `test_validate_subject_dicts_small_file()` - Detects empty/corrupt file
- `test_validate_subject_dicts_not_loadable()` - Detects bad NIfTI

### Integration Tests

**End-to-End Pipelines:**
- `test_training_pipeline_folders()` - Folders → shuffle → split → save split_lists.json
- `test_training_pipeline_presplit()` - Load split_lists.json → validate → use
- `test_validation_pipeline()` - Folders → NO shuffle → flat list → route to validation
- `test_segmentation_pipeline()` - Folders → NO shuffle → flat list → route to segmentation

**Backward Compatibility:**
- `test_existing_split_lists_unchanged()` - Same shuffle behavior produces same splits
- `test_resume_training_works()` - Auto-load splits, no reshuffling

### Test Data Requirements

**Minimal test dataset:**
- 10 synthetic subjects (sub-001 through sub-010)
- 2 modalities (dwi, adc)
- 1 label class (stroke)
- 3 controls (ctr-001 through ctr-003)

**Test fixtures:**
- `tmp_path` (pytest) for creating temporary folder structures
- `create_mock_nifti()` helper for generating test .nii.gz files
- `create_test_split_lists_json()` for pre-split testing

---

## Risk Assessment

### High Risk
- **Breaking existing training workflows**: Extensive testing with real data needed
- **Shuffle reproducibility**: Must verify same seed produces same splits

### Medium Risk
- **Validation/segmentation routing**: Multiple code paths, easy to miss edge cases
- **Multi-modal transform adaptation**: Depends on split_lists format

### Low Risk
- **Stage function extraction**: Clear boundaries, well-defined inputs/outputs
- **Controls handling**: Already separate pipeline, reuses same functions

### Mitigation Strategy
1. **Incremental extraction**: Extract one stage at a time, test before next
2. **Comprehensive tests**: Cover all routing paths, edge cases
3. **Real data validation**: Test with user's actual DWI+ADC dataset
4. **Backward compatibility tests**: Verify existing split_lists.json unchanged

---

## Performance Considerations

### No Performance Impact Expected

**Reasons:**
- Same underlying operations (list files, match, shuffle, split)
- Just reorganized into separate functions
- No new I/O operations
- No new dependencies

**Potential Improvement:**
- Validation function could cache nibabel loads if used multiple times
- Not implementing initially (YAGNI principle)

---

## Future Extensibility

### Easy to Add
- **Database input**: New function `load_from_database()` → Stage 1 format
- **S3 buckets**: New function `list_from_s3()` → Stage 1 format
- **BIDS format**: New function `parse_bids_dataset()` → Stage 1 format
- **Preprocessing pipeline**: Insert between stages

### Example: Adding Database Support
```python
def load_from_database(connection_string, query):
    # Query database for paths
    paths = execute_query(connection_string, query)

    # Convert to Stage 1 format
    modality_mappings = group_by_modality(paths)

    # Rest of pipeline unchanged
    subject_dicts = match_modalities_by_subject(modality_mappings)
    # ...
```

This requires only ONE new function, reuses all existing stages.

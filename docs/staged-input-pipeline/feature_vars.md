# Feature Variables

## Feature Identification
```bash
FEATURE_SLUG="staged-input-pipeline"
PROJECT_NAME="lesseg_unet"
FEATURE_DESCRIPTION="Refactor input handling into modular, composable stages for training/validation/segmentation pipelines"
```

## Feature Requirements

### Problem Statement
Current input handling in main.py always calls `folder_mode_to_split_lists()` which:
- Always shuffles data (inappropriate for validation/segmentation)
- Always creates fold splits (unnecessary for inference)
- Combines multiple responsibilities (listing, matching, validating, splitting)

This causes:
- Test data being shuffled (non-deterministic validation results)
- Validation/segmentation creating unnecessary nested structures
- Poor modularity and code reuse
- Difficulty adding new input formats

### Inputs
**Stage 0: Raw Input Types**
- Folders: `-p dwi adc`, `-lp label`, `-cp controls`
- List files: `-li paths.csv`, `-lli labels.csv`, `-lctr controls.csv`
- Pre-matched JSON: `--subject-list matched.json` (NEW)
- Pre-split JSON: `-psl split_lists.json` (existing)

**Processing Context**
- Subject pattern: `--subject-pattern '(sub-.+?)\.nii\.gz'`
- Control pattern: `--control-pattern '(ctr-.+?)\.nii\.gz'`
- Mode: Training (`--checkpoint` absent) vs Inference (`--checkpoint` present)

### Outputs
**Stage 1: Path Mappings**
```python
{
    'dwi': {'sub-001': '/abs/path/dwi.nii.gz', 'sub-002': '...'},
    'adc': {'sub-001': '/abs/path/adc.nii.gz', 'sub-002': '...'}
}
```

**Stage 2: Subject Dicts (Flat List)**
```python
[
    {'image_dwi': '/path1', 'image_adc': '/path2', 'label_stroke': '/path3'},
    {'image_dwi': '/path4', 'image_adc': '/path5', 'label_stroke': '/path6'}
]
```

**Stage 3: Split Lists (Training Only)**
```python
[
    [fold0_dicts],  # Training fold 0
    [fold1_dicts],  # Training fold 1
    ...
]
```

### Constraints
- **Backward compatibility**: Existing training runs must work unchanged
- **Zero functional changes**: Same shuffle behavior (seed=42), same splits
- **No checkpoint format changes**: Resume training must work
- **Preserve existing CLI**: All current flags continue to work
- **Controls separate**: Controls use same functions but separate pipeline

### Acceptance Criteria
1. **Functional Requirements**:
   - [ ] Training with folders: Shuffles (seed=42), creates folds, saves split_lists.json
   - [ ] Training with `-psl`: Loads pre-split, validates paths, no reshuffling
   - [ ] Validation with folders: NO shuffle, flat list, routes to validation
   - [ ] Segmentation with folders: NO shuffle, flat list, routes to segmentation
   - [ ] Resume training: Auto-loads splits, no reshuffling
   - [ ] New `--subject-list`: Accepts pre-matched flat JSON

2. **Quality Requirements**:
   - [ ] All existing tests pass
   - [ ] New tests for each stage function (≥90% coverage)
   - [ ] Flake8 compliant (max-complexity 10)
   - [ ] NumPy-style docstrings for all new functions

3. **Architecture Requirements**:
   - [ ] Each stage is a separate, testable function
   - [ ] Functions compose: Stage 0→1→2→3
   - [ ] Controls use same functions, separate pipeline
   - [ ] Clear separation: generic (matching) vs training-specific (shuffle/split)

## Architecture Design

### Refinement Stages
```
Stage 0: Raw Input (folders, lists, JSONs)
    ↓
Stage 1: Path Mappings {modality: {subject_id: path}}
    ↓
Stage 2: Subject Dicts [{image_mod: path, label: path}]
    ↓
Stage 3: Split Lists [[fold0], [fold1], ...] (Training only)
```

### Key Functions (data_utils.py)
1. `list_nifti_from_folders()` - Stage 0→1 for folders
2. `load_path_lists_from_files()` - Stage 0→1 for list files
3. `match_modalities_by_subject()` - Stage 1→2 (generic matching)
4. `validate_subject_dicts()` - Validate stage 2 output
5. `shuffle_and_split_subjects()` - Stage 2→3 (training only)

### Integration Points
- **main.py**: Pipeline orchestration, routing to training/validation/segmentation
- **training.py**: Unchanged (receives split_lists or flat list)
- **segmentation.py**: New `validation_loop_split_lists()`, `segmentation_loop_split_lists()`
- **data_loading.py**: Unchanged (already handles dicts via Dataset)

## Quality Standards
- **Complexity**: max-complexity 10 per function
- **Coverage**: ≥90% for new code
- **Docstrings**: NumPy-style required
- **Testing**: TDD approach, stage-by-stage

## Related Files
**Will be modified:**
- `lesseg_unet/data_utils.py` - Extract and add stage functions
- `lesseg_unet/main.py` - Update routing logic
- `lesseg_unet/segmentation.py` - Add split_lists-aware loops
- `lesseg_unet/utils.py` - Add validation function

**Will NOT be modified:**
- `lesseg_unet/training.py` - Already handles both formats
- `lesseg_unet/data_loading.py` - Already handles dicts
- `lesseg_unet/transformations.py` - Already adapted for multi-modal

## Dependencies
- Existing: `_build_subject_to_file_mapping()` (data_utils.py)
- Existing: `file_to_list()` (utils.py)
- Existing: `check_inputs()` (utils.py)
- Existing: `adapt_transforms_for_multimodal()` (data_utils.py)
- Existing: `create_dataset()` (data_loading.py)

## Implementation Notes
- Extract lines 556-733 from `folder_mode_to_split_lists()` to create generic matching
- Refactor `folder_mode_to_split_lists()` to call generic + shuffle/split
- Add CLI argument `--subject-list` for pre-matched JSON (new)
- Routing: `is_training = args.checkpoint is None`

## Session Context
- **Previous work**: Fixed resume data leakage bug, sorting bug, early stopping restoration
- **Current issue**: Validation/segmentation incorrectly shuffles/splits test data
- **Root cause**: `folder_mode_to_split_lists()` always shuffles regardless of use case

# Multi-Modal Support Design

**Status:** Design Complete | Implementation Pending
**Version:** 2.0 (Refined)
**Target:** lesseg_unet v2.1.0

---

## Executive Summary

**Goal:** Enable lesseg_unet to train on multiple imaging modalities (DWI + ADC + FLAIR, etc.) while maintaining backward compatibility with single-modality workflows.

**Current Status:** ❌ NOT SUPPORTED - lesseg_unet is currently designed for single-channel input only.

**Proposed Solution:** Suffix-based key naming with `list[list[dict]]` canonical format and converter functions for multiple input modes.

**Implementation Complexity:**
- **Difficulty:** Medium (leverages existing patterns)
- **Breaking Changes:** Low (backward compatible)
- **Confidence:** High (80-85%)
- **Time Estimate:** 2-3 days (MVP) to 4-5 days (full implementation)

---

## 1. Current Limitations

### Problem 1: Single Modality File Matching
**File:** `data_loading.py:16-73`

Current behavior:
```python
{'image': '/path/dwi.nii.gz', 'label': '/path/mask.nii.gz'}  # ONE image only
```

Missing:
- Cannot associate DWI + ADC files for same subject
- No grouping by subject ID across modalities

### Problem 2: Transform Pipeline
**File:** `data/transform_dicts.py`

Current behavior:
```python
'first_transform': [
    {'LoadImaged': {'keys': ['image', 'label']}},  # Loads ONE image
]
```

Missing:
- No concatenation transform to stack modalities
- No channel-wise concatenation after loading

### Problem 3: Model Configuration
**File:** `net.py:8-44`

Current behavior:
```python
default_unetr_hyper_params = {
    'in_channels': 1,  # Hardcoded to 1
}
```

Missing:
- No auto-detection of channel count from data
- Must manually specify `in_channels` for multi-modal

---

## 2. Proposed Architecture

### 2.1 Data Structure

**Type Definitions:**
```python
from typing import TypeAlias

SubjectDict = dict[str, str]  # {'image_dwi': '/path', 'label': '/path'}
Fold = list[SubjectDict]      # List of subjects in one fold
SplitLists = list[Fold]       # Multiple folds
```

**In code:** Use existing `split_lists` convention
**In docs/discussion:** "split structure" or "fold structure"

### 2.2 Suffix-Based Key Naming

**Pattern:** `{category}_{identifier}` or just `{category}`

**Categories:**
- `image` - Primary imaging modality
- `control` - Control/healthy subject imaging
- `label` - Segmentation masks/annotations

**Examples:**

#### Single Modality (Backward Compatible):
```python
{
    'image': '/path/image.nii.gz',
    'label': '/path/mask.nii.gz',
    'control': '/path/control.nii.gz'
}
```

#### Multi-Modal Images:
```python
{
    'image_dwi': '/path/dwi.nii.gz',
    'image_adc': '/path/adc.nii.gz',
    'image_flair': '/path/flair.nii.gz',
    'label': '/path/mask.nii.gz'
}
```

#### Multi-Class Labels:
```python
{
    'image_dwi': '/path/dwi.nii.gz',
    'label_class0': '/path/lesion_type1.nii.gz',
    'label_class1': '/path/lesion_type2.nii.gz',
    'label_class2': '/path/edema.nii.gz'
}
```

#### Multi-Modal Controls:
```python
{
    'image_dwi': '/path/patient_dwi.nii.gz',
    'image_adc': '/path/patient_adc.nii.gz',
    'control_dwi': '/path/healthy_dwi.nii.gz',
    'control_adc': '/path/healthy_adc.nii.gz',
    'label': '/path/mask.nii.gz'
}
```

### 2.3 Key Parsing Rules

```python
def parse_key(key: str) -> tuple[str, str | None]:
    """Parse key into (category, identifier)"""
    if '_' in key:
        category, identifier = key.split('_', 1)
        return category, identifier
    else:
        return key, None

# Examples:
parse_key('image_dwi')   → ('image', 'dwi')
parse_key('image')       → ('image', None)
parse_key('label_class0') → ('label', 'class0')
```

### 2.4 Schema Validation (Decision Tree)

**Build schema from first subject:**
```python
def build_schema(first_subject: dict) -> dict:
    """Extract expected structure from first subject"""
    schema = {
        'image_keys': [],
        'label_keys': [],
        'control_keys': [],
        'has_multi_modal_images': False,
        'has_multi_class_labels': False,
        'has_controls': False,
    }

    for key in first_subject:
        category, identifier = parse_key(key)

        if category == 'image':
            schema['image_keys'].append(key)
            if identifier:
                schema['has_multi_modal_images'] = True
        elif category == 'label':
            schema['label_keys'].append(key)
            if identifier:
                schema['has_multi_class_labels'] = True
        elif category == 'control':
            schema['control_keys'].append(key)
            schema['has_controls'] = True

    return schema
```

**Validate all subjects match schema:**
```python
def validate_against_schema(subject: dict, schema: dict, subject_id: str):
    """Ensure subject matches expected schema"""
    for key in schema['image_keys']:
        if key not in subject:
            raise ValueError(
                f"Subject '{subject_id}' missing expected key '{key}'\n"
                f"Expected keys: {schema['image_keys']}\n"
                f"Found keys: {list(subject.keys())}"
            )
```

---

## 3. Input Modes

### Mode 1: Folder-per-Modality (Simplest, MVP)

**Directory Structure:**
```
data/
  dwi/
    sub-001_dwi.nii.gz
    sub-002_dwi.nii.gz
  adc/
    sub-001_adc.nii.gz
    sub-002_adc.nii.gz
  lesion_masks/
    sub-001_lesion.nii.gz
    sub-002_lesion.nii.gz
```

**CLI:**
```bash
lesseg_unet \
  --image-folder dwi:data/dwi adc:data/adc \
  --label-folder data/lesion_masks \
  --modality-suffixes dwi adc \
  ...
```

**Converter:**
```python
def folder_mode_to_split_lists(
    image_folders: dict[str, Path],  # {'dwi': Path('data/dwi'), 'adc': ...}
    label_folder: Path,
    n_folds: int = 5
) -> SplitLists:
    # Match files by subject ID
    # Build SubjectDict with image_dwi, image_adc, label keys
    # Split into folds
```

### Mode 2: BIDS-like Structure

**Directory Structure:**
```
data/
  sub-001/
    anat/
      sub-001_T1w.nii.gz
      sub-001_T2w.nii.gz
    dwi/
      sub-001_dwi.nii.gz
  derivatives/
    sub-001/
      sub-001_lesion.nii.gz
```

**CLI:**
```bash
lesseg_unet \
  --bids-root data \
  --image-suffixes T1w T2w dwi \
  --label-suffix lesion \
  ...
```

### Mode 3: One-Folder Pattern Matching

**Directory Structure:**
```
data/
  sub-001_dwi.nii.gz
  sub-001_adc.nii.gz
  sub-001_lesion.nii.gz
  sub-002_dwi.nii.gz
  sub-002_adc.nii.gz
  sub-002_lesion.nii.gz
```

**CLI:**
```bash
lesseg_unet \
  --data-folder data \
  --image-patterns _dwi _adc \
  --label-pattern _lesion \
  ...
```

**Error Handling:**
```python
# If pattern matching fails, provide detailed diagnostics:
raise ValueError(
    f"Pattern matching failed!\n"
    f"  Total files: 100\n"
    f"  Matched files: 45\n"
    f"  Unmatched files: 55\n"
    f"  Subjects with all images: 10\n"
    f"  Subjects missing modalities: 5\n"
    f"  Missing patterns for subjects: ['sub-001' (missing _adc), ...]\n"
    f"  Unmatched files: ['file1.nii.gz', 'file2.nii.gz', ...]\n"
)
```

---

## 4. Transform Pipeline Updates

### 4.1 Loading Keys

**Before (single modality):**
```python
{'LoadImaged': {'keys': ['image', 'label']}}
```

**After (multi-modal):**
```python
# Auto-detect from data structure
image_keys = get_category_keys(split_lists[0][0], 'image')  # ['image_dwi', 'image_adc']
label_keys = get_category_keys(split_lists[0][0], 'label')  # ['label']

{'LoadImaged': {'keys': image_keys + label_keys}}
```

### 4.2 Concatenation Transform

**Add AFTER spatial transforms, BEFORE model input:**
```python
{
    'ConcatImagesd': {
        'keys': image_keys,  # ['image_dwi', 'image_adc']
        'output_key': 'image',  # Merge into single 'image' with N channels
        'channel_dim': 0  # Concatenate along channel dimension
    }
}
```

**Result:**
- Input: `image_dwi` (H×W×D×1), `image_adc` (H×W×D×1)
- Output: `image` (H×W×D×2)

### 4.3 Updated Transform Dict

**Example for DWI+ADC:**
```python
{
    'first_transform': [
        {'LoadImaged': {'keys': ['image_dwi', 'image_adc', 'label']}},
        {'EnsureChannelFirstd': {'keys': ['image_dwi', 'image_adc', 'label']}},
        {'Orientationd': {'keys': ['image_dwi', 'image_adc', 'label'], 'axcodes': 'RAS'}},
    ],
    'preprocessing': [
        # Spatial transforms on individual modalities
        {'Spacingd': {'keys': ['image_dwi', 'image_adc', 'label'], 'pixdim': [1.5, 1.5, 1.5]}},
        {'NormalizeIntensityd': {'keys': ['image_dwi', 'image_adc']}},
    ],
    'concatenation': [
        # NEW: Merge modalities before augmentation
        {'ConcatImagesd': {
            'keys': ['image_dwi', 'image_adc'],
            'output_key': 'image',
            'channel_dim': 0
        }},
    ],
    'augmentation': [
        # Now 'image' has 2 channels
        {'RandSpatialCropSamplesd': {
            'keys': ['image', 'label'],
            'roi_size': [96, 96, 96],
            'num_samples': 2
        }},
    ],
}
```

---

## 5. Model Configuration Auto-Detection

### 5.1 Extract Config from Data

```python
def extract_model_config(split_lists: SplitLists) -> dict:
    """Auto-configure model from data structure"""
    first_subject = split_lists[0][0]

    image_keys = get_category_keys(first_subject, 'image')
    label_keys = get_category_keys(first_subject, 'label')

    return {
        'in_channels': len(image_keys),      # 2 for DWI+ADC
        'out_channels': len(label_keys),     # 1 for single label, N for multi-class
        'image_keys': image_keys,
        'label_keys': label_keys,
    }
```

### 5.2 Model Initialization

**Before:**
```python
model = get_model_from_hyper_params(
    hyper_params={'in_channels': 1, 'out_channels': 1, ...}
)
```

**After:**
```python
# Auto-detect if not specified
config = extract_model_config(split_lists)
if 'in_channels' not in hyper_params:
    hyper_params['in_channels'] = config['in_channels']
if 'out_channels' not in hyper_params:
    hyper_params['out_channels'] = config['out_channels']

model = get_model_from_hyper_params(hyper_params)
```

---

## 6. Implementation Plan

### Phase 1: Core Infrastructure (Day 1, ~6 hours)
- [ ] Create `lesseg_unet/data_utils.py` with type definitions
- [ ] Implement `parse_key()`, `get_category_keys()`, `build_schema()`, `validate_against_schema()`
- [ ] Unit tests for key parsing and schema validation
- [ ] **Files:** New `data_utils.py` (~200 lines)
- [ ] **Risk:** Low (new code, no dependencies)

### Phase 2: Folder-per-Modality Converter (Day 1-2, ~6 hours)
- [ ] Implement `folder_mode_to_split_lists()` converter
- [ ] Add CLI arguments: `--image-folder`, `--modality-suffixes`
- [ ] Update `data_loading.py` to call converter
- [ ] Unit tests with sample data
- [ ] **Files:** `data_loading.py` (+150 lines), `main.py` (+50 lines)
- [ ] **Risk:** Low (isolated feature)

### Phase 3: Transform Pipeline Updates (Day 2, ~4 hours)
- [ ] Modify `transform_dicts.py` to accept `image_keys`, `label_keys`
- [ ] Add `ConcatImagesd` transform after preprocessing
- [ ] Update all transform dicts to use dynamic keys
- [ ] **Files:** `data/transform_dicts.py` (~100 line changes)
- [ ] **Risk:** Medium (affects all training)

### Phase 4: Model Config Auto-Detection (Day 2-3, ~4 hours)
- [ ] Implement `extract_model_config()` in `data_utils.py`
- [ ] Update `training.py` to auto-configure model
- [ ] Update checkpoint saving to include multi-modal metadata
- [ ] **Files:** `training.py` (+80 lines), `data_utils.py` (+50 lines)
- [ ] **Risk:** Medium (affects model initialization)

### Phase 5: Testing & Validation (Day 3, ~4 hours)
- [ ] Create test dataset with DWI+ADC
- [ ] End-to-end training test
- [ ] Backward compatibility test (single modality)
- [ ] Edge case tests (missing files, mismatched modalities)
- [ ] **Files:** `tests/` (new test files)
- [ ] **Risk:** Low (testing only)

### Phase 6: BIDS & One-Folder Modes (Optional, Day 4-5)
- [ ] Implement `bids_mode_to_split_lists()` converter
- [ ] Implement `one_folder_to_split_lists()` with detailed error messages
- [ ] Add CLI arguments for all modes
- [ ] **Files:** `data_loading.py` (+300 lines), `main.py` (+100 lines)
- [ ] **Risk:** Low-Medium (new features)

---

## 7. CLI Design

### Option A: Separate Flags per Mode (Recommended)

```bash
# Mode 1: Folder-per-modality
lesseg_unet \
  --image-folder dwi:/data/dwi adc:/data/adc \
  --label-folder /data/lesions \
  -o /output

# Mode 2: BIDS
lesseg_unet \
  --bids-root /data \
  --image-suffixes T1w T2w dwi \
  --label-suffix lesion \
  -o /output

# Mode 3: One-folder pattern matching
lesseg_unet \
  --data-folder /data \
  --image-patterns _dwi _adc \
  --label-pattern _lesion \
  -o /output

# Backward compatible: Single modality
lesseg_unet \
  -p /data/images \
  -lp /data/labels \
  -o /output
```

---

## 8. Backward Compatibility

**Guaranteed:**
- Existing single-modality workflows work unchanged
- Old checkpoints load correctly
- Transform dicts with `'image'` and `'label'` keys continue to work

**Migration Path:**
```python
# Old code (still works):
{'image': '/path/dwi.nii.gz', 'label': '/path/mask.nii.gz'}
in_channels = 1

# New code (multi-modal):
{'image_dwi': '/path/dwi.nii.gz', 'image_adc': '/path/adc.nii.gz', 'label': '/path/mask.nii.gz'}
in_channels = 2  # Auto-detected
```

---

## 9. Testing Strategy

### Unit Tests
- Key parsing (`parse_key`, `get_category_keys`)
- Schema validation (`build_schema`, `validate_against_schema`)
- Model config extraction (`extract_model_config`)

### Integration Tests
- Folder-per-modality converter with sample data
- Transform pipeline with 2-channel input
- Model initialization with auto-config

### End-to-End Tests
- Train SwinUNETR on DWI+ADC for 5 epochs
- Validate checkpoint saving/loading
- Test inference with multi-modal model

### Backward Compatibility Tests
- Single-modality training still works
- Old checkpoints load correctly
- Existing CLI arguments unchanged

---

## 10. Risk Mitigation

### Risk 1: Transform Dict Compatibility
**Mitigation:** Support both old (`['image', 'label']`) and new (`['image_dwi', 'image_adc', 'label']`) formats

### Risk 2: Missing Modalities
**Mitigation:** Schema validation with clear error messages showing which subjects/files are missing

### Risk 3: Memory Usage
**Mitigation:** Multi-modal increases memory ~linearly with channel count (2 channels ≈ 2× memory). Document requirements.

### Risk 4: Breaking Existing Workflows
**Mitigation:** Keep all existing CLI arguments working. New features are additive only.

---

## 11. Documentation Requirements

### User Documentation
- [ ] Multi-modal training tutorial (DWI+ADC example)
- [ ] CLI reference for all input modes
- [ ] Example datasets and scripts
- [ ] Memory requirements guide

### Developer Documentation
- [ ] Architecture diagram (data flow)
- [ ] API reference for `data_utils.py`
- [ ] Transform pipeline explanation
- [ ] Checkpoint format updates

---

## 12. Open Questions

1. **Channel ordering:** Should it be alphabetical (`adc, dwi`) or user-specified order?
   - **Recommendation:** User-specified order (order of `--image-folder` arguments)

2. **Mixed single/multi-modal in same dataset:** Should we support subjects with different modality counts?
   - **Recommendation:** No - enforce consistent schema across all subjects

3. **Modality-specific normalization:** Should normalization be per-modality or global?
   - **Recommendation:** Per-modality (apply `NormalizeIntensityd` before concatenation)

4. **Control images:** How to handle multi-modal controls (control_dwi, control_adc)?
   - **Recommendation:** Same concatenation logic as images

---

## 13. Success Criteria

- [ ] Can train SwinUNETR on DWI+ADC data
- [ ] Auto-detects `in_channels=2` from data structure
- [ ] Checkpoint saves/loads multi-modal metadata correctly
- [ ] Validation and segmentation work with multi-modal models
- [ ] Backward compatible: single-modality workflows unchanged
- [ ] Clear error messages for common mistakes
- [ ] Documentation complete with examples

---

## Next Steps

1. **Review this design with user** - Get approval before implementation
2. **Set up test dataset** - Create small DWI+ADC dataset for testing
3. **Start Phase 1** - Implement core infrastructure (type defs, key parsing)
4. **Iterate** - Test each phase before moving to next

---

**Last Updated:** 2025-12-20
**Design Version:** 2.0 (Refined & Consolidated)

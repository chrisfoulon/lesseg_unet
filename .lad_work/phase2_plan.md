# Phase 2: Folder-per-Modality Converter - Implementation Plan

**Status:** In Progress
**Started:** 2025-12-20
**Estimated Time:** ~6 hours

---

## Objectives

1. Implement `folder_mode_to_split_lists()` converter function
2. Add CLI arguments for folder mode
3. Update `data_loading.py` to support multi-modal inputs
4. Write comprehensive unit tests

---

## Function Signature

```python
def folder_mode_to_split_lists(
    image_folders: dict[str, Path | str],  # {'dwi': Path('data/dwi'), 'adc': ...}
    label_folder: Path | str,
    n_folds: int = 5,
    subject_pattern: str = r'(sub-\d+)',  # Regex to extract subject ID
    random_seed: int = 42
) -> SplitLists:
    """Convert folder-per-modality structure to SplitLists format.

    Parameters
    ----------
    image_folders : dict[str, Path | str]
        Dictionary mapping modality names to folder paths.
        Example: {'dwi': 'data/dwi', 'adc': 'data/adc'}
    label_folder : Path | str
        Path to folder containing label files
    n_folds : int, default=5
        Number of folds for cross-validation
    subject_pattern : str, default=r'(sub-\d+)'
        Regex pattern to extract subject ID from filenames
    random_seed : int, default=42
        Random seed for reproducible fold splitting

    Returns
    -------
    SplitLists
        List of folds, each containing SubjectDict entries

    Raises
    ------
    ValueError
        If subjects have missing modalities or labels
        If no valid subjects found
    """
```

---

## Algorithm

### Step 1: Extract Subject IDs from Each Modality

```python
# For each modality folder:
# - List all .nii.gz files
# - Extract subject ID using regex
# - Store mapping: {subject_id: file_path}

modality_files = {
    'dwi': {
        'sub-001': 'data/dwi/sub-001_dwi.nii.gz',
        'sub-002': 'data/dwi/sub-002_dwi.nii.gz'
    },
    'adc': {
        'sub-001': 'data/adc/sub-001_adc.nii.gz',
        'sub-002': 'data/adc/sub-002_adc.nii.gz'
    }
}

label_files = {
    'sub-001': 'data/labels/sub-001_lesion.nii.gz',
    'sub-002': 'data/labels/sub-002_lesion.nii.gz'
}
```

### Step 2: Find Complete Subjects

```python
# Subject is complete if:
# - Has file for ALL modalities
# - Has label file

all_modalities = set(image_folders.keys())
subject_ids = set(label_files.keys())

complete_subjects = []
for subject_id in subject_ids:
    has_all_modalities = all(
        subject_id in modality_files[mod]
        for mod in all_modalities
    )
    if has_all_modalities:
        complete_subjects.append(subject_id)
```

### Step 3: Build SubjectDict Entries

```python
subject_list = []
for subject_id in complete_subjects:
    subject_dict = {}

    # Add all image modalities
    for modality in image_folders.keys():
        key = f'image_{modality}'
        subject_dict[key] = modality_files[modality][subject_id]

    # Add label
    subject_dict['label'] = label_files[subject_id]

    subject_list.append(subject_dict)

# Example result:
# [
#   {'image_dwi': 'data/dwi/sub-001_dwi.nii.gz',
#    'image_adc': 'data/adc/sub-001_adc.nii.gz',
#    'label': 'data/labels/sub-001_lesion.nii.gz'},
#   ...
# ]
```

### Step 4: Split into Folds

```python
# Shuffle subjects (with seed for reproducibility)
np.random.seed(random_seed)
shuffled_indices = np.random.permutation(len(subject_list))

# Split into n_folds
split_lists = []
fold_size = len(subject_list) // n_folds

for i in range(n_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(subject_list)

    fold_indices = shuffled_indices[start_idx:end_idx]
    fold = [subject_list[idx] for idx in fold_indices]
    split_lists.append(fold)

return split_lists
```

### Step 5: Error Handling & Validation

**Error Cases:**
1. **Missing modalities:** Subject has some but not all modalities
2. **Missing labels:** Subject has all modalities but no label
3. **No valid subjects:** No subjects with complete data
4. **Empty folders:** Folder contains no .nii.gz files

**Validation:**
- All folds have at least one subject
- Schema consistency (first subject defines expected keys)
- File paths exist

---

## Test Cases

### 1. Basic Multi-Modal (2 modalities)
```python
def test_basic_two_modalities():
    """Test with DWI + ADC modalities."""
    # Create temp folders with mock files
    # Run converter
    # Assert:
    # - len(split_lists) == n_folds
    # - All subjects have 'image_dwi', 'image_adc', 'label' keys
    # - Schema validation passes
```

### 2. Single Modality (Backward Compatible)
```python
def test_single_modality():
    """Test with single modality (backward compatible)."""
    # Only one modality in image_folders
    # Should produce: [{'image_dwi': '...', 'label': '...'}]
```

### 3. Three Modalities
```python
def test_three_modalities():
    """Test with DWI + ADC + FLAIR."""
    # Three modalities
    # Assert all subjects have all three
```

### 4. Missing Modality File
```python
def test_missing_modality_file():
    """Test error when subject missing one modality."""
    # sub-001 has DWI but not ADC
    # Should raise ValueError with descriptive message
```

### 5. Missing Label File
```python
def test_missing_label_file():
    """Test error when subject has images but no label."""
    # sub-001 has DWI+ADC but no label
    # Should raise ValueError
```

### 6. Custom Subject Pattern
```python
def test_custom_subject_pattern():
    """Test custom regex pattern for subject ID extraction."""
    # Files like: patient_001_dwi.nii.gz
    # Pattern: r'(patient_\d+)'
```

### 7. Fold Distribution
```python
def test_fold_distribution():
    """Test subjects are evenly distributed across folds."""
    # 23 subjects, 5 folds
    # Should be: [5, 5, 5, 5, 3] or similar
```

### 8. Reproducibility
```python
def test_reproducibility():
    """Test same random_seed produces same folds."""
    # Run twice with same seed
    # Assert identical fold assignments
```

### 9. Empty Folder
```python
def test_empty_folder():
    """Test error when folder is empty."""
    # Modality folder has no .nii.gz files
    # Should raise ValueError
```

---

## Implementation Order (TDD)

1. ✅ **Create test file structure**
   - `tests/test_folder_converter.py`
   - Mock file creation utilities

2. ✅ **Write all tests**
   - 9+ test cases covering all scenarios
   - Use pytest fixtures for temp directories

3. ✅ **Implement helper functions**
   - `_extract_subject_id(filename: str, pattern: str) -> str | None`
   - `_list_nifti_files(folder: Path) -> list[Path]`
   - `_build_subject_to_file_mapping(folder: Path, pattern: str) -> dict[str, str]`

4. ✅ **Implement main converter**
   - `folder_mode_to_split_lists()`
   - Iterative development to pass tests

5. ✅ **Add to data_utils.py**
   - Export function
   - Update module docstring

---

## CLI Integration (After Converter Works)

### New Arguments in main.py

```python
parser.add_argument(
    '--image-folders',
    nargs='+',
    help='Folder-per-modality mode: modality:path pairs. '
         'Example: dwi:data/dwi adc:data/adc'
)

parser.add_argument(
    '--label-folder',
    type=str,
    help='Path to folder containing label files'
)

parser.add_argument(
    '--modality-suffixes',
    nargs='+',
    help='Suffixes for image modalities (must match --image-folders order)'
)

parser.add_argument(
    '--subject-pattern',
    type=str,
    default=r'(sub-\d+)',
    help='Regex pattern to extract subject ID from filenames'
)
```

### Usage Example

```bash
lesseg_unet \
  --image-folders dwi:data/dwi adc:data/adc \
  --label-folder data/lesion_masks \
  --modality-suffixes dwi adc \
  --subject-pattern '(sub-\d+)' \
  -nf 5 \
  -ne 100 \
  -bs 2 \
  -o output/
```

---

## Success Criteria

- ✅ All unit tests pass (9+ tests)
- ✅ NumPy docstrings on all functions
- ✅ Type hints on all functions
- ✅ Handles edge cases gracefully
- ✅ Descriptive error messages
- ✅ Backward compatible (single modality works)
- ✅ CLI integration complete
- ✅ Ready for Phase 3

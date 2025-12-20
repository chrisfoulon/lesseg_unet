# Phase 1: Core Infrastructure - Implementation Plan

**Status:** In Progress
**Started:** 2025-12-20
**Estimated Time:** ~6 hours

---

## Objectives

1. Create `lesseg_unet/data_utils.py` with type definitions
2. Implement core functions for multi-modal support
3. Write comprehensive unit tests (TDD approach)

---

## Functions to Implement

### 1. `parse_key(key: str) -> tuple[str, str | None]`
Parse a key into (category, identifier).

**Examples:**
- `parse_key('image_dwi')` → `('image', 'dwi')`
- `parse_key('image')` → `('image', None)`
- `parse_key('label_class0')` → `('label', 'class0')`

**Test Cases:**
- Single-word keys (no underscore)
- Multi-part keys with one underscore
- Keys with multiple underscores (split on first only)
- Empty string handling
- Edge cases

### 2. `get_category_keys(subject: dict, category: str) -> list[str]`
Extract all keys belonging to a specific category from a subject dict.

**Examples:**
- `get_category_keys({'image_dwi': '...', 'image_adc': '...', 'label': '...'}, 'image')`
  → `['image_dwi', 'image_adc']`
- `get_category_keys({'image': '...', 'label': '...'}, 'image')`
  → `['image']`

**Test Cases:**
- Single modality (backward compatible)
- Multiple modalities
- No matching keys
- Mixed keys (some with identifiers, some without)

### 3. `build_schema(first_subject: dict) -> dict`
Build a schema from the first subject.

**Returns:**
```python
{
    'image_keys': list[str],
    'label_keys': list[str],
    'control_keys': list[str],
    'has_multi_modal_images': bool,
    'has_multi_class_labels': bool,
    'has_controls': bool,
}
```

**Test Cases:**
- Single modality (no identifiers)
- Multi-modal images
- Multi-class labels
- Controls present
- All features combined

### 4. `validate_against_schema(subject: dict, schema: dict, subject_id: str) -> None`
Validate that a subject matches the expected schema.

**Raises:** `ValueError` if keys don't match

**Test Cases:**
- Valid subjects (should not raise)
- Missing image key
- Missing label key
- Extra keys (should be allowed? or rejected?)
- Descriptive error messages

---

## Type Definitions

```python
from typing import TypeAlias

SubjectDict: TypeAlias = dict[str, str]  # {'image_dwi': '/path', ...}
Fold: TypeAlias = list[SubjectDict]      # List of subjects in one fold
SplitLists: TypeAlias = list[Fold]       # Multiple folds
```

---

## TDD Workflow

1. **Write tests FIRST** (tests/test_data_utils.py)
   - All test cases for parse_key()
   - All test cases for get_category_keys()
   - All test cases for build_schema()
   - All test cases for validate_against_schema()

2. **Run tests** (expect failures)
   ```bash
   pytest tests/test_data_utils.py -v
   ```

3. **Implement functions** (lesseg_unet/data_utils.py)
   - One function at a time
   - Run tests after each function
   - Iterate until all tests pass

4. **Review and verify**
   - Code quality check
   - Documentation complete (NumPy docstrings)
   - All tests passing
   - Ready for Phase 2

---

## Success Criteria

- ✅ All unit tests pass
- ✅ NumPy docstrings for all functions
- ✅ Type hints on all functions
- ✅ Code coverage > 95%
- ✅ No breaking changes to existing code
- ✅ Ready to proceed to Phase 2

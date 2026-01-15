# Phase 3 Design Plan - Unified Input Pipeline

**Date**: 2026-01-15
**Status**: DESIGN phase - Awaiting approval before implementation

---

## 1. Overview

### Goal
Create a unified conditional pipeline where:
- **Loading functions** are input-type aware, return `list[Path]` or `list[dict]` or `list[list[dict]]`
- **Formatting functions** handle matching and are input-type agnostic
- **Pipeline logic is explicit** in main.py (not hidden in wrapper functions)

### Key Changes from Current Implementation
| Aspect | Current (Wrong) | New (Correct) |
|--------|-----------------|---------------|
| Pattern default | `r'(sub-\d+)'` forced | `None` (no filter) |
| Matching strategy | Subject ID extraction via regex | Residual matching (remove modality name) |
| Pipeline visibility | Hidden in `folder_mode_to_split_lists()` | Explicit conditional in main.py |
| Function responsibility | Mixed (loading + matching + splitting) | Separated (loading vs formatting) |

---

## 2. New Function Signatures

### 2.1 Loading Functions (Stage 0)

#### `read_folder()`
```python
def read_folder(
    folder: str | Path,
    pattern: str | None = None,
    recursive: bool = False
) -> list[Path]:
    """List NIfTI files from a folder.

    Parameters
    ----------
    folder : str | Path
        Directory to scan for NIfTI files.
    pattern : str | None, optional
        Glob-style pattern to filter files (e.g., 'sub1*', '*dwi*').
        If None, all NIfTI files are returned.
    recursive : bool, optional
        If True, search recursively in subdirectories.

    Returns
    -------
    list[Path]
        Sorted list of NIfTI file paths.

    Raises
    ------
    ValueError
        If folder doesn't exist or contains no matching NIfTI files.

    Examples
    --------
    >>> read_folder('/data/dwi')  # All NIfTI files
    >>> read_folder('/data/images', pattern='patient*')  # Filtered
    """
```

#### `read_list_file()`
```python
def read_list_file(
    filepath: str | Path,
    check_exists: bool = True
) -> list[Path]:
    """Read file paths from a text file.

    Parameters
    ----------
    filepath : str | Path
        Path to text file containing one file path per line.
    check_exists : bool, optional
        If True, verify each path exists (default: True).

    Returns
    -------
    list[Path]
        List of file paths from the text file.

    Raises
    ------
    FileNotFoundError
        If filepath doesn't exist.
    ValueError
        If check_exists=True and any listed path doesn't exist.
    """
```

#### `read_list_dicts()`
```python
def read_list_dicts(
    filepath: str | Path,
    check_exists: bool = True
) -> list[dict[str, Path]]:
    """Read pre-matched subject dictionaries from a JSON file.

    Parameters
    ----------
    filepath : str | Path
        Path to JSON file containing list of subject dicts.
        Format: [{"image_dwi": "/path/to/dwi.nii", "label_lesion": "/path/to/label.nii"}, ...]
    check_exists : bool, optional
        If True, verify each path in dicts exists (default: True).

    Returns
    -------
    list[dict[str, Path]]
        List of subject dictionaries with validated paths.

    Raises
    ------
    ValueError
        If JSON format is invalid or paths don't exist.
    """
```

#### `read_presplit_json()`
```python
def read_presplit_json(
    filepath: str | Path,
    check_exists: bool = True
) -> list[list[dict[str, Path]]]:
    """Read pre-split subject lists from a JSON file.

    Parameters
    ----------
    filepath : str | Path
        Path to JSON file containing split lists.
        Format: [[{fold0_subject1}, {fold0_subject2}], [{fold1_subject1}, ...], ...]
    check_exists : bool, optional
        If True, verify each path in dicts exists (default: True).

    Returns
    -------
    list[list[dict[str, Path]]]
        Nested list of subject dictionaries organized by fold.

    Raises
    ------
    ValueError
        If JSON format is invalid or paths don't exist.
    """
```

### 2.2 Formatting Functions (Stage 1-2)

#### `match_lists_to_dicts()`
```python
def match_lists_to_dicts(
    image_lists: dict[str, list[Path]],
    label_lists: dict[str, list[Path]] | None = None,
    control_lists: dict[str, list[Path]] | None = None,
    strip_pattern: str | None = None,
    extract_pattern: str | None = None,
    control_strip_pattern: str | None = None,
    control_extract_pattern: str | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Match files across modalities to create subject dictionaries.

    Two matching mechanisms are available (mutually exclusive):

    1. STRIP: Remove pattern from filename, match by residual.
       Use when filenames differ by modality-specific prefixes/suffixes.

    2. EXTRACT: Extract pattern from filename as matching key.
       Use when filenames contain a subject identifier pattern.

    Parameters
    ----------
    image_lists : dict[str, list[Path]]
        Dictionary mapping modality names to lists of file paths.
        Example: {'dwi': [path1, path2], 'adc': [path3, path4]}
    label_lists : dict[str, list[Path]] | None, optional
        Dictionary mapping label class names to lists of file paths.
    control_lists : dict[str, list[Path]] | None, optional
        Dictionary mapping control modality names to lists of file paths.
    strip_pattern : str | None, optional
        Pattern to REMOVE from filenames for residual matching.
        - None (default): exact filename match (no processing)
        - string/regex: remove this pattern, match residuals
    extract_pattern : str | None, optional
        Regex pattern to EXTRACT from filenames as matching key.
        Cannot be used together with strip_pattern.
    control_strip_pattern : str | None, optional
        Strip pattern for controls. Default: use control modality names.
    control_extract_pattern : str | None, optional
        Extract pattern for controls. Cannot be used with control_strip_pattern.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (subject_dicts, control_dicts) where each dict has format:
        {'image_<modality>': path, 'label_<class>': path, ...}

    Raises
    ------
    ValueError
        If both strip_pattern and extract_pattern are provided.
        If matching fails (no matches, ambiguous matches, or missing modalities).

    Examples
    --------
    # Default: strip modality names, match residuals
    >>> image_lists = {'dwi': [Path('dwi_pat001.nii')], 'adc': [Path('adc_pat001.nii')]}
    >>> match_lists_to_dicts(image_lists)
    # 'dwi_pat001.nii' -> strip 'dwi' -> '_pat001.nii'
    # 'adc_pat001.nii' -> strip 'adc' -> '_pat001.nii'
    # Match! -> {'image_dwi': 'dwi_pat001.nii', 'image_adc': 'adc_pat001.nii'}

    # Exact filename match
    >>> match_lists_to_dicts(image_lists, strip_pattern='exact')

    # Extract subject ID pattern
    >>> match_lists_to_dicts(image_lists, extract_pattern=r'subj\\d+')
    # 'scan_subj123_dwi.nii' -> extract 'subj123'
    # 'scan_subj123_adc.nii' -> extract 'subj123'
    # Match!
    """
```

#### `shuffle_and_split_subjects()` - Already exists, keep as-is

---

## 3. Matching Algorithms

### 3.1 Two Distinct Mechanisms

#### Mechanism 1: STRIP & Match Residual
Remove what makes files **DIFFERENT**, match by what's **LEFT** (residual).

```
"DWI_B1000_patient001.nii"  → strip "DWI_B1000" → "_patient001.nii" (residual)
"Diff_ADC_patient001.nii"   → strip "Diff_ADC"  → "_patient001.nii" (residual)
                                                   ↑ MATCH!
```

#### Mechanism 2: EXTRACT & Match Key
Extract what makes files **THE SAME**, match by what's **EXTRACTED** (key).

```
"scan_subj123_dwi.nii"  → extract r"subj\d+" → "subj123" (key)
"scan_subj123_adc.nii"  → extract r"subj\d+" → "subj123" (key)
                                                ↑ MATCH!
```

### 3.2 Matching Cases

#### Case A: Default (no pattern provided) - EXACT MATCH
- Filenames must be IDENTICAL across folders
- Simplest case, no pattern processing

```python
# Folder /data/dwi/: patient001.nii, patient002.nii
# Folder /data/adc/: patient001.nii, patient002.nii
#
# Direct match by filename - files must have same names
# Match! Create: {'image_dwi': '/data/dwi/patient001.nii', 'image_adc': '/data/adc/patient001.nii'}
```

#### Case B: Strip pattern (--strip-pattern "pattern")
- User provides pattern to REMOVE (literal string OR regex)
- Match by residual (what remains after stripping)

```python
# --strip-pattern "dwi" (or use modality name)
# "dwi_patient001.nii" → strip "dwi" → "_patient001.nii"
# "adc_patient001.nii" → strip "adc" → "_patient001.nii"
# Match! Same residual

# --strip-pattern r"_run-\d+"
# "patient001_run-01.nii" → strip "_run-01" → "patient001.nii"
# "patient001_run-02.nii" → strip "_run-02" → "patient001.nii"
```

#### Case C: Extract pattern (--extract-pattern "regex")
- User provides regex pattern to EXTRACT as matching key
- Useful when subject ID follows a pattern

```python
# --extract-pattern r"subj\d+"
# "scan_subj123_dwi_b1000.nii" → extract → "subj123"
# "processed_subj123_adc.nii"  → extract → "subj123"
# Match! Same extracted key
```

### 3.3 Algorithm Pseudocode

```python
def match_lists_to_dicts(
    image_lists: dict[str, list[Path]],
    label_lists: dict[str, list[Path]] | None = None,
    control_lists: dict[str, list[Path]] | None = None,
    strip_pattern: str | None = None,
    extract_pattern: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Match files across modalities using strip or extract mechanism."""

    # Validate: can't use both
    if strip_pattern and extract_pattern:
        raise ValueError("Cannot use both --strip-pattern and --extract-pattern")

    # Determine key extraction function
    if extract_pattern:
        # EXTRACT mechanism: extract pattern as key
        def get_key(path: Path, modality: str) -> str:
            match = re.search(extract_pattern, path.name)
            if not match:
                raise ValueError(
                    f"Extract pattern '{extract_pattern}' not found in '{path.name}'"
                )
            return match.group(0)

    elif strip_pattern:
        # STRIP mechanism: remove pattern, match residual
        def get_key(path: Path, modality: str) -> str:
            return re.sub(strip_pattern, '', path.name, count=1)

    else:
        # DEFAULT: exact filename match (no processing)
        def get_key(path: Path, modality: str) -> str:
            return path.name

    # Build {key: {modality: path}} mapping
    image_by_key = defaultdict(dict)
    for modality, paths in image_lists.items():
        for path in paths:
            key = get_key(path, modality)
            if modality in image_by_key[key]:
                raise ValueError(
                    f"Duplicate key '{key}' for modality '{modality}': "
                    f"both '{image_by_key[key][modality]}' and '{path}'"
                )
            image_by_key[key][modality] = path

    # Same for labels and controls...

    # Find complete matches
    subject_dicts = []
    all_image_modalities = set(image_lists.keys())

    for key, modalities_found in image_by_key.items():
        if set(modalities_found.keys()) == all_image_modalities:
            # All image modalities present for this key
            subject_dict = {f'image_{mod}': str(path)
                           for mod, path in sorted(modalities_found.items())}
            # Add labels if present...
            subject_dicts.append(subject_dict)

    # Error reporting
    if not subject_dicts:
        first_unmatched = next(iter(next(iter(image_lists.values()))))
        raise ValueError(
            f"Matching failed for {sum(len(v) for v in image_lists.values())} files.\n"
            f"  First file: '{first_unmatched.name}'\n"
            f"  Pattern used: '{strip_pattern or extract_pattern or 'modality names'}'\n"
            f"  Hint: Check that the pattern correctly identifies matching files."
        )

    return subject_dicts, control_dicts
```

---

## 4. Main.py Conditional Pipeline

### 4.1 Current State (WRONG - to delete)

```python
# Lines 637-725: Duplicated logic, training vs inference branching
# This will be DELETED
```

### 4.2 New Pipeline Flow

```python
# ===== INPUT LOADING (Stage 0) =====
# Detect input type and load accordingly

if args.pretrained_split_list:
    # Already split - skip all stages
    img_list = read_presplit_json(args.pretrained_split_list)
    # Format: list[list[dict]]

elif args.input_list:
    # List file mode
    image_paths = {'default': read_list_file(args.input_list)}
    label_paths = {'default': read_list_file(args.lesion_input_list)} if args.lesion_input_list else None
    control_paths = {'default': read_list_file(args.controls_list)} if args.controls_list else None
    needs_matching = True

elif args.input_path:
    if is_multi_modal:  # len(args.input_path) > 1
        # Folder-per-modality mode
        image_paths = {name: read_folder(folder) for name, folder in zip(modality_names, args.input_path)}
        label_paths = {name: read_folder(folder) for name, folder in zip(label_names, args.lesion_input_path)} if args.lesion_input_path else None
        control_paths = {name: read_folder(folder) for name, folder in zip(control_names, args.controls_path)} if args.controls_path else None
        needs_matching = True
    else:
        # Single folder scan (backward compatible)
        img_list = create_input_path_list_from_root(args.input_path[0])
        needs_matching = False

# ===== MATCHING (Stage 1) =====
if needs_matching:
    subject_dicts, control_dicts = match_lists_to_dicts(
        image_lists=image_paths,
        label_lists=label_paths,
        control_lists=control_paths,
        matching_pattern=args.matching_pattern
    )
    all_subjects = subject_dicts + control_dicts

# ===== SHUFFLE + SPLIT (Stage 2 - Training only) =====
is_training = args.checkpoint is None

if is_training and needs_matching:
    img_list = shuffle_and_split_subjects(
        subject_dicts=all_subjects,
        n_folds=args.folds_number,
        shuffle=True,
        random_seed=42
    )
    # Format: list[list[dict]]
elif needs_matching:
    # Inference: flat list, no shuffle
    img_list = all_subjects
    # Format: list[dict]
```

### 4.3 CLI Argument Changes

```python
# REMOVE these (wrong defaults):
# --subject-pattern default=r'(sub-\d+)'
# --control-pattern default=r'(ctr-\d+)'

# ADD these:

# Filter patterns (one per type, or one per modality)
parser.add_argument(
    '--image-filter',
    type=str,
    nargs='*',
    default=None,
    help='Glob pattern(s) to filter image files during loading. '
         'If 1 pattern: applied to all image modalities. '
         'If N patterns: must match N image modalities (one-to-one). '
         'Default: None (load all NIfTI files)'
)

parser.add_argument(
    '--label-filter',
    type=str,
    nargs='*',
    default=None,
    help='Glob pattern(s) to filter label files. Same rules as --image-filter.'
)

parser.add_argument(
    '--control-filter',
    type=str,
    nargs='*',
    default=None,
    help='Glob pattern(s) to filter control files. Same rules as --image-filter.'
)

# Matching patterns - TWO DISTINCT MECHANISMS:
#
# 1. STRIP-PATTERN: Remove what makes files DIFFERENT, match by RESIDUAL
#    Example: "DWI_patient001.nii" strip "DWI_" -> "patient001.nii" (residual)
#
# 2. EXTRACT-PATTERN: Extract what makes files the SAME, match by EXTRACTED KEY
#    Example: "scan_subj123_dwi.nii" extract r"subj\d+" -> "subj123" (key)

parser.add_argument(
    '--strip-pattern',
    type=str,
    default=None,
    help='Pattern to REMOVE from filenames for residual matching. '
         'After stripping, files with identical residuals are matched. '
         'Can be literal string ("DWI_") or regex (r"ses-\\d+_"). '
         'Default: None (exact filename matching - filenames must be identical).'
)

parser.add_argument(
    '--extract-pattern',
    type=str,
    default=None,
    help='Regex pattern to EXTRACT from filenames as matching key. '
         'Files with identical extracted keys are matched. '
         'Example: r"subj\\d+" extracts "subj123" from "scan_subj123_dwi.nii". '
         'Cannot be used together with --strip-pattern.'
)

parser.add_argument(
    '--control-strip-pattern',
    type=str,
    default=None,
    help='Strip pattern for control files. Default: use control modality names.'
)

parser.add_argument(
    '--control-extract-pattern',
    type=str,
    default=None,
    help='Extract pattern for control files. Cannot be used with --control-strip-pattern.'
)
```

### 4.4 Filter Pattern Logic

#### Folder × Filter Combinations

| Folders | Modalities | Filters | Valid? | Behavior |
|---------|------------|---------|--------|----------|
| N folders | N modalities | None | ✅ | Each folder scanned independently |
| N folders | N modalities | 1 filter | ✅ | Same filter applied to each folder |
| N folders | N modalities | N filters | ✅ | One filter per folder |
| **1 folder** | **N modalities** | **None** | ❌ | Error: Can't distinguish modalities |
| **1 folder** | **N modalities** | **1 filter** | ❌ | Error: Would list same files N times |
| **1 folder** | **N modalities** | **N filters** | ✅ | Filters partition files by modality |

**Key insight**: Single folder + multiple modalities requires N DISTINCT filters to partition files.

```python
def load_modality_files(
    folders: list[Path],
    modality_names: list[str],
    filters: list[str] | None
) -> dict[str, list[Path]]:
    """Load files for each modality from folders with optional filtering.

    Parameters
    ----------
    folders : list[Path]
        Either N folders (one per modality) or 1 folder (shared).
    modality_names : list[str]
        Names for each modality.
    filters : list[str] | None
        Glob patterns for filtering. Rules depend on folder count.

    Returns
    -------
    dict[str, list[Path]]
        {modality_name: [list of file paths]}

    Raises
    ------
    ValueError
        If filter/folder/modality combination is invalid.
    """
    n_folders = len(folders)
    n_modalities = len(modality_names)
    n_filters = len(filters) if filters else 0

    # Case 1: Separate folder per modality
    if n_folders == n_modalities:
        if n_filters == 0:
            # No filtering, scan each folder
            return {mod: read_folder(folder)
                    for mod, folder in zip(modality_names, folders)}
        elif n_filters == 1:
            # Same filter for all folders
            return {mod: read_folder(folder, pattern=filters[0])
                    for mod, folder in zip(modality_names, folders)}
        elif n_filters == n_modalities:
            # One filter per folder
            return {mod: read_folder(folder, pattern=filt)
                    for mod, folder, filt in zip(modality_names, folders, filters)}
        else:
            raise ValueError(
                f"Filter count mismatch: got {n_filters} filters for "
                f"{n_modalities} modalities. Provide 0, 1, or {n_modalities} filters."
            )

    # Case 2: Single folder for all modalities
    elif n_folders == 1:
        if n_filters != n_modalities:
            raise ValueError(
                f"Single folder with {n_modalities} modalities requires exactly "
                f"{n_modalities} filters to distinguish them. Got {n_filters} filters.\n"
                f"Example: --image-filter '*dwi*' '*adc*' for modalities dwi, adc"
            )
        # Each filter selects files for one modality from the shared folder
        return {mod: read_folder(folders[0], pattern=filt)
                for mod, filt in zip(modality_names, filters)}

    else:
        raise ValueError(
            f"Invalid folder count: got {n_folders} folders for {n_modalities} modalities. "
            f"Provide either 1 folder (shared) or {n_modalities} folders (one per modality)."
        )
```

---

## 5. Test Plan (TDD)

### 5.1 Loading Function Tests

#### `test_read_folder.py`
```python
class TestReadFolder:
    def test_list_all_nifti_files(self, tmp_path):
        """No pattern: returns all NIfTI files sorted."""

    def test_filter_by_glob_pattern(self, tmp_path):
        """Pattern 'patient*' filters correctly."""

    def test_empty_folder_raises_error(self, tmp_path):
        """Empty folder raises ValueError."""

    def test_nonexistent_folder_raises_error(self):
        """Nonexistent path raises ValueError."""

    def test_recursive_search(self, tmp_path):
        """recursive=True finds files in subdirectories."""
```

#### `test_read_list_file.py`
```python
class TestReadListFile:
    def test_read_valid_list_file(self, tmp_path):
        """Reads paths from text file."""

    def test_check_exists_validates_paths(self, tmp_path):
        """check_exists=True raises error for missing files."""

    def test_skip_check_exists(self, tmp_path):
        """check_exists=False skips validation."""

    def test_handles_empty_lines(self, tmp_path):
        """Skips empty lines and comments."""
```

#### `test_read_list_dicts.py` and `test_read_presplit_json.py`
```python
# Similar structure: valid JSON, invalid JSON, path validation
```

### 5.2 Matching Function Tests

#### `test_match_lists_to_dicts.py`
```python
class TestMatchListsToDictsDefault:
    """Tests for default exact filename matching."""

    def test_default_exact_match_identical_filenames(self, tmp_path):
        """No pattern: filenames must be identical across folders."""

    def test_default_exact_match_different_filenames_fails(self, tmp_path):
        """No pattern + different filenames -> no matches -> error."""


class TestMatchListsToDictsStripMechanism:
    """Tests for STRIP & match residual mechanism."""

    def test_strip_pattern_literal_string(self, tmp_path):
        """strip_pattern='dwi_' removes literal string, match residuals."""

    def test_strip_pattern_in_middle_of_filename(self, tmp_path):
        """pat001_dwi_session1.nii -> strip 'dwi' -> 'pat001__session1.nii'"""

    def test_strip_pattern_regex(self, tmp_path):
        """strip_pattern=r'_run-\\d+' removes regex match."""

    def test_strip_pattern_no_match_keeps_filename(self, tmp_path):
        """If pattern not in filename, filename unchanged (may cause mismatch)."""


class TestMatchListsToDictsExtractMechanism:
    """Tests for EXTRACT & match key mechanism."""

    def test_extract_subject_id_pattern(self, tmp_path):
        """extract_pattern=r'subj\\d+' extracts 'subj123' as key."""

    def test_extract_bids_style_pattern(self, tmp_path):
        """extract_pattern=r'sub-[a-zA-Z0-9]+' for BIDS-like naming."""

    def test_extract_pattern_not_found_raises_error(self, tmp_path):
        """File without extractable pattern raises ValueError."""


class TestMatchListsToDictsValidation:
    """Tests for validation and error handling."""

    def test_both_strip_and_extract_raises_error(self):
        """Cannot use both strip_pattern and extract_pattern."""

    def test_ambiguous_match_raises_error(self, tmp_path):
        """Multiple files produce same key -> error with details."""

    def test_incomplete_match_raises_error(self, tmp_path):
        """Missing modality for some subjects -> error."""

    def test_no_matches_error_shows_first_file(self, tmp_path):
        """Error message includes first unmatched file and pattern."""


class TestMatchListsToDictsWithLabelsAndControls:
    """Tests for multi-type matching."""

    def test_match_images_and_labels(self, tmp_path):
        """Images + labels matched correctly."""

    def test_match_images_labels_and_controls(self, tmp_path):
        """Full subject dict with all types."""

    def test_controls_use_separate_pattern(self, tmp_path):
        """control_strip_pattern different from strip_pattern."""

    def test_controls_with_extract_pattern(self, tmp_path):
        """control_extract_pattern for controls."""
```

### 5.3 Integration Tests

```python
class TestConditionalPipeline:
    def test_training_mode_shuffles_and_splits(self):
        """Training mode produces list[list[dict]]."""

    def test_inference_mode_no_shuffle(self):
        """Inference mode produces list[dict], order preserved."""

    def test_presplit_mode_skips_all_stages(self):
        """Pre-split JSON used directly."""

    def test_backward_compatible_single_folder(self):
        """Single folder scan still works."""
```

---

## 6. Cleanup Plan

### 6.1 Code to Delete (After new pipeline works)

| File | Lines | Description |
|------|-------|-------------|
| `main.py` | 637-725 | Wrong Phase 3 conditional logic |
| `main.py` | 223-229 | Old --subject-pattern and --control-pattern args |

### 6.2 Functions to Deprecate

```python
# In data_utils.py, add deprecation warnings:

import warnings

def list_nifti_from_folders(...):
    """DEPRECATED: Use read_folder() + match_lists_to_dicts() instead."""
    warnings.warn(
        "list_nifti_from_folders() is deprecated and will be removed in v3.0. "
        "Use read_folder() for loading and match_lists_to_dicts() for matching.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation ...

def match_modalities_by_subject(...):
    """DEPRECATED: Use match_lists_to_dicts() instead."""
    warnings.warn(
        "match_modalities_by_subject() is deprecated and will be removed in v3.0. "
        "Use match_lists_to_dicts() which supports residual matching.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation ...

def folder_mode_to_split_lists(...):
    """DEPRECATED: Use explicit pipeline in main.py instead."""
    warnings.warn(
        "folder_mode_to_split_lists() is deprecated and will be removed in v3.0. "
        "The staged pipeline is now explicit in main.py.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation ...
```

### 6.3 Tests to Review

| Test File | Action |
|-----------|--------|
| `tests/staged_input/test_list_nifti.py` | Keep for deprecated function, add warning suppression |
| `tests/staged_input/test_match.py` | Keep for deprecated function, add warning suppression |
| `tests/test_folder_converter.py` | Keep for backward compat, add warning suppression |

### 6.4 Documentation Updates

| File | Action |
|------|--------|
| `docs/staged-input-pipeline/COMPACT_CONTEXT.md` | Update with new architecture |
| `docs/staged-input-pipeline/unified_pipeline_vision.md` | Mark as implemented |
| `README.md` | Update CLI argument documentation |

---

## 7. Implementation Order

### Phase 3A: Foundation (TDD)
1. Write tests for `read_folder()`
2. Implement `read_folder()`
3. Write tests for `read_list_file()`
4. Implement `read_list_file()`
5. Write tests for `read_list_dicts()` and `read_presplit_json()`
6. Implement both functions

### Phase 3B: Core Matching (TDD)
7. Write tests for `match_lists_to_dicts()` (all cases)
8. Implement `match_lists_to_dicts()` with residual matching
9. Implement helper `remove_pattern()`

### Phase 3C: Main.py Refactor
10. **DELETE wrong code** at main.py:637-725
11. **REMOVE** old CLI args (--subject-pattern, --control-pattern)
12. **ADD** new CLI args (--filter-pattern, --matching-pattern)
13. Implement explicit conditional pipeline
14. Run all tests

### Phase 3D: Cleanup
15. Add deprecation warnings to old functions
16. Suppress warnings in old tests
17. Update documentation
18. Final test run (45 staged + 13 folder_converter + new tests)

---

## 8. Verification Checklist

- [ ] All 45 existing staged_input tests pass
- [ ] All 13 folder_converter tests pass (with deprecation warnings)
- [ ] New loading function tests pass
- [ ] New matching function tests pass
- [ ] Training mode produces shuffled split_lists
- [ ] Inference mode produces flat list without shuffle
- [ ] Single folder backward compatibility works
- [ ] Pre-split JSON mode works
- [ ] Flake8 clean (complexity < 10)
- [ ] Coverage >= 90% for new code

---

## 9. Design Decisions (Resolved)

1. **Filter pattern scope**: Separate per type (`--image-filter`, `--label-filter`, `--control-filter`). Each accepts 1 or N patterns (N must match modality count).

2. **Matching pattern for controls**: Separate `--control-matching-pattern` for controls.

3. **Error verbosity**: Show count + pattern + first unmatched filename:
   ```
   ValueError: Matching failed for 12 files with pattern 'dwi'.
     First unmatched: 'MRI_patient001_b1000.nii.gz'
     Hint: The pattern 'dwi' was not found in this filename.
   ```

4. **Pattern type**: Matching patterns support both literal strings AND regex (Python's `re.sub` handles both naturally).

---

*Awaiting approval before proceeding to implementation.*

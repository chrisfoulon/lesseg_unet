"""Data utilities for multi-modal imaging support.

This module provides type definitions and utility functions for handling
multi-modal medical imaging data in lesseg_unet. It implements suffix-based
key naming and schema validation to enable training on multiple imaging
modalities (e.g., DWI + ADC + FLAIR) while maintaining backward compatibility
with single-modality workflows.

Key Concepts
------------
- **Suffix-based naming**: Keys follow the pattern `{category}_{identifier}`
  (e.g., 'image_dwi', 'label_class0') or just `{category}` for single modality
- **Categories**: 'image' (patient imaging), 'control' (healthy subject),
  'label' (segmentation masks)
- **Schema validation**: First subject defines expected structure; all other
  subjects must match

Examples
--------
Single modality (backward compatible):
    >>> subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
    >>> schema = build_schema(subject)
    >>> schema['has_multi_modal_images']
    False

Multi-modal imaging:
    >>> subject = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }
    >>> schema = build_schema(subject)
    >>> schema['has_multi_modal_images']
    True
    >>> schema['image_keys']
    ['image_dwi', 'image_adc']
"""

import re
import json
import warnings
import numpy as np
from pathlib import Path
from typing import List, TypeAlias
from copy import deepcopy

# Type definitions for multi-modal data structures
SubjectDict: TypeAlias = dict[str, str]
"""Dictionary mapping keys to file paths for a single subject.

Examples
--------
Single modality:
    {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}

Multi-modal:
    {'image_dwi': '/path/dwi.nii.gz', 'image_adc': '/path/adc.nii.gz',
     'label': '/path/mask.nii.gz'}
"""

Fold: TypeAlias = list[SubjectDict]
"""List of subjects in a single fold for cross-validation."""

SplitLists: TypeAlias = list[Fold]
"""Multiple folds for cross-validation (canonical data format)."""


# ============================================================================
# Stage 0: Loading Functions
# ============================================================================

def read_folder(
    folder: str | Path,
    pattern: str | None = None,
    recursive: bool = False
) -> list[Path]:
    """List NIfTI files from a folder with optional glob filtering.

    This is a Stage 0 loading function that returns a flat list of file paths.
    Use this to load files before matching across modalities.

    Parameters
    ----------
    folder : str | Path
        Directory to scan for NIfTI files.
    pattern : str | None, optional
        Glob-style pattern to filter files (e.g., 'patient*', '*dwi*').
        If None, all NIfTI files are returned.
    recursive : bool, optional
        If True, search recursively in subdirectories.
        Default: False.

    Returns
    -------
    list[Path]
        Sorted list of NIfTI file paths (.nii or .nii.gz).

    Raises
    ------
    ValueError
        If folder doesn't exist or contains no matching NIfTI files.

    Examples
    --------
    >>> # List all NIfTI files
    >>> paths = read_folder('/data/dwi')

    >>> # Filter by pattern
    >>> paths = read_folder('/data/images', pattern='patient*')

    >>> # Recursive search
    >>> paths = read_folder('/data/bids', recursive=True)
    """
    folder = Path(folder)

    # Validate folder exists
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    # Helper to check if file is NIfTI
    def is_nifti(path: Path) -> bool:
        name = path.name.lower()
        return name.endswith('.nii') or name.endswith('.nii.gz')

    # List files based on pattern and recursion
    if pattern:
        # Use glob with pattern
        if recursive:
            # rglob for recursive
            all_paths = list(folder.rglob(pattern))
        else:
            # glob for non-recursive
            all_paths = list(folder.glob(pattern))
        # Filter to only NIfTI files
        nifti_paths = [p for p in all_paths if p.is_file() and is_nifti(p)]
    else:
        # No pattern - list all NIfTI files
        if recursive:
            nifti_paths = [p for p in folder.rglob('*') if p.is_file() and is_nifti(p)]
        else:
            nifti_paths = [p for p in folder.iterdir() if p.is_file() and is_nifti(p)]

    # Validate we found files
    if not nifti_paths:
        if pattern:
            raise ValueError(
                f"No NIfTI files found in '{folder}' matching pattern '{pattern}'"
            )
        else:
            raise ValueError(f"No NIfTI files found in '{folder}'")

    # Return sorted list
    return sorted(nifti_paths)


def read_list_file(
    filepath: str | Path,
    check_exists: bool = True
) -> list[Path]:
    """Read file paths from a text file.

    This is a Stage 0 loading function that reads paths from a list file.
    Each line in the file should contain one file path.

    Parameters
    ----------
    filepath : str | Path
        Path to text file containing one file path per line.
    check_exists : bool, optional
        If True, verify each path exists (default: True).

    Returns
    -------
    list[Path]
        List of file paths in the order they appear in the file.

    Raises
    ------
    FileNotFoundError
        If the list file doesn't exist.
    ValueError
        If the list file is empty or contains only comments.
        If check_exists=True and any listed path doesn't exist.

    Notes
    -----
    - Empty lines are skipped
    - Lines starting with '#' are treated as comments
    - Leading/trailing whitespace is stripped from paths

    Examples
    --------
    >>> # paths.txt contains:
    >>> # /data/patient001.nii.gz
    >>> # /data/patient002.nii.gz
    >>> paths = read_list_file('paths.txt')
    """
    filepath = Path(filepath)

    # Check list file exists
    if not filepath.exists():
        raise FileNotFoundError(f"List file not found: {filepath}")

    # Read and parse lines
    paths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            paths.append(Path(line))

    # Validate we have paths
    if not paths:
        raise ValueError(f"No paths found in list file: {filepath}")

    # Optionally validate paths exist
    if check_exists:
        for path in paths:
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")

    return paths


def read_list_dicts(
    filepath: str | Path,
    check_exists: bool = True
) -> list[dict[str, Path]]:
    """Read pre-matched subject dictionaries from a JSON file.

    This is a Stage 0 loading function for pre-matched data.
    Use this when users provide already-matched subject data.

    Parameters
    ----------
    filepath : str | Path
        Path to JSON file containing list of subject dicts.
        Format: [{"image_dwi": "/path/dwi.nii", "label": "/path/mask.nii"}, ...]
    check_exists : bool, optional
        If True, verify each path in dicts exists (default: True).

    Returns
    -------
    list[dict[str, Path]]
        List of subject dictionaries with Path values.

    Raises
    ------
    FileNotFoundError
        If the JSON file doesn't exist.
    ValueError
        If JSON format is invalid (not a list, empty, or parse error).
        If check_exists=True and any path doesn't exist.

    Examples
    --------
    >>> # subjects.json contains:
    >>> # [{"image": "/data/img1.nii", "label": "/data/mask1.nii"}, ...]
    >>> subjects = read_list_dicts('subjects.json')
    """
    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    # Parse JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")

    # Validate structure
    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON list of dicts, got {type(data).__name__} in {filepath}"
        )

    if not data:
        raise ValueError(f"Empty list in JSON file: {filepath}")

    # Convert string paths to Path objects
    result = []
    for subject_dict in data:
        converted = {key: Path(value) for key, value in subject_dict.items()}
        result.append(converted)

    # Optionally validate paths exist
    if check_exists:
        for subject_dict in result:
            for key, path in subject_dict.items():
                if not path.exists():
                    raise ValueError(f"Path does not exist: {path} (key: {key})")

    return result


def read_presplit_json(
    filepath: str | Path,
    check_exists: bool = True
) -> list[list[dict[str, Path]]]:
    """Read pre-split subject lists from a JSON file.

    This is a Stage 0 loading function for pre-split data.
    Use this when users provide already-split cross-validation folds.

    Parameters
    ----------
    filepath : str | Path
        Path to JSON file containing split lists.
        Format: [[{fold0_subj1}, {fold0_subj2}], [{fold1_subj1}, ...], ...]
    check_exists : bool, optional
        If True, verify each path in dicts exists (default: True).

    Returns
    -------
    list[list[dict[str, Path]]]
        Nested list: folds -> subjects -> {key: Path}.

    Raises
    ------
    FileNotFoundError
        If the JSON file doesn't exist.
    ValueError
        If JSON format is invalid (not nested lists, empty folds).
        If check_exists=True and any path doesn't exist.

    Examples
    --------
    >>> # split_lists.json contains:
    >>> # [[{"image": "/data/img1.nii"}, {"image": "/data/img2.nii"}], ...]
    >>> folds = read_presplit_json('split_lists.json')
    >>> len(folds)  # Number of folds
    5
    """
    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    # Parse JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")

    # Validate top-level structure
    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON list of folds, got {type(data).__name__} in {filepath}"
        )

    if not data:
        raise ValueError(f"Empty list in JSON file: {filepath}")

    # Validate nested structure and convert
    result = []
    for fold_idx, fold in enumerate(data):
        if not isinstance(fold, list):
            raise ValueError(
                f"Expected fold {fold_idx} to be a list of dicts, "
                f"got {type(fold).__name__}. Format should be: "
                f"[[{{subj1}}, {{subj2}}], [{{subj3}}, ...], ...]"
            )

        if not fold:
            raise ValueError(f"Empty fold at index {fold_idx} in {filepath}")

        # Convert each subject dict
        converted_fold = []
        for subject_dict in fold:
            if not isinstance(subject_dict, dict):
                raise ValueError(
                    f"Expected dict in fold {fold_idx}, got {type(subject_dict).__name__}"
                )
            converted = {key: Path(value) for key, value in subject_dict.items()}
            converted_fold.append(converted)

        result.append(converted_fold)

    # Optionally validate paths exist
    if check_exists:
        for fold_idx, fold in enumerate(result):
            for subj_idx, subject_dict in enumerate(fold):
                for key, path in subject_dict.items():
                    if not path.exists():
                        raise ValueError(
                            f"Path does not exist: {path} "
                            f"(fold {fold_idx}, subject {subj_idx}, key: {key})"
                        )

    return result


# ============================================================================
# Stage 1: Matching Functions
# ============================================================================

def match_lists_to_dicts(
    image_lists: dict[str, list[Path]],
    label_lists: dict[str, list[Path]] | None = None,
    control_lists: dict[str, list[Path]] | None = None,
    strip_pattern: str | dict[str, str] | None = None,
    extract_pattern: str | None = None,
    control_strip_pattern: str | dict[str, str] | None = None,
    control_extract_pattern: str | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Match files across modalities to create subject dictionaries.

    Two matching mechanisms are available (mutually exclusive):

    1. Default (no patterns): Exact filename match across folders
    2. STRIP: Remove pattern from filename, match by residual
    3. EXTRACT: Extract pattern from filename as matching key

    Parameters
    ----------
    image_lists : dict[str, list[Path]]
        Dictionary mapping modality names to lists of file paths.
    label_lists : dict[str, list[Path]] | None, optional
        Dictionary mapping label class names to lists of file paths.
    control_lists : dict[str, list[Path]] | None, optional
        Dictionary mapping control modality names to lists of file paths.
    strip_pattern : str | dict[str, str] | None, optional
        Pattern to REMOVE from filenames for residual matching.
        - None (default): exact filename match
        - str: same pattern applied to all modalities
        - dict: {modality: pattern} for per-modality patterns
    extract_pattern : str | None, optional
        Regex pattern to EXTRACT from filenames as matching key.
        Cannot be used together with strip_pattern.
    control_strip_pattern : str | dict[str, str] | None, optional
        Strip pattern for controls. Default: use control modality names.
    control_extract_pattern : str | None, optional
        Extract pattern for controls.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (subject_dicts, control_dicts) where each dict has format:
        {'image_<modality>': path, 'label_<class>': path, ...}
    """
    from collections import defaultdict

    # Validate: can't use both strip and extract for subjects
    if strip_pattern is not None and extract_pattern is not None:
        raise ValueError(
            "Cannot use both strip_pattern and extract_pattern. "
            "Choose one matching mechanism."
        )

    # Validate: can't use both strip and extract for controls
    if control_strip_pattern is not None and control_extract_pattern is not None:
        raise ValueError(
            "Cannot use both control_strip_pattern and control_extract_pattern."
        )

    def get_key_func(pattern, extract_pat, modality):
        """Create key extraction function based on pattern type."""
        if extract_pat is not None:
            # EXTRACT: extract pattern as key
            def get_key(path: Path) -> str:
                match = re.search(extract_pat, path.name)
                if not match:
                    raise ValueError(
                        f"Extract pattern '{extract_pat}' not found in '{path.name}'"
                    )
                return match.group(0)
            return get_key

        elif pattern is not None:
            # STRIP: get pattern for this modality
            if isinstance(pattern, dict):
                mod_pattern = pattern.get(modality, modality)
            else:
                mod_pattern = pattern

            def get_key(path: Path) -> str:
                return re.sub(mod_pattern, '', path.name, count=1)
            return get_key

        else:
            # DEFAULT: exact filename match
            def get_key(path: Path) -> str:
                return path.name
            return get_key

    def build_key_mapping(lists, pattern, extract_pat, key_prefix):
        """Build {key: {modality: path}} mapping."""
        by_key = defaultdict(dict)

        for modality, paths in lists.items():
            get_key = get_key_func(pattern, extract_pat, modality)

            for path in paths:
                key = get_key(path)
                if modality in by_key[key]:
                    raise ValueError(
                        f"Duplicate key '{key}' for {key_prefix} '{modality}': "
                        f"both '{by_key[key][modality]}' and '{path}'"
                    )
                by_key[key][modality] = path

        return by_key

    # Build mappings for images
    image_by_key = build_key_mapping(
        image_lists, strip_pattern, extract_pattern, 'image'
    )

    # Build mappings for labels (if provided)
    label_by_key = {}
    if label_lists:
        label_by_key = build_key_mapping(
            label_lists, strip_pattern, extract_pattern, 'label'
        )

    # Find complete subject matches
    all_image_modalities = set(image_lists.keys())
    all_label_classes = set(label_lists.keys()) if label_lists else set()

    subject_dicts = []
    for key, modalities_found in image_by_key.items():
        # Check all image modalities present
        if set(modalities_found.keys()) != all_image_modalities:
            continue  # Incomplete, skip

        # Check all labels present (if required)
        if label_lists:
            if key not in label_by_key:
                continue
            labels_found = label_by_key[key]
            if set(labels_found.keys()) != all_label_classes:
                continue  # Incomplete labels

        # Build subject dict
        subject_dict = {}
        for mod, path in sorted(modalities_found.items()):
            subject_dict[f'image_{mod}'] = str(path)

        if label_lists and key in label_by_key:
            for cls, path in sorted(label_by_key[key].items()):
                subject_dict[f'label_{cls}'] = str(path)

        subject_dicts.append(subject_dict)

    # Handle controls separately
    control_dicts = []
    if control_lists:
        ctrl_pattern = control_strip_pattern
        ctrl_extract = control_extract_pattern

        control_by_key = build_key_mapping(
            control_lists, ctrl_pattern, ctrl_extract, 'control'
        )

        all_control_modalities = set(control_lists.keys())

        for key, modalities_found in control_by_key.items():
            if set(modalities_found.keys()) != all_control_modalities:
                continue

            control_dict = {}
            for mod, path in sorted(modalities_found.items()):
                control_dict[f'control_{mod}'] = str(path)

            control_dicts.append(control_dict)

    # Error if no matches found
    if not subject_dicts and image_lists:
        first_mod = next(iter(image_lists.keys()))
        first_file = image_lists[first_mod][0] if image_lists[first_mod] else "unknown"
        pattern_desc = (
            f"extract_pattern='{extract_pattern}'" if extract_pattern
            else f"strip_pattern='{strip_pattern}'" if strip_pattern
            else "exact filename match"
        )
        raise ValueError(
            f"No complete matches found using {pattern_desc}.\n"
            f"  First file: '{first_file}'\n"
            f"  Total files: {sum(len(v) for v in image_lists.values())}\n"
            f"  Hint: Check that filenames match across modalities, "
            f"or provide a strip_pattern/extract_pattern."
        )

    return subject_dicts, control_dicts


def parse_key(key: str) -> tuple[str, str | None]:
    """Parse a key into category and identifier components.

    Keys follow the pattern `{category}_{identifier}` for multi-modal data
    or just `{category}` for single-modality backward compatibility.

    Parameters
    ----------
    key : str
        The key to parse (e.g., 'image_dwi', 'image', 'label_class0')

    Returns
    -------
    category : str
        The category portion (e.g., 'image', 'label', 'control')
    identifier : str or None
        The identifier portion (e.g., 'dwi', 'class0') or None if no underscore

    Notes
    -----
    - If key contains multiple underscores, splits on the first one only
    - This allows identifiers to contain underscores (e.g., 'label_class0_subtype')

    Examples
    --------
    >>> parse_key('image_dwi')
    ('image', 'dwi')
    >>> parse_key('image')
    ('image', None)
    >>> parse_key('label_class0_subtype')
    ('label', 'class0_subtype')
    """
    if '_' in key:
        category, identifier = key.split('_', 1)
        return category, identifier
    else:
        return key, None


def get_category_keys(subject: SubjectDict, category: str) -> list[str]:
    """Extract all keys belonging to a specific category from a subject.

    Parameters
    ----------
    subject : SubjectDict
        Dictionary mapping keys to file paths
    category : str
        Category to filter by ('image', 'label', or 'control')

    Returns
    -------
    list[str]
        List of keys belonging to the specified category. Order is preserved
        from the subject dictionary. Returns empty list if no matches.

    Examples
    --------
    >>> subject = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }
    >>> get_category_keys(subject, 'image')
    ['image_dwi', 'image_adc']
    >>> get_category_keys(subject, 'control')
    []

    Backward compatibility:
    >>> subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
    >>> get_category_keys(subject, 'image')
    ['image']
    """
    matching_keys = []
    for key in subject.keys():
        key_category, _ = parse_key(key)
        if key_category == category:
            matching_keys.append(key)
    return matching_keys


def build_schema(first_subject: SubjectDict) -> dict:
    """Build a schema from the first subject to validate subsequent subjects.

    The schema extracts the expected data structure from the first subject,
    including which keys are present and whether multi-modal/multi-class
    features are used. This implements a "decision tree" validation approach
    where all subjects must match the structure of the first.

    Parameters
    ----------
    first_subject : SubjectDict
        The first subject in the dataset, used as the reference structure

    Returns
    -------
    dict
        Schema dictionary with the following keys:

        - **image_keys** (list[str]): List of image keys
        - **label_keys** (list[str]): List of label keys
        - **control_keys** (list[str]): List of control keys
        - **has_multi_modal_images** (bool): True if multiple image modalities
        - **has_multi_class_labels** (bool): True if multiple label classes
        - **has_controls** (bool): True if control subjects present

    Examples
    --------
    Single modality:
    >>> subject = {'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}
    >>> schema = build_schema(subject)
    >>> schema['image_keys']
    ['image']
    >>> schema['has_multi_modal_images']
    False

    Multi-modal:
    >>> subject = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }
    >>> schema = build_schema(subject)
    >>> schema['image_keys']
    ['image_dwi', 'image_adc']
    >>> schema['has_multi_modal_images']
    True

    Multi-class labels:
    >>> subject = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'label_class0': '/path/lesion1.nii.gz',
    ...     'label_class1': '/path/lesion2.nii.gz'
    ... }
    >>> schema = build_schema(subject)
    >>> schema['label_keys']
    ['label_class0', 'label_class1']
    >>> schema['has_multi_class_labels']
    True
    """
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
        elif category == 'label':
            schema['label_keys'].append(key)
        elif category == 'control':
            schema['control_keys'].append(key)
            schema['has_controls'] = True

    # Set multi-modal/multi-class flags based on counts
    schema['has_multi_modal_images'] = len(schema['image_keys']) > 1
    schema['has_multi_class_labels'] = len(schema['label_keys']) > 1

    return schema


def validate_against_schema(subject: SubjectDict, schema: dict, subject_id: str) -> None:
    """Validate that a subject matches the expected schema.

    Checks that the subject contains all required keys defined in the schema.
    Extra keys are allowed for extensibility (e.g., metadata fields).

    Parameters
    ----------
    subject : SubjectDict
        The subject to validate
    schema : dict
        Schema built from the first subject (see build_schema)
    subject_id : str
        Identifier for the subject (used in error messages)

    Raises
    ------
    ValueError
        If any required key from the schema is missing from the subject.
        Error message includes subject_id, missing key, expected keys, and
        found keys for easy debugging.

    Examples
    --------
    Valid subject:
    >>> schema = {
    ...     'image_keys': ['image_dwi', 'image_adc'],
    ...     'label_keys': ['label'],
    ...     'control_keys': []
    ... }
    >>> subject = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }
    >>> validate_against_schema(subject, schema, 'subject_001')  # No error

    Missing key:
    >>> subject_bad = {
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }
    >>> validate_against_schema(subject_bad, schema, 'subject_002')
    Traceback (most recent call last):
        ...
    ValueError: Subject 'subject_002' missing expected key 'image_adc'
    Expected keys: ['image_dwi', 'image_adc']
    Found keys: ['image_dwi', 'label']
    """
    # Validate all required keys are present
    all_required_keys = (
        schema['image_keys'] + schema['label_keys'] + schema['control_keys']
    )

    for key in all_required_keys:
        if key not in subject:
            raise ValueError(
                f"Subject '{subject_id}' missing expected key '{key}'\n"
                f"Expected keys: {all_required_keys}\n"
                f"Found keys: {list(subject.keys())}"
            )


# ============================================================================
# Folder-per-Modality Converter
# ============================================================================


def _extract_subject_id(filename: str, pattern: str) -> str | None:
    r"""Extract subject ID from filename using regex pattern.

    Parameters
    ----------
    filename : str
        Filename to extract subject ID from (e.g., 'sub-001_dwi.nii.gz')
    pattern : str
        Regex pattern with one capture group for subject ID
        (e.g., r'(sub-\d+)')

    Returns
    -------
    str or None
        Subject ID if pattern matches, None otherwise

    Examples
    --------
    >>> _extract_subject_id('sub-001_dwi.nii.gz', r'(sub-\d+)')
    'sub-001'
    >>> _extract_subject_id('patient_042_t1.nii.gz', r'(patient_\d+)')
    'patient_042'
    >>> _extract_subject_id('no_match.nii.gz', r'(sub-\d+)')
    None
    """
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def _list_nifti_files(folder: Path) -> list[Path]:
    """List all NIfTI files in a folder.

    Parameters
    ----------
    folder : Path
        Folder to search for NIfTI files

    Returns
    -------
    list[Path]
        List of Path objects for .nii.gz and .nii files

    Examples
    --------
    >>> folder = Path('data/dwi')
    >>> files = _list_nifti_files(folder)
    >>> [f.name for f in files]
    ['sub-001_dwi.nii.gz', 'sub-002_dwi.nii.gz']
    """
    nifti_files = []
    # Search for both .nii.gz and .nii files
    nifti_files.extend(folder.glob('*.nii.gz'))
    nifti_files.extend(folder.glob('*.nii'))
    return sorted(nifti_files)


def _build_subject_to_file_mapping(
    folder: Path,
    pattern: str
) -> dict[str, str]:
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

    Examples
    --------
    >>> folder = Path('data/dwi')
    >>> mapping = _build_subject_to_file_mapping(folder, r'(sub-\d+)')
    >>> mapping
    {'sub-001': '/abs/path/data/dwi/sub-001_dwi.nii.gz',
     'sub-002': '/abs/path/data/dwi/sub-002_dwi.nii.gz'}
    """
    files = _list_nifti_files(folder)
    mapping = {}

    for file_path in files:
        subject_id = _extract_subject_id(file_path.name, pattern)
        if subject_id:
            if subject_id in mapping:
                raise ValueError(
                    f"Duplicate subject ID '{subject_id}' found in {folder}\n"
                    f"Files: {mapping[subject_id]} and {file_path}"
                )
            mapping[subject_id] = str(file_path.absolute())

    return mapping


def _validate_entity_completeness(
    entity_ids: set[str],
    modality_mappings: dict[str, dict[str, str]],
    label_mappings: dict[str, dict[str, str]] | None = None,
    key_prefix: str = 'image'
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Validate that entities have all required modalities and labels.

    Parameters
    ----------
    entity_ids : set[str]
        Set of entity IDs to validate (subjects or controls)
    modality_mappings : dict[str, dict[str, str]]
        Modality name to entity-to-file mappings
    label_mappings : dict[str, dict[str, str]] or None
        Label class to entity-to-file mappings (only for subjects)
    key_prefix : str
        Prefix for missing item names ('image' or 'control')

    Returns
    -------
    complete_entities : list[str]
        List of entity IDs with complete data
    incomplete_entities : list[tuple[str, list[str]]]
        List of (entity_id, missing_items) tuples
    """
    all_modalities = set(modality_mappings.keys())
    all_labels = set(label_mappings.keys()) if label_mappings is not None else set()

    complete_entities = []
    incomplete_entities = []

    for entity_id in entity_ids:
        missing_items = []

        # Check modalities
        for modality in all_modalities:
            if entity_id not in modality_mappings[modality]:
                missing_items.append(f"{key_prefix}_{modality}")

        # Check labels (if provided)
        if label_mappings is not None:
            for label_class in all_labels:
                if entity_id not in label_mappings[label_class]:
                    missing_items.append(f"label_{label_class}")

        if missing_items:
            incomplete_entities.append((entity_id, missing_items))
        else:
            complete_entities.append(entity_id)

    return complete_entities, incomplete_entities


def _build_entity_dicts(
    entity_ids: list[str],
    modality_mappings: dict[str, dict[str, str]],
    label_mappings: dict[str, dict[str, str]] | None = None,
    key_prefix: str = 'image'
) -> list[dict[str, str]]:
    """Build entity dictionaries with sorted keys.

    Parameters
    ----------
    entity_ids : list[str]
        List of entity IDs (should be sorted for reproducibility)
    modality_mappings : dict[str, dict[str, str]]
        Modality name to entity-to-file mappings
    label_mappings : dict[str, dict[str, str]] or None
        Label class to entity-to-file mappings
    key_prefix : str
        Prefix for keys ('image' or 'control')

    Returns
    -------
    entity_dicts : list[dict[str, str]]
        List of dictionaries with sorted keys
    """
    all_modalities = set(modality_mappings.keys())
    all_labels = set(label_mappings.keys()) if label_mappings is not None else set()

    entity_dicts = []
    for entity_id in entity_ids:
        entity_dict = {}

        # Add modalities (sorted for consistent ordering)
        for modality in sorted(all_modalities):
            key = f"{key_prefix}_{modality}"
            entity_dict[key] = modality_mappings[modality][entity_id]

        # Add labels (if provided, sorted for consistent ordering)
        if label_mappings is not None:
            for label_class in sorted(all_labels):
                key = f"label_{label_class}"
                entity_dict[key] = label_mappings[label_class][entity_id]

        entity_dicts.append(entity_dict)

    return entity_dicts


def _build_no_valid_subjects_error(
    all_modalities: set[str],
    all_label_classes: set[str],
    incomplete_subjects_count: int,
    all_control_modalities: set[str] | None = None,
    incomplete_controls_count: int = 0
) -> str:
    """Build error message when no valid subjects are found."""
    error_parts = ["No valid subjects found."]

    if all_label_classes:
        error_parts.append(f"Image modalities required: {sorted(all_modalities)}")
        error_parts.append(f"Label classes required: {sorted(all_label_classes)}")
    else:
        error_parts.append(f"Image modalities required: {sorted(all_modalities)}")

    if all_control_modalities is not None:
        error_parts.append(f"Control modalities required: {sorted(all_control_modalities)}")

    error_parts.append(f"Subjects with incomplete data: {incomplete_subjects_count}")
    if all_control_modalities is not None:
        error_parts.append(f"Controls with incomplete data: {incomplete_controls_count}")

    return '\n'.join(error_parts)


def _build_incomplete_data_error(
    incomplete_subjects: list[tuple[str, list[str]]],
    incomplete_controls: list[tuple[str, list[str]]]
) -> str:
    """Build error message for incomplete subjects/controls."""
    error_parts = ["Some subjects have incomplete data:"]

    if incomplete_subjects:
        error_parts.append(f"\nPatients missing data ({len(incomplete_subjects)} total):")
        for subj_id, missing in incomplete_subjects[:10]:  # Show first 10
            error_parts.append(f"  - {subj_id}: missing {missing}")
        if len(incomplete_subjects) > 10:
            error_parts.append(f"  ... and {len(incomplete_subjects) - 10} more")

    if incomplete_controls:
        error_parts.append(f"\nControls missing data ({len(incomplete_controls)} total):")
        for ctrl_id, missing in incomplete_controls[:10]:
            error_parts.append(f"  - {ctrl_id}: missing {missing}")
        if len(incomplete_controls) > 10:
            error_parts.append(f"  ... and {len(incomplete_controls) - 10} more")

    return '\n'.join(error_parts)


def validate_subject_dicts(
    subject_dicts: list[dict[str, str | Path]],
    check_loadable: bool = False,
    min_size: int = 100
) -> None:
    """Validate that all paths in subject dictionaries exist and are loadable.

    Parameters
    ----------
    subject_dicts : list[dict[str, str or Path]]
        List of subject dictionaries to validate.
        Each value should be a file path (str or Path).

        Example:
            [
                {'image_dwi': '/data/sub-001.nii.gz', 'label': '/labels/sub-001.nii.gz'},
                {'image_dwi': '/data/sub-002.nii.gz', 'label': '/labels/sub-002.nii.gz'}
            ]

    check_loadable : bool, default=False
        If True, attempt to load each file with nibabel to verify it's valid NIfTI.
        If False, only check file existence and size.

    min_size : int, default=100
        Minimum file size in bytes. Files smaller than this are considered
        potentially corrupt. Set to 0 to disable size checking.

    Raises
    ------
    ValueError
        If any file does not exist, is too small, or cannot be loaded (if check_loadable=True).

    Notes
    -----
    - Validates on first error (fails fast)
    - Accepts both str and Path objects
    - Empty list is valid (nothing to check)
    - Loadability check uses nibabel.load()

    Examples
    --------
    Basic existence check:

    >>> subject_dicts = [
    ...     {'image': '/data/sub-001.nii.gz'},
    ...     {'image': '/data/sub-002.nii.gz'}
    ... ]
    >>> validate_subject_dicts(subject_dicts, check_loadable=False)

    With loadability check:

    >>> validate_subject_dicts(subject_dicts, check_loadable=True, min_size=1000)
    """
    import nibabel as nib  # Import here to avoid dependency for non-validation use

    for subject_dict in subject_dicts:
        for key, file_path in subject_dict.items():
            # Convert to Path for consistent handling
            path = Path(file_path)

            # Check existence
            if not path.exists():
                raise ValueError(
                    f"File for key '{key}' does not exist: {path}"
                )

            # Check size
            if min_size > 0:
                file_size = path.stat().st_size
                if file_size < min_size:
                    raise ValueError(
                        f"File for key '{key}' is too small ({file_size} bytes < {min_size} bytes): {path}. "
                        f"This may indicate a corrupt or incomplete file."
                    )

            # Check loadability
            if check_loadable:
                try:
                    nib.load(path)
                except Exception as e:
                    raise ValueError(
                        f"File for key '{key}' cannot be loaded with nibabel: {path}. "
                        f"Error: {e}"
                    ) from e


def _validate_nonempty_modality_mappings(
    modality_mappings: dict[str, dict[str, str]],
    folder_paths: dict[str, Path | str],
    pattern: str,
    mapping_type: str = "image"
) -> None:
    """Validate that modality mappings are not empty.

    Parameters
    ----------
    modality_mappings : dict[str, dict[str, str]]
        Dictionary of modality-to-subject mappings to validate
    folder_paths : dict[str, Path | str]
        Original folder paths for error messages
    pattern : str
        Regex pattern used for error messages
    mapping_type : str
        Type of mapping for error messages ('image', 'label', or 'control')

    Raises
    ------
    ValueError
        If any modality mapping is empty (no files found)
    """
    for modality, mapping in modality_mappings.items():
        if not mapping:
            raise ValueError(
                f"No NIfTI files found in {folder_paths[modality]} matching pattern '{pattern}'"
            )


def list_nifti_from_folders(
    folders: dict[str, Path | str],
    pattern: str
) -> dict[str, dict[str, str]]:
    """List NIfTI files from multiple folders organized by modality.

    This function performs Stage 0→1 transformation in the input pipeline:
    converts folder paths into subject-to-file mappings.

    Parameters
    ----------
    folders : dict[str, Path or str]
        Dictionary mapping modality names to folder paths.
        Format: {modality_name: folder_path}

        Example:
            {
                'dwi': Path('/data/dwi'),
                'adc': '/data/adc',
                'flair': Path('/data/flair')
            }

    pattern : str
        Regex pattern to extract subject ID from filenames.
        Must contain exactly one capture group.

        Example:
            r'(sub-\\d+)' matches 'sub-001' in 'sub-001_dwi.nii.gz'

    Returns
    -------
    modality_mappings : dict[str, dict[str, str]]
        Dictionary mapping modality names to subject-to-file mappings.
        Format: {modality: {subject_id: absolute_path}}

        Example:
            {
                'dwi': {
                    'sub-001': '/data/dwi/sub-001_dwi.nii.gz',
                    'sub-002': '/data/dwi/sub-002_dwi.nii.gz'
                },
                'adc': {
                    'sub-001': '/data/adc/sub-001_adc.nii.gz',
                    'sub-002': '/data/adc/sub-002_adc.nii.gz'
                }
            }

    Raises
    ------
    ValueError
        If any folder does not exist or is not a directory.

    Notes
    -----
    - Finds both .nii and .nii.gz files
    - Returns absolute paths
    - Empty folders return empty subject mappings (not an error)
    - Uses _build_subject_to_file_mapping() internally

    Examples
    --------
    List files from multiple modality folders:

    >>> folders = {
    ...     'dwi': '/data/images/dwi',
    ...     'adc': '/data/images/adc'
    ... }
    >>> pattern = r'(sub-\\d+)'
    >>> mappings = list_nifti_from_folders(folders, pattern)
    >>> 'dwi' in mappings
    True
    >>> 'sub-001' in mappings['dwi']
    True

    .. deprecated:: 3.0
        Use :func:`read_folder` for loading and :func:`match_lists_to_dicts`
        for matching instead.
    """
    warnings.warn(
        "list_nifti_from_folders() is deprecated and will be removed in v3.0. "
        "Use read_folder() for loading and match_lists_to_dicts() for matching.",
        DeprecationWarning,
        stacklevel=2
    )
    modality_mappings = {}

    for modality_name, folder_path in folders.items():
        # Convert to Path object if str
        folder = Path(folder_path)

        # Validate folder exists
        if not folder.exists():
            raise ValueError(
                f"Folder for modality '{modality_name}' does not exist: {folder}"
            )

        if not folder.is_dir():
            raise ValueError(
                f"Path for modality '{modality_name}' is not a directory: {folder}"
            )

        # List files and build subject mapping
        subject_mapping = _build_subject_to_file_mapping(folder, pattern)
        modality_mappings[modality_name] = subject_mapping

    return modality_mappings


def shuffle_and_split_subjects(
    subject_dicts: list[dict[str, str]],
    n_folds: int,
    shuffle: bool = True,
    random_seed: int = 42
) -> list[list[dict[str, str]]]:
    """Shuffle and split subject dictionaries into cross-validation folds.

    This function performs Stage 2→3 transformation in the input pipeline:
    converts a flat list of subject dictionaries into nested fold lists.

    Parameters
    ----------
    subject_dicts : list[dict[str, str]]
        Flat list of subject dictionaries to split.
        Each dict contains paths for one subject.

        Example:
            [
                {'image_dwi': '/data/sub-001.nii.gz', 'label': '/labels/sub-001.nii.gz'},
                {'image_dwi': '/data/sub-002.nii.gz', 'label': '/labels/sub-002.nii.gz'},
                ...
            ]

    n_folds : int
        Number of folds to split into. Must be >= 1.
        If n_folds > len(subject_dicts), some folds will be empty.

    shuffle : bool, default=True
        If True, randomly shuffle subjects before splitting.
        If False, preserve original order.

    random_seed : int, default=42
        Random seed for reproducible shuffling.
        Only used when shuffle=True.

    Returns
    -------
    split_lists : list[list[dict[str, str]]]
        Nested list of subject dictionaries organized by fold.
        Format: [[fold0_dicts], [fold1_dicts], ...]

        Example (2 subjects, 2 folds):
            [
                [{'image_dwi': '/data/sub-001.nii.gz', ...}],  # Fold 0
                [{'image_dwi': '/data/sub-002.nii.gz', ...}]   # Fold 1
            ]

    Raises
    ------
    ValueError
        If subject_dicts is empty.

    Notes
    -----
    - Uses numpy.array_split for even distribution across folds
    - With 10 subjects and 3 folds: fold sizes will be [4, 3, 3]
    - Empty folds possible when n_folds > len(subject_dicts)
    - Maintains dict structure within subjects
    - Same random_seed produces same splits (reproducible)

    Examples
    --------
    Basic split with shuffle:

    >>> subjects = [
    ...     {'image': '/data/sub-001.nii.gz'},
    ...     {'image': '/data/sub-002.nii.gz'},
    ...     {'image': '/data/sub-003.nii.gz'}
    ... ]
    >>> splits = shuffle_and_split_subjects(subjects, n_folds=2, shuffle=True, random_seed=42)
    >>> len(splits)
    2
    >>> sum(len(fold) for fold in splits)
    3

    Split without shuffle (preserves order):

    >>> splits = shuffle_and_split_subjects(subjects, n_folds=2, shuffle=False)
    >>> splits[0][0]
    {'image': '/data/sub-001.nii.gz'}
    """
    # Validate input
    if not subject_dicts:
        raise ValueError("Cannot split empty subject list.")

    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(subject_dicts))
        subjects_to_split = [subject_dicts[idx] for idx in shuffled_indices]
    else:
        subjects_to_split = subject_dicts

    # Split into folds using numpy's array_split for even distribution
    split_arrays = np.array_split(np.array(subjects_to_split, dtype=object), n_folds)
    split_lists = [list(fold) for fold in split_arrays]

    return split_lists


def match_modalities_by_subject(
    image_modalities: dict[str, dict[str, str]],
    label_classes: dict[str, dict[str, str]] | None = None,
    control_modalities: dict[str, dict[str, str]] | None = None,
    require_all: bool = True
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Match imaging modalities and labels by subject ID.

    This function performs Stage 1→2 transformation in the input pipeline:
    converts subject-to-file mappings into complete subject dictionaries.

    Each subject must have ALL required modalities and labels (if labels
    provided). Controls are matched separately and returned in a separate list.

    Parameters
    ----------
    image_modalities : dict[str, dict[str, str]]
        Dictionary mapping modality names to subject-to-file mappings.
        Format: {modality: {subject_id: absolute_path}}

        Example:
            {
                'dwi': {'sub-001': '/data/dwi/sub-001.nii.gz', ...},
                'adc': {'sub-001': '/data/adc/sub-001.nii.gz', ...}
            }

    label_classes : dict[str, dict[str, str]] or None, optional
        Dictionary mapping label class names to subject-to-file mappings.
        Format: {label_class: {subject_id: absolute_path}}

        Example:
            {'stroke': {'sub-001': '/data/labels/sub-001.nii.gz', ...}}

        If None, creates subject dicts without labels (for segmentation).

    control_modalities : dict[str, dict[str, str]] or None, optional
        Dictionary mapping control modality names to control-to-file mappings.
        Format: {modality: {control_id: absolute_path}}

        Controls are healthy subjects without labels. Returned separately
        from patient subjects.

    require_all : bool, default=True
        If True, raises ValueError if any subject is missing modalities/labels.
        If False, incomplete subjects are silently excluded.

    Returns
    -------
    subject_dicts : list[dict[str, str]]
        List of subject dictionaries with keys:
        - 'image_{modality}' for each image modality (sorted alphabetically)
        - 'label_{class}' for each label class (sorted alphabetically)

        Example:
            [
                {
                    'image_adc': '/data/adc/sub-001.nii.gz',
                    'image_dwi': '/data/dwi/sub-001.nii.gz',
                    'label_stroke': '/data/labels/sub-001.nii.gz'
                },
                ...
            ]

    control_dicts : list[dict[str, str]]
        List of control dictionaries with keys:
        - 'control_{modality}' for each control modality (sorted alphabetically)

        Example:
            [
                {'control_dwi': '/data/controls/ctr-001.nii.gz'},
                ...
            ]

    Raises
    ------
    ValueError
        If require_all=True and any subject has incomplete data (missing
        modality or label). Error message lists all incomplete subjects.

    ValueError
        If no valid subjects found (all subjects are incomplete).

    Notes
    -----
    - Subject IDs are sorted alphabetically for reproducible output
    - Modality keys are sorted alphabetically for consistent key ordering
    - Label keys are sorted alphabetically for consistent key ordering
    - Controls are returned separately from subjects

    Examples
    --------
    Basic two-modality matching:

    >>> image_mods = {
    ...     'dwi': {'sub-001': '/data/dwi/sub-001.nii.gz'},
    ...     'adc': {'sub-001': '/data/adc/sub-001.nii.gz'}
    ... }
    >>> label_cls = {
    ...     'stroke': {'sub-001': '/data/labels/sub-001.nii.gz'}
    ... }
    >>> subjects, controls = match_modalities_by_subject(image_mods, label_cls)
    >>> subjects[0]
    {
        'image_adc': '/data/adc/sub-001.nii.gz',
        'image_dwi': '/data/dwi/sub-001.nii.gz',
        'label_stroke': '/data/labels/sub-001.nii.gz'
    }

    Segmentation mode (no labels):

    >>> subjects, _ = match_modalities_by_subject(image_mods, label_classes=None)
    >>> subjects[0]
    {'image_adc': '/data/adc/sub-001.nii.gz', 'image_dwi': '/data/dwi/sub-001.nii.gz'}

    With controls:

    >>> control_mods = {
    ...     'dwi': {'ctr-001': '/data/controls/ctr-001.nii.gz'}
    ... }
    >>> subjects, controls = match_modalities_by_subject(
    ...     image_mods, label_cls, control_mods
    ... )
    >>> controls[0]
    {'control_dwi': '/data/controls/ctr-001.nii.gz'}

    .. deprecated:: 3.0
        Use :func:`match_lists_to_dicts` instead.
    """
    warnings.warn(
        "match_modalities_by_subject() is deprecated and will be removed in v3.0. "
        "Use match_lists_to_dicts() which supports residual matching.",
        DeprecationWarning,
        stacklevel=2
    )
    # Step 1: Identify all required modalities and labels
    all_modalities = set(image_modalities.keys())
    all_label_classes = set(label_classes.keys()) if label_classes is not None else set()

    # Step 2: Find all subject IDs to check
    subjects_to_check = set()
    for modality_mapping in image_modalities.values():
        subjects_to_check.update(modality_mapping.keys())
    if label_classes is not None:
        for label_mapping in label_classes.values():
            subjects_to_check.update(label_mapping.keys())

    # Step 3: Validate subjects completeness
    complete_subjects, incomplete_subjects = _validate_entity_completeness(
        entity_ids=subjects_to_check,
        modality_mappings=image_modalities,
        label_mappings=label_classes,
        key_prefix='image'
    )

    # Step 4: Validate controls completeness (if provided)
    complete_controls = []
    incomplete_controls = []
    all_control_modalities = None

    if control_modalities is not None:
        all_control_modalities = set(control_modalities.keys())
        controls_to_check = set()
        for modality_mapping in control_modalities.values():
            controls_to_check.update(modality_mapping.keys())

        complete_controls, incomplete_controls = _validate_entity_completeness(
            entity_ids=controls_to_check,
            modality_mappings=control_modalities,
            label_mappings=None,
            key_prefix='control'
        )

    # Step 5: Check if any valid subjects found
    if not complete_subjects and not complete_controls:
        error_msg = _build_no_valid_subjects_error(
            all_modalities=all_modalities,
            all_label_classes=all_label_classes,
            incomplete_subjects_count=len(incomplete_subjects),
            all_control_modalities=all_control_modalities,
            incomplete_controls_count=len(incomplete_controls)
        )
        raise ValueError(error_msg)

    # Step 6: Raise error if require_all and incomplete subjects exist
    if require_all and (incomplete_subjects or incomplete_controls):
        error_msg = _build_incomplete_data_error(
            incomplete_subjects=incomplete_subjects,
            incomplete_controls=incomplete_controls
        )
        raise ValueError(error_msg)

    # Step 7: Build subject dictionaries (sorted for reproducibility)
    subject_dicts = _build_entity_dicts(
        entity_ids=sorted(complete_subjects),
        modality_mappings=image_modalities,
        label_mappings=label_classes,
        key_prefix='image'
    )

    # Step 8: Build control dictionaries (sorted for reproducibility)
    control_dicts = []
    if control_modalities is not None:
        control_dicts = _build_entity_dicts(
            entity_ids=sorted(complete_controls),
            modality_mappings=control_modalities,
            label_mappings=None,
            key_prefix='control'
        )

    return subject_dicts, control_dicts


def folder_mode_to_split_lists(
    image_folders: dict[str, Path | str],
    label_folders: dict[str, Path | str] | None = None,
    control_folders: dict[str, Path | str] | None = None,
    n_folds: int = 5,
    subject_pattern: str = r'(sub-\d+)',
    control_pattern: str = r'(ctr-\d+)',
    random_seed: int = 42
) -> SplitLists:
    r"""Convert folder-per-modality structure to SplitLists format.

    This function implements the "folder-per-modality" input mode where each
    imaging modality is stored in a separate folder. It matches files by
    subject ID across modalities and labels, then splits subjects into folds
    for cross-validation.

    Parameters
    ----------
    image_folders : dict[str, Path | str]
        Dictionary mapping modality names to folder paths.
        Keys become identifiers in 'image_{modality}' format.

        Example:
            {'dwi': 'data/dwi', 'adc': 'data/adc'}
            → creates 'image_dwi' and 'image_adc' keys

    label_folders : dict[str, Path | str] or None, optional
        Dictionary mapping label class names to folder paths.
        Keys become identifiers in 'label_{class}' format.

        Example:
            {'lesion': 'data/lesion', 'edema': 'data/edema'}
            → creates 'label_lesion' and 'label_edema' keys

        If None, subjects without labels are included (for control-only datasets).

    control_folders : dict[str, Path | str] or None, optional
        Dictionary mapping control modality names to folder paths.
        Keys become identifiers in 'control_{modality}' format.
        Controls are healthy subjects without labels.

        Example:
            {'dwi': 'data/controls_dwi', 'adc': 'data/controls_adc'}
            → creates 'control_dwi' and 'control_adc' keys

    n_folds : int, default=5
        Number of folds for cross-validation split

    subject_pattern : str, default=r'(sub-\d+)'
        Regex pattern to extract subject ID from patient filenames.
        Must contain exactly one capture group.

        Examples:
            r'(sub-\d+)' → matches 'sub-001', 'sub-042'
            r'(patient_\d+)' → matches 'patient_001', 'patient_123'

    control_pattern : str, default=r'(ctr-\d+)'
        Regex pattern to extract subject ID from control filenames.
        Must contain exactly one capture group.

        Examples:
            r'(ctr-\d+)' → matches 'ctr-001', 'ctr-042'
            r'(control_\d+)' → matches 'control_001', 'control_123'

    random_seed : int, default=42
        Random seed for reproducible fold splitting

    Returns
    -------
    SplitLists
        List of folds, each containing SubjectDict entries with:
        - 'image_{modality}' keys for each image modality
        - 'label_{class}' keys for each label class (if label_folders provided)
        - 'control_{modality}' keys for control subjects (if control_folders provided)

    Raises
    ------
    ValueError
        - If folder is empty (no NIfTI files found)
        - If subject is missing modality files
        - If subject is missing label files (when labels required)
        - If no valid subjects found (no complete data)
        - If multiple files match same subject ID in same folder

    Examples
    --------
    Basic usage with two modalities and single label:

    >>> image_folders = {
    ...     'dwi': Path('data/dwi'),
    ...     'adc': Path('data/adc')
    ... }
    >>> label_folders = {'lesion': Path('data/lesion_masks')}
    >>> split_lists = folder_mode_to_split_lists(
    ...     image_folders=image_folders,
    ...     label_folders=label_folders,
    ...     n_folds=5
    ... )
    >>> split_lists[0][0]  # First subject in first fold
    {'image_adc': '/path/data/adc/sub-001_adc.nii.gz',
     'image_dwi': '/path/data/dwi/sub-001_dwi.nii.gz',
     'label_lesion': '/path/data/lesion_masks/sub-001_lesion.nii.gz'}

    Multi-class labels:

    >>> split_lists = folder_mode_to_split_lists(
    ...     image_folders={'dwi': 'data/dwi'},
    ...     label_folders={'lesion': 'data/lesion', 'edema': 'data/edema'},
    ...     n_folds=3
    ... )
    >>> split_lists[0][0]
    {'image_dwi': '/path/data/dwi/sub-001_dwi.nii.gz',
     'label_edema': '/path/data/edema/sub-001_edema.nii.gz',
     'label_lesion': '/path/data/lesion/sub-001_lesion.nii.gz'}

    With controls:

    >>> split_lists = folder_mode_to_split_lists(
    ...     image_folders={'dwi': 'data/dwi'},
    ...     label_folders={'lesion': 'data/lesion'},
    ...     control_folders={'dwi': 'data/controls_dwi'},
    ...     subject_pattern=r'(sub-\d+)',
    ...     control_pattern=r'(ctr-\d+)',
    ...     n_folds=3
    ... )
    >>> # Patients have images + labels, controls have images only
    >>> split_lists[0][0]  # Patient
    {'image_dwi': '/path/sub-001_dwi.nii.gz', 'label_lesion': '/path/sub-001_lesion.nii.gz'}
    >>> split_lists[0][1]  # Control
    {'control_dwi': '/path/ctr-001_dwi.nii.gz'}

    Notes
    -----
    - All file paths in returned SubjectDict are absolute paths as strings
    - Subjects are randomly shuffled before splitting (controlled by random_seed)
    - If n_subjects % n_folds != 0, later folds may have one fewer subject
    - Only subjects with ALL modalities AND label are included

    .. deprecated:: 3.0
        Use explicit pipeline in main.py with :func:`read_folder`,
        :func:`match_lists_to_dicts`, and :func:`shuffle_and_split_subjects`.
    """
    warnings.warn(
        "folder_mode_to_split_lists() is deprecated and will be removed in v3.0. "
        "The staged pipeline is now explicit in main.py using read_folder(), "
        "match_lists_to_dicts(), and shuffle_and_split_subjects().",
        DeprecationWarning,
        stacklevel=2
    )
    # Validate current limitations (multi-class/multi-modal not yet implemented)
    if label_folders is not None and len(label_folders) > 1:
        raise NotImplementedError(
            f"Multi-class label segmentation not yet implemented. "
            f"Received {len(label_folders)} label classes: {list(label_folders.keys())}"
        )

    if control_folders is not None and len(control_folders) > 1:
        raise NotImplementedError(
            f"Multi-modal controls not yet implemented. "
            f"Received {len(control_folders)} control modalities: {list(control_folders.keys())}"
        )

    # Step 1: List NIfTI files from folders (Stage 0→1)
    image_modalities = list_nifti_from_folders(image_folders, subject_pattern)
    _validate_nonempty_modality_mappings(image_modalities, image_folders, subject_pattern)

    # List label files (if provided)
    label_classes = None
    if label_folders is not None:
        label_classes = list_nifti_from_folders(label_folders, subject_pattern)
        _validate_nonempty_modality_mappings(label_classes, label_folders, subject_pattern)

    # List control files (if provided)
    control_modalities = None
    if control_folders is not None:
        control_modalities = list_nifti_from_folders(control_folders, control_pattern)
        _validate_nonempty_modality_mappings(control_modalities, control_folders, control_pattern)

    # Step 2: Match modalities by subject ID (Stage 1→2)
    try:
        subject_dicts, control_dicts = match_modalities_by_subject(
            image_modalities=image_modalities,
            label_classes=label_classes,
            control_modalities=control_modalities,
            require_all=True
        )
    except ValueError as e:
        # Add pattern hint to error message if no valid subjects
        error_msg = str(e)
        if 'No valid subjects found' in error_msg:
            error_msg += f"\nCheck that filenames match patterns: subject='{subject_pattern}', control='{control_pattern}'"
        raise ValueError(error_msg) from e

    # Step 3: Merge subjects and controls into single list for shuffling
    # Controls must be merged BEFORE shuffle to maintain backward compatibility
    subject_list = subject_dicts + control_dicts

    # Step 4: Shuffle and split into folds (Stage 2→3)
    split_lists = shuffle_and_split_subjects(
        subject_dicts=subject_list,
        n_folds=n_folds,
        shuffle=True,
        random_seed=random_seed
    )

    return split_lists


def adapt_transforms_for_multimodal(transform_dict: dict, split_lists: SplitLists) -> dict:
    """
    Adapt transform dictionary for multi-modal data.

    This function modifies transform dictionaries to work with multi-modal data by:
    1. Detecting all image modalities and label classes from the first subject
    2. Replacing 'image' key with list of image_* keys in early transforms
    3. Replacing 'label' key with list of label_* keys in early transforms
    4. Expanding 'modality_intensity' transforms to per-modality versions (if present)
    5. Inserting expanded transforms and ConcatItemsd in correct order

    If single modality is detected, returns unchanged (backward compatible).

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary with structure like:
        {'first_transform': [...], 'modality_intensity': [...], 'monai_transform': [...], ...}
    split_lists : SplitLists
        Cross-validation fold splits containing SubjectDict entries

    Returns
    -------
    dict
        Modified transform dictionary with multi-modal support

    Examples
    --------
    Single modality with unnamed keys (unchanged):
    >>> split_lists = [[{'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}]]
    >>> adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)
    >>> # Returns transform_dict unchanged

    Multi-modal images with named label:
    >>> split_lists = [[{
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label_stroke': '/path/stroke.nii.gz'
    ... }]]
    >>> adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)
    >>> # Replaces 'image' with ['image_adc', 'image_dwi'] and adds ConcatItemsd
    >>> # Replaces 'label' with ['label_stroke']
    >>> # Expands modality_intensity transforms to per-modality versions

    Notes
    -----
    - Image keys are sorted alphabetically for consistent ordering
    - Label keys are sorted alphabetically for consistent ordering
    - If 'modality_intensity' section exists, it's expanded via expand_per_modality_transforms
    - Expanded modality transforms are inserted BEFORE ConcatItemsd
    - ConcatItemsd is inserted after modality transforms in first_transform
    - Output from ConcatItemsd is named 'image' (standard key)
    - Single label classes are not concatenated (just renamed)
    - Multi-label concatenation is not yet implemented (raises NotImplementedError)
    - Control keys are not included in concatenation
    """
    # Step 1: Get first subject to detect keys
    if not split_lists or not split_lists[0]:
        # Empty split_lists, return unchanged
        return transform_dict

    first_subject = split_lists[0][0]
    image_keys = get_category_keys(first_subject, 'image')
    label_keys = get_category_keys(first_subject, 'label')

    # Step 2: Check if adaptation is needed
    # Adapt if: (1) multi-modal images, OR (2) named labels, OR (3) both
    has_multi_modal = len(image_keys) > 1 or (len(image_keys) == 1 and image_keys[0] != 'image')
    has_named_labels = len(label_keys) > 0 and (len(label_keys) > 1 or label_keys[0] != 'label')

    if not has_multi_modal and not has_named_labels:
        # Standard single modality with 'image' and 'label' keys - no adaptation needed
        return transform_dict

    # Step 3: Sort keys for consistent ordering
    image_keys_sorted = sorted(image_keys)
    label_keys_sorted = sorted(label_keys) if label_keys else []

    # Step 4: Deep copy to avoid modifying original
    adapted_dict = deepcopy(transform_dict)

    # Step 4b: Expand modality_intensity transforms if present
    if 'modality_intensity' in adapted_dict and len(image_keys) > 1:
        adapted_dict = expand_per_modality_transforms(adapted_dict, image_keys_sorted)

    # Step 5: Find insertion point and replace pattern keys in early transforms
    # Only LoadImaged and EnsureChannelFirstd need actual modality/label keys (they load raw files)
    # After ConcatItemsd (for images), all other transforms use 'image' (the concatenated result)
    insertion_index = None
    if 'first_transform' in adapted_dict:
        first_transform = adapted_dict['first_transform']

        # Find insertion point (after EnsureChannelFirstd, or after LoadImaged if not found)
        transforms_to_update = ['LoadImaged', 'EnsureChannelFirstd']

        for i, transform_dict_item in enumerate(first_transform):
            transform_name = list(transform_dict_item.keys())[0]
            if transform_name in transforms_to_update:
                insertion_index = i
                # Replace pattern keys ('image', 'label') with actual keys
                params = transform_dict_item[transform_name]
                if 'keys' in params:
                    new_keys = []
                    for key in params['keys']:
                        if key == 'image' and image_keys_sorted:
                            # Expand 'image' pattern to actual image modality keys
                            new_keys.extend(image_keys_sorted)
                        elif key == 'label' and label_keys_sorted:
                            # Expand 'label' pattern to actual label class keys
                            new_keys.extend(label_keys_sorted)
                        else:
                            # Keep specific keys as-is (e.g., 'image_dwi', 'label_stroke')
                            new_keys.append(key)
                    params['keys'] = new_keys

    # Step 6: Insert expanded modality transforms and ConcatItemsd
    if 'first_transform' in adapted_dict and len(image_keys) > 1 and insertion_index is not None:
        current_insert_index = insertion_index + 1

        # Insert expanded modality_intensity transforms (if any)
        if 'expanded_modality_intensity' in adapted_dict:
            expanded_transforms = adapted_dict.pop('expanded_modality_intensity')
            for transform in expanded_transforms:
                adapted_dict['first_transform'].insert(current_insert_index, transform)
                current_insert_index += 1

        # Insert ConcatItemsd after modality transforms
        concat_transform = {
            'ConcatItemsd': {
                'keys': image_keys_sorted,
                'name': 'image',  # Output key
                'dim': 0  # Concatenate along channel dimension
            }
        }
        adapted_dict['first_transform'].insert(current_insert_index, concat_transform)
        current_insert_index += 1

        # Update insertion_index for subsequent inserts
        insertion_index = current_insert_index - 1

    # Step 7: For single named label, create alias 'label' → 'label_xxx'
    # This allows subsequent transforms to use generic 'label' key
    # (Similar to how ConcatItemsd creates 'image' for multi-modal)
    if ('first_transform' in adapted_dict and len(label_keys_sorted) == 1
        and label_keys_sorted[0] != 'label' and insertion_index is not None):
        # Single named label (e.g., 'label_stroke') - create 'label' alias
        copy_label_transform = {
            'CopyItemsd': {
                'keys': label_keys_sorted[0],  # Source: 'label_stroke'
                'times': 1,
                'names': 'label',  # Destination: 'label'
                'allow_missing_keys': False
            }
        }
        # Insert after ConcatItemsd
        adapted_dict['first_transform'].insert(insertion_index + 1, copy_label_transform)

    return adapted_dict


def extract_model_config(split_lists: SplitLists) -> dict:
    """Extract model configuration from split_lists structure.

    Automatically detects the number of input channels (image modalities) and
    output channels (label classes) from the split_lists data structure.

    Parameters
    ----------
    split_lists : SplitLists
        List of cross-validation folds containing subject dictionaries.

    Returns
    -------
    dict
        Configuration dictionary with keys:
        - 'in_channels': int - Number of input image channels (modalities)
        - 'out_channels': int - Number of output label channels (classes)

    Examples
    --------
    Single modality, single class:
    >>> split_lists = [[{'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}]]
    >>> extract_model_config(split_lists)
    {'in_channels': 1, 'out_channels': 1}

    Multi-modal (DWI + ADC), single class:
    >>> split_lists = [[{
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }]]
    >>> extract_model_config(split_lists)
    {'in_channels': 2, 'out_channels': 1}

    Single modality, multi-class:
    >>> split_lists = [[{
    ...     'image': '/path/img.nii.gz',
    ...     'label_class0': '/path/mask0.nii.gz',
    ...     'label_class1': '/path/mask1.nii.gz'
    ... }]]
    >>> extract_model_config(split_lists)
    {'in_channels': 1, 'out_channels': 2}

    Notes
    -----
    - Returns default config (1 input, 1 output) for empty split_lists
    - Backward compatible with single-modality workflows
    - Uses first subject from first fold for detection
    """
    # Default configuration for empty split_lists
    default_config = {'in_channels': 1, 'out_channels': 1}

    # Check for empty split_lists or empty first fold
    if not split_lists or not split_lists[0]:
        return default_config

    # Extract first subject to analyze keys
    first_subject = split_lists[0][0]

    # Get image and label keys using existing utility function
    image_keys = get_category_keys(first_subject, 'image')
    label_keys = get_category_keys(first_subject, 'label')

    # Calculate channel counts
    in_channels = len(image_keys) if image_keys else 1
    out_channels = len(label_keys) if label_keys else 1

    return {
        'in_channels': in_channels,
        'out_channels': out_channels
    }


def adapt_transforms_for_resolution(
    transform_dict: dict,
    base_resolution: int = 2,
    target_resolution: int = 1,
) -> dict:
    """Scale voxel-based parameters for different image resolutions.

    Elastic deformation parameters (sigma, magnitude, translation) are specified
    in voxels. When changing resolution, these need to be scaled to maintain
    equivalent physical deformation.

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary to adapt.
    base_resolution : int
        Resolution (in mm) the base parameters were designed for. Default 2mm.
    target_resolution : int
        Target resolution (in mm) to scale parameters for.

    Returns
    -------
    dict
        New transform dict with scaled parameters. Original is not modified.

    Notes
    -----
    Scaling formula: new_value = base_value * (target_resolution / base_resolution)

    For example, at 1mm resolution with 2mm base:
    - sigma_range (3, 15) -> (1.5, 7.5)
    - magnitude_range (3, 10) -> (1.5, 5)
    - translate_range (0.5, 3) -> (0.25, 1.5)

    Only affects Rand3DElasticd parameters. Other transforms use scale-invariant
    parameters (fractions, angles, etc.).

    See Also
    --------
    transform_dicts_references.md : Literature references for augmentation parameters
    """
    if base_resolution == target_resolution:
        return transform_dict  # No scaling needed

    adapted = deepcopy(transform_dict)
    scale_factor = target_resolution / base_resolution

    # Parameters to scale in Rand3DElasticd
    voxel_params = ['sigma_range', 'magnitude_range', 'translate_range']

    # Iterate through all transform lists
    for list_name in adapted:
        if not isinstance(adapted[list_name], list):
            continue

        for transform_entry in adapted[list_name]:
            if not isinstance(transform_entry, dict):
                continue

            if 'Rand3DElasticd' in transform_entry:
                params = transform_entry['Rand3DElasticd']
                for param_name in voxel_params:
                    if param_name in params:
                        value = params[param_name]
                        if isinstance(value, (list, tuple)):
                            # Scale each element in tuple/list
                            scaled = tuple(v * scale_factor for v in value)
                            params[param_name] = scaled
                        elif isinstance(value, (int, float)):
                            params[param_name] = value * scale_factor

    return adapted


# Modality-specific parameter adjustments based on MRI physics literature
# See transform_dicts_references.md for citations
MODALITY_PARAMS = {
    'dwi': {
        # DWI (TRACE): Direct acquisition, full noise/artifact effects
        'RandRicianNoised': {'std': 0.03},
        'RandBiasFieldd': {'coeff_range': (0.0, 0.05)},
        'RandKSpaceSpikeNoised': {'prob': 0.1},
        'RandGibbsNoised': {'alpha': (0.5, 0.7)},
    },
    'adc': {
        # ADC: Calculated map, noise propagated (heteroscedastic), bias partially cancelled
        'RandRicianNoised': {'std': 0.02},  # Propagated, not direct
        'RandBiasFieldd': {'coeff_range': (0.0, 0.02)},  # Residual only
        'RandKSpaceSpikeNoised': {'prob': 0.05},  # Indirect effect
        'RandGibbsNoised': {'alpha': (0.4, 0.6)},  # Propagates through calculation
    },
    # Default fallback for unknown modalities
    '_default': {
        'RandRicianNoised': {'std': 0.025},
        'RandBiasFieldd': {'coeff_range': (0.0, 0.03)},
        'RandKSpaceSpikeNoised': {'prob': 0.08},
        'RandGibbsNoised': {'alpha': (0.45, 0.65)},
    }
}


def expand_per_modality_transforms(
    transform_dict: dict,
    image_keys: List[str],
) -> dict:
    """Expand modality_intensity transforms to per-modality versions.

    Transforms in the 'modality_intensity' section are designed to be applied
    independently to each image modality BEFORE concatenation. This function
    expands them into separate transforms for each modality with appropriate
    modality-specific parameters.

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary containing 'modality_intensity' section.
    image_keys : List[str]
        List of image keys, e.g., ['image_dwi', 'image_adc'].
        The modality is extracted from the suffix (after 'image_').

    Returns
    -------
    dict
        New transform dict with expanded per-modality transforms.
        The 'modality_intensity' section is replaced with 'expanded_modality_intensity'.

    Notes
    -----
    For single modality (len(image_keys) == 1), returns the dict with minimal
    changes - transforms use the single image key directly.

    Modality-specific parameters are defined in MODALITY_PARAMS based on
    MRI physics literature. See transform_dicts_references.md.

    Examples
    --------
    Input:
    >>> transform_dict = {
    ...     'modality_intensity': [
    ...         {'RandHistogramShiftd': {'keys': ['image'], 'prob': 0.1}},
    ...     ]
    ... }
    >>> expand_per_modality_transforms(transform_dict, ['image_dwi', 'image_adc'])

    Output (simplified):
    >>> {
    ...     'expanded_modality_intensity': [
    ...         {'RandHistogramShiftd': {'keys': ['image_dwi'], 'prob': 0.1}},
    ...         {'RandHistogramShiftd': {'keys': ['image_adc'], 'prob': 0.1}},
    ...     ]
    ... }
    """
    if 'modality_intensity' not in transform_dict:
        # No modality_intensity section, nothing to expand
        return transform_dict

    adapted = deepcopy(transform_dict)
    modality_transforms = adapted.pop('modality_intensity')

    # Extract modality names from keys (e.g., 'image_dwi' -> 'dwi')
    modalities = []
    for key in image_keys:
        if key.startswith('image_'):
            modalities.append(key.replace('image_', ''))
        elif key == 'image':
            modalities.append('_default')
        else:
            modalities.append(key)

    expanded_transforms = []

    for transform_entry in modality_transforms:
        if not isinstance(transform_entry, dict):
            continue

        transform_name = list(transform_entry.keys())[0]
        base_params = transform_entry[transform_name]

        # Expand to each modality
        for i, image_key in enumerate(image_keys):
            modality = modalities[i] if i < len(modalities) else '_default'

            # Create modality-specific params
            new_params = deepcopy(base_params)

            # Replace 'image' key with specific modality key
            if 'keys' in new_params:
                new_keys = []
                for k in new_params['keys']:
                    if k == 'image':
                        new_keys.append(image_key)
                    else:
                        new_keys.append(k)
                new_params['keys'] = new_keys

            # Apply modality-specific parameter overrides
            modality_overrides = MODALITY_PARAMS.get(
                modality, MODALITY_PARAMS['_default']
            )
            if transform_name in modality_overrides:
                new_params.update(modality_overrides[transform_name])

            expanded_transforms.append({transform_name: new_params})

    # Store expanded transforms in new section
    adapted['expanded_modality_intensity'] = expanded_transforms

    return adapted

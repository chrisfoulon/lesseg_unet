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
import numpy as np
from pathlib import Path
from typing import TypeAlias
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


def folder_mode_to_split_lists(
    image_folders: dict[str, Path | str],
    label_folder: Path | str,
    n_folds: int = 5,
    subject_pattern: str = r'(sub-\d+)',
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

    label_folder : Path | str
        Path to folder containing label files

    n_folds : int, default=5
        Number of folds for cross-validation split

    subject_pattern : str, default=r'(sub-\d+)'
        Regex pattern to extract subject ID from filenames.
        Must contain exactly one capture group.

        Examples:
            r'(sub-\d+)' → matches 'sub-001', 'sub-042'
            r'(patient_\d+)' → matches 'patient_001', 'patient_123'

    random_seed : int, default=42
        Random seed for reproducible fold splitting

    Returns
    -------
    SplitLists
        List of folds, each containing SubjectDict entries with
        'image_{modality}' and 'label' keys

    Raises
    ------
    ValueError
        - If folder is empty (no NIfTI files found)
        - If subject is missing modality files
        - If subject is missing label file
        - If no valid subjects found (no complete data)
        - If multiple files match same subject ID in same folder

    Examples
    --------
    Basic usage with two modalities:

    >>> image_folders = {
    ...     'dwi': Path('data/dwi'),
    ...     'adc': Path('data/adc')
    ... }
    >>> label_folder = Path('data/lesion_masks')
    >>> split_lists = folder_mode_to_split_lists(
    ...     image_folders=image_folders,
    ...     label_folder=label_folder,
    ...     n_folds=5
    ... )
    >>> split_lists[0][0]  # First subject in first fold
    {'image_dwi': '/path/data/dwi/sub-001_dwi.nii.gz',
     'image_adc': '/path/data/adc/sub-001_adc.nii.gz',
     'label': '/path/data/lesion_masks/sub-001_lesion.nii.gz'}

    Custom subject pattern:

    >>> split_lists = folder_mode_to_split_lists(
    ...     image_folders={'t1': 'data/t1'},
    ...     label_folder='data/masks',
    ...     subject_pattern=r'(patient_\d+)',  # Match 'patient_001'
    ...     n_folds=3
    ... )

    Notes
    -----
    - All file paths in returned SubjectDict are absolute paths as strings
    - Subjects are randomly shuffled before splitting (controlled by random_seed)
    - If n_subjects % n_folds != 0, later folds may have one fewer subject
    - Only subjects with ALL modalities AND label are included
    """
    # Convert paths to Path objects
    image_folders = {
        modality: Path(folder) for modality, folder in image_folders.items()
    }
    label_folder = Path(label_folder)

    # Step 1: Build subject-to-file mappings for each modality
    modality_files = {}
    for modality, folder in image_folders.items():
        mapping = _build_subject_to_file_mapping(folder, subject_pattern)
        if not mapping:
            raise ValueError(
                f"No NIfTI files found in {folder} matching pattern '{subject_pattern}'"
            )
        modality_files[modality] = mapping

    # Step 2: Build subject-to-file mapping for labels
    label_files = _build_subject_to_file_mapping(label_folder, subject_pattern)
    if not label_files:
        raise ValueError(
            f"No NIfTI files found in {label_folder} matching pattern '{subject_pattern}'"
        )

    # Step 3: Find subjects with complete data (all modalities + label)
    all_modalities = set(image_folders.keys())
    complete_subjects = []
    incomplete_subjects = []

    for subject_id in label_files.keys():
        # Check if subject has all modalities
        missing_modalities = []
        for modality in all_modalities:
            if subject_id not in modality_files[modality]:
                missing_modalities.append(modality)

        if missing_modalities:
            incomplete_subjects.append((subject_id, missing_modalities))
        else:
            complete_subjects.append(subject_id)

    # Also check for subjects with images but no label
    subjects_with_images = set()
    for modality_mapping in modality_files.values():
        subjects_with_images.update(modality_mapping.keys())

    subjects_missing_labels = subjects_with_images - set(label_files.keys())

    # Check if no valid subjects found FIRST
    if not complete_subjects:
        raise ValueError(
            f"No valid subjects found with all modalities and labels.\n"
            f"Modalities required: {list(all_modalities)}\n"
            f"Subjects with labels: {len(label_files)}\n"
            f"Subjects with images: {len(subjects_with_images)}\n"
            f"Subjects with incomplete data: {len(incomplete_subjects) + len(subjects_missing_labels)}\n"
            f"Check that filenames match pattern '{subject_pattern}'"
        )

    # Warn about incomplete subjects (if any)
    if incomplete_subjects or subjects_missing_labels:
        error_parts = ["Some subjects have incomplete data:"]

        if incomplete_subjects:
            error_parts.append("\nSubjects missing modalities:")
            for subj_id, missing_mods in incomplete_subjects[:10]:  # Show first 10
                error_parts.append(f"  - {subj_id}: missing {missing_mods}")
            if len(incomplete_subjects) > 10:
                error_parts.append(f"  ... and {len(incomplete_subjects) - 10} more")

        if subjects_missing_labels:
            error_parts.append("\nSubjects missing labels:")
            missing_list = sorted(list(subjects_missing_labels))[:10]
            for subj_id in missing_list:
                error_parts.append(f"  - {subj_id}")
            if len(subjects_missing_labels) > 10:
                error_parts.append(f"  ... and {len(subjects_missing_labels) - 10} more")

        raise ValueError('\n'.join(error_parts))

    # Step 4: Build SubjectDict entries for complete subjects
    subject_list = []
    for subject_id in complete_subjects:
        subject_dict = {}

        # Add all image modalities
        for modality in sorted(all_modalities):  # Sort for consistent ordering
            key = f'image_{modality}'
            subject_dict[key] = modality_files[modality][subject_id]

        # Add label
        subject_dict['label'] = label_files[subject_id]

        subject_list.append(subject_dict)

    # Step 5: Shuffle and split into folds
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(subject_list))
    shuffled_subjects = [subject_list[idx] for idx in shuffled_indices]

    # Use numpy's array_split for even distribution (same as existing codebase)
    split_arrays = np.array_split(np.array(shuffled_subjects, dtype=object), n_folds)
    split_lists = [list(fold) for fold in split_arrays]

    return split_lists


def adapt_transforms_for_multimodal(transform_dict: dict, split_lists: SplitLists) -> dict:
    """
    Adapt transform dictionary for multi-modal data.

    This function modifies transform dictionaries to work with multi-modal data by:
    1. Detecting all image modalities from the first subject
    2. Replacing single 'image' key with list of image_* keys in all transforms
    3. Inserting ConcatItemsd after LoadImaged to merge modalities into single tensor

    If single modality is detected, returns unchanged (backward compatible).

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary with structure like:
        {'first_transform': [...], 'monai_transform': [...], ...}
    split_lists : SplitLists
        Cross-validation fold splits containing SubjectDict entries

    Returns
    -------
    dict
        Modified transform dictionary with multi-modal support

    Examples
    --------
    Single modality (unchanged):
    >>> split_lists = [[{'image': '/path/img.nii.gz', 'label': '/path/mask.nii.gz'}]]
    >>> adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)
    >>> # Returns transform_dict unchanged

    Multi-modal (adapted):
    >>> split_lists = [[{
    ...     'image_dwi': '/path/dwi.nii.gz',
    ...     'image_adc': '/path/adc.nii.gz',
    ...     'label': '/path/mask.nii.gz'
    ... }]]
    >>> adapted = adapt_transforms_for_multimodal(transform_dict, split_lists)
    >>> # Replaces 'image' with ['image_adc', 'image_dwi'] and adds ConcatItemsd

    Notes
    -----
    - Image keys are sorted alphabetically for consistent ordering
    - ConcatItemsd is inserted after LoadImaged in first_transform
    - Output from ConcatItemsd is named 'image' (standard key)
    - Label keys are not modified
    - Control keys are not included in concatenation
    """
    # Step 1: Get first subject to detect keys
    if not split_lists or not split_lists[0]:
        # Empty split_lists, return unchanged
        return transform_dict

    first_subject = split_lists[0][0]
    image_keys = get_category_keys(first_subject, 'image')

    # Step 2: Check if adaptation is needed
    if len(image_keys) == 1 and image_keys[0] == 'image':
        # Single modality with standard key - no adaptation needed
        return transform_dict

    # Step 3: Sort image keys for consistent ordering
    image_keys_sorted = sorted(image_keys)

    # Step 4: Deep copy to avoid modifying original
    adapted_dict = deepcopy(transform_dict)

    # Step 5: Find insertion point and replace 'image' only in early transforms
    # Only LoadImaged and EnsureChannelFirstd need image_* keys (they load raw files)
    # After ConcatItemsd, all other transforms use 'image' (the concatenated result)
    insertion_index = None
    if 'first_transform' in adapted_dict:
        first_transform = adapted_dict['first_transform']

        # Find insertion point (after EnsureChannelFirstd, or after LoadImaged if not found)
        transforms_to_update = ['LoadImaged', 'EnsureChannelFirstd']

        for i, transform_dict_item in enumerate(first_transform):
            transform_name = list(transform_dict_item.keys())[0]
            if transform_name in transforms_to_update:
                insertion_index = i
                # Replace 'image' with image_* keys in this transform
                params = transform_dict_item[transform_name]
                if 'keys' in params and 'image' in params['keys']:
                    new_keys = []
                    for key in params['keys']:
                        if key == 'image':
                            new_keys.extend(image_keys_sorted)
                        else:
                            new_keys.append(key)
                    params['keys'] = new_keys

    # Step 6: Insert ConcatItemsd after the last early transform
    if 'first_transform' in adapted_dict and len(image_keys) > 1 and insertion_index is not None:
        concat_transform = {
            'ConcatItemsd': {
                'keys': image_keys_sorted,
                'name': 'image',  # Output key
                'dim': 0  # Concatenate along channel dimension
            }
        }
        # Insert after the last transform that was updated
        adapted_dict['first_transform'].insert(insertion_index + 1, concat_transform)

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

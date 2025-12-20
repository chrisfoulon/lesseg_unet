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

from typing import TypeAlias

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

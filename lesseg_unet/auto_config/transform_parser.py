"""Parser for transform dictionaries to detect training mode and tunable parameters."""

import logging
from typing import Literal, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Parsed transform dictionary configuration.

    Attributes
    ----------
    mode : {'patching', 'full-image'}
        Training mode detected from transform dict.
    patch_size : tuple[int, int, int] | None
        Fixed patch size if specified by user, None if tunable.
    num_samples : int | None
        Number of samples per image for patching mode.
    has_wildcard : bool
        Whether transform dict has None wildcards for auto-tuning.
    """

    mode: Literal['patching', 'full-image']
    patch_size: Optional[tuple[int, int, int]]
    num_samples: Optional[int]
    has_wildcard: bool


def parse_transform_dict(transform_dict: dict) -> TransformConfig:
    """Parse transform dictionary to extract training mode and constraints.

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary with structure:
        {
            'first_transform': [...],
            'patches': [...],  # RandCropByPosNegLabeld if patching
            'last_transform': [...]
        }

    Returns
    -------
    TransformConfig
        Parsed configuration with mode, patch_size, and tunable flags.

    Examples
    --------
    >>> # Patching mode with wildcard
    >>> transform_dict = {
    ...     'patches': [{
    ...         'RandCropByPosNegLabeld': {
    ...             'spatial_size': None,  # Wildcard!
    ...             'num_samples': 4
    ...         }
    ...     }]
    ... }
    >>> config = parse_transform_dict(transform_dict)
    >>> config.mode
    'patching'
    >>> config.has_wildcard
    True

    >>> # Patching mode with fixed size
    >>> transform_dict = {
    ...     'patches': [{
    ...         'RandCropByPosNegLabeld': {
    ...             'spatial_size': [96, 96, 96],  # Fixed
    ...             'num_samples': 4
    ...         }
    ...     }]
    ... }
    >>> config = parse_transform_dict(transform_dict)
    >>> config.patch_size
    (96, 96, 96)
    >>> config.has_wildcard
    False

    >>> # Full-image mode (no patching)
    >>> transform_dict = {
    ...     'first_transform': [...],
    ...     'patches': [],  # Empty!
    ...     'last_transform': [...]
    ... }
    >>> config = parse_transform_dict(transform_dict)
    >>> config.mode
    'full-image'
    """
    if not isinstance(transform_dict, dict):
        raise ValueError(f"transform_dict must be dict, got {type(transform_dict)}")

    # Extract patches section
    patches = transform_dict.get('patches', [])

    # Detect mode: patching or full-image
    if not patches or len(patches) == 0:
        # No patching transforms → full-image mode
        logger.debug("Detected full-image training mode (no patching transforms)")
        return TransformConfig(
            mode='full-image',
            patch_size=None,
            num_samples=None,
            has_wildcard=False
        )

    # Patching mode: look for RandCropByPosNegLabeld or similar
    patch_transform = None
    for transform_entry in patches:
        if isinstance(transform_entry, dict):
            # Check for common patching transforms
            if 'RandCropByPosNegLabeld' in transform_entry:
                patch_transform = transform_entry['RandCropByPosNegLabeld']
                break
            elif 'RandSpatialCropd' in transform_entry:
                patch_transform = transform_entry['RandSpatialCropd']
                break
            elif 'RandCropByLabelClassesd' in transform_entry:
                patch_transform = transform_entry['RandCropByLabelClassesd']
                break

    if patch_transform is None:
        # Patches list exists but no recognized patching transform
        logger.warning("Patches list exists but no recognized patching transform found")
        return TransformConfig(
            mode='full-image',
            patch_size=None,
            num_samples=None,
            has_wildcard=False
        )

    # Extract spatial_size (patch size)
    spatial_size = patch_transform.get('spatial_size')
    num_samples = patch_transform.get('num_samples', 1)

    # Check if spatial_size is a wildcard (None)
    if spatial_size is None:
        logger.debug("Detected wildcard (None) for spatial_size - auto-config will tune")
        return TransformConfig(
            mode='patching',
            patch_size=None,  # Tunable
            num_samples=num_samples,
            has_wildcard=True
        )

    # spatial_size is fixed by user
    if isinstance(spatial_size, (list, tuple)):
        patch_size_tuple = tuple(spatial_size)
        logger.debug(f"Detected fixed patch_size: {patch_size_tuple}")
        return TransformConfig(
            mode='patching',
            patch_size=patch_size_tuple,
            num_samples=num_samples,
            has_wildcard=False
        )

    raise ValueError(
        f"Invalid spatial_size in transform_dict: {spatial_size}. "
        f"Must be None (wildcard) or list/tuple of 3 ints."
    )


def fill_wildcards(transform_dict: dict, patch_size: tuple[int, int, int]) -> dict:
    """Fill None wildcards in transform dict with tuned values.

    Parameters
    ----------
    transform_dict : dict
        Transform dictionary potentially containing None wildcards.
    patch_size : tuple[int, int, int]
        Tuned patch size to fill in.

    Returns
    -------
    dict
        Updated transform dictionary with wildcards filled.

    Notes
    -----
    Modifies transform_dict in place and returns it for convenience.

    Examples
    --------
    >>> transform_dict = {
    ...     'patches': [{
    ...         'RandCropByPosNegLabeld': {
    ...             'spatial_size': None,
    ...             'num_samples': 4
    ...         }
    ...     }]
    ... }
    >>> updated = fill_wildcards(transform_dict, (96, 96, 96))
    >>> updated['patches'][0]['RandCropByPosNegLabeld']['spatial_size']
    [96, 96, 96]
    """
    patches = transform_dict.get('patches', [])

    if not patches:
        # No patching, nothing to fill
        return transform_dict

    # Find patching transform and fill spatial_size if it's None
    for transform_entry in patches:
        if isinstance(transform_entry, dict):
            for transform_name in ['RandCropByPosNegLabeld', 'RandSpatialCropd', 'RandCropByLabelClassesd']:
                if transform_name in transform_entry:
                    patch_transform = transform_entry[transform_name]
                    if patch_transform.get('spatial_size') is None:
                        patch_transform['spatial_size'] = list(patch_size)
                        logger.info(f"Filled wildcard spatial_size with {list(patch_size)}")

    return transform_dict

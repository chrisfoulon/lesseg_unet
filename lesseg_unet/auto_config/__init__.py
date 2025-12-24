"""Auto-configuration for hardware-aware training."""

from lesseg_unet.auto_config.configurator import (
    AutoConfigurator,
    AutoConfigResult,
    DatasetProfile
)
from lesseg_unet.auto_config.config_manager import TrainingConfig
from lesseg_unet.auto_config.transform_parser import (
    parse_transform_dict,
    fill_wildcards,
    TransformConfig
)

__all__ = [
    'AutoConfigurator',
    'AutoConfigResult',
    'DatasetProfile',
    'TrainingConfig',
    'parse_transform_dict',
    'fill_wildcards',
    'TransformConfig'
]

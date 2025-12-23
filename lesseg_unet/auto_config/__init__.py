"""Auto-configuration for hardware-aware training."""

from lesseg_unet.auto_config.configurator import (
    AutoConfigurator,
    AutoConfigResult,
    DatasetProfile
)
from lesseg_unet.auto_config.config_manager import TrainingConfig

__all__ = [
    'AutoConfigurator',
    'AutoConfigResult',
    'DatasetProfile',
    'TrainingConfig'
]

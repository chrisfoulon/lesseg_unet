"""Configuration management for training runs.

Handles saving, loading, and serialization of complete training configurations
including hardware profiles, auto-config results, and metadata.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import yaml

from lesseg_unet.auto_config.configurator import AutoConfigResult


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration with metadata.

    Attributes
    ----------
    batch_size : int
        Batch size for training.
    patch_size : tuple[int, int, int]
        Patch size for training (H, W, D).
    num_workers : int
        Number of dataloader workers.
    network_depth : int
        Network depth (4 or 5).
    feature_size : int
        Base feature size for network.
    use_amp : bool
        Whether to use automatic mixed precision.
    num_gpus : int
        Number of GPUs to use.
    vram_safety_margin : float
        VRAM safety margin (0.0-1.0).
    learning_rate : Optional[float]
        Learning rate (if specified).
    num_epochs : Optional[int]
        Number of training epochs (if specified).
    model_type : Optional[str]
        Model type ('swinunetr' or 'unet').
    hardware_profile : Optional[dict]
        Hardware profile snapshot.
    command : Optional[str]
        Command that was run to generate this config.
    timestamp : Optional[str]
        ISO format timestamp when config was created.
    reasoning : dict[str, str]
        Auto-config reasoning (if applicable).
    memory_estimate : dict[str, float]
        Memory estimate breakdown (if applicable).
    user_overrides : dict[str, any]
        Any user-specified overrides applied.
    """

    # Required training parameters
    batch_size: int
    val_batch_size: int
    patch_size: tuple[int, int, int]
    num_workers: int
    network_depth: int
    feature_size: int
    use_amp: bool
    num_gpus: int
    vram_safety_margin: float

    # Optional training parameters
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    model_type: Optional[str] = None

    # Metadata
    hardware_profile: Optional[dict] = None
    command: Optional[str] = None
    timestamp: Optional[str] = None
    reasoning: dict[str, str] = field(default_factory=dict)
    memory_estimate: dict[str, float] = field(default_factory=dict)
    user_overrides: dict = field(default_factory=dict)

    @classmethod
    def from_auto_config(
        cls,
        auto_config: AutoConfigResult,
        hardware_profile: Optional[dict] = None,
        command: Optional[str] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        model_type: Optional[str] = None,
        user_overrides: Optional[dict] = None
    ) -> 'TrainingConfig':
        """Create TrainingConfig from AutoConfigResult.

        Parameters
        ----------
        auto_config : AutoConfigResult
            Auto-configuration result.
        hardware_profile : dict, optional
            Hardware profile to store with config.
        command : str, optional
            Command that was run.
        learning_rate : float, optional
            Learning rate.
        num_epochs : int, optional
            Number of training epochs.
        model_type : str, optional
            Model type ('swinunetr' or 'unet').
        user_overrides : dict, optional
            User-specified overrides applied.

        Returns
        -------
        TrainingConfig
            Complete training configuration.

        Examples
        --------
        >>> from lesseg_unet.auto_config import AutoConfigurator, DatasetProfile
        >>> from lesseg_unet.hardware import get_hardware_profile
        >>>
        >>> hw = get_hardware_profile()
        >>> dataset = DatasetProfile(
        ...     median_image_size=(181, 217, 181),
        ...     num_subjects=100,
        ...     in_channels=2,
        ...     out_channels=1
        ... )
        >>> configurator = AutoConfigurator(hw, dataset)
        >>> auto_config = configurator.suggest_config()
        >>>
        >>> config = TrainingConfig.from_auto_config(
        ...     auto_config,
        ...     hardware_profile=hw.to_dict(),
        ...     learning_rate=1e-4,
        ...     num_epochs=100
        ... )
        """
        return cls(
            batch_size=auto_config.batch_size,
            val_batch_size=auto_config.val_batch_size,
            patch_size=auto_config.patch_size,
            num_workers=auto_config.num_workers,
            network_depth=auto_config.network_depth,
            feature_size=auto_config.feature_size,
            use_amp=auto_config.use_amp,
            num_gpus=auto_config.num_gpus,
            vram_safety_margin=auto_config.vram_safety_margin,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            model_type=model_type,
            hardware_profile=hardware_profile,
            command=command,
            timestamp=datetime.now().isoformat(),
            reasoning=auto_config.reasoning,
            memory_estimate=auto_config.memory_estimate,
            user_overrides=user_overrides or {}
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of config.

        Examples
        --------
        >>> config = TrainingConfig(
        ...     batch_size=4,
        ...     patch_size=(96, 96, 96),
        ...     num_workers=8,
        ...     network_depth=5,
        ...     feature_size=48,
        ...     use_amp=True,
        ...     num_gpus=1,
        ...     vram_safety_margin=0.95
        ... )
        >>> d = config.to_dict()
        >>> d['batch_size']
        4
        """
        data = asdict(self)
        # Convert tuples to lists for YAML/JSON compatibility
        if 'patch_size' in data and isinstance(data['patch_size'], tuple):
            data['patch_size'] = list(data['patch_size'])

        # Also convert tuples in user_overrides
        if 'user_overrides' in data and data['user_overrides']:
            for key, value in data['user_overrides'].items():
                if isinstance(value, tuple):
                    data['user_overrides'][key] = list(value)

        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """Create from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of config.

        Returns
        -------
        TrainingConfig
            Reconstructed config.

        Examples
        --------
        >>> data = {
        ...     'batch_size': 4,
        ...     'patch_size': [96, 96, 96],
        ...     'num_workers': 8,
        ...     'network_depth': 5,
        ...     'feature_size': 48,
        ...     'use_amp': True,
        ...     'num_gpus': 1,
        ...     'vram_safety_margin': 0.95
        ... }
        >>> config = TrainingConfig.from_dict(data)
        >>> config.batch_size
        4
        """
        # Convert patch_size from list to tuple if needed
        if 'patch_size' in data and isinstance(data['patch_size'], list):
            data['patch_size'] = tuple(data['patch_size'])

        return cls(**data)

    def save(
        self,
        output_path: Path | str,
        format: Literal['yaml', 'json'] = 'yaml',
        overwrite: bool = False
    ) -> None:
        """Save config to file.

        Parameters
        ----------
        output_path : Path or str
            Output file path.
        format : {'yaml', 'json'}
            Output format. Default: 'yaml'.
        overwrite : bool
            Whether to overwrite existing file. Default: False.

        Raises
        ------
        FileExistsError
            If file exists and overwrite=False.
        ValueError
            If format is not 'yaml' or 'json'.

        Examples
        --------
        >>> config.save('output/config.yaml')
        >>> config.save('output/config.json', format='json')
        """
        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Config file already exists: {output_path}. "
                "Use overwrite=True to replace."
            )

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {output_path}")
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved config to {output_path}")
        else:
            raise ValueError(f"Invalid format: {format}. Must be 'yaml' or 'json'.")

    @classmethod
    def load(cls, config_path: Path | str) -> 'TrainingConfig':
        """Load config from file.

        Parameters
        ----------
        config_path : Path or str
            Path to config file (.yaml or .json).

        Returns
        -------
        TrainingConfig
            Loaded config.

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist.
        ValueError
            If file format is unsupported.

        Examples
        --------
        >>> config = TrainingConfig.load('output/config.yaml')
        >>> config.batch_size
        4
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        elif suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. "
                "Must be .yaml, .yml, or .json."
            )

        return cls.from_dict(data)

    def apply_to_namespace(self, args) -> None:
        """Apply config values to argparse namespace.

        Updates an argparse.Namespace object with values from this config.
        Only updates attributes that exist in the namespace.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments namespace to update.

        Examples
        --------
        >>> import argparse
        >>> args = argparse.Namespace(batch_size=1, patch_size=(64, 64, 64))
        >>> config.apply_to_namespace(args)
        >>> args.batch_size
        4
        """
        # Map config attributes to common arg names
        attr_map = {
            'batch_size': 'batch_size',
            'patch_size': 'patch_size',
            'num_workers': 'num_workers',
            'network_depth': 'network_depth',
            'feature_size': 'feature_size',
            'use_amp': 'amp',  # Common alternative name
            'num_gpus': 'num_gpus',
            'learning_rate': 'lr',  # Common alternative name
            'num_epochs': 'num_epochs',
        }

        for config_attr, arg_name in attr_map.items():
            value = getattr(self, config_attr, None)
            if value is not None and hasattr(args, arg_name):
                setattr(args, arg_name, value)
                logger.debug(f"Applied {arg_name}={value} from config")

        # Also try direct attribute names
        for attr in ['batch_size', 'patch_size', 'num_workers', 'network_depth',
                     'feature_size', 'use_amp', 'num_gpus', 'learning_rate', 'num_epochs']:
            value = getattr(self, attr, None)
            if value is not None and hasattr(args, attr):
                setattr(args, attr, value)
                logger.debug(f"Applied {attr}={value} from config")

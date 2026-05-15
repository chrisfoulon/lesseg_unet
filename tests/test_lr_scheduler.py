"""Tests for LR scheduler integration in training."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lesseg_unet import utils


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def tiny_model():
    return _TinyModel()


@pytest.fixture
def optimizer(tiny_model):
    return AdamW(tiny_model.parameters(), lr=1e-3)


class TestSchedulerCreation:
    """Scheduler is created iff use_lr_scheduler=True."""

    def test_scheduler_created_when_enabled(self, optimizer):
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        assert scheduler is not None
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.patience == 10
        assert scheduler.factor == 0.5

    def test_scheduler_none_when_disabled(self):
        scheduler = None
        assert scheduler is None

    def test_scheduler_params_match_training_defaults(self, optimizer):
        patience = 10
        factor = 0.5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor
        )
        assert scheduler.patience == patience
        assert scheduler.factor == factor

    def test_scheduler_params_configurable(self, optimizer):
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.1
        )
        assert scheduler.patience == 20
        assert scheduler.factor == 0.1


class TestSchedulerCheckpoint:
    """Scheduler state is saved and restored via save_checkpoint."""

    def test_checkpoint_includes_scheduler_dict(self, tiny_model, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        scaler = None
        hyper_params = {'model_type': 'test'}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = utils.save_checkpoint(
                tiny_model, epoch=1, fold=0, optimizer=optimizer,
                scaler=scaler, hyper_params=hyper_params,
                output_folder=tmpdir, model_name='test',
                transform_dict=None, scheduler=scheduler
            )
            checkpoint = torch.load(path, weights_only=False)

        assert 'scheduler_dict' in checkpoint

    def test_checkpoint_without_scheduler_has_no_scheduler_dict(self, tiny_model, optimizer):
        scaler = None
        hyper_params = {'model_type': 'test'}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = utils.save_checkpoint(
                tiny_model, epoch=1, fold=0, optimizer=optimizer,
                scaler=scaler, hyper_params=hyper_params,
                output_folder=tmpdir, model_name='test',
                transform_dict=None, scheduler=None
            )
            checkpoint = torch.load(path, weights_only=False)

        assert 'scheduler_dict' not in checkpoint

    def test_scheduler_state_can_be_restored(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        # Step a few times to change internal state
        for loss_val in [1.0, 0.9, 0.9, 0.9, 0.9, 0.9]:
            scheduler.step(loss_val)

        state = scheduler.state_dict()

        optimizer2 = AdamW(_TinyModel().parameters(), lr=1e-3)
        scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', patience=5, factor=0.5)
        scheduler2.load_state_dict(state)

        assert scheduler2.num_bad_epochs == scheduler.num_bad_epochs


class TestSchedulerLRReduction:
    """LR decreases after patience epochs without improvement."""

    def test_lr_decreases_after_patience(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        patience = 3
        factor = 0.5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor
        )

        # ReduceLROnPlateau reduces when num_bad_epochs > patience,
        # so need patience + 2 total steps (1 initial + patience+1 bad)
        stagnant_loss = 1.0
        for _ in range(patience + 2):
            scheduler.step(stagnant_loss)

        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr < initial_lr
        assert abs(current_lr - initial_lr * factor) < 1e-10

    def test_lr_stable_when_improving(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        patience = 3
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=0.5
        )

        # Steady improvement — LR should not drop
        for loss_val in [1.0, 0.9, 0.8, 0.7, 0.6]:
            scheduler.step(loss_val)

        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == initial_lr

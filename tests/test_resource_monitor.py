"""Unit tests for lesseg_unet.resource_monitor module."""
import csv
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lesseg_unet.resource_monitor import ResourceMonitor, _worker_count, _gpu_utilization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(tmp_path, stop_after_epochs=3, interval=0.05):
    return ResourceMonitor(
        output_path=tmp_path / 'monitor.csv',
        interval=interval,
        stop_after_epochs=stop_after_epochs,
        device=None,
    )


def _read_csv(path: Path) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# _gpu_utilization helper
# ---------------------------------------------------------------------------

class TestGpuUtilization:
    def test_returns_float_on_success(self):
        with patch('torch.cuda.utilization', return_value=42):
            assert _gpu_utilization(0) == 42.0

    def test_returns_minus_one_on_exception(self):
        with patch('torch.cuda.utilization', side_effect=RuntimeError):
            assert _gpu_utilization(0) == -1.0


# ---------------------------------------------------------------------------
# _worker_count helper
# ---------------------------------------------------------------------------

class TestWorkerCount:
    def test_returns_int(self):
        result = _worker_count()
        assert isinstance(result, int)

    def test_returns_minus_one_on_exception(self):
        with patch('psutil.Process', side_effect=Exception):
            assert _worker_count() == -1


# ---------------------------------------------------------------------------
# ResourceMonitor — CSV creation
# ---------------------------------------------------------------------------

class TestResourceMonitorCsv:
    def test_creates_csv_with_headers(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.start()
        time.sleep(0.15)
        mon.stop()
        assert (tmp_path / 'monitor.csv').exists()
        rows = _read_csv(tmp_path / 'monitor.csv')
        assert len(rows) > 0
        assert 'epoch' in rows[0]
        assert 'phase' in rows[0]
        assert 'ram_used_gb' in rows[0]
        assert 'gpu_mem_allocated_gb' in rows[0]
        assert 'worker_count' in rows[0]

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / 'fold_0' / 'monitor.csv'
        mon = ResourceMonitor(output_path=nested, interval=0.05, stop_after_epochs=2)
        mon.start()
        time.sleep(0.1)
        mon.stop()
        assert nested.exists()


# ---------------------------------------------------------------------------
# ResourceMonitor — set_epoch / set_phase
# ---------------------------------------------------------------------------

class TestResourceMonitorState:
    def test_epoch_and_phase_appear_in_csv(self, tmp_path):
        mon = _make_monitor(tmp_path, stop_after_epochs=5)
        mon.start()
        time.sleep(0.08)
        mon.set_epoch(2)
        mon.set_phase('train')
        time.sleep(0.12)
        mon.stop()
        rows = _read_csv(tmp_path / 'monitor.csv')
        epochs = {r['epoch'] for r in rows}
        phases = {r['phase'] for r in rows}
        assert '2' in epochs
        assert 'train' in phases

    def test_phase_transitions(self, tmp_path):
        mon = _make_monitor(tmp_path, stop_after_epochs=5)
        mon.start()
        mon.set_phase('train')
        time.sleep(0.08)
        mon.set_phase('val')
        time.sleep(0.08)
        mon.stop()
        rows = _read_csv(tmp_path / 'monitor.csv')
        phases = {r['phase'] for r in rows}
        assert 'train' in phases
        assert 'val' in phases


# ---------------------------------------------------------------------------
# ResourceMonitor — stop_after_epochs
# ---------------------------------------------------------------------------

class TestResourceMonitorStopping:
    def test_stops_after_configured_epochs(self, tmp_path):
        mon = _make_monitor(tmp_path, stop_after_epochs=2, interval=0.05)
        mon.start()
        mon.set_epoch(1)
        time.sleep(0.15)
        mon.set_epoch(3)   # exceeds stop_after_epochs=2
        time.sleep(0.3)    # give thread time to exit
        assert not mon._thread.is_alive()

    def test_stop_method_joins_thread(self, tmp_path):
        mon = _make_monitor(tmp_path, stop_after_epochs=10)
        mon.start()
        time.sleep(0.1)
        mon.stop()
        assert not mon._thread.is_alive()

    def test_stop_after_zero_runs_until_explicitly_stopped(self, tmp_path):
        """stop_after_epochs=0 means no epoch-based cutoff; runs until stop() is called.
        The caller (training.py) is responsible for not creating the monitor when disabled."""
        mon = ResourceMonitor(
            output_path=tmp_path / 'monitor.csv',
            interval=0.05,
            stop_after_epochs=0,
        )
        mon.start()
        time.sleep(0.15)
        assert mon._thread.is_alive()
        mon.stop()
        assert not mon._thread.is_alive()

    def test_daemon_thread(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.start()
        assert mon._thread.daemon is True
        mon.stop()


# ---------------------------------------------------------------------------
# ResourceMonitor — no GPU (device=None)
# ---------------------------------------------------------------------------

class TestResourceMonitorNoGpu:
    def test_gpu_fields_minus_one_without_device(self, tmp_path):
        mon = ResourceMonitor(
            output_path=tmp_path / 'monitor.csv',
            interval=0.05,
            stop_after_epochs=5,
            device=None,
        )
        # Patch torch.cuda.is_available so _gpu_index stays None
        with patch('torch.cuda.is_available', return_value=False):
            mon2 = ResourceMonitor(
                output_path=tmp_path / 'monitor2.csv',
                interval=0.05,
                stop_after_epochs=5,
                device=None,
            )
        mon2.start()
        time.sleep(0.15)
        mon2.stop()
        rows = _read_csv(tmp_path / 'monitor2.csv')
        assert len(rows) > 0
        assert rows[0]['gpu_mem_allocated_gb'] == '-1.0'

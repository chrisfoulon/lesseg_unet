"""Background resource monitor: samples CPU/RAM/GPU/worker stats to a CSV file."""
import csv
import os
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
import torch


def _gpu_utilization(device_index: int) -> float:
    """Return GPU utilisation % or -1.0 if unavailable."""
    try:
        return float(torch.cuda.utilization(device_index))
    except Exception:
        return -1.0


def _worker_count() -> int:
    """Count live DataLoader spawn_main child processes."""
    try:
        proc = psutil.Process(os.getpid())
        return sum(
            1 for p in proc.children(recursive=True)
            if any('spawn_main' in part for part in p.cmdline())
        )
    except Exception:
        return -1


_FIELDS = [
    'timestamp', 'epoch', 'phase',
    'ram_used_gb', 'ram_available_gb', 'swap_used_gb', 'swap_total_gb',
    'cpu_percent',
    'gpu_mem_allocated_gb', 'gpu_mem_reserved_gb', 'gpu_mem_total_gb', 'gpu_util_pct',
    'worker_count',
]


class ResourceMonitor:
    """Daemon thread that samples resources every *interval* seconds.

    Writes one CSV row per sample to *output_path*.
    Stops automatically once the epoch reported via :meth:`set_epoch` exceeds
    *stop_after_epochs*.  Pass ``stop_after_epochs=0`` to disable entirely.
    """

    def __init__(self, output_path: Path, interval: float = 1.0,
                 stop_after_epochs: int = 5, device=None):
        self.output_path = Path(output_path)
        self.interval = interval
        self.stop_after_epochs = stop_after_epochs

        # Resolve GPU device index once
        if device is not None and hasattr(device, 'index') and device.index is not None:
            self._gpu_index = device.index
        elif isinstance(device, str) and device.startswith('cuda:'):
            self._gpu_index = int(device.split(':')[1])
        elif torch.cuda.is_available():
            self._gpu_index = torch.cuda.current_device()
        else:
            self._gpu_index = None

        self._epoch = 0
        self._phase = 'init'
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='ResourceMonitor')

    def start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2 + 2)

    def set_epoch(self, epoch: int):
        with self._lock:
            self._epoch = epoch

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def _sample(self) -> dict:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        row = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ram_used_gb': round(vm.used / 1024 ** 3, 2),
            'ram_available_gb': round(vm.available / 1024 ** 3, 2),
            'swap_used_gb': round(sw.used / 1024 ** 3, 2),
            'swap_total_gb': round(sw.total / 1024 ** 3, 2),
            'cpu_percent': psutil.cpu_percent(),
            'gpu_mem_allocated_gb': -1.0,
            'gpu_mem_reserved_gb': -1.0,
            'gpu_mem_total_gb': -1.0,
            'gpu_util_pct': -1.0,
            'worker_count': _worker_count(),
        }
        if self._gpu_index is not None:
            try:
                row['gpu_mem_allocated_gb'] = round(
                    torch.cuda.memory_allocated(self._gpu_index) / 1024 ** 3, 2)
                row['gpu_mem_reserved_gb'] = round(
                    torch.cuda.memory_reserved(self._gpu_index) / 1024 ** 3, 2)
                props = torch.cuda.get_device_properties(self._gpu_index)
                row['gpu_mem_total_gb'] = round(props.total_memory / 1024 ** 3, 2)
                row['gpu_util_pct'] = _gpu_utilization(self._gpu_index)
            except Exception:
                pass
        return row

    def _run(self):
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            f.flush()
            while not self._stop_event.is_set():
                with self._lock:
                    epoch = self._epoch
                    phase = self._phase
                if self.stop_after_epochs > 0 and epoch > self.stop_after_epochs:
                    break
                row = self._sample()
                row['epoch'] = epoch
                row['phase'] = phase
                writer.writerow(row)
                f.flush()
                self._stop_event.wait(self.interval)

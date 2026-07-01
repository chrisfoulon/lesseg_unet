#!/usr/bin/env python3
"""Lightweight training resource monitor. Run in a separate terminal while training.

Usage:
    python monitor.py              # print to terminal, also log to /tmp/monitor.log
    python monitor.py --interval 5 # sample every 5s (default 10s)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

LOG_PATH = Path("/tmp/monitor.log")
DISK = "sda"   # /alpha lives here


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split(", ")
        util, used, total = int(out[0]), int(out[1]), int(out[2])
        return util, used / 1024, total / 1024  # pct, GB used, GB total
    except Exception:
        return None, None, None


def disk_read_mbs(disk, prev_bytes, elapsed):
    try:
        counters = psutil.disk_io_counters(perdisk=True)
        if disk not in counters:
            return 0.0, prev_bytes
        read_bytes = counters[disk].read_bytes
        mbs = (read_bytes - prev_bytes) / elapsed / 1024**2 if prev_bytes else 0.0
        return max(mbs, 0.0), read_bytes
    except Exception:
        return 0.0, prev_bytes


def worker_count():
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            if "spawn_main" in " ".join(p.info["cmdline"] or []):
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    header = f"{'time':>8}  {'RAM':>12}  {'GPU%':>5}  {'VRAM':>10}  {'disk_r MB/s':>11}  {'workers':>7}  {'CPU%':>5}"
    print(header)
    with open(LOG_PATH, "w") as f:
        f.write(header + "\n")

    prev_disk_bytes = 0
    prev_time = time.time()

    while True:
        time.sleep(args.interval)
        now = time.time()
        elapsed = now - prev_time
        prev_time = now

        ram = psutil.virtual_memory()
        ram_str = f"{ram.used/1024**3:.1f}/{ram.total/1024**3:.0f}GB"

        gpu_util, gpu_used, gpu_total = gpu_stats()
        if gpu_util is not None:
            gpu_str = f"{gpu_util:3d}%"
            vram_str = f"{gpu_used:.1f}/{gpu_total:.0f}GB"
        else:
            gpu_str, vram_str = "  N/A", "       N/A"

        disk_mbs, prev_disk_bytes = disk_read_mbs(DISK, prev_disk_bytes, elapsed)
        disk_str = f"{disk_mbs:7.1f}"

        workers = worker_count()
        cpu = psutil.cpu_percent(interval=None)

        ts = time.strftime("%H:%M:%S")
        line = f"{ts:>8}  {ram_str:>12}  {gpu_str:>5}  {vram_str:>10}  {disk_str:>11}  {workers:>7}  {cpu:>4.1f}%"
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nLog saved to {LOG_PATH}")
        sys.exit(0)

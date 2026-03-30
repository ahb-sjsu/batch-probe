# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Thermal-aware thread tuning — find max thread count that keeps CPU under a target temperature.

Usage:
    from batch_probe import probe_threads

    threads = probe_threads(
        work_fn=lambda n: run_my_workload(n_threads=n),
        max_temp=85.0,       # target max CPU temp (Celsius)
        low=1, high=48,      # thread range
        settle_time=5.0,     # seconds to let temp stabilize
    )
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time

log = logging.getLogger(__name__)


def _read_cpu_temp() -> float | None:
    """Read the highest CPU package temperature in Celsius.

    Tries (in order):
    1. lm-sensors (``sensors``)
    2. /sys/class/hwmon thermal zones
    3. /sys/class/thermal
    """
    # Method 1: lm-sensors
    try:
        out = subprocess.check_output(["sensors"], stderr=subprocess.DEVNULL, text=True, timeout=5)
        temps = []
        for line in out.splitlines():
            if "Package" in line or "Tctl" in line or "Tdie" in line:
                m = re.search(r"\+(\d+\.?\d*)", line)
                if m:
                    temps.append(float(m.group(1)))
        if temps:
            return max(temps)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Method 2: hwmon
    try:
        hwmon_dir = "/sys/class/hwmon"
        if os.path.isdir(hwmon_dir):
            temps = []
            for hw in os.listdir(hwmon_dir):
                name_path = os.path.join(hwmon_dir, hw, "name")
                if os.path.exists(name_path):
                    with open(name_path) as f:
                        if "coretemp" not in f.read():
                            continue
                for fname in os.listdir(os.path.join(hwmon_dir, hw)):
                    if fname.endswith("_input") and fname.startswith("temp"):
                        with open(os.path.join(hwmon_dir, hw, fname)) as f:
                            temps.append(float(f.read().strip()) / 1000.0)
            if temps:
                return max(temps)
    except (OSError, ValueError):
        pass

    # Method 3: thermal zones
    try:
        tz_dir = "/sys/class/thermal"
        if os.path.isdir(tz_dir):
            temps = []
            for tz in os.listdir(tz_dir):
                temp_path = os.path.join(tz_dir, tz, "temp")
                if os.path.exists(temp_path):
                    with open(temp_path) as f:
                        temps.append(float(f.read().strip()) / 1000.0)
            if temps:
                return max(temps)
    except (OSError, ValueError):
        pass

    return None


def probe_threads(
    work_fn,
    *,
    max_temp: float = 85.0,
    low: int = 1,
    high: int | None = None,
    settle_time: float = 5.0,
    work_time: float = 10.0,
    cooldown_time: float = 15.0,
    verbose: bool = True,
) -> int:
    """Find the maximum thread count that keeps CPU temperature under ``max_temp``.

    Binary search: run ``work_fn(n_threads)`` for ``work_time`` seconds, wait
    ``settle_time`` for thermal ramp, read temperature. If over ``max_temp``,
    search lower; otherwise search higher.

    Args:
        work_fn: Callable ``f(n_threads: int) -> None``. Should run a
            CPU workload using ``n_threads`` threads for at least
            ``work_time`` seconds. It will be interrupted via a timeout
            if needed.
        max_temp: Maximum acceptable CPU temperature in Celsius.
        low: Minimum thread count.
        high: Maximum thread count. Default: ``os.cpu_count()``.
        settle_time: Seconds to wait before reading temperature.
        work_time: Seconds to run the workload.
        cooldown_time: Seconds to wait between probes for cooling.
        verbose: Print progress.

    Returns:
        Safe thread count (``int``), guaranteed to keep CPU under ``max_temp``
        during sustained workload.

    Example::

        from batch_probe import probe_threads
        import numpy as np

        def stress(n):
            import os
            os.environ["OMP_NUM_THREADS"] = str(n)
            # Simulate heavy CPU work
            for _ in range(100):
                a = np.random.randn(2000, 2000)
                _ = a @ a.T

        threads = probe_threads(stress, max_temp=85.0)
        print(f"Safe thread count: {threads}")
    """
    if high is None:
        high = os.cpu_count() or 48

    # Check if we can read temperature
    baseline_temp = _read_cpu_temp()
    if baseline_temp is None:
        if verbose:
            print("batch-probe thermal: cannot read CPU temperature, returning high")
        return high

    if verbose:
        print(
            f"batch-probe thermal: probing (range=[{low}, {high}], "
            f"max_temp={max_temp}°C, baseline={baseline_temp:.1f}°C)...",
            flush=True,
        )

    best = low

    while low <= high:
        mid = (low + high) // 2

        # Let CPU cool before each probe
        if verbose:
            print(f"  cooling ({cooldown_time}s)...", end="", flush=True)
        time.sleep(cooldown_time)
        pre_temp = _read_cpu_temp()
        if verbose:
            print(f" {pre_temp:.1f}°C", flush=True)

        # Run workload
        if verbose:
            print(f"  trying {mid} threads ({work_time}s)...", end="", flush=True)

        import threading

        stop_event = threading.Event()

        def _run():
            try:
                work_fn(mid)
            except Exception:
                pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Wait for thermal ramp
        time.sleep(work_time + settle_time)
        stop_event.set()

        peak_temp = _read_cpu_temp()
        if verbose:
            print(f" peak={peak_temp:.1f}°C", flush=True)

        if peak_temp is not None and peak_temp <= max_temp:
            best = mid
            low = mid + 1
            if verbose:
                print(f"  → OK (under {max_temp}°C)")
        else:
            high = mid - 1
            if verbose:
                print(f"  → TOO HOT (over {max_temp}°C)")

    if verbose:
        print(f"batch-probe thermal: safe thread count = {best}")

    return best

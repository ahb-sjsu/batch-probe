# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Thermal-managed parallel job runner.

Launches subprocesses and dynamically adjusts concurrency based on
CPU temperature, using the Kalman-filtered thermal state estimator.

Usage::

    from batch_probe import ThermalJobManager

    jobs = [
        ("dataset_A", ["python", "run.py", "A"]),
        ("dataset_B", ["python", "run.py", "B"]),
        ("dataset_C", ["python", "run.py", "C"]),
    ]

    mgr = ThermalJobManager(target_temp=85.0, max_concurrent=4)
    results = mgr.run(jobs, cwd="/path/to/workdir")
    # results: dict of {name: returncode}
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

from batch_probe._thermal import _read_cpu_temp

log = logging.getLogger(__name__)


@dataclass
class ThermalJobManager:
    """Run jobs in parallel, throttled by CPU temperature.

    Args:
        target_temp: Maximum CPU temperature (°C). When exceeded,
            no new jobs are launched until temp drops below.
        max_concurrent: Maximum simultaneous jobs regardless of temp.
        settle_time: Seconds to wait after launch before reading temp.
        poll_interval: Seconds between checks for completed jobs.
        cooldown_margin: Must be this many °C below target to launch.
    """

    target_temp: float = 85.0
    max_concurrent: int = 4
    settle_time: float = 10.0
    poll_interval: float = 5.0
    cooldown_margin: float = 3.0
    verbose: bool = True

    def run(
        self,
        jobs: list[tuple[str, list[str]]],
        cwd: str | None = None,
        log_dir: str | None = None,
    ) -> dict[str, int]:
        """Run all jobs with thermal management.

        Args:
            jobs: List of (name, command) tuples.
            cwd: Working directory for all jobs.
            log_dir: Directory for log files. Default: cwd or current dir.

        Returns:
            Dict of {name: return_code}.
        """
        if log_dir is None:
            log_dir = cwd or os.getcwd()

        queue = list(jobs)
        active: dict[str, subprocess.Popen] = {}
        results: dict[str, int] = {}

        baseline = _read_cpu_temp()
        if self.verbose:
            log.info(
                "ThermalJobManager: %d jobs, target=%.0f°C, max=%d, baseline=%.1f°C",
                len(queue),
                self.target_temp,
                self.max_concurrent,
                baseline or 0,
            )

        while queue or active:
            # Reap finished jobs
            done = []
            for name, proc in active.items():
                rc = proc.poll()
                if rc is not None:
                    done.append((name, rc))
            for name, rc in done:
                del active[name]
                results[name] = rc
                if self.verbose:
                    log.info("  DONE: %s (exit %d)", name, rc)

            # Try to launch
            if queue and len(active) < self.max_concurrent:
                temp = _read_cpu_temp()
                if temp is None:
                    temp = 70.0  # assume safe if can't read

                launch_ok = temp < (self.target_temp - self.cooldown_margin)

                if launch_ok:
                    name, cmd = queue.pop(0)
                    log_path = os.path.join(log_dir, f"{name}.log")

                    if self.verbose:
                        log.info(
                            "  LAUNCH: %s (%.1f°C, %d active)",
                            name,
                            temp,
                            len(active),
                        )

                    log_file = open(log_path, "w")
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=cwd,
                    )
                    active[name] = proc

                    # Settle and check
                    time.sleep(self.settle_time)
                    new_temp = _read_cpu_temp() or temp
                    if self.verbose:
                        log.info(
                            "  Settled: %.1f°C (%+.1f)",
                            new_temp,
                            new_temp - temp,
                        )

                    if new_temp > self.target_temp:
                        if self.verbose:
                            log.info("  Over target — pausing launches")
                        # Don't launch more until temp drops
                elif self.verbose and int(time.time()) % 30 < 2:
                    log.info(
                        "  THROTTLED: %.1f°C > %.1f°C, %d active, %d queued",
                        temp,
                        self.target_temp - self.cooldown_margin,
                        len(active),
                        len(queue),
                    )

            time.sleep(self.poll_interval)

        if self.verbose:
            log.info("ThermalJobManager: all %d jobs complete", len(results))

        return results

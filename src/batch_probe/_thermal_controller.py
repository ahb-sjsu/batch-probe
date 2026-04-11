# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Adaptive thermal thread controller with Kalman filter.

Continuously adjusts thread count to maintain CPU temperature at a
setpoint, using Kalman-filtered state estimation for smooth,
predictive control.

State model:
    x = [temp, temp_rate]  (temperature and rate of change)
    u = thread_count       (control input — more threads = more heat)

The Kalman filter smooths noisy sensor readings and predicts thermal
trajectory. A PI controller then adjusts threads proactively:
- Reduces threads BEFORE overshoot (using predicted rate)
- Increases threads during cooldown (using predicted headroom)

Usage:
    from batch_probe import ThermalController

    ctrl = ThermalController(target_temp=82.0, max_threads=48)
    ctrl.start()

    # In your workload loop:
    while work_remaining:
        n = ctrl.get_threads()
        run_workload(n_threads=n)

    ctrl.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from batch_probe._thermal import _read_cpu_temp

log = logging.getLogger(__name__)


@dataclass
class KalmanThermal:
    """Kalman filter for CPU thermal state estimation.

    State: [temperature (°C), rate (°C/s)]
    Measurement: temperature sensor reading
    """

    dt: float = 2.0  # measurement interval (seconds)

    # State estimate
    x: np.ndarray = field(default_factory=lambda: np.array([60.0, 0.0]))

    # State covariance
    P: np.ndarray = field(default_factory=lambda: np.diag([4.0, 1.0]))

    # Process noise (temp wanders ��0.5°C, rate wanders ±0.2°C/s)
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.25, 0.04]))

    # Measurement noise (sensor ±1°C)
    R: float = 1.0

    def predict(self):
        """Predict next state (constant-rate model)."""
        F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: float):
        """Update with temperature measurement."""
        H = np.array([[1.0, 0.0]])
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + self.R  # innovation covariance
        K = self.P @ H.T / S  # Kalman gain
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

    @property
    def temp(self) -> float:
        return float(self.x[0])

    @property
    def rate(self) -> float:
        return float(self.x[1])

    def predicted_temp(self, horizon: float) -> float:
        """Predict temperature `horizon` seconds into the future."""
        return self.temp + self.rate * horizon


class ThermalController:
    """Adaptive thread controller using Kalman-filtered thermal state.

    PI controller with feedforward:
      threads = base - Kp * error - Ki * integral - Kd * rate

    where:
      error = filtered_temp - target_temp
      rate = Kalman-estimated temperature rate (°C/s)
      integral = accumulated error over time

    Args:
        target_temp: Desired CPU temperature (°C). Default 82.
        max_threads: Maximum thread count. Default: os.cpu_count().
        min_threads: Minimum thread count. Default: 1.
        poll_interval: Seconds between sensor reads. Default: 2.
        Kp: Proportional gain (threads per °C over target). Default: 3.
        Ki: Integral gain. Default: 0.1.
        Kd: Derivative gain (threads per °C/s rate). Default: 10.
        lookahead: Seconds to predict ahead for proactive control.
    """

    def __init__(
        self,
        target_temp: float = 82.0,
        max_threads: int | None = None,
        min_threads: int = 1,
        poll_interval: float = 2.0,
        Kp: float = 3.0,
        Ki: float = 0.1,
        Kd: float = 10.0,
        lookahead: float = 5.0,
        verbose: bool = True,
        auto_apply: bool = False,
    ):
        self.target = target_temp
        self.max_threads = max_threads or (os.cpu_count() or 48)
        self.min_threads = min_threads
        self.poll_interval = poll_interval
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.lookahead = lookahead
        self.verbose = verbose
        self.auto_apply = auto_apply
        self._pid = os.getpid()

        self.kf = KalmanThermal(dt=poll_interval)
        self._threads = self.max_threads
        self._integral = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._history: list[tuple[float, float, int]] = []  # (time, temp, threads)

    def get_threads(self) -> int:
        """Get the current recommended thread count (thread-safe)."""
        with self._lock:
            return self._threads

    def start(self):
        """Start the background thermal monitoring thread."""
        # Initialize Kalman with current reading
        t0 = _read_cpu_temp()
        if t0 is not None:
            self.kf.x = np.array([t0, 0.0])

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.auto_apply:
            self._apply_threads(self._threads)
        if self.verbose:
            log.info(
                "ThermalController started: target=%.0f°C, max=%d threads",
                self.target,
                self.max_threads,
            )

    def stop(self):
        """Stop the background thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.verbose:
            log.info("ThermalController stopped")

    def _loop(self):
        """Background loop: read sensor → Kalman update → PI control."""
        while not self._stop.is_set():
            z = _read_cpu_temp()
            if z is None:
                time.sleep(self.poll_interval)
                continue

            # Kalman predict + update
            self.kf.predict()
            self.kf.update(z)

            # PI + D control
            error = self.kf.temp - self.target
            predicted_error = self.kf.predicted_temp(self.lookahead) - self.target
            self._integral += error * self.poll_interval
            # Anti-windup: clamp integral
            self._integral = max(-50.0, min(50.0, self._integral))

            # Control signal: reduce threads when hot, increase when cool
            adjustment = self.Kp * error + self.Ki * self._integral + self.Kd * self.kf.rate

            # Use predicted error for proactive control
            if predicted_error > 2.0:
                adjustment += self.Kp * predicted_error * 0.5

            new_threads = int(round(self.max_threads - adjustment))
            new_threads = max(self.min_threads, min(self.max_threads, new_threads))

            with self._lock:
                old = self._threads
                self._threads = new_threads

            if self.auto_apply and new_threads != old:
                self._apply_threads(new_threads)

            self._history.append((time.time(), self.kf.temp, new_threads))

            if self.verbose and len(self._history) % 5 == 0:
                log.info(
                    "Thermal: %.1f°C (rate=%+.2f°C/s, pred=%.1f°C) → %d threads",
                    self.kf.temp,
                    self.kf.rate,
                    self.kf.predicted_temp(self.lookahead),
                    new_threads,
                )

            self._stop.wait(self.poll_interval)

    def _apply_threads(self, n: int) -> None:
        """Apply thread count to OMP, MKL, torch, and CPU affinity."""
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n)
        try:
            import torch

            torch.set_num_threads(n)
        except ImportError:
            pass
        try:
            os.sched_setaffinity(self._pid, set(range(n)))
        except (AttributeError, OSError):
            pass

    def summary(self) -> dict:
        """Return control history summary."""
        if not self._history:
            return {}
        temps = [h[1] for h in self._history]
        threads = [h[2] for h in self._history]
        return {
            "samples": len(self._history),
            "temp_mean": float(np.mean(temps)),
            "temp_max": float(np.max(temps)),
            "temp_min": float(np.min(temps)),
            "threads_mean": float(np.mean(threads)),
            "threads_min": int(np.min(threads)),
            "threads_max": int(np.max(threads)),
            "time_over_target": sum(1 for t in temps if t > self.target) / len(temps),
        }

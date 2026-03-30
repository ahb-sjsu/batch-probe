# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for _thermal_controller.py — KalmanThermal and ThermalController."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from batch_probe._thermal_controller import KalmanThermal, ThermalController


# ---------------------------------------------------------------------------
# KalmanThermal tests
# ---------------------------------------------------------------------------


class TestKalmanThermal:
    """Test the Kalman filter for thermal state estimation."""

    def test_initial_state(self):
        """Default state is [60.0, 0.0] (60C, no rate change)."""
        kf = KalmanThermal()
        assert kf.temp == pytest.approx(60.0)
        assert kf.rate == pytest.approx(0.0)

    def test_custom_initial_state(self):
        """Custom initial state is set correctly."""
        kf = KalmanThermal(dt=1.0)
        kf.x = np.array([75.0, 0.5])
        assert kf.temp == pytest.approx(75.0)
        assert kf.rate == pytest.approx(0.5)

    def test_predict_constant_rate(self):
        """Predict step extrapolates temperature using the rate."""
        kf = KalmanThermal(dt=2.0)
        kf.x = np.array([70.0, 1.0])  # 70C, rising 1C/s
        kf.predict()
        # After dt=2s at 1C/s: expected temp = 70 + 2*1 = 72
        assert kf.temp == pytest.approx(72.0, abs=0.5)
        # Rate should remain approximately 1.0 (process noise adds uncertainty)
        assert kf.rate == pytest.approx(1.0, abs=0.5)

    def test_predict_zero_rate(self):
        """Predict with zero rate keeps temperature stable."""
        kf = KalmanThermal(dt=2.0)
        kf.x = np.array([70.0, 0.0])
        kf.predict()
        assert kf.temp == pytest.approx(70.0, abs=0.5)

    def test_update_corrects_estimate(self):
        """Update step moves estimate toward the measurement."""
        kf = KalmanThermal(dt=2.0)
        kf.x = np.array([70.0, 0.0])
        kf.P = np.diag([4.0, 1.0])

        # Measurement says temp is actually 80
        kf.predict()
        kf.update(80.0)

        # After update, estimate should move toward 80
        assert kf.temp > 70.0
        assert kf.temp < 80.0  # Not all the way due to process noise weighting

    def test_repeated_updates_converge(self):
        """Multiple updates with same reading converge to that reading."""
        kf = KalmanThermal(dt=1.0)
        kf.x = np.array([50.0, 0.0])

        for _ in range(20):
            kf.predict()
            kf.update(80.0)

        # After many updates, should be very close to 80
        assert kf.temp == pytest.approx(80.0, abs=1.0)
        # Rate should converge toward 0 since measurement is constant
        assert abs(kf.rate) < 1.0

    def test_tracks_rising_temperature(self):
        """Filter tracks a linearly rising temperature."""
        kf = KalmanThermal(dt=1.0)
        kf.x = np.array([60.0, 0.0])

        # Feed linearly rising readings: 60, 61, 62, ...
        for i in range(20):
            kf.predict()
            kf.update(60.0 + i)

        # Should be near the latest reading
        assert kf.temp == pytest.approx(79.0, abs=2.0)
        # Rate should be close to 1.0 C/s
        assert kf.rate == pytest.approx(1.0, abs=0.5)

    def test_predicted_temp_uses_rate(self):
        """predicted_temp extrapolates using current rate."""
        kf = KalmanThermal()
        kf.x = np.array([75.0, 2.0])  # 75C, rising 2C/s
        # 5 seconds ahead: 75 + 2*5 = 85
        assert kf.predicted_temp(5.0) == pytest.approx(85.0)
        assert kf.predicted_temp(0.0) == pytest.approx(75.0)

    def test_covariance_shrinks_with_updates(self):
        """Covariance should decrease as more measurements arrive."""
        kf = KalmanThermal(dt=1.0)
        initial_trace = np.trace(kf.P)

        for _ in range(10):
            kf.predict()
            kf.update(70.0)

        # Covariance should have reduced (more confident)
        assert np.trace(kf.P) < initial_trace

    def test_dt_parameter(self):
        """dt parameter controls the predict step size."""
        kf_fast = KalmanThermal(dt=0.5)
        kf_slow = KalmanThermal(dt=5.0)

        kf_fast.x = np.array([70.0, 1.0])
        kf_slow.x = np.array([70.0, 1.0])

        kf_fast.predict()
        kf_slow.predict()

        # Slow dt should predict further ahead
        assert kf_slow.temp > kf_fast.temp


# ---------------------------------------------------------------------------
# ThermalController tests
# ---------------------------------------------------------------------------


class TestThermalController:
    """Test the adaptive thermal thread controller."""

    def test_default_construction(self):
        """Default parameters are sane."""
        ctrl = ThermalController(verbose=False)
        assert ctrl.target == 82.0
        assert ctrl.min_threads == 1
        assert ctrl.max_threads > 0
        assert ctrl.Kp == 3.0
        assert ctrl.Ki == 0.1
        assert ctrl.Kd == 10.0

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""
        ctrl = ThermalController(
            target_temp=75.0,
            max_threads=24,
            min_threads=2,
            poll_interval=1.0,
            Kp=2.0,
            Ki=0.05,
            Kd=5.0,
            lookahead=3.0,
            verbose=False,
        )
        assert ctrl.target == 75.0
        assert ctrl.max_threads == 24
        assert ctrl.min_threads == 2
        assert ctrl.poll_interval == 1.0
        assert ctrl.Kp == 2.0

    def test_get_threads_returns_max_initially(self):
        """Before start(), get_threads returns max_threads."""
        ctrl = ThermalController(max_threads=16, verbose=False)
        assert ctrl.get_threads() == 16

    def test_start_and_stop(self):
        """Controller can start and stop cleanly."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=65.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=8,
                poll_interval=0.05,
                verbose=False,
            )
            ctrl.start()
            assert ctrl._thread is not None
            assert ctrl._thread.is_alive()

            ctrl.stop()
            assert not ctrl._thread.is_alive()

    def test_adjusts_threads_when_hot(self):
        """Controller reduces threads when temperature exceeds target."""
        # Return 90C (well over 82C target)
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=90.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=16,
                poll_interval=0.01,
                verbose=False,
            )
            ctrl.start()
            # Give the loop enough iterations to adjust
            time.sleep(0.3)
            threads = ctrl.get_threads()
            ctrl.stop()

        # Should have reduced from 16
        assert threads < 16

    def test_maintains_threads_when_cool(self):
        """Controller keeps threads high when temperature is well below target."""
        # Return 60C (well under 82C target)
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=60.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=16,
                poll_interval=0.01,
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.3)
            threads = ctrl.get_threads()
            ctrl.stop()

        # Should stay at or near max
        assert threads >= 12  # Allow some fluctuation from Kalman dynamics

    def test_get_threads_is_thread_safe(self):
        """get_threads can be called from multiple threads concurrently."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=70.0):
            ctrl = ThermalController(
                max_threads=16,
                poll_interval=0.02,
                verbose=False,
            )
            ctrl.start()

            results = []
            errors = []

            def reader():
                try:
                    for _ in range(50):
                        t = ctrl.get_threads()
                        results.append(t)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=reader) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            ctrl.stop()

        assert not errors
        assert len(results) == 200
        assert all(1 <= r <= 16 for r in results)

    def test_skips_none_temperature(self):
        """Controller handles None temp readings gracefully."""
        call_count = 0

        def temp_alternating():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return None
            return 70.0

        with patch("batch_probe._thermal_controller._read_cpu_temp", side_effect=temp_alternating):
            ctrl = ThermalController(
                max_threads=8,
                poll_interval=0.02,
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.15)
            threads = ctrl.get_threads()
            ctrl.stop()

        # Should not crash, should eventually get a valid reading
        assert 1 <= threads <= 8

    def test_summary_empty_when_no_data(self):
        """summary() returns empty dict before any data is collected."""
        ctrl = ThermalController(verbose=False)
        assert ctrl.summary() == {}

    def test_summary_after_running(self):
        """summary() returns valid statistics after running."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=75.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=8,
                poll_interval=0.01,
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.3)
            ctrl.stop()

        s = ctrl.summary()
        assert "samples" in s
        assert s["samples"] > 0
        assert "temp_mean" in s
        assert "temp_max" in s
        assert "temp_min" in s
        assert "threads_mean" in s
        assert "threads_min" in s
        assert "threads_max" in s
        assert "time_over_target" in s

    def test_integral_antiwindup(self):
        """Integral term is clamped to prevent windup."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=95.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=16,
                poll_interval=0.02,
                Ki=1.0,  # High Ki to exercise windup
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.2)
            ctrl.stop()

        # Integral should be clamped at 50
        assert ctrl._integral <= 50.0
        assert ctrl._integral >= -50.0

    def test_threads_clamped_to_range(self):
        """Thread count never goes below min or above max."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=99.0):
            ctrl = ThermalController(
                target_temp=82.0,
                max_threads=16,
                min_threads=2,
                poll_interval=0.02,
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.15)
            threads = ctrl.get_threads()
            ctrl.stop()

        assert threads >= 2
        assert threads <= 16

    def test_stop_without_start(self):
        """Calling stop() without start() does not raise."""
        ctrl = ThermalController(verbose=False)
        ctrl.stop()  # Should not raise

    def test_history_accumulates(self):
        """History entries are appended each iteration."""
        with patch("batch_probe._thermal_controller._read_cpu_temp", return_value=70.0):
            ctrl = ThermalController(
                max_threads=8,
                poll_interval=0.01,
                verbose=False,
            )
            ctrl.start()
            time.sleep(0.3)
            ctrl.stop()

        assert len(ctrl._history) > 0
        # Each entry is (time, temp, threads)
        for entry in ctrl._history:
            assert len(entry) == 3
            timestamp, temp, threads = entry
            assert isinstance(timestamp, float)
            assert isinstance(temp, float)
            assert isinstance(threads, int)

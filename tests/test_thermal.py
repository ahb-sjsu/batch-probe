# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for _thermal.py — CPU temperature reading and probe_threads()."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import mock_open, patch


from batch_probe._thermal import _read_cpu_temp, probe_threads


# ---------------------------------------------------------------------------
# _read_cpu_temp tests
# ---------------------------------------------------------------------------


class TestReadCpuTemp:
    """Test the three temperature-reading fallback methods."""

    def test_method1_sensors_parses_package_temp(self):
        """lm-sensors output with 'Package' line is parsed correctly."""
        sensors_output = (
            "coretemp-isa-0000\n"
            "Adapter: ISA adapter\n"
            "Package id 0:  +72.0\u00b0C  (high = +100.0\u00b0C, crit = +110.0\u00b0C)\n"
            "Core 0:        +68.0\u00b0C  (high = +100.0\u00b0C, crit = +110.0\u00b0C)\n"
            "Core 1:        +70.0\u00b0C  (high = +100.0\u00b0C, crit = +110.0\u00b0C)\n"
        )
        with patch("subprocess.check_output", return_value=sensors_output):
            result = _read_cpu_temp()
        assert result == 72.0

    def test_method1_sensors_parses_tctl(self):
        """AMD Tctl output is parsed correctly."""
        sensors_output = (
            "k10temp-pci-00c3\n"
            "Adapter: PCI adapter\n"
            "Tctl:         +65.5\u00b0C\n"
            "Tdie:         +63.2\u00b0C\n"
        )
        with patch("subprocess.check_output", return_value=sensors_output):
            result = _read_cpu_temp()
        assert result == 65.5

    def test_method1_sensors_parses_tdie_only(self):
        """When only Tdie is present, it is returned."""
        sensors_output = "k10temp-pci-00c3\nTdie:         +58.0\u00b0C\n"
        with patch("subprocess.check_output", return_value=sensors_output):
            result = _read_cpu_temp()
        assert result == 58.0

    def test_method1_sensors_no_match_falls_through(self):
        """If sensors output has no recognized lines, fall through to method 2."""
        sensors_output = "some-sensor\nAdapter: PCI\nfan1: 1200 RPM\n"
        with (
            patch("subprocess.check_output", return_value=sensors_output),
            patch("os.path.isdir", return_value=False),
        ):
            result = _read_cpu_temp()
        assert result is None

    def test_method1_sensors_not_found_falls_to_method2(self):
        """FileNotFoundError from sensors falls through to hwmon."""
        with (
            patch(
                "subprocess.check_output",
                side_effect=FileNotFoundError("sensors not found"),
            ),
            patch("os.path.isdir", return_value=False),
        ):
            result = _read_cpu_temp()
        assert result is None

    def test_method1_sensors_timeout_falls_through(self):
        """Subprocess timeout falls through to method 2."""
        with (
            patch(
                "subprocess.check_output",
                side_effect=subprocess.TimeoutExpired("sensors", 5),
            ),
            patch("os.path.isdir", return_value=False),
        ):
            result = _read_cpu_temp()
        assert result is None

    def test_method2_hwmon_reads_coretemp(self):
        """hwmon coretemp files are read and max is returned."""
        # Use os.path.join to match platform-specific path separators
        hwmon_dir = "/sys/class/hwmon"
        hw0 = os.path.join(hwmon_dir, "hwmon0")
        name_path = os.path.join(hw0, "name")
        temp1_path = os.path.join(hw0, "temp1_input")
        temp2_path = os.path.join(hw0, "temp2_input")

        def mock_isdir(path):
            return path == hwmon_dir

        def mock_listdir(path):
            if path == hwmon_dir:
                return ["hwmon0"]
            if path == hw0:
                return ["name", "temp1_input", "temp2_input"]
            return []

        def mock_exists(path):
            return path in (name_path, temp1_path, temp2_path)

        file_contents = {
            name_path: "coretemp\n",
            temp1_path: "75000\n",  # 75.0 C
            temp2_path: "73000\n",  # 73.0 C
        }

        def mock_file_open(path, *args, **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError(path)

        with (
            patch(
                "batch_probe._thermal.subprocess.check_output",
                side_effect=FileNotFoundError("no sensors"),
            ),
            patch("batch_probe._thermal.os.path.isdir", side_effect=mock_isdir),
            patch("batch_probe._thermal.os.listdir", side_effect=mock_listdir),
            patch("batch_probe._thermal.os.path.exists", side_effect=mock_exists),
            patch("builtins.open", side_effect=mock_file_open),
        ):
            result = _read_cpu_temp()
        assert result == 75.0

    def test_method3_thermal_zone(self):
        """Thermal zone files are read as last resort."""
        # Use os.path.join to match platform-specific path separators
        tz_dir = "/sys/class/thermal"
        tz0_temp = os.path.join(tz_dir, "thermal_zone0", "temp")
        tz1_temp = os.path.join(tz_dir, "thermal_zone1", "temp")

        def mock_isdir(path):
            if path == "/sys/class/hwmon":
                return False
            if path == tz_dir:
                return True
            return False

        def mock_listdir(path):
            if path == tz_dir:
                return ["thermal_zone0", "thermal_zone1"]
            return []

        def mock_exists(path):
            return path in (tz0_temp, tz1_temp)

        file_contents = {
            tz0_temp: "68000\n",
            tz1_temp: "71500\n",
        }

        def mock_file_open(path, *args, **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError(path)

        with (
            patch(
                "batch_probe._thermal.subprocess.check_output",
                side_effect=FileNotFoundError("no sensors"),
            ),
            patch("batch_probe._thermal.os.path.isdir", side_effect=mock_isdir),
            patch("batch_probe._thermal.os.listdir", side_effect=mock_listdir),
            patch("batch_probe._thermal.os.path.exists", side_effect=mock_exists),
            patch("builtins.open", side_effect=mock_file_open),
        ):
            result = _read_cpu_temp()
        assert result == 71.5

    def test_all_methods_fail_returns_none(self):
        """When no method can read temperature, return None."""
        with (
            patch(
                "subprocess.check_output",
                side_effect=FileNotFoundError("no sensors"),
            ),
            patch("os.path.isdir", return_value=False),
        ):
            result = _read_cpu_temp()
        assert result is None


# ---------------------------------------------------------------------------
# probe_threads tests
# ---------------------------------------------------------------------------


class TestProbeThreads:
    """Test the thermal binary search for thread count."""

    def test_returns_high_when_temp_unreadable(self):
        """If temperature cannot be read, return high (max threads)."""
        with patch("batch_probe._thermal._read_cpu_temp", return_value=None):
            result = probe_threads(
                lambda n: None,
                max_temp=85.0,
                low=1,
                high=32,
                verbose=False,
            )
        assert result == 32

    def test_finds_safe_thread_count_under_target(self):
        """Binary search finds a thread count that keeps temp under max."""
        # Simulate: temp rises proportionally with thread count
        # At n threads, temp = 60 + n * 1.0
        # max_temp = 85 -> safe at n <= 25, too hot at n > 25
        call_count = 0

        def mock_temp():
            nonlocal call_count
            call_count += 1
            # Return baseline first, then simulate rising temp per mid value
            return 60.0

        temp_readings = []

        def mock_temp_for_search():
            # We need to track what mid value was being tested
            if temp_readings:
                return temp_readings[-1]
            return 60.0

        # Simpler approach: mock _read_cpu_temp to return values based
        # on call sequence. probe_threads does:
        #   1. baseline read
        #   2. per iteration: pre_temp read (after cooldown), peak read (after work)
        # With low=1, high=8, binary search tries mid=4, then adjusts.

        # Let's say: any thread count <= 6 is safe (temp <= 85),
        # thread count > 6 is too hot (temp > 85)
        readings = iter(
            [
                60.0,  # baseline
                55.0,  # pre_temp for mid=4
                78.0,  # peak for mid=4 -> OK (78 < 85)
                55.0,  # pre_temp for mid=6
                83.0,  # peak for mid=6 -> OK (83 < 85)
                55.0,  # pre_temp for mid=7
                89.0,  # peak for mid=7 -> TOO HOT
                55.0,  # pre_temp for mid=... (won't be reached if search ends)
                84.0,
            ]
        )

        with (
            patch("batch_probe._thermal._read_cpu_temp", side_effect=readings),
            patch("batch_probe._thermal.time.sleep"),
        ):
            result = probe_threads(
                lambda n: None,
                max_temp=85.0,
                low=1,
                high=8,
                settle_time=0,
                work_time=0,
                cooldown_time=0,
                verbose=False,
            )
        assert result == 6

    def test_all_temps_safe_returns_high(self):
        """If all probes are under max_temp, return high."""
        # For low=1, high=4: mid=2 (ok), mid=3 (ok), mid=4 (ok)
        readings = iter(
            [
                60.0,  # baseline
                55.0,
                70.0,  # mid=2 ok
                55.0,
                72.0,  # mid=3 ok
                55.0,
                74.0,  # mid=4 ok
                55.0,
                75.0,  # extra safety
            ]
        )
        with (
            patch("batch_probe._thermal._read_cpu_temp", side_effect=readings),
            patch("batch_probe._thermal.time.sleep"),
        ):
            result = probe_threads(
                lambda n: None,
                max_temp=85.0,
                low=1,
                high=4,
                settle_time=0,
                work_time=0,
                cooldown_time=0,
                verbose=False,
            )
        assert result == 4

    def test_all_temps_too_hot_returns_low(self):
        """If even 1 thread is too hot, return low."""
        readings = iter(
            [
                60.0,  # baseline
                55.0,
                90.0,  # mid=2 too hot
                55.0,
                88.0,  # mid=1 too hot
            ]
        )
        with (
            patch("batch_probe._thermal._read_cpu_temp", side_effect=readings),
            patch("batch_probe._thermal.time.sleep"),
        ):
            result = probe_threads(
                lambda n: None,
                max_temp=85.0,
                low=1,
                high=4,
                settle_time=0,
                work_time=0,
                cooldown_time=0,
                verbose=False,
            )
        # best stays at initial value = low = 1
        assert result == 1

    def test_default_high_uses_cpu_count(self):
        """When high is None, os.cpu_count() is used."""
        with (
            patch("batch_probe._thermal._read_cpu_temp", return_value=None),
            patch("os.cpu_count", return_value=16),
        ):
            result = probe_threads(lambda n: None, high=None, verbose=False)
        assert result == 16

    def test_verbose_output(self, capsys):
        """Verbose mode prints progress information."""
        readings = iter(
            [
                60.0,  # baseline
                55.0,
                70.0,  # mid ok
                55.0,
                72.0,  # mid ok
                55.0,
                74.0,
            ]
        )
        with (
            patch("batch_probe._thermal._read_cpu_temp", side_effect=readings),
            patch("batch_probe._thermal.time.sleep"),
        ):
            probe_threads(
                lambda n: None,
                max_temp=85.0,
                low=1,
                high=4,
                settle_time=0,
                work_time=0,
                cooldown_time=0,
                verbose=True,
            )
        captured = capsys.readouterr()
        assert "batch-probe thermal" in captured.out
        assert "safe thread count" in captured.out

    def test_work_fn_exception_handled(self):
        """Exceptions in work_fn do not crash the probe."""

        def bad_work(n):
            raise RuntimeError("work failed")

        readings = iter(
            [
                60.0,  # baseline
                55.0,
                70.0,  # mid=1
            ]
        )
        with (
            patch("batch_probe._thermal._read_cpu_temp", side_effect=readings),
            patch("batch_probe._thermal.time.sleep"),
        ):
            # Should not raise — the thread catches the exception
            result = probe_threads(
                bad_work,
                max_temp=85.0,
                low=1,
                high=1,
                settle_time=0,
                work_time=0,
                cooldown_time=0,
                verbose=False,
            )
        assert result >= 1

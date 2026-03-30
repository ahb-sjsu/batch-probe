# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for _thermal_jobs.py — ThermalJobManager."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch


from batch_probe._thermal_jobs import ThermalJobManager


class TestThermalJobManager:
    """Test the thermal-managed parallel job runner."""

    def test_default_parameters(self):
        """Default configuration is sane."""
        mgr = ThermalJobManager()
        assert mgr.target_temp == 85.0
        assert mgr.max_concurrent == 4
        assert mgr.settle_time == 10.0
        assert mgr.poll_interval == 5.0
        assert mgr.cooldown_margin == 3.0

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""
        mgr = ThermalJobManager(
            target_temp=80.0,
            max_concurrent=2,
            settle_time=5.0,
            poll_interval=2.0,
            cooldown_margin=5.0,
        )
        assert mgr.target_temp == 80.0
        assert mgr.max_concurrent == 2
        assert mgr.settle_time == 5.0

    def test_empty_job_list(self):
        """Running with no jobs returns empty dict."""
        mgr = ThermalJobManager(verbose=False)
        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
        ):
            results = mgr.run([])
        assert results == {}

    def test_single_job_success(self):
        """Single job runs and returns exit code 0."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        # First poll returns None (running), second returns 0 (done)
        mock_proc.poll.side_effect = [None, 0]

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            results = mgr.run([("job1", ["echo", "hello"])], cwd="/tmp")

        assert results == {"job1": 0}

    def test_multiple_jobs_all_succeed(self):
        """Multiple jobs all complete with exit code 0."""
        procs = []
        for _ in range(3):
            p = MagicMock(spec=subprocess.Popen)
            # Running, running, done
            p.poll.side_effect = [None, None, 0]
            procs.append(p)

        proc_iter = iter(procs)

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", side_effect=lambda *a, **kw: next(proc_iter)),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                max_concurrent=3,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            jobs = [
                ("jobA", ["python", "a.py"]),
                ("jobB", ["python", "b.py"]),
                ("jobC", ["python", "c.py"]),
            ]
            results = mgr.run(jobs, cwd="/tmp")

        assert len(results) == 3
        assert all(rc == 0 for rc in results.values())

    def test_job_failure_captured(self):
        """Non-zero exit code is captured in results."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.side_effect = [None, 1]  # exit code 1

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            results = mgr.run([("failing_job", ["false"])], cwd="/tmp")

        assert results["failing_job"] == 1

    def test_throttles_when_hot(self):
        """Jobs are not launched when temperature exceeds threshold."""
        # Temperature starts hot (above target - margin), preventing launch
        # Then drops to allow launch
        temp_readings = iter(
            [
                60.0,  # baseline
                # Loop 1: active=0, queue has job, check temp
                90.0,  # too hot (90 > 85-3=82), don't launch
                # Loop 2: still hot
                90.0,
                # Loop 3: cooled down
                75.0,  # ok (75 < 82), launch
                70.0,  # settle check
                # Loop 4: job running, check poll
            ]
        )

        mock_proc = MagicMock(spec=subprocess.Popen)
        # Not done, not done, not done, done
        mock_proc.poll.side_effect = [None, None, None, 0]

        launch_count = 0

        def counting_popen(*args, **kwargs):
            nonlocal launch_count
            launch_count += 1
            return mock_proc

        with (
            patch(
                "batch_probe._thermal_jobs._read_cpu_temp",
                side_effect=temp_readings,
            ),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", side_effect=counting_popen),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                max_concurrent=4,
                cooldown_margin=3.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            results = mgr.run([("job1", ["echo", "hi"])], cwd="/tmp")

        # Job should eventually complete
        assert "job1" in results
        assert launch_count == 1  # Only launched once (after cool-down)

    def test_respects_max_concurrent(self):
        """No more than max_concurrent jobs run simultaneously."""
        max_active_seen = 0
        current_active = 0

        def make_proc():
            nonlocal current_active
            current_active += 1

            p = MagicMock()
            poll_count = [0]

            def mock_poll():
                nonlocal current_active
                poll_count[0] += 1
                if poll_count[0] >= 3:
                    current_active -= 1
                    return 0
                return None

            p.poll.side_effect = mock_poll
            return p

        popen_calls = []

        def tracking_popen(*args, **kwargs):
            nonlocal max_active_seen
            p = make_proc()
            popen_calls.append(p)
            if current_active > max_active_seen:
                max_active_seen = current_active
            return p

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("batch_probe._thermal_jobs.subprocess.Popen", side_effect=tracking_popen),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                max_concurrent=2,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            jobs = [
                ("j1", ["echo", "1"]),
                ("j2", ["echo", "2"]),
                ("j3", ["echo", "3"]),
                ("j4", ["echo", "4"]),
            ]
            results = mgr.run(jobs, cwd="/tmp")

        assert len(results) == 4
        assert max_active_seen <= 2

    def test_log_dir_defaults_to_cwd(self):
        """Log directory defaults to cwd when not specified."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.side_effect = [0]

        opened_paths = []
        real_mock = mock_open()

        def tracking_open(path, *args, **kwargs):
            opened_paths.append(path)
            return real_mock(path, *args, **kwargs)

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("builtins.open", side_effect=tracking_open),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            mgr.run([("test_job", ["echo"])], cwd="/work/dir")

        # Log file should be at cwd/test_job.log
        assert any("test_job.log" in p for p in opened_paths)

    def test_custom_log_dir(self):
        """Custom log_dir is used for log files."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.side_effect = [0]

        opened_paths = []
        real_mock = mock_open()

        def tracking_open(path, *args, **kwargs):
            opened_paths.append(path)
            return real_mock(path, *args, **kwargs)

        with (
            patch("batch_probe._thermal_jobs._read_cpu_temp", return_value=60.0),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("builtins.open", side_effect=tracking_open),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            mgr.run(
                [("myjob", ["echo"])],
                cwd="/work",
                log_dir="/logs",
            )

        assert any("/logs/myjob.log" in p.replace("\\", "/") for p in opened_paths)

    def test_none_baseline_temp_handled(self):
        """If baseline temp is None, manager still runs."""
        temp_call_count = [0]

        def temp_returns_none_then_value():
            temp_call_count[0] += 1
            if temp_call_count[0] == 1:
                return None  # baseline
            return 60.0  # subsequent reads

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.side_effect = [None, 0]

        with (
            patch(
                "batch_probe._thermal_jobs._read_cpu_temp",
                side_effect=temp_returns_none_then_value,
            ),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            results = mgr.run([("j1", ["echo"])], cwd="/tmp")

        assert "j1" in results

    def test_settle_detects_overshoot(self):
        """After launch, if temp exceeds target, no more jobs launch immediately."""
        temp_sequence = iter(
            [
                60.0,  # baseline
                70.0,  # check before launch job1 (ok, 70 < 82)
                88.0,  # settle read after job1 launch (over target)
                # Now another job is queued but won't launch because active check
                # Next iteration: reap job1, then check temp for job2
                70.0,  # check before launch job2
                72.0,  # settle after job2
            ]
        )

        procs = []
        for _ in range(2):
            p = MagicMock(spec=subprocess.Popen)
            p.poll.side_effect = [None, 0]
            procs.append(p)
        proc_iter = iter(procs)

        with (
            patch(
                "batch_probe._thermal_jobs._read_cpu_temp",
                side_effect=temp_sequence,
            ),
            patch("batch_probe._thermal_jobs.time.sleep"),
            patch("subprocess.Popen", side_effect=lambda *a, **kw: next(proc_iter)),
            patch("builtins.open", mock_open()),
        ):
            mgr = ThermalJobManager(
                target_temp=85.0,
                max_concurrent=2,
                settle_time=0,
                poll_interval=0,
                verbose=False,
            )
            results = mgr.run(
                [("j1", ["echo"]), ("j2", ["echo"])],
                cwd="/tmp",
            )

        assert len(results) == 2

# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for the generic probe() API."""

import pytest
from batch_probe import probe


class TestGenericProbe:
    """Tests for probe(work_fn) — the framework-agnostic API."""

    def test_finds_max_batch_simple(self):
        """Binary search finds the right batch size for a simple workload."""

        # Simulate: works up to batch=500, OOM above
        def work(n):
            if n > 500:
                raise MemoryError("out of memory")

        result = probe(work, low=1, high=1000, headroom=0.0, verbose=False)
        assert result == 500

    def test_headroom_applied(self):
        """Headroom reduces the returned batch size."""

        def work(n):
            if n > 1000:
                raise MemoryError("out of memory")

        result = probe(work, low=1, high=2000, headroom=0.2, verbose=False)
        assert result == 800  # 1000 * 0.8

    def test_headroom_zero(self):
        """Zero headroom returns the exact max."""

        def work(n):
            if n > 200:
                raise MemoryError("out of memory")

        result = probe(work, low=1, high=500, headroom=0.0, verbose=False)
        assert result == 200

    def test_all_oom_returns_low(self):
        """If everything OOMs, returns at least 1."""

        def work(n):
            raise MemoryError("out of memory")

        result = probe(work, low=1, high=100, headroom=0.0, verbose=False)
        assert result >= 1

    def test_runtime_error_oom(self):
        """RuntimeError with 'out of memory' is treated as OOM."""

        def work(n):
            if n > 100:
                raise RuntimeError("CUDA out of memory. Tried to allocate 7GB.")

        result = probe(work, low=1, high=500, headroom=0.0, verbose=False)
        assert result == 100

    def test_non_oom_runtime_error_propagates(self):
        """RuntimeError without 'out of memory' is re-raised."""

        def work(n):
            raise RuntimeError("something else broke")

        with pytest.raises(RuntimeError, match="something else broke"):
            probe(work, low=1, high=100, verbose=False)

    def test_non_memory_error_propagates(self):
        """Non-memory exceptions propagate."""

        def work(n):
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            probe(work, low=1, high=100, verbose=False)

    def test_verbose_output(self, capsys):
        """Verbose mode prints progress."""

        def work(n):
            if n > 50:
                raise MemoryError("oom")

        probe(work, low=1, high=100, verbose=True)
        captured = capsys.readouterr()
        assert "batch-probe" in captured.out
        assert "max=" in captured.out
        assert "safe=" in captured.out

    def test_low_equals_high(self):
        """Works when low == high."""
        calls = []

        def work(n):
            calls.append(n)

        result = probe(work, low=42, high=42, headroom=0.0, verbose=False)
        assert result == 42
        assert 42 in calls

    def test_large_range(self):
        """Handles large search ranges efficiently (log2 steps)."""
        call_count = 0

        def work(n):
            nonlocal call_count
            call_count += 1
            if n > 123456:
                raise MemoryError("oom")

        result = probe(work, low=1, high=1_000_000, headroom=0.0, verbose=False)
        assert result == 123456
        # Binary search on 1M range should take ~20 steps
        assert call_count <= 25

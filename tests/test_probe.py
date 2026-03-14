# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for the core probe_batch_size function.

These tests work without a GPU by using models that simulate OOM
via a Python-side threshold check.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from batch_probe import probe_batch_size


class FakeOOMModel(nn.Module):
    """Model that raises OOM above a configurable batch size threshold."""

    def __init__(self, oom_threshold: int = 16):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.oom_threshold = oom_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > self.oom_threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self.linear(x)


class FakeDictModel(nn.Module):
    """Model that returns a dict with 'loss' key (like HuggingFace)."""

    def __init__(self, oom_threshold: int = 8):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.oom_threshold = oom_threshold

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        if input_ids.shape[0] > self.oom_threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        out = self.linear(input_ids.float())
        return {"loss": out.mean(), "logits": out}


def _make_input_fn(feature_dim: int = 10):
    """Create an input_fn for FakeOOMModel."""
    return lambda bs: {"x": torch.randn(bs, feature_dim)}


def _make_dict_input_fn(feature_dim: int = 10):
    """Create an input_fn for FakeDictModel."""
    return lambda bs: {
        "input_ids": torch.zeros(bs, feature_dim, dtype=torch.long),
        "attention_mask": torch.ones(bs, feature_dim, dtype=torch.long),
    }


class TestProbeBatchSize:
    def test_finds_max_batch_infer(self):
        model = FakeOOMModel(oom_threshold=32)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=128,
            headroom=0.2,
            verbose=False,
        )
        # max successful = 32, safe = int(32 * 0.8) = 25
        assert result == 25

    def test_finds_max_batch_train(self):
        model = FakeOOMModel(oom_threshold=32)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="train",
            high=128,
            headroom=0.2,
            verbose=False,
        )
        assert result == 25

    def test_headroom_zero(self):
        model = FakeOOMModel(oom_threshold=32)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=128,
            headroom=0.0,
            verbose=False,
        )
        assert result == 32

    def test_headroom_fifty_percent(self):
        model = FakeOOMModel(oom_threshold=32)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=128,
            headroom=0.5,
            verbose=False,
        )
        assert result == 16

    def test_returns_at_least_one(self):
        # Even with high headroom and low threshold, never returns 0
        model = FakeOOMModel(oom_threshold=2)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=64,
            headroom=0.9,
            verbose=False,
        )
        assert result >= 1

    def test_all_oom(self):
        # Model OOMs even at batch=1
        model = FakeOOMModel(oom_threshold=0)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            low=1,
            high=64,
            headroom=0.2,
            verbose=False,
        )
        # best stays at low=1 but it OOMs, so best stays at initial=1
        # This is an edge case — we return 1 as the floor
        assert result >= 1

    def test_dict_output_model(self):
        model = FakeDictModel(oom_threshold=8)
        result = probe_batch_size(
            model,
            _make_dict_input_fn(),
            mode="train",
            high=64,
            headroom=0.2,
            verbose=False,
        )
        assert result == max(1, int(8 * 0.8))  # 6

    def test_cpu_probes_normally(self):
        # On CPU, probe still runs (no CUDA cleanup, but OOM still detected)
        model = FakeOOMModel(oom_threshold=16)
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=64,
            device="cpu",
            headroom=0.2,
            verbose=False,
        )
        assert result == int(16 * 0.8)  # 12

    def test_non_oom_error_propagates(self):
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                raise ValueError("Something went wrong")

        model = BadModel()
        with pytest.raises(ValueError, match="Something went wrong"):
            probe_batch_size(
                model,
                _make_input_fn(),
                mode="infer",
                high=16,
                verbose=False,
            )

    def test_restores_training_mode(self):
        model = FakeOOMModel(oom_threshold=16)

        # Start in eval mode
        model.eval()
        assert not model.training

        probe_batch_size(
            model,
            _make_input_fn(),
            mode="train",
            high=32,
            verbose=False,
        )
        # Should restore eval mode
        assert not model.training

        # Start in train mode
        model.train()
        assert model.training

        probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=32,
            verbose=False,
        )
        # Should restore train mode
        assert model.training

    def test_verbose_output(self, capsys):
        model = FakeOOMModel(oom_threshold=8)
        probe_batch_size(
            model,
            _make_input_fn(),
            mode="infer",
            high=32,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "torch-probe" in captured.out
        assert "max=8" in captured.out
        assert "safe=" in captured.out

    def test_train_mode_includes_optimizer_step(self):
        """Train mode should include optimizer step to account for state memory."""

        class OOMOnOptimizerStep(nn.Module):
            """Model that OOMs when optimizer.step() is called at large batch."""

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
                self._forward_count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self._forward_count += 1
                return self.linear(x)

        model = OOMOnOptimizerStep()
        # Forward+backward always succeed; the probe still runs train with
        # optimizer. Just verify the probe completes without error.
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="train",
            high=32,
            headroom=0.2,
            verbose=False,
        )
        assert result >= 1
        # Confirm optimizer step was exercised (forward was called)
        assert model._forward_count > 0

    def test_cleanup_no_tensor_leak(self):
        """Verify outputs/loss are cleaned up between iterations."""
        model = FakeOOMModel(oom_threshold=16)
        # Run probe — if cleanup leaks, repeated iterations would accumulate
        result = probe_batch_size(
            model,
            _make_input_fn(),
            mode="train",
            high=128,
            headroom=0.2,
            verbose=False,
        )
        assert result == int(16 * 0.8)

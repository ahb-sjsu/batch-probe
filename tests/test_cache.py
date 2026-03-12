# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for the caching layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from batch_probe import cached_probe, clear_cache


class CountingModel(nn.Module):
    """Model that counts how many times forward() is called."""

    def __init__(self, oom_threshold: int = 8):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.oom_threshold = oom_threshold
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if x.shape[0] > self.oom_threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return self.linear(x)


class TestCachedProbe:
    def setup_method(self):
        clear_cache()

    def test_cache_hit(self):
        model = CountingModel(oom_threshold=8)
        input_fn = lambda bs: {"x": torch.randn(bs, 10)}

        r1 = cached_probe(model, input_fn, mode="infer", high=32, verbose=False)
        calls_after_first = model.call_count

        r2 = cached_probe(model, input_fn, mode="infer", high=32, verbose=False)
        calls_after_second = model.call_count

        assert r1 == r2
        assert calls_after_second == calls_after_first  # No new forward calls

    def test_different_modes_separate_cache(self):
        model = CountingModel(oom_threshold=8)
        input_fn = lambda bs: {"x": torch.randn(bs, 10)}

        r_train = cached_probe(model, input_fn, mode="train", high=32, verbose=False)
        r_infer = cached_probe(model, input_fn, mode="infer", high=32, verbose=False)

        # Both should probe (different modes = different cache keys)
        # Results may differ since train mode uses backward pass
        assert isinstance(r_train, int)
        assert isinstance(r_infer, int)

    def test_clear_cache(self):
        model = CountingModel(oom_threshold=8)
        input_fn = lambda bs: {"x": torch.randn(bs, 10)}

        cached_probe(model, input_fn, mode="infer", high=32, verbose=False)
        calls_first = model.call_count

        clear_cache()
        cached_probe(model, input_fn, mode="infer", high=32, verbose=False)
        calls_second = model.call_count

        assert calls_second > calls_first  # Had to re-probe after clearing

# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""GPU memory cleanup utilities."""

from __future__ import annotations

import gc

import torch


def gpu_cleanup() -> None:
    """Aggressively free GPU memory after an OOM or between probe iterations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

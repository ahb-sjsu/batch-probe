# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""In-memory cache for probe results."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

import torch
import torch.nn as nn

from batch_probe._probe import probe_batch_size

_cache: dict[str, int] = {}


def _make_key(
    model: nn.Module,
    input_fn: Callable[[int], Dict[str, torch.Tensor]],
    mode: str,
) -> str:
    """Build a cache key from model identity and input shape."""
    # Model class + param count gives a stable identity
    model_id = f"{model.__class__.__name__}_{sum(p.numel() for p in model.parameters())}"

    # Probe input shapes at batch=1
    try:
        sample = input_fn(1)
        shapes = "_".join(f"{k}:{tuple(v.shape)}:{v.dtype}" for k, v in sorted(sample.items()))
        # Clean up sample tensors
        del sample
    except Exception:
        shapes = "unknown"

    return f"{model_id}__{mode}__{shapes}"


def cached_probe(
    model: nn.Module,
    input_fn: Callable[[int], Dict[str, torch.Tensor]],
    *,
    mode: Literal["train", "infer"] = "train",
    **kwargs: Any,
) -> int:
    """Like :func:`probe_batch_size` but caches results.

    Same arguments as :func:`probe_batch_size`. Returns a cached result
    if the same model class, parameter count, input shapes, and mode
    have been probed before.
    """
    key = _make_key(model, input_fn, mode)
    if key not in _cache:
        _cache[key] = probe_batch_size(model, input_fn, mode=mode, **kwargs)
    return _cache[key]


def clear_cache() -> None:
    """Clear all cached probe results."""
    _cache.clear()

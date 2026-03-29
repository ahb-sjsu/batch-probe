# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Generic GPU memory probe — works with any framework (CuPy, JAX, PyTorch, raw CUDA)."""

from __future__ import annotations

import gc
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)


def _gpu_cleanup_generic(backend: str = "auto") -> None:
    """Free GPU memory across frameworks."""
    gc.collect()

    if backend in ("auto", "torch"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    if backend in ("auto", "cupy"):
        try:
            import cupy as cp
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            pinned_pool = cp.get_default_pinned_memory_pool()
            pinned_pool.free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
        except ImportError:
            pass

    if backend in ("auto", "jax"):
        try:
            import jax
            jax.clear_caches()
        except (ImportError, AttributeError):
            pass


def probe(
    work_fn: Callable[[int], None],
    *,
    low: int = 1,
    high: int = 1_000_000,
    headroom: float = 0.2,
    backend: str = "auto",
    gpu_id: int = 0,
    verbose: bool = True,
) -> int:
    """Find the maximum batch size for any GPU workload via binary search.

    Unlike ``probe_batch_size`` (which requires a PyTorch nn.Module), this
    function accepts any callable that takes a batch size and runs the
    workload on the GPU. If it raises an OOM error, binary search continues.

    Args:
        work_fn: A callable ``f(batch_size: int) -> None`` that runs your
            GPU workload at the given batch size. It should allocate GPU
            memory proportional to ``batch_size`` and raise an
            ``OutOfMemoryError`` (or ``RuntimeError`` with "out of memory")
            if the GPU is full. Any return value is ignored.
        low: Minimum batch size to try.
        high: Starting upper bound for binary search.
        headroom: Safety margin. ``0.2`` means return ``int(max * 0.8)``.
        backend: ``"auto"``, ``"torch"``, ``"cupy"``, or ``"jax"``.
            Controls which framework's memory pool is cleaned between probes.
        gpu_id: GPU device index (used for logging only; ``work_fn`` should
            target the correct device internally).
        verbose: Print progress.

    Returns:
        Safe batch size (``int``), guaranteed ``>= low``.

    Example — CuPy::

        import cupy as cp
        from batch_probe import probe

        def my_work(n):
            x = cp.random.randn(n, 1000, dtype=cp.float64)
            y = cp.linalg.svd(x, compute_uv=False)
            cp.cuda.runtime.deviceSynchronize()

        batch = probe(my_work, low=100, high=500_000, backend="cupy")

    Example — PyTorch (without nn.Module)::

        import torch
        from batch_probe import probe

        def my_work(n):
            x = torch.randn(n, 768, device="cuda")
            y = x @ x.T  # big matmul
            torch.cuda.synchronize()

        batch = probe(my_work, low=1, high=100_000, backend="torch")
    """
    # Detect OOM exceptions for each framework
    oom_exceptions = [MemoryError]
    try:
        import torch
        oom_exceptions.append(torch.cuda.OutOfMemoryError)
    except (ImportError, AttributeError):
        pass
    try:
        import cupy as cp
        oom_exceptions.append(cp.cuda.memory.OutOfMemoryError)
    except (ImportError, AttributeError):
        pass
    oom_tuple = tuple(oom_exceptions)

    best = low

    if verbose:
        print(
            f"batch-probe: probing (range=[{low}, {high}], "
            f"headroom={headroom:.0%}, backend={backend})...",
            end="",
            flush=True,
        )

    while low <= high:
        mid = (low + high) // 2

        try:
            _gpu_cleanup_generic(backend)
            work_fn(mid)
            best = mid
            low = mid + 1
        except oom_tuple:
            high = mid - 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
            else:
                raise
        finally:
            _gpu_cleanup_generic(backend)

    safe = max(1, int(best * (1.0 - headroom)))

    if verbose:
        print(f" max={best}, safe={safe}")

    return safe

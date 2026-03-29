# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""batch-probe: Find the maximum batch size that fits in GPU memory.

Two APIs:
  - ``probe_batch_size(model, input_fn)`` — PyTorch nn.Module (original API)
  - ``probe(work_fn)`` — any GPU workload (CuPy, JAX, PyTorch, raw CUDA)
"""

from batch_probe._probe_generic import probe

__all__ = ["probe"]
__version__ = "0.3.0"

# PyTorch-specific API — lazy import so torch is not required
try:
    from batch_probe._probe import probe_batch_size
    from batch_probe._cache import cached_probe, clear_cache
    __all__ += ["probe_batch_size", "cached_probe", "clear_cache"]
except ImportError:
    pass

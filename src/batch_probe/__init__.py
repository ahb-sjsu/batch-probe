# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""batch-probe: GPU memory probing and thermal-aware CPU thread control.

Four APIs:
  - ``probe_batch_size(model, input_fn)`` — PyTorch nn.Module (original API)
  - ``probe(work_fn)`` — any GPU workload (CuPy, JAX, PyTorch, raw CUDA)
  - ``probe_threads(work_fn)`` — one-shot thermal thread search
  - ``ThermalController(target_temp)`` — continuous Kalman-filtered thread control
"""

from batch_probe._probe_generic import probe
from batch_probe._thermal import probe_threads
from batch_probe._thermal_controller import ThermalController
from batch_probe._thermal_jobs import ThermalJobManager

__all__ = ["probe", "probe_threads", "ThermalController", "ThermalJobManager"]
__version__ = "0.4.0"

# PyTorch-specific API — lazy import so torch is not required
try:
    from batch_probe._probe import probe_batch_size
    from batch_probe._cache import cached_probe, clear_cache

    __all__ += ["probe_batch_size", "cached_probe", "clear_cache"]
except ImportError:
    pass

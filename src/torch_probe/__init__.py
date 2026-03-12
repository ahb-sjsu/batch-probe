# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""torch-probe: Find the maximum batch size that fits in GPU memory."""

from torch_probe._cache import cached_probe, clear_cache
from torch_probe._probe import probe_batch_size

__all__ = ["probe_batch_size", "cached_probe", "clear_cache"]
__version__ = "0.1.0"

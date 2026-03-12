# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Core binary-search GPU memory probe."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
import torch.nn as nn

from batch_probe._cleanup import gpu_cleanup


def _extract_loss(outputs: Any) -> torch.Tensor:
    """Extract a scalar loss from model outputs for the backward pass.

    Handles:
      - HuggingFace ModelOutput / dataclass with .loss attribute
      - dict with "loss" key
      - plain Tensor
      - tuple (uses first element)
      - dict without "loss" (uses first value)
    """
    # .loss attribute (HuggingFace ModelOutput, dataclasses)
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss

    # dict with "loss" key
    if isinstance(outputs, dict):
        if "loss" in outputs:
            return outputs["loss"]
        # Fall back to first tensor value
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.mean()

    # plain Tensor
    if isinstance(outputs, torch.Tensor):
        return outputs.mean()

    # tuple / list
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first.mean()

    raise TypeError(
        f"Cannot extract a loss from model output of type {type(outputs)}. "
        "Ensure your model returns a tensor, a dict with a 'loss' key, "
        "or an object with a .loss attribute."
    )


def probe_batch_size(
    model: nn.Module,
    input_fn: Callable[[int], Dict[str, torch.Tensor]],
    *,
    mode: Literal["train", "infer"] = "train",
    low: int = 1,
    high: int = 4096,
    headroom: float = 0.2,
    device: Optional[Union[torch.device, str]] = None,
    verbose: bool = True,
) -> int:
    """Find the maximum batch size that fits in GPU memory.

    Uses binary search with OOM recovery. Tries a forward pass (and backward
    pass in train mode) at each candidate batch size. Returns the largest
    successful size minus a safety margin.

    Args:
        model: Any ``nn.Module``, already on the target device.
        input_fn: A callable that takes a batch size ``int`` and returns a dict
            of tensors to pass as ``**kwargs`` to ``model()``. Tensors must
            already be on the correct device.
        mode: ``"train"`` runs forward + backward (2-4x more memory).
            ``"infer"`` runs forward only under ``torch.no_grad()``.
        low: Minimum batch size to try (and the floor for the return value).
        high: Starting upper bound for binary search.
        headroom: Fraction of headroom to subtract. ``0.2`` (default) means
            the returned batch size is ``int(max_successful * 0.8)``.
        device: Device to check. Defaults to the device of the model's first
            parameter. On CPU the probe still runs but skips CUDA-specific cleanup.
        verbose: Print probe progress.

    Returns:
        Safe batch size (``int``), guaranteed ``>= low``.

    Example::

        from batch_probe import probe_batch_size

        batch_size = probe_batch_size(
            model,
            lambda bs: {
                "input_ids": torch.zeros(bs, 512, dtype=torch.long, device="cuda"),
                "attention_mask": torch.ones(bs, 512, dtype=torch.long, device="cuda"),
            },
        )
    """
    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device) if isinstance(device, str) else device

    is_cuda = device.type == "cuda"

    # Save and restore model state
    was_training = model.training
    best = low

    if verbose:
        print(
            f"torch-probe: probing batch size (mode={mode}, range=[{low}, {high}], "
            f"headroom={headroom:.0%})...",
            end="",
            flush=True,
        )

    while low <= high:
        mid = (low + high) // 2
        success = False
        inputs = None

        try:
            if is_cuda:
                gpu_cleanup()
            inputs = input_fn(mid)

            if mode == "train":
                model.train()
                outputs = model(**inputs)
                loss = _extract_loss(outputs)
                loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                model.eval()
                with torch.no_grad():
                    model(**inputs)

            success = True

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            if "out of memory" not in err_msg and "cuda" not in err_msg:
                # Not an OOM — re-raise
                model.train(was_training)
                raise
        finally:
            # Always clean up tensors
            if inputs is not None:
                del inputs
            if is_cuda:
                gpu_cleanup()

        if success:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    # Restore model state
    model.train(was_training)

    safe = max(1, int(best * (1.0 - headroom)))
    # Never go below the user's requested minimum
    safe = max(safe, 1)

    if verbose:
        print(f" max={best}, safe={safe}")

    return safe

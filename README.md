# torch-probe

Find the maximum batch size that fits in GPU memory.

Binary search with OOM recovery, configurable safety headroom, no framework required.

## The Problem

Every ML practitioner has done this:

```
batch_size = 64   # OOM
batch_size = 32   # OOM
batch_size = 16   # OOM
batch_size = 8    # works... but am I leaving GPU memory on the table?
```

`torch-probe` automates this. It binary-searches for the largest batch size your model can handle, with a safety margin so you don't OOM during real training.

## Install

```bash
pip install torch-probe
```

## Quick Start

```python
from torch_probe import probe_batch_size

batch_size = probe_batch_size(
    model,
    lambda bs: {
        "input_ids": torch.zeros(bs, 512, dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(bs, 512, dtype=torch.long, device="cuda"),
    },
)
# torch-probe: probing batch size (mode=train, range=[1, 4096], headroom=20%)... max=6, safe=4
```

That's it. Three lines. Works with any `nn.Module`.

## Usage

### Encoder models (BERT, RoBERTa, etc.)

```python
batch_size = probe_batch_size(
    model,
    lambda bs: {
        "input_ids": torch.zeros(bs, 128, dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(bs, 128, dtype=torch.long, device="cuda"),
    },
    mode="train",
)
```

### Seq2seq models (T5, BART, etc.)

```python
batch_size = probe_batch_size(
    model,
    lambda bs: {
        "input_ids": torch.zeros(bs, 512, dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(bs, 512, dtype=torch.long, device="cuda"),
        "labels": torch.zeros(bs, 512, dtype=torch.long, device="cuda"),
    },
    mode="train",
)
```

### Vision models

```python
batch_size = probe_batch_size(
    model,
    lambda bs: {"x": torch.randn(bs, 3, 224, 224, device="cuda")},
    mode="infer",
)
```

### Inference-only probing

Inference uses ~2-4x less memory than training (no gradients stored):

```python
infer_batch = probe_batch_size(model, input_fn, mode="infer")
train_batch = probe_batch_size(model, input_fn, mode="train")
# infer_batch >> train_batch
```

### Custom headroom

Default is 20% safety margin. Adjust for your risk tolerance:

```python
# Conservative (40% headroom) — for long training runs
batch_size = probe_batch_size(model, input_fn, headroom=0.4)

# Aggressive (5% headroom) — squeeze every last sample
batch_size = probe_batch_size(model, input_fn, headroom=0.05)
```

### Caching

Use `cached_probe` to avoid re-probing the same model:

```python
from torch_probe import cached_probe, clear_cache

batch_size = cached_probe(model, input_fn, mode="train")  # probes
batch_size = cached_probe(model, input_fn, mode="train")  # cache hit

clear_cache()  # reset if model changed
```

## How It Works

1. Binary search between `low` (default 1) and `high` (default 4096)
2. At each midpoint, create dummy tensors via your `input_fn`
3. Run a forward pass (+ backward pass in train mode)
4. If OOM: upper bound ← midpoint − 1, clean GPU memory
5. If success: lower bound ← midpoint + 1
6. Return `int(max_successful × (1 − headroom))`

The OOM recovery uses `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` to fully reclaim memory between iterations.

## vs. Alternatives

| Feature | torch-probe | Lightning BatchSizeFinder | HF `auto_find_batch_size` |
|---|---|---|---|
| Works with raw PyTorch | Yes | No (needs LightningModule) | No (needs HF Trainer) |
| Algorithm | Binary search | Power-of-2 scaling | Halve on OOM |
| Configurable headroom | Yes | No | No |
| Train + infer modes | Yes | Train only | Train only |
| Dependencies | torch only | pytorch-lightning | accelerate |

## API Reference

### `probe_batch_size(model, input_fn, *, mode, low, high, headroom, device, verbose)`

Find the maximum safe batch size.

- **model** (`nn.Module`): Your model, already on the target device.
- **input_fn** (`Callable[[int], dict[str, Tensor]]`): Takes batch size, returns dict of tensors for `model(**inputs)`.
- **mode** (`"train"` | `"infer"`): Train mode runs forward + backward. Default: `"train"`.
- **low** (`int`): Minimum batch size. Default: `1`.
- **high** (`int`): Upper bound for search. Default: `4096`.
- **headroom** (`float`): Safety margin. Default: `0.2` (20%).
- **device** (`str | torch.device | None`): Override device. Default: model's device.
- **verbose** (`bool`): Print progress. Default: `True`.

Returns: `int` — safe batch size.

### `cached_probe(model, input_fn, *, mode, **kwargs)`

Same as `probe_batch_size` but caches results keyed on model class, param count, input shapes, and mode.

### `clear_cache()`

Clear all cached probe results.

## License

MIT

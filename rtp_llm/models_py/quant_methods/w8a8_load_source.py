"""CPU scratch view of a floating checkpoint as W8A8 weights and channel scales.

Both real and virtual keys are consumed by the usual NewLoader parameter
owners. Quantize each requested row over its complete K axis *before* TP
slicing. Never materialize the floating stacked expert tensor.
"""

import re

import torch

from rtp_llm.utils.w8a8_int8_quant import (
    STACKED_EXPERT_SUFFIXES,
    is_recipe_weight,
    quantize_weight_per_output_channel,
    scale_name_for,
)


class W8A8LoadSource:
    def __init__(self, chunk_rows, weights):
        self.chunk_rows = chunk_rows
        self.weights = weights

    @classmethod
    def from_files(cls, files, runtime, catalog=None):
        from safetensors import safe_open

        rows = runtime.source_config.load_chunk_rows
        if type(rows) is not int or rows <= 0:
            raise ValueError("W8A8 load_chunk_rows must be a positive integer")
        if not files or any(not path.endswith(".safetensors") for path in files):
            raise ValueError(
                "load-time W8A8 requires a floating safetensors checkpoint"
            )
        names, weights = set(), {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = handle.keys()
                if catalog is not None:
                    keys = catalog.shard_keys(path, keys)
                for name in keys:
                    if name in names:
                        raise ValueError(f"Duplicate W8A8 checkpoint key {name}")
                    names.add(name)
                    if re.search(
                        r"\.mlp\.experts\.\d+\.(?:up|gate|down)_proj\.weight$", name
                    ):
                        raise ValueError(
                            "load-time W8A8 requires fused stacked experts, not split experts"
                        )
                    if not is_recipe_weight(name):
                        continue
                    source = handle.get_slice(name)
                    shape = tuple(source.get_shape())
                    expected_dims = 3 if name.endswith(STACKED_EXPERT_SUFFIXES) else 2
                    if len(shape) != expected_dims or min(shape) <= 0:
                        raise ValueError(
                            f"Invalid W8A8 source shape for {name}: {shape}"
                        )
                    if source.get_dtype() not in ("BF16", "F16", "F32"):
                        raise ValueError(
                            f"load-time W8A8 requires a floating weight: {name}"
                        )
                    if not runtime.is_layer_ignored(name.removesuffix(".weight")):
                        weights[name] = scale_name_for(name)
        if not weights:
            raise ValueError(
                "load-time W8A8 found no weights covered by the Qwen MoE recipe"
            )
        conflicts = {scale_name_for(k) for k in names if is_recipe_weight(k)} & names
        if conflicts:
            raise ValueError(f"load-time W8A8 source already contains {min(conflicts)}")
        return cls(rows, weights)

    def shard(self, handle, allowed):
        return _Shard(self, handle, allowed)

    def validate_overlay(self, overlay):
        if overlay is None:
            return
        # ptuning_path is a PyTorch archive. Main's online recipe only
        # supports safetensor slices; never pair an unconverted shadow
        # weight with a scale calculated from the base file.
        scale_names = set(self.weights.values())
        for key in overlay.names:
            if is_recipe_weight(key) or key in scale_names:
                raise ValueError(
                    "load-time W8A8 requires recipe weights and scales in safetensors, not a finetune overlay"
                )


class _Rows:
    def __init__(self, source, name, rows):
        self.source, self.name, self.rows = source, name, rows
        self.shape = tuple(source.get_shape())
        # Allocate small scale caches only for experts actually assigned to the rank.
        self.scales = {}

    def read(self, key, scale_only):
        shape = (*self.shape[:-1], 1) if scale_only else self.shape
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > len(shape):
            raise IndexError("too many indices for W8A8 source")
        key += (slice(None),) * (len(shape) - len(key))
        cuts, squeeze = [], []
        for dim, (item, size) in enumerate(zip(key, shape)):
            if isinstance(item, int):
                item = item + size if item < 0 else item
                if not 0 <= item < size:
                    raise IndexError("W8A8 source index out of range")
                squeeze.append(dim)
                item = slice(item, item + 1)
            if not isinstance(item, slice) or item.step not in (None, 1):
                raise ValueError("W8A8 source requires contiguous slices")
            lo, hi, _ = item.indices(size)
            cuts.append(slice(lo, max(lo, hi)))
        out = torch.empty(
            [x.stop - x.start for x in cuts],
            dtype=torch.float32 if scale_only else torch.int8,
        )
        experts = range(cuts[0].start, cuts[0].stop) if len(shape) == 3 else (None,)
        rows, cols = cuts[-2:]
        for slot, expert in enumerate(experts):
            prefix, dst = ((), ()) if expert is None else ((expert,), (slot,))
            if expert not in self.scales:
                self.scales[expert] = (
                    torch.empty((shape[-2], 1), dtype=torch.float32),
                    torch.zeros(shape[-2], dtype=torch.bool),
                )
            saved, valid = self.scales[expert]
            for start in range(rows.start, rows.stop, self.rows):
                stop = min(start + self.rows, rows.stop)
                span = slice(start, stop)
                target = (*dst, slice(start - rows.start, stop - rows.start))
                if scale_only and valid[span].all():
                    out[target] = saved[span, cols]
                    continue
                raw = self.source[(*prefix, span, slice(None))]
                if not torch.isfinite(raw).all():
                    raise ValueError(f"non-finite W8A8 source values in {self.name}")
                q, scales = quantize_weight_per_output_channel(raw)
                saved[span] = scales
                valid[span] = True
                out[target] = (scales if scale_only else q)[:, cols]
        for dim in reversed(squeeze):
            out = out.squeeze(dim)
        return out


class _Slice:
    def __init__(self, rows, scale):
        self.rows, self.scale = rows, scale

    def get_shape(self):
        return (*self.rows.shape[:-1], 1) if self.scale else self.rows.shape

    def get_dtype(self):
        return "F32" if self.scale else "I8"

    def __getitem__(self, key):
        return self.rows.read(key, self.scale)


class _Shard:
    def __init__(self, config, handle, allowed):
        self.config, self.handle = config, handle
        self.physical = [k for k in handle.keys() if allowed is None or k in allowed]
        self.sources = {
            config.weights[k]: k for k in self.physical if k in config.weights
        }
        self.last = None

    def keys(self):
        for name in self.physical:
            yield name
            if name in self.config.weights:
                yield self.config.weights[name]

    def get_slice(self, name):
        weight = self.sources.get(name, name)
        if weight not in self.config.weights:
            return self.handle.get_slice(name)
        if self.last is None or self.last.name != weight:
            self.last = _Rows(
                self.handle.get_slice(weight), weight, self.config.chunk_rows
            )
        return _Slice(self.last, name != weight)

    def get_tensor(self, name):
        if name not in self.config.weights and name not in self.sources:
            return self.handle.get_tensor(name)
        return self.get_slice(name)[:]

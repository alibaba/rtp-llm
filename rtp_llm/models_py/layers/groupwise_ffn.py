"""Pad the FFN intermediate axis before TP, while reading only real checkpoint lanes."""

import math

import torch

from rtp_llm.models_py.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.quant_methods.groupwise import groupwise_settings, uses_gptq_v2
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor, ReadSpec, unwrap_checkpoint_tensor


def _linear(cls, padded_cls, width, widths, kwargs):
    config = kwargs.get("quant_config")
    if config is None or config.quant_type not in ("awq", "gptq"):
        return cls(**widths(width), **kwargs)
    method, bits, group = groupwise_settings(config)
    tp = kwargs["tp_size"]
    if type(tp) is not int or tp <= 0:
        raise ValueError(f"Invalid FFN TP size {tp!r}")
    parent = kwargs["prefix"].rsplit(".", 1)[0]
    down = not config.is_layer_ignored(parent + ".down_proj")
    gate_up = not (
        config.is_layer_ignored(parent + ".gate_up_proj")
        or all(config.is_layer_ignored(parent + "." + part) for part in ("gate_proj", "up_proj"))
    )
    if not (down or gate_up):
        return cls(**widths(width), **kwargs)
    alignment = tp * math.lcm(
        group if down else 1, 32 // bits, 8 if method == "awq" and gate_up else 1
    )
    padded = (width + alignment - 1) // alignment * alignment
    if padded == width:
        return cls(**widths(width), **kwargs)
    layer = padded_cls(**widths(padded), **kwargs)
    layer.checkpoint_intermediate_size = width
    layer.padded_intermediate_size = padded
    return layer


def merged_ffn_linear(*, output_size, **kwargs):
    parts = len(kwargs["shard_names"])
    if output_size % parts:
        raise ValueError("Merged FFN projections must have equal widths")
    return _linear(MergedColumnParallelLinear, _PaddedGateUp, output_size // parts,
                   lambda width: {"output_size": parts * width}, kwargs)


def row_ffn_linear(*, input_size, **kwargs):
    return _linear(RowParallelLinear, _PaddedDown, input_size,
                   lambda width: {"input_size": width}, kwargs)


class _PaddedFFN:
    def _axis(self, name, shape):
        """Return (axis, channels per stored element, projection count)."""
        part = name.rsplit(".", 1)[-1]
        up = isinstance(self, MergedColumnParallelLinear)
        count = len(self.shard_names) if up and "." not in name else 1
        if part == "weight":  # Unquantized exclusion inside a groupwise model.
            width = count * self.checkpoint_intermediate_size
            axis = 0 if up else 1
            if len(shape) == 2 and shape[axis] != width:
                axis = 1 - axis
            return axis, 1, count
        if part == "bias":
            return (0, 1, count) if up else None
        if part == "g_idx":
            return None if up else (0, 1, 1)
        if part not in ("qweight", "qzeros", "scales"):
            return None
        method, bits, group = groupwise_settings(self.quant_config)
        if up:
            packed = part == "qzeros" or (method == "awq" and part == "qweight")
            return 1, (32 // bits if packed else 1), count
        return 0, (group if part != "qweight" else 32 // bits if method == "gptq" else 1), 1

    def _plans(self, name, shape):
        info = self._axis(name, shape)
        if info is None:
            return None
        axis, unit, count = info
        raw = (self.checkpoint_intermediate_size + unit - 1) // unit
        padded = self.padded_intermediate_size // unit
        if axis >= len(shape) or shape[axis] != count * raw:
            raise ValueError(f"Invalid checkpoint FFN shape for {name}: {shape}")
        physical = list(shape)
        physical[axis] = count * padded
        dest = super().checkpoint_read_spec(name, tuple(physical))
        # Fused gate+up has two independent padded regions. The normal fused
        # loader likewise reads it whole and splits the two projections locally.
        if dest is None or not dest.is_local_shard:
            return ReadSpec.full(name, shape), dest, axis, count, padded
        slices = list(dest.slices)
        span = slices[axis]
        start, stop, _ = span.indices(physical[axis])
        if count != 1:
            raise ValueError(f"Cannot linearly shard fused FFN source {name}")
        slices[axis] = slice(min(start, raw), min(stop, raw))
        return ReadSpec(name, name, shape, slices=tuple(slices), partition="tp"), dest, axis, 1, stop - start

    def checkpoint_read_spec(self, name, shape):
        plans = self._plans(name, tuple(shape))
        return plans[0] if plans else super().checkpoint_read_spec(name, shape)

    def load_weights(self, weights):
        for name, value in weights.items():
            tensor, spec = unwrap_checkpoint_tensor(value)
            shape = spec.source_shape if spec is not None else tuple(tensor.shape)
            plans = self._plans(name, shape)
            if plans is not None:
                source, dest, axis, count, width = plans
                if spec is not None and spec.for_checkpoint_name(name) != source:
                    raise ValueError(f"Incorrect padded FFN read slice for {name}")
                if spec is None and source.slices is not None:
                    tensor = tensor[source.slices]
                if source.slices is not None:
                    expected = tuple(
                        len(range(*s.indices(n))) for s, n in zip(source.slices, shape)
                    )
                    if tuple(tensor.shape) != expected:
                        raise ValueError(f"Truncated padded FFN checkpoint slice for {name}")
                part = name.rsplit(".", 1)[-1]
                if tensor.shape[axis] < count * width:
                    method, bits, group = groupwise_settings(self.quant_config)
                    # Positive scales and exactly zero decoded weights for all
                    # added groups/output channels. A partially occupied down
                    # group keeps its original zero point; its extra activation
                    # lanes are zero from gate/up. GPTQ v1 stores zero_point - 1.
                    fill = 1 if part == "scales" else 0
                    if part == "qweight" and method == "gptq" and not uses_gptq_v2(self.quant_config):
                        fill = 0xffffffff // ((1 << bits) - 1)
                    out_shape = list(tensor.shape)
                    out_shape[axis] = count * width
                    result = tensor.new_full(out_shape, fill)
                    if part == "g_idx":
                        result = torch.arange(width, dtype=tensor.dtype, device=tensor.device) // group
                    actual = tensor.shape[axis] // count
                    for i in range(count):
                        result.narrow(axis, i * width, actual).copy_(tensor.narrow(axis, i * actual, actual))
                    tensor = result
                value = LoadedCheckpointTensor(tensor, dest) if dest is not None and dest.is_local_shard else tensor
            super().load_weights({name: value})


class _PaddedGateUp(_PaddedFFN, MergedColumnParallelLinear):
    def _checked_lora_factors(self, a, b, input_size, output_size):
        # PEFT describes the checkpoint, not TP's padded parameter. A fused B
        # contains [gate, up]; insert zeros after EACH projection before the
        # normal merged-column sharder selects this rank's gate and up rows.
        parts = output_size // self.padded_intermediate_size
        width = self.checkpoint_intermediate_size
        a, b = super()._checked_lora_factors(a, b, input_size, parts * width)
        b = torch.nn.functional.pad(
            b.reshape(parts, width, -1), (0, 0, 0, self.padded_intermediate_size - width)
        )
        return a, b.flatten(0, 1)


class _PaddedDown(_PaddedFFN, RowParallelLinear):
    def _checked_lora_factors(self, a, b, input_size, output_size):
        width = self.checkpoint_intermediate_size
        a, b = super()._checked_lora_factors(a, b, width, output_size)
        return torch.nn.functional.pad(a, (0, input_size - width)), b

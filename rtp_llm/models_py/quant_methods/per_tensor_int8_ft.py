"""Decode the per-tensor INT8 writer's rank-local byte layout, without requantizing."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding


def static_int8_bindings(
    linear, source, destination, projection, layer_prefix, swizzled
):
    if swizzled:
        raise ValueError("Swizzled static INT8 FT storage has no declared decoder")
    rows, columns = linear.weight.shape
    scale_name = source.removesuffix(".kernel") + ".weight_only_quant_scale"
    stem = {
        "qkv": "pre_layernorm_weights",
        "out": "self_attention_weights.attention_output_weight",
        "gate_up": "post_layernorm_weights",
        "down": "ffn_weights.intermediate_weight2",
    }[projection]
    base = layer_prefix + stem

    def kernel(values):
        (value,) = values
        # Old writer first transposes input-major HF matrices into output-major
        # rows, then *reshapes* the contiguous bytes (not a second transpose).
        if projection == "gate_up":
            # transpose_w13 concatenates the two transposes along INPUT.
            half = rows // 2
            if value.shape != (2 * columns, half):
                raise ValueError(
                    f"Invalid static INT8 FT gate/up layout for {source}: {value.shape}"
                )
            gate, up = value.reshape(half, 2 * columns).chunk(2, dim=1)
            return torch.cat((gate, up))
        if value.shape != (columns, rows):
            raise ValueError(
                f"Invalid static INT8 FT kernel layout for {source}: {value.shape}"
            )
        return value.reshape(rows, columns)

    def positive(value):
        if (
            value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or not (value > 0).all()
        ):
            raise ValueError(
                f"Static INT8 FT scales must be finite, positive FP32: {source}"
            )
        return value.reshape(-1)

    def scales(values):
        (raw,) = values
        value = positive(raw)
        if projection == "gate_up" and raw.shape == (rows // 2, 2):
            # Historical transpose_w13 on [1, width] gate/up input scales
            # wrote adjacent gate,up pairs per channel, not two flat halves.
            value = raw.T.reshape(-1)
        if projection == "gate_up" and value.numel() == 2:
            value = value.repeat_interleave(rows // 2)
        if value.numel() not in (1, rows):
            raise ValueError(
                f"Invalid static INT8 FT channel scales for {source}: {raw.shape}"
            )
        return value.expand(rows).reshape(rows, 1)

    def step(values):
        inverse, actual = map(positive, values)
        if (
            inverse.numel() != 1
            or actual.numel() != 1
            or not torch.allclose(
                inverse * actual, torch.ones_like(actual), atol=0, rtol=1e-5
            )
        ):
            raise ValueError(
                f"Invalid reciprocal static INT8 FT activation pair for {source}"
            )
        return actual

    return (
        RuntimeTensorBinding((source,), destination + ".weight", kernel),
        RuntimeTensorBinding((scale_name,), destination + ".weight_scale", scales),
        RuntimeTensorBinding(
            (base + ".static_quant", base + ".static_quant_reciprocal"),
            destination + ".input_scale",
            step,
        ),
    )

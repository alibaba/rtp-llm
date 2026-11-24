"""BERT's calibrated W8A8 FT bytes, as emitted by PerTensorInt8QuantWeight."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding

from .ft_checkpoint import vector


def bind_static_projection(bindings, layer, source, target, activation):
    def kernel(values):
        packed = values[0]
        # Export reshapes an already output-major local matrix to [K,N].
        # This is a flat storage view, not a mathematical transpose.
        shape = tuple(layer.weight.shape)
        if packed.dtype != torch.int8 or tuple(packed.shape) != shape[::-1]:
            raise ValueError(f"Invalid static INT8 FT kernel for {target}")
        return packed.reshape(shape)

    def scale(values):
        value = values[0]
        if value.ndim > 2 or (value.ndim == 2 and 1 not in value.shape):
            raise ValueError(f"Expected static INT8 FT scale vector for {target}")
        steps = value.reshape(-1)
        if (
            steps.dtype != torch.float32
            or not torch.isfinite(steps).all()
            or not (steps > 0).all()
        ):
            raise ValueError(f"Invalid static INT8 FT weight scales for {target}")
        return steps.expand_as(layer.weight_scale) if steps.numel() == 1 else steps

    def input_scale(values):
        inverse, step = values
        if (
            step.dtype != torch.float32
            or inverse.dtype != torch.float32
            or step.shape != (1,)
            or inverse.shape != (1,)
            or not torch.isfinite(step).all()
            or not (step > 0).all()
            or not torch.isfinite(inverse).all()
            or not torch.allclose(inverse, step.reciprocal(), rtol=1e-5, atol=0)
        ):
            raise ValueError(
                f"Invalid static INT8 FT activation scale pair for {target}"
            )
        return step

    bindings.field(source + ".kernel", target + ".weight", kernel)
    bindings.field(source + ".bias", target + ".bias", vector)
    bindings.field(source + ".weight_only_quant_scale", target + ".weight_scale", scale)
    bindings.append(
        RuntimeTensorBinding(
            (activation + ".static_quant", activation + ".static_quant_reciprocal"),
            target + ".input_scale",
            input_scale,
        )
    )

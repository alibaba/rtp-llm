"""Format-boundary operations shared by the GDN and KDA text owners."""

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight
from rtp_llm.models_py.quant_methods.fp8 import (
    Fp8BlockLinearMethod,
    Fp8PerChannelLinearMethod,
)
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def identity(values):
    return values[0]


def matrix(values, swizzled=False):
    raw = values[0]
    if raw.ndim != 2:
        raise ValueError("FT projection must be a matrix")
    return decode_dense_weight(raw.T) if swizzled else raw.T


def convolution(values):
    value = values[0]
    if value.ndim == 3 and value.shape[1] == 1:
        return value.squeeze(1)
    if value.ndim == 2:
        return value
    raise ValueError(
        "FT convolution must have shape [channels, 1, kernel] or [channels, kernel]"
    )


def linear_bindings(checkpoint, module, source, target, scale_key, swizzled, partition):
    """The exporter writes dense [K,N] or a quant-method-specific packed pair."""
    if partition != (checkpoint.topology.tp_size, checkpoint.topology.tp_rank):
        raise ValueError("Hybrid FT projection topology does not match the producer")
    if isinstance(module.quant_method, UnquantizedLinearMethod):
        return [
            RuntimeTensorBinding(
                (source,), target + ".weight", lambda v: matrix(v, swizzled)
            )
        ]
    if isinstance(module.quant_method, Fp8BlockLinearMethod):
        from rtp_llm.models_py.quant_methods.block_fp8_ft import block_ft_bindings

        return block_ft_bindings(
            module, source, scale_key, target, checkpoint, swizzled=swizzled
        )
    if isinstance(module.quant_method, Fp8PerChannelLinearMethod):
        from rtp_llm.models_py.quant_methods.fp8_ft import PerChannelFP8FTProjection

        state = PerChannelFP8FTProjection(
            module,
            checkpoint,
            source,
            scale_key,
            "local",
            swizzled=swizzled,
            owner_partition=partition,
        )
        return [
            RuntimeTensorBinding(
                (source, scale_key),
                target + "." + attr,
                identity,
                lambda fn=fn: (fn(),),
            )
            for attr, fn in (("weight", state.weight), ("weight_scale", state.scale))
        ]
    raise ValueError(
        f"No hybrid FT producer layout for {target}: {type(module.quant_method).__name__}"
    )

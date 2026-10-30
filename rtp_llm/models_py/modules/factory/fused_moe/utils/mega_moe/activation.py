"""Explicit SiTU argument mapping for supported DeepGEMM interfaces."""

import inspect
import math


def mega_moe_functions(backend, activation):
    """Keep the kernel, weight transform and workspace in one library ABI.

    DeepGEMM 2.8.2 separates SiTU/BF16-shared experts from the generic FP8
    shared-expert implementation. Their symmetric layouts are not interchangeable.
    """
    suffix = "_k3" if activation == "situ" and callable(
        getattr(backend, "fp8_fp4_mega_moe_k3", None)
    ) else ""
    names = ("fp8_fp4_mega_moe", "transform_weights_for_mega_moe",
             "get_symm_buffer_for_mega_moe")
    functions = tuple(getattr(backend, name + suffix, None) for name in names)
    if not all(callable(function) for function in functions):
        raise RuntimeError("DeepGEMM lacks a matching MegaMoE kernel/weights/workspace API")
    return functions


def situ_kwargs(kernel, gate_beta, up_beta):
    """Bind the independent gate/up tanh scales of the SiTU kernel API."""
    if gate_beta is None or not math.isfinite(gate_beta) or gate_beta <= 0:
        raise ValueError("SiTU requires a finite positive gate beta")
    if up_beta is not None and (not math.isfinite(up_beta) or up_beta <= 0):
        raise ValueError("SiTU up beta must be finite and positive, or None")
    parameters = inspect.signature(kernel).parameters
    scale_names = ("activation_beta", "activation_linear_beta")
    if all(
        name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for name in scale_names
    ):
        # DeepGEMM's SiTU gate is beta*tanh(gate/beta)*sigmoid(gate).
        # A None linear beta leaves up unsaturated; zero is not equivalent.
        return dict(activation_beta=gate_beta, activation_linear_beta=up_beta)
    raise RuntimeError(
        "DeepGEMM SiTU requires keyword parameters activation_beta and "
        "activation_linear_beta; this kernel API is not supported"
    )

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
    if gate_beta is None or not math.isfinite(gate_beta) or gate_beta <= 0:
        raise ValueError("SiTU requires a finite positive gate beta")
    if up_beta is not None and (not math.isfinite(up_beta) or up_beta <= 0):
        raise ValueError("SiTU up beta must be finite and positive, or None")
    parameters = inspect.signature(kernel).parameters
    if {"activation_beta", "activation_linear_beta"}.issubset(parameters):
        return dict(activation_beta=gate_beta, activation_linear_beta=up_beta)
    if {"situ_beta", "situ_linear_beta"}.issubset(parameters):
        # main's kernel computes beta*tanh(gate/beta)*sigmoid(gate),
        # multiplied by linear_beta*tanh(up/linear_beta).
        if up_beta is None:
            raise RuntimeError(
                "This DeepGEMM SiTU backend requires a saturated up branch"
            )
        return dict(situ_beta=gate_beta, situ_linear_beta=up_beta)
    if {"activation_alpha", "activation_beta"}.issubset(parameters):
        # vllm-project/DeepGEMM a6bbb800, csrc/apis/mega_moe.hpp:190-193
        # explicitly defines beta=0 as an unsaturated up branch. The optional
        # native backend verifies that source/patch manifest before this call.
        return dict(activation_alpha=gate_beta,
                    activation_beta=0.0 if up_beta is None else up_beta)
    raise RuntimeError("DeepGEMM lacks a verified SiTU gate/up beta interface")

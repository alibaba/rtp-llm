"""Explicit SiTU argument mapping for supported DeepGEMM interfaces."""

import inspect
import math


def situ_kwargs(kernel, gate_beta, up_beta):
    if gate_beta is None or not math.isfinite(gate_beta) or gate_beta <= 0:
        raise ValueError("SiTU requires a finite positive gate beta")
    if up_beta is not None and (not math.isfinite(up_beta) or up_beta <= 0):
        raise ValueError("SiTU up beta must be finite and positive, or None")
    parameters = inspect.signature(kernel).parameters
    if {"situ_beta", "situ_linear_beta"}.issubset(parameters):
        # main's kernel computes beta*tanh(gate/beta)*sigmoid(gate),
        # multiplied by linear_beta*tanh(up/linear_beta).
        if up_beta is None:
            raise RuntimeError(
                "This DeepGEMM SiTU backend requires a saturated up branch"
            )
        return dict(situ_beta=gate_beta, situ_linear_beta=up_beta)
    if {"activation_alpha", "activation_beta"}.issubset(parameters):
        # The reference K3 backend calls the gate scale alpha and up scale beta.
        return dict(activation_alpha=gate_beta, activation_beta=up_beta or 0.0)
    raise RuntimeError("K3 requires a DeepGEMM backend with SiTU support")

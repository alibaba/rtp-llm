"""Qwen checkpoint storage, compute copies and model-local linear assembly.

Storage owners remain alive with the compute view. Updating graph-bound FP8 or
copied parameters requires restarting the model; no view advertises live update.
"""

import copy

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX


class PreparedWeightDict(dict):
    """Compute weights whose copied projections require a model restart to update."""


def prepare_compute_weights(weights):
    view = copy.copy(weights)
    view._fp8_storage_owner = weights

    def convert(items):
        result = PreparedWeightDict()
        for name, value in items.items():
            scale = items.get(name + FP8_STORAGE_SCALE_SUFFIX)
            compute = (
                (value.float() * scale).to(torch.bfloat16)
                if value.dtype == torch.float8_e4m3fn
                and value.ndim != 2
                and scale is not None
                else value
            )
            restart = value.dtype == torch.float8_e4m3fn or compute is not value
            if restart:
                compute._rtp_requires_restart_for_update = True
            result[name] = compute
        return result

    view.weights = [convert(items) for items in weights.weights]
    view.global_weights = convert(weights.global_weights)
    return view


def mark_projection_for_restart(weights, module):
    """Mark model-local FP8 consumers and BF16 copies as requiring restart."""
    if not isinstance(weights, PreparedWeightDict) or module is None:
        return
    compute = (
        module.compute_weight if module.compute_weight is not None else module.weight
    )
    compute._rtp_requires_restart_for_update = True


class Qwen4LinearFactory:
    @classmethod
    def create_linear_from_weights(
        cls,
        weights,
        weight_key,
        scale_key=None,
        bias_key=None,
        quant_config=None,
        hw_kernel_config=None,
        weight_scale_2_key=None,
        input_scale_key=None,
    ):
        storage_scale = weights.get(weight_key + FP8_STORAGE_SCALE_SUFFIX)
        if storage_scale is not None:
            module = Fp8Linear(
                weights[weight_key].t().contiguous(),
                storage_scale,
                weights.get(bias_key) if bias_key else None,
            )
            mark_projection_for_restart(weights, module)
            return module
        return LinearFactory.create_linear_from_weights(
            weights,
            weight_key,
            scale_key,
            bias_key,
            quant_config,
            hw_kernel_config,
            weight_scale_2_key,
            input_scale_key,
        )

    create_linear = LinearFactory.create_linear
    create_merged_linear = LinearFactory.create_merged_linear

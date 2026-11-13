"""Qwen checkpoint storage, compute copies and model-local linear assembly.

Storage owners remain alive with the compute view. Updating graph-bound FP8 or
copied parameters requires restarting the model; no view advertises live update.
"""

import copy

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.qwen4_exp.mxfp8_linear import maybe_make_mxfp8_projection
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
from rtp_llm.utils.model_weight import W


class PreparedWeightDict(dict):
    """Compute weights whose copied projections require a model restart to update."""

    def __init__(self, owner=None):
        super().__init__(owner or {})
        self._storage_owner = owner if owner is not None else {}

    def __setitem__(self, name, value):
        original = self._storage_owner.get(name)
        if original is not None and original is not value:
            # WeightManager retains the owner dictionary. Rebinding a compute
            # entry must also prevent updates to its now detached source.
            original._rtp_requires_restart_for_update = True
            value._rtp_requires_restart_for_update = True
        super().__setitem__(name, value)


def prepare_compute_weights(weights):
    view = copy.copy(weights)
    view._fp8_storage_owner = weights

    def convert(items):
        result = PreparedWeightDict(items)
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
                value._rtp_requires_restart_for_update = True
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
    def __init__(self, online_mxfp8=False):
        self.online_mxfp8 = online_mxfp8

    def create_linear_from_weights(
        self,
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
        # Router logits and the shared scalar gate retain their original BF16
        # producer. Shared FFNs use their separately configured MoE strategy.
        if (
            self.online_mxfp8
            and weight_key
            in (
                W.attn_qkv_w,
                W.attn_o_w,
                W.attn_gate_w,
                W.linear_attn_qkvz_w,
                W.linear_attn_ba_w,
                W.linear_attn_out_w,
            )
            and (scale_key is None or weights.get(scale_key) is None)
            and (weight_scale_2_key is None or weights.get(weight_scale_2_key) is None)
            and (input_scale_key is None or weights.get(input_scale_key) is None)
        ):
            module = maybe_make_mxfp8_projection(
                weights[weight_key].t(), weights.get(bias_key) if bias_key else None
            )
            if module is not None:
                weights[weight_key]._rtp_requires_restart_for_update = True
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

    def create_linear(
        self,
        weight,
        bias,
        weight_scales,
        quant_config,
        hw_kernel_config=None,
        weight_scale_2=None,
        input_scale=None,
    ):
        # Qwen3NextGatedDeltaNet builds its merged BF16 QKVZ/BA projection
        # through this entry rather than create_linear_from_weights.
        if (
            self.online_mxfp8
            and weight_scales is None
            and weight_scale_2 is None
            and input_scale is None
        ):
            module = maybe_make_mxfp8_projection(weight.t(), bias)
            if module is not None:
                weight._rtp_requires_restart_for_update = True
                return module
        return LinearFactory.create_linear(
            weight,
            bias,
            weight_scales,
            quant_config,
            hw_kernel_config,
            weight_scale_2,
            input_scale,
        )

    create_merged_linear = staticmethod(LinearFactory.create_merged_linear)

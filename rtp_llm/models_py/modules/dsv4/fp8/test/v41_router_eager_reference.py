"""Select the maintained eager epilogue for explicit SM120 Gate owners."""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe import gate as native_module


@contextmanager
def router_eager_reference(enabled, *, gates=()):
    if type(enabled) is not bool:
        raise ValueError("Eager router selection requires a bool")
    if not enabled:
        yield
        return
    gates = tuple(gates)
    if not gates or len({id(owner) for owner in gates}) != len(gates):
        raise ValueError("Select distinct Gate owners")
    for owner in gates:
        if type(owner) is not native_module.Gate:
            raise ValueError("Eager router selection requires Gate owners")
        weight, bias = owner.weight, owner.bias
        if (
            type(owner.dim) is not int
            or owner.dim not in (512, 5120)
            or owner.hash is not False
            or owner.fp32_logits is not True
            or owner.score_func != "sqrtsoftplus"
            or owner._v41_cp_ctx is not None
            or not isinstance(weight, torch.Tensor)
            or not isinstance(bias, torch.Tensor)
            or weight.ndim != 2
            or weight.shape[1] != owner.dim
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or not weight.is_contiguous()
            or bias.shape != (weight.shape[0],)
            or bias.dtype != torch.float32
            or bias.device != weight.device
            or not bias.is_contiguous()
            or type(owner.topk) is not int
            or not 0 < owner.topk <= weight.shape[0]
            or type(owner.route_scale) not in (int, float)
            or not math.isfinite(owner.route_scale)
            or owner.route_scale <= 0
            or torch.cuda.get_device_capability(weight.device) != (12, 0)
        ):
            raise ValueError("Eager router selection requires admitted SM120 Gates")
    selected = {id(owner) for owner in gates}
    active = ContextVar("v41_selected_eager_router", default=False)
    original_forward = native_module.Gate._forward_local
    original_policy = native_module._use_fused_gate

    def forward(owner, x, input_ids=None):
        chosen = id(owner) in selected
        if chosen and (
            x.ndim != 2
            or x.shape[1] != owner.dim
            or x.dtype != torch.bfloat16
            or x.device != owner.weight.device
            or not x.is_contiguous()
        ):
            raise ValueError(
                "Eager router selection requires contiguous BF16 token rows"
            )
        # Reset to this owner's membership: a nested unselected call must not
        # inherit a selected parent's policy. ContextVar also isolates threads.
        token = active.set(chosen)
        try:
            return original_forward(owner, x, input_ids)
        finally:
            active.reset(token)

    def policy(score_func, rows):
        return False if active.get() else original_policy(score_func, rows)

    with patch.object(native_module.Gate, "_forward_local", forward), patch.object(
        native_module, "_use_fused_gate", policy
    ):
        yield

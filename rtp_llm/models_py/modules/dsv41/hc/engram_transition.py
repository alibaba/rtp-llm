"""Opt-in Engram transition with independent producer and row-reduction owners.

The optional backend supplies WKV/Post, Gate and pre-only CUDA operations.
Lookup and input quantization stay under the existing Engram/WKV contracts.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _backend():
    try:
        module = importlib.import_module("rtp_engram_fusion")
    except ModuleNotFoundError as error:
        if error.name == "rtp_engram_fusion":
            return None
        raise
    return module if getattr(module, "ABI_VERSION", None) == 1 else None


def enabled() -> bool:
    return os.environ.get("DSV41_ENGRAM_RETAINED_FUSION", "0") == "1" and (
        _backend() is not None
    )


def try_fused_engram_post_pre(layer, pending):
    """Return a native-QKV prepared tuple, or leave all pending inputs intact.

    A state owns scratch for one layer, shape and CUDA stream. New stream/shape
    states must warm up before capture; a capture miss keeps the native path.
    """
    if not enabled() or torch.is_grad_enabled():
        return None
    from rtp_llm.models_py.modules.dsv41.engram import Engram
    from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
    from rtp_llm.models_py.modules.dsv41.hc.v41_mega_mhc import _units_supported
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

    module = layer.engram
    x, residual, post, comb, previous = pending
    next_hc = layer.attn_hc
    if (
        type(module) is not Engram
        or type(module.wkv) is not V41MXFP8Linear
        or type(previous) is not DelayedHCUnit
        or type(next_hc) is not DelayedHCUnit
        or next_hc._previous_ref is None
        or next_hc._previous_ref() is not previous
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim not in (2, 3)
        or x.shape[-1] != 5120
        or residual.shape != (*x.shape[:-1], 4, 5120)
        or residual.dtype != x.dtype
        or residual.device != x.device
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or module.wkv.K != 6144
        or module.wkv.N != 25600
        or previous.pre_mix_out is None
        or layer.engram_hashes is None
    ):
        return None
    m = x.numel() // 5120
    if not 1 <= m <= 512:
        return None
    if not _units_supported(previous, next_hc, layer.attn_norm, x.device):
        return None
    if any(
        value.device != x.device
        or value.dtype != torch.float32
        or value.numel() != count
        or not value.is_contiguous()
        for value, count in (
            (post, m * 4),
            (comb, m * 16),
            (previous.pre_mix_out, m * 4),
        )
    ):
        return None
    if (
        torch.cuda.get_device_capability(x.device) != (10, 3)
        or torch.cuda.get_device_properties(x.device).multi_processor_count != 148
    ):
        return None
    # Key on tensor ownership as well as stream. A replacement model/weight
    # must never consume a state holding its predecessor's coefficients.
    key = (
        x.device,
        torch.cuda.current_stream(x.device).cuda_stream,
        m,
        id(previous),
        id(next_hc),
        id(module.wkv.weight),
        id(module.wkv.weight_scales),
        id(next_hc.fn),
        None if torch.is_inference(next_hc.fn) else next_hc.fn._version,
        id(next_hc.scale),
        id(next_hc.base),
        id(layer.attn_norm.weight),
        module.eps,
        next_hc.norm_eps,
        next_hc.hc_eps,
        next_hc.hc_sinkhorn_iters,
        layer.attn_norm.variance_epsilon,
    )
    states = getattr(module, "_retained_transition_states", None)
    state = None if states is None else states.get(key)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import prepare_tf32_weight

        state = _backend().State(
            module.wkv.weight,
            module.wkv.weight_scales,
            prepare_tf32_weight(next_hc.fn),
            next_hc.scale,
            next_hc.base,
            layer.attn_norm.weight,
            rows=m,
            gate_eps=module.eps,
            hc_norm_eps=next_hc.norm_eps,
            hc_eps=next_hc.hc_eps,
            sinkhorn_iters=next_hc.hc_sinkhorn_iters,
            rmsnorm_eps=layer.attn_norm.variance_epsilon,
        )
        if states is None:
            states = module._retained_transition_states = {}
        states[key] = state
    # Consume the same forward-local lookup owner/event as the native path.
    # In particular, keep packed FP8 rows quantized instead of re-quantizing.
    rows = module.projection_input(residual.reshape(m, 4, 5120), layer.engram_hashes)
    if isinstance(rows, tuple):
        a, sf = rows
    else:
        a, sf = module.wkv._quantize_input(rows)
    out = state.forward(
        a,
        sf,
        x.reshape(m, 5120),
        residual.reshape(m, 4, 5120),
        post.reshape(m, 4, 1),
        comb.reshape(m, 4, 4),
        previous.pre_mix_out.reshape(m, 4, 1),
        module.q_weight,
        module.k_weight,
        layer.engram_token_mask,
    )
    new_h, y, new_post, new_comb, pre, q, qsf = out
    leading = x.shape[:-1]
    next_hc.pre_mix_out = pre.reshape(*leading, 4)
    return (
        new_h.reshape_as(residual),
        y.reshape_as(x),
        new_post.reshape(*leading, 4, 1),
        new_comb.reshape(*leading, 4, 4),
        (q, qsf),
    )

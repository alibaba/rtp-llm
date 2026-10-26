"""Default-on DeepGEMM Reindex Q-A/KV + BF16 index-head projection on B300.

Warm up admitted shapes before CUDA Graph capture. No weight duplication or
forward-time validation kernels.
Unsupported configurations use the original projections. Reload is performed
at init_rope_cache with inference quiesced, like the other graph-owned weights.
Set DSV41_FUSED_FRONT_PROJECTION=0 to retain the original projections.
FULL/source layers retain their existing projection and overlap paths.
"""

import os

import torch

from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import is_supported
from rtp_llm.models_py.modules.dsv41._v41_query_quant import query_norm_quant
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def is_reindex(attn):
    """An index producer reusing an earlier layer's KV, not a FULL source."""
    return getattr(attn, "is_index_source", False) and not getattr(
        attn, "is_kv_source", True
    )


def _weights(attn):
    return attn.wq_a_wkv.weight, attn.wq_a_wkv.weight_scales, attn.index_weights


def _layout(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


class FrontWeights:
    def __init__(self, attn, op):
        # Independent Tensor objects retain the original storage even if a loader
        # replaces Parameter.data. These are aliases, not weight copies.
        self.tensors = tuple(t.detach() for t in _weights(attn))
        self.op = op

    def reload(self, attn):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("front weight reload is not allowed during capture")
        incoming = _weights(attn)
        if any(_layout(a) != _layout(b) for a, b in zip(self.tensors, incoming)):
            raise ValueError("front weight layout changed; rebuild model and CUDA Graphs")
        # Validate every layout before copying anything; preserve graph addresses.
        with torch.no_grad():
            for old, new in zip(self.tensors, incoming):
                if old.data_ptr() != new.data_ptr():
                    old.copy_(new)
        attn.wq_a_wkv.weight, attn.wq_a_wkv.weight_scales, attn.index_weights = self.tensors


def prepare(attn):
    if not is_reindex(attn):
        return
    state = getattr(attn, "_front_weights", None)
    if state is not None:
        state.reload(attn)
        return
    if os.environ.get("DSV41_FUSED_FRONT_PROJECTION", "1") != "1":
        return
    linear = getattr(attn, "wq_a_wkv", None)
    head = getattr(attn, "index_weights", None)
    if (not isinstance(linear, V41MXFP8Linear) or (linear.N, linear.K) != (1792, 5120)
        or head is None or head.shape != (32, 5120) or head.dtype != torch.bfloat16
        or not head.is_cuda or not head.is_contiguous()):
        return
    props = torch.cuda.get_device_properties(head.device)
    if (props.major, props.minor, props.multi_processor_count) != (10, 3, 148):
        return
    import deep_gemm

    op = getattr(deep_gemm, "dsv41_front_gemm", None)
    if op is not None:
        attn._front_weights = FrontWeights(attn, op)


def project(attn, x, *, quantized_input=None, pdl=False):
    if not is_reindex(attn):
        return None
    state = getattr(attn, "_front_weights", None)
    if state is None or os.environ.get("DSV41_FUSED_FRONT_PROJECTION", "1") != "1":
        return None
    linear = attn.wq_a_wkv
    rows = x.numel() // x.shape[-1]
    if (rows not in (128, 256, 384, 512) or not x.is_contiguous()
        or not is_supported(linear, x, attn.q_norm, attn.q_lora_rank)
        or not isinstance(attn.wq_b, V41MXFP8Linear) or attn.q_lora_rank != 1280
        or os.environ.get("DSV41_FUSED_QUERY_QUANT", "1") != "1"
        or os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower() not in ("auto", "legacy")):
        return None
    if any(a.data_ptr() != b.data_ptr() or _layout(a) != _layout(b)
           for a, b in zip(_weights(attn), state.tensors)):
        raise RuntimeError("front weights replaced without init_rope_cache reload")
    xq, xs = linear._quantize_input(x) if quantized_input is None else quantized_input
    weight, scales, head = state.tensors
    projected, raw_head = state.op(x.view(rows, 5120), xq.view(rows, 5120), xs, weight, scales, head)
    projected = projected.view(*x.shape[:-1], 1792)
    qr, quantized = query_norm_quant(projected[..., :1280], attn.q_norm, attn.eps, pdl=pdl)
    return qr, projected[..., 1280:], quantized, raw_head

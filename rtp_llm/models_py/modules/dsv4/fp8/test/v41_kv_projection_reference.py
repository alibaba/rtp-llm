"""Select released WKV GEMM arithmetic on explicit SM120 attention owners."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    _dense_down,
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


@contextmanager
def kv_projection_reference(enabled, *, attentions=(), kernel=None):
    if type(enabled) is not bool:
        raise ValueError("WKV selection requires a bool")
    if not enabled:
        yield
        return
    attentions = tuple(attentions)
    if not attentions or len({id(owner) for owner in attentions}) != len(attentions):
        raise ValueError("Select distinct V4.1 attention owners")
    providers = []
    for owner in attentions:
        if type(owner) is not AttentionV41FP8:
            raise ValueError("WKV selection requires V4.1 attention owners")
        provider = getattr(owner.wkv, "_sm120_linear", None)
        if (
            type(provider) is not V41MXFP8Linear
            or owner.dim not in (512, 5120)
            or (owner.head_dim, owner.rope_head_dim) != (512, 64)
            or (provider.N, provider.K) != (512, owner.dim)
            or provider.backend != "flashinfer"
            or provider.weight.dtype != torch.float8_e4m3fn
            or not provider.weight.is_cuda
            or not provider.weight.is_contiguous()
            or provider.scales.device != provider.weight.device
            or torch.cuda.get_device_capability(provider.weight.device) != (12, 0)
        ):
            raise ValueError("WKV selection requires admitted SM120 group-32 providers")
        providers.append(provider)
    selected = {id(provider) for provider in providers}
    if len(selected) != len(providers):
        raise ValueError("Select distinct WKV providers")
    kernel = load_reference_kernel() if kernel is None else kernel
    original = V41MXFP8Linear.forward_quantized

    def forward(provider, values, scales, *, out=None):
        if id(provider) in selected:
            return _dense_down(provider, values, scales, kernel, out=out)
        return original(provider, values, scales, out=out)

    with patch.object(V41MXFP8Linear, "forward_quantized", forward):
        yield

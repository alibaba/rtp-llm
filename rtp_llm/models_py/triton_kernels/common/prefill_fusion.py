"""Independent default-on optimizations, scoped to ordinary Qwen3.5 prefill only."""

import os
from contextlib import contextmanager
from contextvars import ContextVar

_PREFILL = ContextVar("qwen35_ordinary_prefill", default=False)
SIGMOID = "RTP_QWEN35_PREFILL_SIGMOID_FP8"
MROPE = "RTP_QWEN35_PREFILL_MROPE_CACHE"
GATING = "RTP_QWEN35_PREFILL_FLASHINFER_GATING"
METADATA = "RTP_QWEN35_PREFILL_CHECKPOINT_METADATA"


def enabled(name):
    return os.environ.get(name, "1").strip().lower() in {"1", "true", "yes", "on"}


def in_prefill():
    return _PREFILL.get()


@contextmanager
def prefill_fusion_scope(ordinary_prefill):
    token = _PREFILL.set(bool(ordinary_prefill))
    try:
        yield
    finally:
        _PREFILL.reset(token)


def prefill_quantized_linear(linear, x):
    # Preserve LinearFactory's backend decision, including its small-M fallback.
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
        CudaFp8DeepGEMMLinear,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )

    if isinstance(linear, CudaFp8GEMMLinear):
        if linear._should_use_flashinfer(x):
            return None
        linear = linear._deepgemm_linear
    if isinstance(linear, CudaFp8DeepGEMMLinear) and linear.scale_ue8m0:
        return linear
    return None


def gdn_prefill_backend(tensor, key_dim=128, value_dim=128):
    """Prefer FlashInfer for its validated contract; explicit choices stay strict."""
    backend = os.getenv("RTP_QWEN35_GDN_PREFILL_BACKEND")
    if backend is not None:
        return backend
    import torch

    if (
        tensor.is_cuda
        and torch.version.hip is None
        and tensor.dtype == torch.bfloat16
        and key_dim == value_dim == 128
        and torch.cuda.get_device_capability(tensor.device)[0] == 10
    ):
        return "flashinfer"
    return "native"

"""Single-pass FP8 input quantization shared by all three QSA input projections.

In each QSA layer, ``hidden_states`` (``[M, 2560]``) feeds three independent FP8
projections:
1. ``qkv_proj`` (``CudaFp8GEMMLinear``, 128-group FP8 block-scale GEMM)
2. ``gate`` (``Fp8Linear``, per-tensor FP8 ``_scaled_mm``)
3. ``qsa_indexer.qk_projection`` (``Fp8Linear``, per-tensor FP8 ``_scaled_mm``)

Reusing ``_quantize_qkvz`` + ``_quantize_ba`` computes the 128-group FP8 codes,
column-major group scales, and partial absmax in one pass over ``hidden_states``,
then derives the exact per-tensor FP8 codes and scale for ``gate`` and
``qk_projection`` without a second absmax pass or a redundant group quantization
inside ``qkv_proj``.
"""

import os

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import FP8_GROUP_QUANT_V2_MIN_ELEMENTS
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.qwen4_exp.gdn_fp8_input_projection import (
    _quantize_ba,
    _quantize_qkvz,
    _unmodified,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_mix_triton import (
    get_prequantized_mix_reduce,
)
from rtp_llm.models_py.utils.device_capability import cached_device_capability


def try_prepare_qsa_triple(x: torch.Tensor, qkv_proj, gate, qk_proj):
    """Quantize ``x`` once for ``qkv_proj``, ``gate``, and ``qk_proj``.

    Returns ``(gate_out, qk_out, qkv_callable)`` or ``None`` when unsupported.
    """
    if os.getenv("RTP_QWEN4_DISABLE_QSA_TRIPLE_QUANT", "0") == "1":
        return None
    if not (
        x.ndim == 2
        and 1 <= x.shape[0] <= 32768
        and x.numel() <= 2**31 - 1
        and x.shape[1] == 2560
        and x.is_cuda
        and torch.version.hip is None
        and cached_device_capability(x.device) == (9, 0)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.storage_offset() % 8 == 0
        and not x.requires_grad
        and type(qkv_proj) is CudaFp8GEMMLinear
        and qkv_proj.K == 2560
        and qkv_proj.N > 0
        and qkv_proj.bias is None
        and type(gate) is Fp8Linear
        and type(qk_proj) is Fp8Linear
        and gate.use_scaled_mm
        and qk_proj.use_scaled_mm
        and gate.compute_weight is None
        and qk_proj.compute_weight is None
        and gate.bias is None
        and qk_proj.bias is None
        and gate.weight.ndim == 2
        and qk_proj.weight.ndim == 2
        and gate.weight.shape[1] == 2560
        and qk_proj.weight.shape[1] == 2560
        and gate.weight.dtype == torch.float8_e4m3fn
        and qk_proj.weight.dtype == torch.float8_e4m3fn
        and x.device == qkv_proj.weight.device == gate.weight.device == qk_proj.weight.device
        and not torch.nn.modules.module._global_forward_hooks
        and not torch.nn.modules.module._global_forward_pre_hooks
        and _unmodified(qkv_proj, ("forward", "_should_use_flashinfer"))
        and _unmodified(gate, ("forward",))
        and _unmodified(qk_proj, ("forward",))
    ):
        return None

    quant_kernel = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if quant_kernel not in ("auto", "legacy", "v2"):
        return None
    v2 = quant_kernel == "v2" or (
        quant_kernel == "auto" and x.numel() >= FP8_GROUP_QUANT_V2_MIN_ELEMENTS
    )
    backend = (
        qkv_proj._flashinfer_linear
        if qkv_proj._should_use_flashinfer(x)
        else qkv_proj._deepgemm_linear
    )
    if getattr(backend, "scale_ue8m0", False) or not _unmodified(
        backend, ("forward", "quantize_input", "forward_quantized")
    ):
        return None

    bundle = get_prequantized_mix_reduce(x, "attn_v2" if v2 else "attn_legacy")
    if bundle is not None:
        q_group, s_group, partial = bundle.q, bundle.s, bundle.partial
    else:
        q_group, s_group, partial = _quantize_qkvz(x, v2=v2)
    q_tensor, s_tensor = _quantize_ba(x, partial, wait_x=bundle is not None)
    rows = x.shape[0]
    prefix = x.shape[:-1]
    gate_out = gate._run_quantized(q_tensor, s_tensor, prefix, rows).to(x.dtype)
    qk_out = qk_proj._run_quantized(q_tensor, s_tensor, prefix, rows).to(x.dtype)

    def run_qkv() -> torch.Tensor:
        return backend.forward_quantized(q_group, s_group)

    return gate_out, qk_out, run_qkv

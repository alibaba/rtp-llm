"""V4.1 inverse RoPE/FP8 quant and batched wo_a projection on SM100.

The caller retains wo_b and the old output projection fallback.  This helper
does not mutate the attention output and preserves the eager BF16 RoPE
intermediate and the selected RTP quantizer's scale policy.
"""

import os

import torch

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)


def is_supported(o, freqs, weight, weight_scale) -> bool:
    """Static gate, safe to evaluate during graph capture; no device reads."""
    if (
        o.device.type != "cuda"
        or o.dtype != torch.bfloat16
        or o.ndim not in (3, 4)
        or o.numel() == 0
        or o.shape[-1] != 512
        or o.stride(-1) != 1
        or freqs.dtype != torch.complex64
        or freqs.ndim != 2
        or freqs.shape[-1] != 32
        or freqs.shape[0] == 0
        or freqs.is_conj()
        or weight.dtype != torch.float8_e4m3fn
        or weight.ndim != 3
        or weight_scale.dtype != torch.int32
        or weight_scale.ndim != 3
        or not weight.is_contiguous()
    ):
        return False
    groups, rank, width = weight.shape
    tokens = o.numel() // (o.shape[-2] * o.shape[-1])
    if o.ndim == 4 and (
        freqs.shape[0] not in (o.shape[0], tokens)
        or (
            o.shape[0] > 1
            and o.shape[1] > 1
            and o.stride(0) != o.shape[1] * o.stride(1)
        )
    ):
        return False
    if (
        groups == 0
        or o.shape[-2] % groups
        or width != o.shape[-2] // groups * o.shape[-1]
        or weight_scale.shape != (groups, rank, width // 128)
        or tokens % freqs.shape[0]
        or any(t.device != o.device for t in (freqs, weight, weight_scale))
    ):
        return False
    return torch.cuda.get_device_capability(o.device)[0] == 10


def quantization_scale_min(tokens: int, group_width: int) -> float:
    """Mirror the per-group V41MXFP8Linear -> sgl quantizer dispatch."""
    backend = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if backend not in ("auto", "legacy", "v2"):
        raise ValueError("DSV4_FP8_QUANT_KERNEL must be auto, legacy, or v2")
    use_v2 = backend == "v2" or (
        backend == "auto" and tokens * group_width >= 4 * 1024 * 1024
    )
    return 0.0 if use_v2 else 1e-10


def grouped_output_projection(o, freqs, weight, weight_scale, *, out=None):
    """Return BF16 [tokens, groups * rank] for the existing wo_b linear.

    Call only after ``is_supported``. CUDA/JIT errors intentionally propagate.
    ``out`` optionally supplies a contiguous [tokens, groups, rank] tensor.
    """
    import deep_gemm

    groups, rank, width = weight.shape
    tokens = o.numel() // (o.shape[-2] * o.shape[-1])
    quantized, scales = fused_inv_rope_fp8_quant(
        o,
        freqs,
        n_groups=groups,
        heads_per_group=o.shape[-2] // groups,
        nope_dim=o.shape[-1] - freqs.shape[-1] * 2,
        rope_head_dim=freqs.shape[-1] * 2,
        quant_group_size=32,
        eps=torch.finfo(torch.float32).tiny,
        round_rope_to_input_dtype=True,
        scale_min=quantization_scale_min(tokens, width),
    )
    if out is None:
        out = torch.empty(tokens, groups, rank, dtype=torch.bfloat16, device=o.device)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (quantized, scales),
        (weight, weight_scale),
        out,
        recipe=(1, 1, 32),
    )
    return out.view(tokens, groups * rank)

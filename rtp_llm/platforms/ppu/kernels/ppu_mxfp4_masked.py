"""Full-slot MXFP4 expert execution for PPU DeepEP low-latency payloads."""

import torch
from rtp_llm.platforms.ppu.kernels.cuda.ppu_mxfp4_moe import silu_mul_masked_mxfp4


def full_slot_mxfp4_views(payload, scales):
    """Validate metadata without copying payloads or synchronizing counts."""
    if payload.ndim != 3 or scales.ndim != 3:
        raise ValueError("Grouped MXFP4 data and scales must have rank 3")
    e, m, packed_k = payload.shape
    if (
        e <= 0
        or m <= 0
        or packed_k <= 0
        or packed_k % 32
        or scales.shape != (e, m, packed_k // 32)
        or payload.dtype != torch.uint8
        or scales.dtype != torch.uint16
        or payload.device != scales.device
        or not payload.is_cuda
        or not payload.is_contiguous()
        or not scales.transpose(-1, -2).is_contiguous()
    ):
        raise ValueError(
            "Grouped MXFP4 requires packed uint8 and mn-major uint16 scales"
        )
    return payload, scales


def tensor_spans_overlap(a, b):
    """Conservative byte-span overlap, including holes in strided views."""
    if a.device != b.device or not a.numel() or not b.numel():
        return False

    def span(tensor):
        start = tensor.data_ptr()
        last = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        return start, start + (last + 1) * tensor.element_size()

    a0, a1 = span(a)
    b0, b1 = span(b)
    return a0 < b1 and b0 < a1


def mxfp4_experts_masked(
    expert_x, weight13, weight2, counts, *, expected_m, swiglu_limit=None, out=None
):
    """Two masked GEMMs and fused SwiGLU; all valid rows retain full capacity.

    DeepEP owns counts in [0, M]. Their values remain on device and are passed
    unchanged to all three kernels. Inactive rows of the result are undefined.
    ``expected_m`` only guides launch geometry; skewed routes may exceed it.
    """
    import deep_gemm

    if not isinstance(expert_x, tuple) or len(expert_x) != 2:
        raise ValueError("Expert input must be a packed MXFP4 (data, scale) tuple")
    x, scale = full_slot_mxfp4_views(*expert_x)
    if torch.cuda.get_device_name(x.device) != "ZW-M890P":
        raise RuntimeError("Masked MXFP4 experts require a PPU M890P")
    w13, s13 = full_slot_mxfp4_views(*weight13)
    w2, s2 = full_slot_mxfp4_views(*weight2)
    e, m, packed_d = x.shape
    d = packed_d * 2
    inter = w13.shape[1] // 2
    if (
        w13.shape != (e, 2 * inter, packed_d)
        or w2.shape != (e, d, inter // 2)
        or inter % 256
        or w13.device != x.device
        or w2.device != x.device
        or counts.shape != (e,)
        or counts.dtype != torch.int32
        or counts.device != x.device
        or not counts.is_contiguous()
    ):
        raise ValueError(
            "Masked MXFP4 expert weight/count geometry does not match input"
        )
    if not isinstance(expected_m, int) or expected_m <= 0:
        raise ValueError("expected_m must be a positive host integer")
    if out is None:
        out = torch.empty((e, m, d), dtype=torch.bfloat16, device=x.device)
    elif (
        out.shape != (e, m, d)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
        or any(
            tensor_spans_overlap(out, t) for t in (x, scale, w13, s13, w2, s2, counts)
        )
    ):
        raise ValueError("Expert output must be a disjoint contiguous BF16 full slot")

    gate_up = torch.empty((e, m, 2 * inter), dtype=torch.bfloat16, device=x.device)
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
        (x, scale), (w13, s13), None, gate_up, counts, expected_m
    )
    hidden = silu_mul_masked_mxfp4(gate_up, counts, swiglu_limit, expected_m)
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
        hidden, (w2, s2), None, out, counts, expected_m
    )
    return out

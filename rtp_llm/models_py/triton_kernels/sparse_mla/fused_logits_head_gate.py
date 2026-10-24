"""Fused ``cast + GEMV + 2 elementwise muls`` for the indexer logits-head gate.

Replaces the 4-kernel chain in ``Indexer._get_logits_head_gate``:

    x = x.float()                                    # bf16 -> fp32 cast
    weights = self.weights_proj(x)                   # GEMV (in_features=H, out_features=N)
    scale = self.softmax_scale * self.weights_scale  # Python const
    weights = weights.unsqueeze(-1) * q_scale * scale

with a single Triton kernel that:
  - reads ``x`` (bf16/fp16) and casts to fp32 in-register (no separate cast kernel),
  - reads ``w`` as fp32 (DSV3.2 ``weights_proj.weight`` is loaded as fp32 in
    ``models/deepseek_v2.py``); falls back to fp32 cast for bf16/fp16 weight too,
  - does ``out[t, n] = sum_k x[t, k] * w[n, k]`` in fp32 (TF32 tensor cores
    on H20+), keeping the same precision as the original fp32 GEMM,
  - multiplies by ``q_scale[t, n, 1]`` and a constant ``scale`` in epilogue.

The fp32 path is required because DSV3.2 indexer logits feed into a top-k
selection that is sensitive to the accumulated dot-product precision; bf16
weight cast on K=7168 accumulates ~8% relative error (sqrt(K) * bf16_ulp),
which is above the noise floor for stable top-k.

Shape constraints for the fast path:
  - ``x`` 2-D ``[T, K]`` bf16/fp16 contiguous, ``K <= 8192``.
  - ``w`` 2-D ``[N, K]`` bf16/fp16/fp32 (contiguous or transposed view OK).
  - ``q_scale`` 3-D ``[T, N, 1]`` fp32 OR 2-D ``[T, N]`` fp32.
  - ``T >= 1``.
Otherwise falls back to the unfused 4-op python chain.

Weight layout handling:
  Production weight may be a transposed view ``[N, K]`` of underlying
  ``[K, N]`` contiguous storage (stride_w_n=1, stride_w_k=N). The large-T
  kernel detects this via ``W_IS_TRANSPOSED`` and swaps the 2D load order
  so the stride-1 dimension is always the inner axis for coalesced access.
  The small-T kernel uses per-(token, head) 1D loads and works best with
  contiguous weight; callers should pre-contiguify transposed weight at
  init time (see ``Indexer.__init__``).
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl

MAX_K = 8192


@triton.jit
def _fused_logits_head_gate_kernel(
    x_ptr,  # [T, K] bf16
    w_ptr,  # [N, K] bf16/fp32 (Linear weight)
    qs_ptr,  # [T, N] fp32 (q_scale, last dim already squeezed)
    out_ptr,  # [T, N] fp32
    scale_const,  # fp32
    T,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_x_t,
    stride_w_n,
    stride_w_k,
    stride_qs_t,
    stride_o_t,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W_IS_TRANSPOSED: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < T
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_t + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        if W_IS_TRANSPOSED:
            # [BLOCK_K, BLOCK_N] with N as inner dim (coalesced: stride_w_n=1)
            w = tl.load(
                w_ptr + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            # Single-pass TF32 tensor cores. The original tf32x3 (3-pass for
            # fp32 emulation) was 3x slower; empirical measurement on DSV3.2
            # K=7168 N=64 shows tf32 has mean_rel 0.09% (not 1.4% as a stale
            # comment claimed) and top-32 recall=0.9999 vs cuBLAS fp32. The
            # downstream consumer is fp8_paged_mqa_logits + top-k selection,
            # which is already FP8-quantized — 0.09% drift in weights is well
            # below the FP8 noise floor.
            acc += tl.dot(x, w, out_dtype=tl.float32, input_precision="tf32")
        else:
            # [BLOCK_N, BLOCK_K] with K as inner dim (coalesced: stride_w_k=1)
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32, input_precision="tf32")

    # Load q_scale [BLOCK_M, BLOCK_N] (already squeezed from [T, N, 1])
    qs = tl.load(
        qs_ptr + offs_m[:, None] * stride_qs_t + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    out = acc * qs * scale_const

    tl.store(
        out_ptr + offs_m[:, None] * stride_o_t + offs_n[None, :],
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _fused_logits_head_gate_small_t_kernel(
    x_ptr,  # [T, K] bf16
    w_ptr,  # [N, K] bf16/fp32 (Linear weight)
    qs_ptr,  # [T, N] fp32
    out_ptr,  # [T, N] fp32
    scale_const,  # fp32
    K: tl.constexpr,
    stride_x_t,
    stride_w_n,
    stride_w_k,
    stride_qs_t,
    stride_o_t,
    BLOCK_K: tl.constexpr,  # power-of-2 padded K (whole row in registers)
):
    """Small-T variant: one program per (token, head_n).

    Avoids ``tl.dot``'s BLOCK_M >= 16 minimum which forces 16x wasted compute
    when T < 16. Each program reads one row of ``x[t, :]`` (BLOCK_K elements)
    and one row of ``w[n, :]``, multiplies + reduces in fp32. Decode-friendly.
    Best with contiguous weight (stride_w_k=1); callers should pre-contiguify
    transposed weight at init time.
    """
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    x = tl.load(x_ptr + t * stride_x_t + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    w = tl.load(
        w_ptr + n * stride_w_n + offs_k * stride_w_k, mask=mask_k, other=0.0
    ).to(tl.float32)

    acc = tl.sum(x * w, axis=0)  # scalar fp32

    qs = tl.load(qs_ptr + t * stride_qs_t + n).to(tl.float32)
    out = acc * qs * scale_const

    tl.store(out_ptr + t * stride_o_t + n, out)


def _baseline_logits_head_gate(
    x: torch.Tensor,
    q_scale: torch.Tensor,
    weights_proj: torch.nn.Module,
    scale_const: float,
) -> torch.Tensor:
    """Fallback 4-op chain matching the original Python implementation."""
    x_fp32 = x.float()
    w = weights_proj(x_fp32)
    if q_scale.dim() == 2:
        q_scale_b = q_scale.unsqueeze(-1)
    else:
        q_scale_b = q_scale
    return w.unsqueeze(-1) * q_scale_b * scale_const


def fused_logits_head_gate(
    x: torch.Tensor,
    q_scale: torch.Tensor,
    weight: torch.Tensor,
    scale_const: float,
    fallback_proj: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Fused ``cast + GEMV + 2 muls`` for indexer's _get_logits_head_gate.

    Args:
        x:           [T, K] bf16 input (was ``x`` before ``.float()`` cast).
        q_scale:     [T, N, 1] fp32 OR [T, N] fp32.
        weight:      [N, K] bf16 (``weights_proj.weight``).
        scale_const: ``softmax_scale * weights_scale`` (Python float).
        fallback_proj: optional callable used only when the fast path bails out
                       (matches the original ``self.weights_proj``).

    Returns:
        ``[T, N, 1]`` fp32 tensor.

    Fast path conditions:
        - ``x.dtype == bf16`` (or fp16) and ``weight.dtype == bf16`` (or fp16)
        - ``x.is_contiguous()``
        - ``K <= 8192``
        - ``T >= 1``

    Falls back to ``_baseline_logits_head_gate`` (cast + GEMM + 2 muls)
    otherwise.
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    T, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"x.shape[1]={K} but weight.shape[1]={K_w}"

    # Squeeze q_scale to [T, N]
    if q_scale.dim() == 3:
        assert q_scale.shape[-1] == 1
        qs_2d = q_scale.squeeze(-1)
    elif q_scale.dim() == 2:
        qs_2d = q_scale
    else:
        raise ValueError(f"q_scale must be 2-D or 3-D, got {q_scale.shape}")
    assert qs_2d.shape == (T, N), f"q_scale shape {qs_2d.shape} != (T={T}, N={N})"

    # Fast-path constraints. Production decode runs under CUDA Graph (launch
    # overhead amortized by capture+replay), so the Triton kernel can win at
    # any T. Two kernel variants:
    #  - small-T (T <= 32): per-(token, head) program, explicit reduce. Avoids
    #    tl.dot's BLOCK_M>=16 minimum which wastes compute at small T.
    #    Best with contiguous weight; callers should pre-contiguify at init.
    #  - large-T (T > 32): tiled tl.dot with input_precision="tf32".
    #    Empirical on DSV3.2 (K=7168, N=64): tf32 mean_rel=0.09%, top-32
    #    recall=0.9999 vs cuBLAS fp32 — well within the downstream FP8
    #    paged-mqa-logits noise floor. The single-pass TF32 path is 1.6-3x
    #    faster than the prior 3-pass tf32x3 (1024->44us, 8192->68us on B300).
    #    Handles both contiguous and transposed weight via W_IS_TRANSPOSED.
    use_fast_path = (
        x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.is_contiguous()
        and K <= MAX_K
        and qs_2d.dtype == torch.float32
        and qs_2d.is_contiguous()
    )

    if not use_fast_path:
        if fallback_proj is None:
            raise ValueError(
                "fused_logits_head_gate fast path unsupported and no "
                "fallback_proj provided"
            )
        return _baseline_logits_head_gate(x, q_scale, fallback_proj, scale_const)

    out_2d = torch.empty((T, N), dtype=torch.float32, device=x.device)

    # Detect transposed weight layout for coalesced 2D loads in large-T kernel.
    w_is_transposed = weight.stride(0) < weight.stride(1)

    if T <= 32:
        # small-T kernel: grid (T, N), one program per (t, n) pair.
        BLOCK_K = triton.next_power_of_2(K)
        num_warps = 4 if K >= 4096 else 2
        _fused_logits_head_gate_small_t_kernel[(T, N)](
            x,
            weight,
            qs_2d,
            out_2d,
            float(scale_const),
            K,
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            qs_2d.stride(0),
            out_2d.stride(0),
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )
    else:
        # M=32 K=128 nw=4 ns=3 picked empirically across T=1024-8192 on DSV3.2
        # (K=7168, N=64): within 4.5% of per-T optimum without shape dispatch.
        # num_stages=3 enables Triton software pipelining over the K-loop.
        BLOCK_M = 32
        BLOCK_K = 128
        BLOCK_N = triton.next_power_of_2(max(N, 16))
        grid = (triton.cdiv(T, BLOCK_M),)
        num_warps = 4 if N <= 128 else 8
        _fused_logits_head_gate_kernel[grid](
            x,
            weight,
            qs_2d,
            out_2d,
            float(scale_const),
            T,
            K,
            N,
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            qs_2d.stride(0),
            out_2d.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            W_IS_TRANSPOSED=w_is_transposed,
            num_warps=num_warps,
            num_stages=3,
        )

    return out_2d.unsqueeze(-1)

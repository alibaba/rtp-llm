# SPDX-License-Identifier: Apache-2.0
"""Fused Indexer head projection with FP32 output.

K3.1 Prefill uses BF16 operands and FP32 accumulation. The IEEE FP32 dot
remains available for the existing verification path.
"""
from typing import Optional
import torch
import triton
import triton.language as tl
MAX_K = 8192

@triton.jit
def _fused_logits_head_gate_kernel(
    x_ptr, w_ptr, qs_ptr, out_ptr, scale_const, T, K: tl.constexpr, N: tl.constexpr,
    stride_x_t, stride_w_n, stride_w_k, stride_qs_t, stride_o_t,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    W_IS_TRANSPOSED: tl.constexpr, TILE_HEADS: tl.constexpr = False,
    APPLY_SCALE: tl.constexpr = True,
    BF16_DOT: tl.constexpr = False,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < T
    offs_n = tl.arange(0, BLOCK_N)
    if TILE_HEADS:
        offs_n += tl.program_id(1) * BLOCK_N
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_t + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        if W_IS_TRANSPOSED:
            w = tl.load(
                w_ptr + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n,
                mask=mask_k[:, None] & mask_n[None, :], other=0.0,
            )
        else:
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0,
            )
            w = tl.trans(w)
        if BF16_DOT:
            acc = tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16), acc, out_dtype=tl.float32)
        else:
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32), out_dtype=tl.float32, input_precision='ieee')
    if APPLY_SCALE:
        qs = tl.load(qs_ptr + offs_m[:, None] * stride_qs_t + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc = acc * qs * scale_const
    tl.store(out_ptr + offs_m[:, None] * stride_o_t + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def project_prefill_head_weights(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Write BF16 dot products to FP32 before the Indexer Q scale exists."""
    rows, width = x.shape
    heads = weight.shape[0]
    if (x.dtype != torch.bfloat16 or weight.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous() or width > MAX_K
            or weight.shape[1] != width or weight.device != x.device
            or out.shape != (rows, heads) or out.dtype != torch.float32
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("invalid Indexer prefill head projection buffers")
    _fused_logits_head_gate_kernel[(triton.cdiv(rows, 32),)](
        x, weight, x, out, 1.0, rows, width, heads,
        x.stride(0), weight.stride(0), weight.stride(1), 0, out.stride(0),
        BLOCK_M=32, BLOCK_K=128, BLOCK_N=triton.next_power_of_2(max(heads, 16)),
        W_IS_TRANSPOSED=weight.stride(0) < weight.stride(1), APPLY_SCALE=False, BF16_DOT=True,
        num_warps=4, num_stages=3,
    )

@triton.jit
def _fused_logits_head_gate_small_t_kernel(x_ptr, w_ptr, qs_ptr, out_ptr, scale_const, K: tl.constexpr, stride_x_t, stride_w_n, stride_w_k, stride_qs_t, stride_o_t, BLOCK_K: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    x = tl.load(x_ptr + t * stride_x_t + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + n * stride_w_n + offs_k * stride_w_k, mask=mask_k, other=0.0).to(tl.float32)
    acc = tl.sum(x * w, axis=0)
    qs = tl.load(qs_ptr + t * stride_qs_t + n).to(tl.float32)
    out = acc * qs * scale_const
    tl.store(out_ptr + t * stride_o_t + n, out)

def _baseline_logits_head_gate(x: torch.Tensor, q_scale: torch.Tensor, weights_proj: torch.nn.Module, scale_const: float) -> torch.Tensor:
    x_fp32 = x.float()
    w = weights_proj(x_fp32)
    if q_scale.dim() == 2:
        q_scale_b = q_scale.unsqueeze(-1)
    else:
        q_scale_b = q_scale
    return w.unsqueeze(-1) * q_scale_b * scale_const

def fused_logits_head_gate(
    x: torch.Tensor, q_scale: torch.Tensor, weight: torch.Tensor, scale_const: float,
    fallback_proj: Optional[torch.nn.Module] = None, *,
    original_rows: Optional[int] = None, tile_decode_output: bool = False, bf16_dot: bool = False,
) -> torch.Tensor:
    assert x.dim() == 2
    assert weight.dim() == 2
    (T, K) = x.shape
    policy_rows = T if original_rows is None else original_rows
    if policy_rows < T:
        raise ValueError('original head-weight row extent cannot be smaller than the chunk')
    (N, K_w) = weight.shape
    assert K == K_w, f'x.shape[1]={K} but weight.shape[1]={K_w}'
    if q_scale.dim() == 3:
        assert q_scale.shape[-1] == 1
        qs_2d = q_scale.squeeze(-1)
    elif q_scale.dim() == 2:
        qs_2d = q_scale
    else:
        raise ValueError(f'q_scale must be 2-D or 3-D, got {q_scale.shape}')
    assert qs_2d.shape == (T, N), f'q_scale shape {qs_2d.shape} != (T={T}, N={N})'
    use_fast_path = x.is_cuda and weight.device == x.device and qs_2d.device == x.device and x.dtype in (torch.bfloat16, torch.float16) and weight.dtype in (torch.bfloat16, torch.float16, torch.float32) and x.is_contiguous() and (K <= MAX_K) and (qs_2d.dtype == torch.float32) and qs_2d.is_contiguous()
    if not use_fast_path:
        if bf16_dot:
            projected = torch.mm(x.bfloat16(), weight.bfloat16().t(), out_dtype=torch.float32)
            return (projected * qs_2d.float() * scale_const).unsqueeze(-1)
        if fallback_proj is None:
            raise ValueError('fused_logits_head_gate fast path unsupported and no fallback_proj provided')
        return _baseline_logits_head_gate(x, q_scale, fallback_proj, scale_const)
    out_2d = torch.empty((T, N), dtype=torch.float32, device=x.device)
    if T == 0:
        return out_2d.unsqueeze(-1)
    w_is_transposed = weight.stride(0) < weight.stride(1)
    if policy_rows <= 32 and not bf16_dot:
        BLOCK_K = triton.next_power_of_2(K)
        num_warps = 4 if K >= 4096 else 2
        _fused_logits_head_gate_small_t_kernel[T, N](x, weight, qs_2d, out_2d, float(scale_const), K, x.stride(0), weight.stride(0), weight.stride(1), qs_2d.stride(0), out_2d.stride(0), BLOCK_K=BLOCK_K, num_warps=num_warps)
    else:
        # A small Decode batch benefits from more CTAs over independent outputs.
        # Keep the same IEEE dot products and 128-wide K reduction per output.
        tile_heads = (tile_decode_output and policy_rows <= 128 and K == 7168 and N == 32
                      and x.dtype == torch.bfloat16 and weight.dtype == torch.float32
                      and w_is_transposed)
        BLOCK_M = 16 if tile_heads else 32
        BLOCK_K = 128
        BLOCK_N = 16 if tile_heads else triton.next_power_of_2(max(N, 16))
        grid = (triton.cdiv(T, BLOCK_M),)
        if tile_heads:
            grid += (triton.cdiv(N, BLOCK_N),)
        num_warps = 4 if N <= 128 else 8
        _fused_logits_head_gate_kernel[grid](x, weight, qs_2d, out_2d, float(scale_const), T, K, N, x.stride(0), weight.stride(0), weight.stride(1), qs_2d.stride(0), out_2d.stride(0), BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, W_IS_TRANSPOSED=w_is_transposed, TILE_HEADS=tile_heads, BF16_DOT=bf16_dot, num_warps=num_warps, num_stages=3)
    return out_2d.unsqueeze(-1)

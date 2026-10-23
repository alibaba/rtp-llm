# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-K BF16 projection for SM120 prefill routing and compression.

Adapted from vLLM's persistent batch-invariant GEMM at
52358e6e192aeae73dd3764046c98fc4b156ea83. Output tiles may change with M,
but every output element accumulates K in the same order without split-K.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.arch import is_sm120


@triton.jit
def _tile_coordinates(tile, group_tiles, m_tiles, GROUP: tl.constexpr):
    group = tile // group_tiles
    first_m = group * GROUP
    actual_group = min(m_tiles - first_m, GROUP)
    return first_m + tile % actual_group, (tile % group_tiles) // actual_group


@triton.jit
def _persistent_linear(
    x,
    weight,
    output,
    M,
    # Weight dimensions are fixed; M stays dynamic across request batches.
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    SMS: tl.constexpr,
):
    first = tl.program_id(0)
    m_tiles, n_tiles = tl.cdiv(M, BM), tl.cdiv(N, BN)
    output_tile = first - SMS
    for tile in tl.range(first, m_tiles * n_tiles, SMS, flatten=True):
        pm, pn = _tile_coordinates(tile, GROUP * n_tiles, m_tiles, GROUP)
        mm, nn = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
        mm = tl.where(mm < M, mm, 0)
        nn = tl.where(nn < N, nn, 0)
        mm = tl.max_contiguous(tl.multiple_of(mm, BM), BM)
        nn = tl.max_contiguous(tl.multiple_of(nn, BN), BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + tl.arange(0, BK)
            a = tl.load(x + mm[:, None] * K + kk[None, :], kk[None, :] < K, other=0)
            b = tl.load(
                weight + nn[None, :] * K + kk[:, None], kk[:, None] < K, other=0
            )
            acc = tl.dot(a, b, acc)
        output_tile += SMS
        pm, pn = _tile_coordinates(output_tile, GROUP * n_tiles, m_tiles, GROUP)
        mm, nn = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
        tl.store(
            output + mm[:, None] * N + nn[None, :],
            acc,
            (mm[:, None] < M) & (nn[None, :] < N),
        )


def is_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.ndim >= 2
        and weight.ndim == 2
        and x.is_cuda
        and is_sm120(x.device)
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == weight.shape[-1]
        and weight.shape[0] > 0
        and weight.shape[1] > 0
        and x.numel() < 2**31
        and weight.numel() < 2**31
    )


def batch_invariant_linear(
    x: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype
) -> torch.Tensor:
    """Project [..., K] by [N, K], preserving per-row arithmetic across M.

    Strided inputs are materialized before the kernel. BF16 inputs and either
    BF16 or FP32 outputs are supported; callers select their ordinary backend
    when ``is_supported`` is false.
    """
    if not is_supported(x, weight):
        raise ValueError("batch_invariant_linear requires SM120 BF16 operands")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("batch_invariant_linear output must be BF16 or FP32")
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    weight = weight.contiguous()
    m, k = flat.shape
    n = weight.shape[0]
    if m * n >= 2**31:
        raise ValueError("batch_invariant_linear output exceeds kernel index range")
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    if m:
        # Both tiles traverse K without split-K. Exact agreement across their
        # different BK values is qualified on SM120 by the 63/64/65-row tests;
        # rerun those tests when changing Triton or these launch parameters.
        bm, bn, bk = (16, 64, 128) if m < 64 else (64, 128, 64)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        _persistent_linear[(min(sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)](
            flat,
            weight,
            out,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            8,
            sms,
            num_warps=4,
            num_stages=3,
        )
    return out.reshape(*shape[:-1], n)

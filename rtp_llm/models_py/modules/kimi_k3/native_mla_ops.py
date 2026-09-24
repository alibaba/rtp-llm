# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 MLA pointwise kernels from vLLM c3b484463; RTP launch adapters."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    # The weight load does not depend on the producer's output, so issue it
    # before the PDL wait: gamma streams in while the producer finishes,
    # leaving a single dependent global round trip (x) after the wait.
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@lru_cache(None)
def _supports_pdl(device_index):
    return torch.cuda.get_device_capability(device_index)[0] >= 9


def fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps):
    if q.ndim != 2 or kv.ndim != 2 or q.shape[0] != kv.shape[0]:
        raise ValueError("K3 MLA norms require rank-2 Q/KV with equal token counts")
    for x, weight in ((q, q_weight), (kv, kv_weight)):
        if not x.is_cuda or x.device != q.device or weight.device != q.device:
            raise ValueError("K3 MLA norms require CUDA tensors on the same device")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("K3 MLA norms require BF16 tensors")
        if x.stride(-1) != 1 or not weight.is_contiguous():
            raise ValueError("K3 MLA norms require contiguous features and weights")
        if weight.ndim != 1 or weight.numel() != x.shape[1] or x.shape[1] <= 0:
            raise ValueError("K3 MLA norm weight shape does not match input")
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    kv_out = torch.empty(kv.shape, device=kv.device, dtype=kv.dtype)
    if q.shape[0]:
        block = triton.next_power_of_2(max(q.shape[1], kv.shape[1]))
        _fused_q_kv_rmsnorm_kernel[(q.shape[0], 2)](
            q, q_out, q_weight, q.stride(0), q_out.stride(0),
            kv, kv_out, kv_weight, kv.stride(0), kv_out.stride(0), eps,
            Q_SIZE=q.shape[1], KV_SIZE=kv.shape[1], BLOCK_SIZE=block,
            launch_pdl=_supports_pdl(q.device.index),
            num_warps=8 if block >= 2048 else 4,
        )
    return q_out, kv_out


@torch.compile(backend="inductor")
def gate_sigmoid_mul(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return attn_out * gate.sigmoid()

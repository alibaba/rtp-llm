# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
# Adapted for RTP-LLM. Some sglang variants are intentionally dropped to keep
# the port portable across SM levels (no TensorDescriptor / TMA / swap_ab) and
# focused on the high-perf Triton fused_moe path used in sglang's profiling
# timeline (fused_moe_kernel + moe_sum_reduce_triton, no DeepEP).
#
# Supported quant modes:
#   * no quant (BF16/FP16 activation, BF16/FP16 weights)
#   * FP8 W8A8 per-block (A: per-token-group fp8, W: per-block fp8)
#   * FP8 W8A8 per-tensor / per-token (A: per-tensor or per-token fp8)
#
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# moe_align_block_size (Triton + torch fallback)
# -----------------------------------------------------------------------------
#
# sgl_kernel exposes a fast CUDA implementation of moe_align_block_size. RTP-LLM
# does not link sgl_kernel so we provide a pure-torch implementation that is
# functionally equivalent. Performance is acceptable for typical inference
# shapes (M*topk in the few thousand range) because the heavy lifting (sort,
# -----------------------------------------------------------------------------
# moe_align_block_size (Triton + torch, CUDA-graph-safe)
# -----------------------------------------------------------------------------
#
# sgl_kernel exposes a fast CUDA implementation of moe_align_block_size. RTP-LLM
# does not link sgl_kernel so we provide an equivalent in torch + a small
# triton kernel. The implementation deliberately avoids ``torch.argsort`` on
# CUDA: stable argsort dispatches to thrust which calls ``cudaMalloc`` for
# scratch storage, bypassing PyTorch's caching allocator and therefore failing
# inside CUDA graph capture (``cudaErrorStreamCaptureUnsupported``). Instead
# we use a one-pass triton kernel that places each token via ``atomic_add`` on
# a per-expert slot counter — this is graph-capture safe.


@triton.jit
def _moe_align_scatter_kernel(
    bucket_ptr,  # int64 [N]
    cum_ptr,  # int64 [E + 1]; cum_ptr[e] = padded prefix sum
    slot_counter_ptr,  # int32 [E]; zero-initialized; receives atomic adds
    sorted_ids_ptr,  # int32 [max_pad]; pre-filled with sentinel
    num_valid_tokens,  # i32
    num_experts,  # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offsets < num_valid_tokens
    bucket = tl.load(bucket_ptr + offsets, mask=in_range, other=num_experts)
    valid = in_range & (bucket < num_experts)
    safe_bucket = tl.where(valid, bucket, 0)
    base = tl.load(cum_ptr + safe_bucket, mask=valid, other=0)
    rank = tl.atomic_add(slot_counter_ptr + safe_bucket, 1, mask=valid)
    dest = base + rank.to(tl.int64)
    tl.store(sorted_ids_ptr + dest, offsets.to(tl.int32), mask=valid)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
):
    """Aligns the per-expert token count to ``block_size`` for fused MoE.

    Args:
        topk_ids: int tensor of shape ``(num_tokens, top_k)``. Values < 0 mark
            tokens that should be skipped (e.g. EP-filtered experts).
        block_size: BLOCK_SIZE_M used by the fused MoE kernel.
        num_experts: total number of experts. The function reserves an extra
            sentinel slot ``num_experts`` for filtered tokens.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_pad)`` matching the
        layout that ``fused_moe_kernel`` expects.

    CUDA Graph compatibility:
        Output buffers are pre-sized to a fixed worst-case bound; the actual
        padded length is published only through the device tensor
        ``num_tokens_post_pad`` which the kernel consumes via a pointer load.
        No host<->device sync (``.item()``) and no ops that bypass PyTorch's
        caching allocator are used, so this is safe inside a captured CUDA
        graph.
    """
    assert topk_ids.dim() == 2
    assert topk_ids.dtype in (torch.int32, torch.int64)
    device = topk_ids.device
    num_valid_tokens = topk_ids.numel()  # host-side python int
    flat = topk_ids.reshape(-1).to(torch.int64)
    # Map negative / out-of-range values to sentinel ``num_experts`` so they
    # form a separate bucket that is never written to ``sorted_ids``.
    bucket = torch.where(
        (flat >= 0) & (flat < num_experts),
        flat,
        torch.full_like(flat, num_experts),
    )

    # Per-expert (and sentinel) raw count. Stays on device.
    # NB: ``torch.bincount`` on CUDA dispatches to thrust which calls
    # ``cudaMalloc`` directly and is forbidden during cuda graph capture.
    # Use ``scatter_add_`` instead — pure tensor ops, allocator-friendly.
    expert_count = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_count.scatter_add_(0, bucket, torch.ones_like(bucket))
    valid_count = expert_count[:num_experts]
    # Pad each valid expert's bucket up to a multiple of block_size.
    padded = ((valid_count + block_size - 1) // block_size) * block_size
    # Exclusive cumsum prefixed with 0. Done as a single device-side op:
    # writing a python scalar to a tensor slice (e.g. ``cum[0] = 0``) issues
    # a host->device ``cudaMemcpy`` which is forbidden during graph capture.
    cum = torch.nn.functional.pad(torch.cumsum(padded, dim=0), (1, 0))  # [E+1]

    # Worst-case padded length: every expert padded by up to (block_size-1)
    # entries. Round up to a multiple of block_size so num_blocks is exact.
    max_pad = num_valid_tokens + num_experts * block_size
    max_pad = ((max_pad + block_size - 1) // block_size) * block_size

    # ``sorted_token_ids`` defaults to the sentinel ``num_valid_tokens`` so
    # masked loads in the kernel will correctly produce zeros.
    sorted_ids = torch.full(
        (max_pad,), num_valid_tokens, dtype=torch.int32, device=device
    )

    # One-pass triton scatter (graph-capture safe; no thrust / argsort).
    if num_valid_tokens > 0:
        slot_counter = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 256
        grid = (triton.cdiv(num_valid_tokens, BLOCK),)
        _moe_align_scatter_kernel[grid](
            bucket,
            cum,
            slot_counter,
            sorted_ids,
            num_valid_tokens,
            num_experts,
            BLOCK_SIZE=BLOCK,
        )

    # expert_ids per BLOCK_SIZE_M-row block. Sized to the worst case so the
    # downstream grid is host-deterministic; padding-only blocks beyond the
    # actual ``total_pad`` are tagged with ``-1`` and skipped by the kernel.
    max_num_blocks = max_pad // block_size
    block_starts = (
        torch.arange(max_num_blocks, device=device, dtype=torch.int64) * block_size
    )
    # ``searchsorted`` with right=True returns ``e+1`` for a block starting at
    # ``cum[e+1]``. Clamp results past ``num_experts`` (i.e. blocks beyond
    # ``total_pad``) to the ``-1`` filter sentinel.
    ids_long = torch.searchsorted(cum[1:].contiguous(), block_starts, right=True)
    expert_ids = torch.where(
        ids_long >= num_experts,
        torch.full_like(ids_long, -1),
        ids_long,
    ).to(torch.int32)

    # Publish total_pad as a device tensor so the kernel can read it without a
    # host sync. Slicing keeps it as a 1-element tensor matching the original
    # contract.
    num_tokens_post_pad = cum[-1:].to(torch.int32)
    return sorted_ids, expert_ids, num_tokens_post_pad


# -----------------------------------------------------------------------------
# Helper kernel: zero out blocks whose expert is filtered (off-rank).
# -----------------------------------------------------------------------------
@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# -----------------------------------------------------------------------------
# Core fused MoE matmul kernel (no TMA / swap_ab / int4 / int8 / fuse_sum).
# -----------------------------------------------------------------------------
@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    filter_expert: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if filter_expert and off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_fp8_w8a8 and (group_k == 0 or group_n == 0):
        accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    filter_expert: bool = True,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    fused_moe_kernel[grid](
        A,
        B,
        bias,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        even_Ks=even_Ks,
        filter_expert=filter_expert,
        **config,
    )


# -----------------------------------------------------------------------------
# Custom moe_sum_reduce kernel (the high-perf reduce sglang uses post-MoE).
# -----------------------------------------------------------------------------
# Modified from https://github.com/ModelTC/lightllm and sglang fused_moe_triton_kernels.
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    inp: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
) -> None:
    """Reduce ``inp`` (shape ``[T, topk, H]``) along the topk dim into
    ``output`` (shape ``[T, H]``) and multiply by ``routed_scaling_factor``."""
    assert inp.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = inp.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        inp,
        *inp.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )


# -----------------------------------------------------------------------------
# Activation + multiply kernel (silu / gelu).
# -----------------------------------------------------------------------------
# RTP-LLM convention: weight w13 is laid out so that the first half of the
# gate-up output is the "value" (up) and the second half is the "gate". This
# matches rtp_llm.models_py.triton_kernels.common.activation.silu_and_mul and
# the reference math in fused_moe_executor_test_util.generate_ref_output. We
# therefore output ``act(second_half) * first_half`` (sglang's convention is
# the opposite, so we swap the slicing here).
@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        # triton requires a definite return; raising in jit context produces
        # a clean compile error so the user sees the bad activation name.
        tl.static_assert(False, "Unsupported activation")
        return x


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)
    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    # RTP-LLM: first half = value (up), second half = gate.
    value_output_ptr = gateup_output_ptr
    gate_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        value_output = tl.load(value_output_ptr + offset, mask=mask)
        gate_output = tl.load(gate_output_ptr + offset, mask=mask)

        # Compute activation in float32 to avoid arith.mulf on FP8 (Triton's
        # LLVM backend rejects fmul between two fp8 operands). Promote
        # ``value_output`` explicitly so the multiply happens in f32.
        gate_output_activated = _apply_activation(
            gate_output.to(tl.float32), ACTIVATION_TYPE
        )
        act_mul_output = gate_output_activated * value_output.to(tl.float32)
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> None:
    """Compute ``act(gate) * value`` over a flattened (N, 2*H) tensor.

    ``topk_ids.view(-1)`` is consulted per-row so that filtered (-1) experts
    skip the multiplication (their downstream contribution is zero anyway).
    """
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    assert topk_ids is not None
    expert_ids_row = topk_ids.reshape(-1)
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        1,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
    )

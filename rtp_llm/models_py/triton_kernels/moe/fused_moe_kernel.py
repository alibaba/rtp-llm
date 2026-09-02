"""Fused MoE Triton kernel, adapted from SGLang/vLLM.

Supports the BF16/FP16 path and dynamic-activation, 128x128 block-scaled FP8.
"""

from typing import Any, Dict

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Fused MoE GEMM kernel.

    A: input tokens [M, K] (indexed via sorted_token_ids)
    B: expert weights [E, N, K] (read as [K, N] tiles per expert)
    C: output [num_valid_tokens, N] (scattered via sorted_token_ids)

    sorted_token_ids, expert_ids, num_tokens_post_padded are produced by
    moe_align_block_size_torch().
    """
    # Map program id to the block of C it should compute (grouped for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early exit for padding blocks
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Load sorted token ids and expert id for this block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Pointers for A and B
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

    if use_fp8_w8a8:
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        if scale_ue8m0:
            # DeepGEMM expands each N-group scale over its 128 output rows
            # before packing. Select a representative expanded row for a
            # single-group tile, or each row when a tile spans groups.
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N
        elif BLOCK_SIZE_N > group_n:
            offs_bsn = offs_bn // group_n
        else:
            offs_bsn = pid_n * BLOCK_SIZE_N // group_n
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    # Accumulate in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        if use_fp8_w8a8:
            offs_ks = k_start // group_k
            if scale_ue8m0:
                scale_pack_idx = offs_ks // 4
                scale_shift = (offs_ks % 4) * 8
                a_scale_pack = tl.load(
                    a_scale_ptrs + scale_pack_idx * stride_ask,
                    mask=token_mask,
                    other=0,
                )
                b_scale_pack = tl.load(b_scale_ptrs + scale_pack_idx * stride_bsk)
                a_scale_bits = ((a_scale_pack >> scale_shift) & 0xFF) << 23
                b_scale_bits = ((b_scale_pack >> scale_shift) & 0xFF) << 23
                a_scale = a_scale_bits.to(tl.float32, bitcast=True)
                b_scale = b_scale_bits.to(tl.float32, bitcast=True)
            else:
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask,
                    mask=token_mask,
                    other=0.0,
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
            if BLOCK_SIZE_N > group_n:
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
        else:
            accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply router weights if requested
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,  # [M, K] input hidden states
    B: torch.Tensor,  # [E, N, K] expert weights
    C: torch.Tensor,  # [M * top_k, N] output (scattered via sorted_token_ids)
    topk_weights: torch.Tensor,  # [M * top_k] flat routing weights
    topk_ids: torch.Tensor,  # [M * top_k] flat (only used for numel)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    scale_ue8m0: bool = False,
) -> None:
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0
    use_fp8_w8a8 = block_shape is not None
    if use_fp8_w8a8:
        assert A.dtype == torch.float8_e4m3fn
        assert B.dtype == torch.float8_e4m3fn
        assert A_scale is not None and B_scale is not None
        assert len(block_shape) == 2
        block_n, block_k = block_shape
        kernel_block_n = config["BLOCK_SIZE_N"]
        kernel_block_k = config["BLOCK_SIZE_K"]
        assert kernel_block_n > block_n or block_n % kernel_block_n == 0, (
            f"BLOCK_SIZE_N={kernel_block_n} must divide FP8 scale group_n={block_n} "
            "unless the tile spans multiple scale groups"
        )
        assert (
            kernel_block_k <= block_k and block_k % kernel_block_k == 0
        ), f"BLOCK_SIZE_K={kernel_block_k} must divide FP8 scale group_k={block_k}"
        assert A_scale.ndim == 2 and B_scale.ndim == 3
        if scale_ue8m0:
            assert A_scale.dtype == torch.int32 and B_scale.dtype == torch.int32
            assert triton.cdiv(A.shape[-1], block_k) <= A_scale.shape[-1] * 4
            # DeepGEMM's packed weight scale is expanded over output rows.
            assert B.shape[-2] == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) <= B_scale.shape[-1] * 4
        else:
            assert A_scale.dtype == torch.float32 and B_scale.dtype == torch.float32
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    else:
        assert A_scale is None and B_scale is None
        block_n, block_k = 0, 0

    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],  # N
        K,  # K
        sorted_token_ids.shape[0],  # EM
        topk_ids.numel(),  # num_valid_tokens
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        block_n,
        block_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_ue8m0=scale_ue8m0,
        even_Ks=even_Ks,
        **config,
    )


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,  # [M, top_k]
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align tokens to experts in blocks of block_size for the fused MoE kernel.

    Returns:
        sorted_token_ids: [max_padded] padded token indices (may be larger than needed)
        expert_ids: [max_blocks] expert id for each block
        num_tokens_post_padded: [1] scalar tensor (actual used count, on GPU)

    Requires every route id to be in ``[0, num_experts)``.  EP padding
    sentinels must be filtered or remapped by the caller before this helper.
    """
    M, top_k = topk_ids.shape
    num_valid = M * top_k
    flat_ids = topk_ids.reshape(-1)  # [M * top_k]

    # Pre-allocate a static worst-case capacity to avoid a GPU→CPU sync.  If
    # A experts are active, their first block costs A blocks and every further
    # block needs another ``block_size`` routes.  The maximum is reached at
    # A=min(num_valid, num_experts).  This is substantially tighter than
    # reserving one extra block for every expert (for Qwen batch=1: 8 instead
    # of 129 blocks), so the static Triton grid launches far fewer empty
    # programs while remaining CUDA-graph safe for every routing distribution.
    max_active_experts = min(num_valid, num_experts)
    max_blocks = max_active_experts + (num_valid - max_active_experts) // block_size
    max_padded = max_blocks * block_size

    # Count tokens per expert via scatter_add (CUDA-graph-safe, no CPU-GPU sync
    # unlike torch.bincount which reads GPU data on CPU to determine output size)
    tokens_per_expert = torch.zeros(
        num_experts, dtype=torch.int32, device=topk_ids.device
    )
    clamped_ids = flat_ids.clamp(min=0, max=num_experts - 1).long()
    tokens_per_expert.scatter_add_(
        0, clamped_ids, torch.ones(num_valid, dtype=torch.int32, device=topk_ids.device)
    )
    padded_counts = ((tokens_per_expert + block_size - 1) // block_size) * block_size

    # expert_offsets[0]=0, expert_offsets[i]=sum(padded_counts[:i])
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    torch.cumsum(padded_counts.int(), dim=0, out=expert_offsets[1:])
    num_tokens_post_padded = expert_offsets[
        num_experts : num_experts + 1
    ]  # [1] view, stays on GPU

    # Sort tokens by expert + compute positions (reuse cumsum for expert_start)
    sorted_order = flat_ids.argsort(stable=True)
    sorted_expert_ids = flat_ids[sorted_order]

    # expert_start[i] = sum(tokens_per_expert[:i]) — reuse tokens_per_expert cumsum
    expert_start = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    torch.cumsum(tokens_per_expert[:-1].int(), dim=0, out=expert_start[1:])
    position_in_expert = (
        torch.arange(num_valid, device=topk_ids.device)
        - expert_start[sorted_expert_ids]
    )

    # Scatter sorted tokens into padded layout
    sorted_token_ids = torch.full(
        (max_padded,), num_valid, dtype=torch.int32, device=topk_ids.device
    )
    dest = expert_offsets[sorted_expert_ids] + position_in_expert.int()
    sorted_token_ids[dest.long()] = sorted_order.int()

    # Expert ids per block: use searchsorted instead of repeat_interleave
    max_blocks = max_padded // block_size
    # block_boundaries[i] = expert_offsets[i+1] / block_size = cumulative blocks for experts 0..i
    block_boundaries = expert_offsets[1:] // block_size  # [E]
    block_indices = torch.arange(max_blocks, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.searchsorted(block_boundaries, block_indices, right=True).int()
    # Blocks beyond actual content get expert_id >= num_experts, which is fine
    # (they'll be skipped by num_tokens_post_padded check in kernel)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# The routing shapes are static for each captured decode batch.  Compiling the
# metadata path folds its elementwise/indexing work into a handful of kernels,
# which keeps the CUDA graph small enough that replay launch cost does not hide
# the routed FP8 GEMM gain on SM120.  ``dynamic=False`` also avoids the stale
# dynamic-shape indirect-indexing assertion seen with older Torch versions.
# The default 1-32 decode capture set has five static shapes.  Custom capture
# lists above Dynamo's recompile limit remain correct but can fall back to eager.
moe_align_block_size_compiled = torch.compile(
    moe_align_block_size_torch,
    fullgraph=True,
    dynamic=False,
)


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
) -> Dict[str, Any]:
    """Select default block sizes based on problem dimensions.

    Tuned on H20 (SM90) with Triton 3.4.0. num_warps and num_stages are
    critical for performance — triton defaults are suboptimal on SM90.
    """
    avg_tokens_per_expert = M * top_k / max(E, 1)

    if avg_tokens_per_expert <= 2:
        # Decode / very small batch: maximize N-parallelism
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        }
    elif avg_tokens_per_expert <= 16:
        # Small batch: large K block for memory bandwidth
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif avg_tokens_per_expert <= 48:
        # Medium batch: BM=32 balances padding waste vs parallelism
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        # Large batch: maximize compute throughput with bigger M blocks
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }

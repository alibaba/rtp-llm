import math
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.arch import is_cuda

if TYPE_CHECKING:
    from rtp_llm.models_py.distributed.collective_torch import Fp8Activation


@torch.compile(backend="inductor")
def gate_sigmoid_mul(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return attn_out * gate.sigmoid()

@triton.jit
def _silu_and_mul_kernel(
    output_ptr,
    input_ptr,
    # Tensor dimensions
    N: tl.int32,
    # Row strides for jumping between batches
    input_row_stride: tl.int32,
    output_row_stride: tl.int32,
    # Meta-parameter for tuning
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)  # Batch dimension
    pid_n_block = tl.program_id(axis=1)  # N-dimension block

    input_row_start_ptr = input_ptr + pid_b * input_row_stride
    output_row_start_ptr = output_ptr + pid_b * output_row_stride

    n_offsets = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    value_ptrs = input_row_start_ptr + n_offsets
    gate_ptrs = input_row_start_ptr + N + n_offsets
    output_ptrs = output_row_start_ptr + n_offsets

    mask = n_offsets < N
    gate = tl.load(gate_ptrs, mask=mask)
    value = tl.load(value_ptrs, mask=mask)

    silu_gate = gate * tl.sigmoid(gate.to(tl.float32))
    output = silu_gate * value

    tl.store(output_ptrs, output, mask=mask)


def silu_and_mul(
    output_tensor: torch.Tensor, input_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes SiLU(gate) * value in a fused Triton kernel.
    Assumes input_tensor has shape [B, 2*N] and is contiguous.
    """
    B, D = input_tensor.shape
    assert D % 2 == 0, "Last dimension must be even (2*N)"
    N = D // 2

    # Kernel launch grid
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    # Heuristic for block size
    BLOCK_SIZE_N = 1024 if N > 1024 else triton.next_power_of_2(N)

    _silu_and_mul_kernel[grid](
        output_tensor,
        input_tensor,
        N,
        input_tensor.stride(0),
        output_tensor.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output_tensor


@triton.jit
def _silu_mul_fp8_quant_deep_gemm_masked(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_up_offset = base_input_offset + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if use_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def silu_mul_fp8_quant_deep_gemm_masked(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    group_size: int = 128,
    use_ue8m0: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales

    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert (
        tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E
    ), "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # allocate outputs
    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided(
        (E, T, G),
        (stride_ys_e, stride_ys_t, stride_ys_g),
        dtype=torch.float32,
        device=y.device,
    )

    stride_cnt_e = tokens_per_expert.stride()[0]

    # Static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = f_info.min

    _silu_mul_fp8_quant_deep_gemm_masked[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_ys_e,
        stride_ys_t,
        stride_ys_g,
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0,
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return y_q, y_s


@triton.jit
def _silu_mul_bf16_deep_gemm_masked(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_o_ptr,  # bf16 output (E, T, H)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_o (elements) -----------------------------------------
    stride_yo_e,
    stride_yo_t,
    stride_yo_h,
    # Stride for counts (elements)
    stride_counts_e,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_up_offset = base_input_offset + cols * stride_i_h
    base_yo_offset = e * stride_yo_e + g * GROUP_SIZE * stride_yo_h + cols * stride_yo_h

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        yo = (gate * up).to(tl.bfloat16)

        tl.store(y_o_ptr + base_yo_offset + t * stride_yo_t, yo, mask=mask)


def silu_mul_bf16_deep_gemm_masked(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    group_size: int = 256,
) -> torch.Tensor:
    """Not quantize BF16 silu(y[..., :H]) * y[..., H:]

    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half.

    Returns `y_o` where
    * `y_o`: BF16 tensor, shape (E, T, H), same layout as y[..., :H]
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert (
        tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E
    ), "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # allocate outputs
    y_o = torch.empty((E, T, H), dtype=torch.bfloat16, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yo_e, stride_yo_t, stride_yo_h = y_o.stride()
    stride_cnt_e = tokens_per_expert.stride()[0]

    # Static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G,)

    _silu_mul_bf16_deep_gemm_masked[grid](
        y,
        y_o,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yo_e,
        stride_yo_t,
        stride_yo_h,
        stride_cnt_e,
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return y_o


# TODO(serina.wzq): 优化两次loop
@triton.jit
def _silu_mul_fp8_per_token_quant_batched(
    input_ptr,
    y_q_ptr,
    y_s_ptr,
    counts_ptr,
    H: tl.constexpr,  # hidden size
    T: tl.constexpr,  # max_num_tokens
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    stride_ys_e,
    stride_ys_t,
    stride_cnt_e,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)

    n_tokens = tl.load(counts_ptr + e * stride_cnt_e).to(tl.int64)

    if t >= n_tokens:
        return

    base_offset_e_t = e * stride_i_e + t * stride_i_t

    max_val = 0.0
    # 计算quant scale
    for h_offset in tl.range(0, H, BLOCK_SIZE):
        h_indices = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        h_actual = h_offset + h_indices
        h_mask = h_actual < H

        gate_offset = base_offset_e_t + (H + h_actual) * stride_i_h
        up_offset = base_offset_e_t + h_actual * stride_i_h

        gate = tl.load(input_ptr + gate_offset, mask=h_mask, other=0.0).to(tl.float32)
        up = tl.load(input_ptr + up_offset, mask=h_mask, other=0.0).to(tl.float32)
        y = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        block_max = tl.max(tl.abs(y))
        max_val = tl.maximum(max_val, block_max)
    scale = tl.maximum(max_val, eps) / fp8_max
    if use_ue8m0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    scale_inv = 1.0 / scale

    # silu_mul and quant
    for h_offset in tl.range(0, H, BLOCK_SIZE):
        h_indices = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        h_actual = h_offset + h_indices
        h_mask = h_actual < H

        gate_offset = base_offset_e_t + (H + h_actual) * stride_i_h
        up_offset = base_offset_e_t + h_actual * stride_i_h

        gate = tl.load(input_ptr + gate_offset, mask=h_mask, other=0.0).to(tl.float32)
        up = tl.load(input_ptr + up_offset, mask=h_mask, other=0.0).to(tl.float32)
        y = gate * (1.0 / (1.0 + tl.exp(-gate))) * up

        y_q = tl.clamp(y * scale_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
        yq_offset = e * stride_yq_e + t * stride_yq_t + h_actual * stride_yq_h
        tl.store(y_q_ptr + yq_offset, y_q, mask=h_mask)

    tl.store(y_s_ptr + e * stride_ys_e + t * stride_ys_t, scale)


def silu_mul_fp8_per_token_quant_batched(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,)
    use_ue8m0: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize silu(y[..., :H]) * y[..., H:] to FP8 with per-token scales
    Args:
        y: (E * T, 2*H) or (E, T, 2*H)
        tokens_per_expert: (E,) 每个expert真正的token数
        use_ue8m0: 是否舍入scale到2的幂次
        eps: 数值稳定性
    Returns:
        y_q: (E, T, H), dtype为fp8
        y_s: (E, T), per token 的scale, dtype为fp32
    """
    if y.ndim == 3:
        E, T, H2 = y.shape
        assert E == tokens_per_expert.shape[0]
    elif y.ndim == 2:
        E = tokens_per_expert.shape[0]
        E_T, H2 = y.shape
        T = E_T // E
        y = y.view(E, T, H2)
    else:
        raise RuntimeError(
            f"unsupported input dim for silu_mul_fp8_per_token_quant_batched"
        )

    assert H2 % 2 == 0
    H = H2 // 2

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)
    y_s = torch.empty((E, T), dtype=torch.float32, device=y.device)

    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
    stride_ys_e, stride_ys_t = y_s.stride()
    stride_cnt_e = tokens_per_expert.stride()[0]

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = f_info.min

    grid = (E, T)
    BLOCK_SIZE = 1024

    _silu_mul_fp8_per_token_quant_batched[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H=H,
        T=T,
        stride_i_e=stride_i_e,
        stride_i_t=stride_i_t,
        stride_i_h=stride_i_h,
        stride_yq_e=stride_yq_e,
        stride_yq_t=stride_yq_t,
        stride_yq_h=stride_yq_h,
        stride_ys_e=stride_ys_e,
        stride_ys_t=stride_ys_t,
        stride_cnt_e=stride_cnt_e,
        eps=eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y_q.view(-1, H), y_s.view(-1)


# copy from https://github.com/ModelTC/lightllm/blob/a000ab69098654df4731f5b12587dd4e7f0a4f41/lightllm/common/fused_moe/moe_silu_and_mul_mix_quant_ep.py
@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        # ours weights first up then gate
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    return


# Modified to integrate pack_ue8m0 logic for Blackwell with UE8M0 scale packing
@triton.jit
def _silu_and_mul_post_quant_packed_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,  # Packed int32 scales (UE8M0 format)
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,  # group_size (e.g., 128)
    NUM_STAGE: tl.constexpr,
):
    """
    Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing.

    This kernel processes 4 consecutive groups at once and packs their scales
    into a single int32 in UE8M0 format (extracting exponent bits from float32).

    Output scale format:
    - 4 consecutive float32 scales are packed into 1 int32
    - packed = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    - Where exp_i = (float32_bits >> 23) & 0xFF
    """
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    # Now this index represents a "packed group" (4 groups packed together)
    packed_group_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    # Base offset for this expert
    input_base = input_ptr + expert_id * stride_input_0
    output_base = output_ptr + expert_id * stride_output_0
    output_scale_base = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + packed_group_index * stride_output_scale_2
    )

    # Process 4 groups at once (each group has BLOCK_N elements)
    # Group indices: packed_group_index * 4 + [0, 1, 2, 3]
    base_group_idx = packed_group_index * 4

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        # Initialize packed scale value
        packed_scale: tl.int32 = 0

        # Process 4 groups
        for g in tl.static_range(4):
            group_idx = base_group_idx + g
            offs_in_d = group_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Check if this group is within bounds
            mask = offs_in_d < size_n

            # Load gate and up values (our weights: first up, then gate)
            gate = tl.load(
                input_base + token_index * stride_input_1 + offs_in_d + size_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                input_base + token_index * stride_input_1 + offs_in_d,
                mask=mask,
                other=0.0,
            )

            # SiLU activation
            gate = gate / (1 + tl.exp(-gate))
            gate = gate.to(input_ptr.dtype.element_ty)
            gate_up = up * gate

            # Compute scale with UE8M0 rounding (power of 2)
            _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
            output_s = _absmax / fp8_max
            # Round to power of 2 (UE8M0 format requires this)
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))

            # Quantize to FP8
            output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
                output_ptr.dtype.element_ty
            )

            # Store quantized output
            tl.store(
                output_base + token_index * stride_output_1 + offs_in_d,
                output_q,
                mask=mask,
            )

            # Extract exponent from float32 scale for UE8M0 packing
            # float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits
            # We extract the 8 exponent bits by bitcasting to int32 and shifting
            scale_bits = output_s.to(tl.int32, bitcast=True)
            exp_bits = (scale_bits >> 23) & 0xFF

            # Pack this exponent into the appropriate byte position
            # Little endian: exp0 at bits[0:7], exp1 at bits[8:15], etc.
            packed_scale = packed_scale | (exp_bits << (g * 8))

        # Store packed scale (one int32 containing 4 UE8M0 exponents)
        tl.store(
            output_scale_base + token_index * stride_output_scale_1,
            packed_scale,
        )


def create_packed_scale_tensor(
    expert_num: int,
    token_num_padded: int,
    hidden_dim: int,
    quant_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a properly-shaped output_scale tensor for UE8M0 packed format.

    The tensor is created with column-major layout for the packed dimension
    to be compatible with deep_gemm's expected scale format.

    Args:
        expert_num: Number of experts (E)
        token_num_padded: Padded token count per expert (T)
        hidden_dim: Hidden dimension (2*H, before split)
        quant_group_size: Quantization group size (typically 128)
        device: Target device

    Returns:
        output_scale: int32 tensor with shape (E, T, G // 4) in column-major layout
                      where G = hidden_dim // 2 // quant_group_size
    """
    H = hidden_dim // 2
    G = H // quant_group_size
    assert G % 4 == 0, "Number of groups must be divisible by 4 for UE8M0 packing"
    G_packed = G // 4

    # Create storage in column-major layout for the packed dimension
    # Storage shape: (E, G_packed, T) to get column-major K dimension
    packed_storage = torch.empty(
        (expert_num, G_packed, token_num_padded),
        device=device,
        dtype=torch.int32,
    )
    # Transpose to get (E, T, G_packed) view with column-major K strides
    # This gives strides: (G_packed * T, 1, T) for (E, T, G_packed) shape
    output_scale = packed_storage.transpose(1, 2)

    return output_scale


def silu_and_mul_masked_post_quant_packed_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
):
    """
    Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing.

    This function integrates the pack_ue8m0 logic directly into the quantization kernel,
    eliminating the need for a separate packing pass.

    Args:
        input: shape [expert_num, token_num_padded, hidden_dim], dtype bf16/fp16
        output: shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
        output_scale: shape [expert_num, token_num_padded, hidden_dim // 2 // group_size // 4],
                      dtype int32 (packed UE8M0 format), use create_packed_scale_tensor() to create
        quant_group_size: int, typically 128
        masked_m: shape [expert_num], number of valid tokens per expert
    """
    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    num_groups = size_n // quant_group_size
    assert (
        num_groups % 4 == 0
    ), "Number of groups must be divisible by 4 for UE8M0 packing"

    num_packed_groups = num_groups // 4

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6

    grid = (
        num_packed_groups,  # Each block processes 4 groups
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_packed_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
    )
    return


_SKEW_HOT_LOAD_MULTIPLIER = 8
_SKEW_TOKENS_PER_WORKER = 32
_SKEW_MAX_WORKERS = 16


class MaskedSiluInputLayout(Enum):
    """Meaning of the masked SiLU input's padded-token dimension."""

    PER_EXPERT_CAPACITY = "per_expert_capacity"
    BATCH_TOKEN_ALIGNMENT = "batch_token_alignment"


def _heuristic_params(
    expert_num: int,
    expected_m: int,
    token_num_padded: int,
    input_layout: MaskedSiluInputLayout,
) -> Tuple[int, int, int]:
    """Select launch parameters for the masked SiLU kernels.

    The baseline rules come from a parameter search over 12,978 points. A
    separate manual correction covers heavily skewed decode traffic measured
    on H20 with 96 local experts, average load 43, and padded capacity 512.

    Args:
        expert_num: Number of local experts handled by this kernel invocation
        expected_m: Expected (average) number of tokens per expert
        token_num_padded: Size of the input's padded-token dimension
        input_layout: Whether ``token_num_padded`` is an independently
            provisioned capacity for every expert or an alignment derived from
            the total contiguous token batch

    Returns:
        (BLOCK_NUM_PER_EXPERT, NUM_STAGES, num_warps)
    """
    # BLOCK_NUM_PER_EXPERT heuristic rules (based on actual test data)
    if expert_num < 16:
        # Small number of experts: use 32 for all expected_m ranges
        block_num_per_expert = 32
    elif expert_num < 64:
        # Medium number of experts: 2 for small expected_m, 16 otherwise
        block_num_per_expert = 2 if expected_m < 128 else 16
    else:  # expert_num >= 64
        # Large number of experts: 1 for small expected_m, 8 otherwise
        block_num_per_expert = 1 if expected_m < 128 else 8

    # NUM_STAGES heuristic rules (based on actual test data)
    # The baseline search uses 2 stages below expected_m 64 and 4 otherwise.
    num_stages = 2 if expected_m < 64 else 4

    # num_warps heuristic rules (based on actual test data)
    # Overwhelming majority (8151/12978) use 1 warp
    num_warps = 1

    # expected_m can underestimate a hot expert in skewed beam-search traffic.
    # Only a per-expert capacity is eligible for the deployment-specific skew
    # correction. In the hybrid executor the same dimension is instead aligned
    # from the whole contiguous batch, so its capacity/average ratio carries no
    # information about expert skew. Reading masked_m on the host is deliberately
    # avoided because this selection also runs while preparing CUDA graphs.
    hot_load_estimate = max(expected_m, 1) * _SKEW_HOT_LOAD_MULTIPLIER
    if (
        input_layout is MaskedSiluInputLayout.PER_EXPERT_CAPACITY
        and expected_m < 128
        and token_num_padded > max(128, hot_load_estimate)
    ):
        rounded_hot_load = hot_load_estimate + _SKEW_TOKENS_PER_WORKER - 1
        workers_for_hot_load = max(1, rounded_hot_load // _SKEW_TOKENS_PER_WORKER)
        # Round up to a power of two, then cap empty-expert launch overhead.
        workers_for_hot_load = 1 << (workers_for_hot_load - 1).bit_length()
        workers_for_hot_load = min(_SKEW_MAX_WORKERS, workers_for_hot_load)
        if workers_for_hot_load > block_num_per_expert:
            block_num_per_expert = workers_for_hot_load
            # Four stages suit shorter per-worker loops; the large skew case is
            # faster with three stages on H20.
            skew_num_stages = 4 if hot_load_estimate <= 128 else 3
            num_stages = max(num_stages, skew_num_stages)

    return (block_num_per_expert, num_stages, num_warps)


@triton.jit
def _silu_mul_masked_fp8_post_quant_fwd(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_mul_masked_fp8_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    expected_m: int,
    input_layout: MaskedSiluInputLayout = MaskedSiluInputLayout.PER_EXPERT_CAPACITY,
    scale_ue8m0: bool = False,
):
    """
    SiLU and multiply with masked FP8 post-quantization forward pass.

    Args:
        input: Input tensor with shape [expert_num, token_num_padded, hidden_dim]
        output: Output tensor with shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
        output_scale: Output scale tensor with shape [expert_num, token_num_padded, hidden_dim // 2 // 128], dtype float32
        quant_group_size: Quantization group size
        masked_m: Mask tensor with shape [expert_num]
        expected_m: Expected number of tokens
        input_layout: Meaning of ``input.shape[1]`` for heuristic selection
        scale_ue8m0: Whether to use ue8m0 scaling format
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    # Use heuristic rules to determine optimal configuration
    BLOCK_NUM_PER_EXPERT, NUM_STAGES, num_warps = _heuristic_params(
        expert_num=expert_num,
        expected_m=expected_m,
        token_num_padded=input.shape[1],
        input_layout=input_layout,
    )

    BLOCK_N = quant_group_size
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_mul_masked_fp8_post_quant_fwd[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    return


@triton.jit
def _silu_mul_masked_bf16_no_post_quant_fwd(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    masked_m_ptr,
    size_n,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = (up * gate).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            gate_up,
            mask=offs_in_d < size_n,
        )


def silu_mul_masked_bf16_no_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    input_layout: MaskedSiluInputLayout = MaskedSiluInputLayout.PER_EXPERT_CAPACITY,
    group_size: int = 256,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype bf16
    masked_m shape [expert_num],
    expected_m int,
    input_layout MaskedSiluInputLayout,
    group_size int,
    """

    assert input.is_contiguous()
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2

    expert_num = len(masked_m)

    # Use heuristic rules to determine optimal configuration
    BLOCK_NUM_PER_EXPERT, NUM_STAGES, num_warps = _heuristic_params(
        expert_num=expert_num,
        expected_m=expected_m,
        token_num_padded=input.shape[1],
        input_layout=input_layout,
    )

    BLOCK_N = group_size
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    _silu_mul_masked_bf16_no_post_quant_fwd[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        masked_m,
        size_n,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    return


@triton.jit
def store_group128(values, out, scales, token, M, K: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    values = values.to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(tl.where(cols < K, values, 0.0), (BLOCK // 128, 128))
    amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.reshape(
        tl.minimum(tl.maximum(grouped / scale[:, None], -448.0), 448.0), (BLOCK,)
    )
    tl.store(out + token * K + cols, quantized, cols < K)
    groups = tl.arange(0, BLOCK // 128)
    exp = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    exp = tl.where(groups < K // 128, exp, 127)
    exp4 = tl.reshape(exp, (BLOCK // 512, 4))
    shifts = tl.arange(0, 4) * 8
    packed = tl.sum(exp4 << shifts[None, :], axis=1)
    pg = tl.arange(0, BLOCK // 512)
    aligned_m = tl.cdiv(M, 4) * 4
    tl.store(scales + pg * aligned_m + token, packed, pg < triton.cdiv(K, 512))
    if token == 0:
        padding = M + tl.arange(0, 4)
        tl.store(
            scales + pg[:, None] * aligned_m + padding[None, :],
            0x7F7F7F7F,
            (pg[:, None] < triton.cdiv(K, 512)) & (padding[None, :] < aligned_m),
        )


def is_fp8_attn_res_supported(
    prefix_sum, block_residual, norm_weight, projection_weight,
    output_norm_weight, delta, active_blocks, block_write_idx,
):
    if (
        not prefix_sum.is_cuda
        or prefix_sum.ndim != 2
        or prefix_sum.shape[0] == 0
        or prefix_sum.shape[1] != 7168
        or prefix_sum.stride(-1) != 1
        or prefix_sum.dtype != torch.bfloat16
        or active_blocks < 0
        or active_blocks > 8
        or block_residual.ndim != 3
        or block_residual.shape[0] != prefix_sum.shape[0]
        or block_residual.shape[2] != prefix_sum.shape[1]
        or block_residual.shape[1] < active_blocks
        or block_residual.stride(-1) != 1
        or block_residual.dtype != prefix_sum.dtype
        or block_residual.device != prefix_sum.device
        or block_write_idx < -1
        or block_write_idx >= block_residual.shape[1]
    ):
        return False
    if delta is not None and (
        delta.shape != prefix_sum.shape
        or delta.stride(-1) != 1
        or delta.dtype != prefix_sum.dtype
        or delta.device != prefix_sum.device
    ):
        return False
    weights = (norm_weight, projection_weight, output_norm_weight)
    return all(
        weight is not None
        and weight.device == prefix_sum.device
        and weight.numel() == prefix_sum.shape[1]
        and weight.reshape(-1).stride(0) == 1
        for weight in weights
    )


@triton.jit(do_not_specialize=["token_count"])
def _multi_block_attn_res_fp8_kernel(
    prefix,
    delta,
    blocks,
    norm_weight,
    projection_weight,
    output_norm_weight,
    output,
    output_scales,
    token_count,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    has_delta: tl.constexpr,
    write_block: tl.constexpr,
    apply_output_norm: tl.constexpr,
    block_l: tl.constexpr,
    block_d: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    hidden_offsets = tl.max_contiguous(tl.arange(0, block_d), block_d)
    hidden_mask = hidden_offsets < hidden_size
    updated_prefix = tl.load(
        prefix + token * stride_prefix_m + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    if has_delta:
        updated_prefix += tl.load(
            delta + token * stride_delta_m + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix = updated_prefix.to(prefix.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix + token * stride_prefix_m + hidden_offsets,
            updated_prefix,
            mask=hidden_mask,
        )
    if write_block:
        tl.store(
            blocks
            + token * stride_block_m
            + block_write_idx * stride_block_r
            + hidden_offsets,
            updated_prefix,
            mask=hidden_mask,
        )

    if num_blocks == 0:
        mixed = updated_prefix
    else:
        if has_delta:
            tl.debug_barrier()
        input_projection_weight = tl.load(
            norm_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32) * tl.load(
            projection_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(
            tl.float32
        )
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((block_d,), tl.float32)
        num_sources: tl.constexpr = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, block_l)):
            source_offsets = source_tile * block_l + tl.arange(0, block_l)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_ptrs = (
                blocks
                + token * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + hidden_offsets[None, :]
            )
            prefix_ptrs = (
                prefix
                + token * stride_prefix_m
                + source_offsets[:, None] * 0
                + hidden_offsets[None, :]
            )
            value_ptrs = tl.where(is_prefix[:, None], prefix_ptrs, block_ptrs)
            values = tl.load(
                value_ptrs,
                mask=source_mask[:, None] & hidden_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            inverse_rms = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = (
                tl.sum(values * input_projection_weight[None, :], axis=1) * inverse_rms
            )
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            source_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(source_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(source_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

        mixed /= denominator
    if apply_output_norm:
        # Preserve both eager BF16 rounding points: the AttnRes output store
        # and RMSNorm's normalized activation before the affine multiply.
        mixed = mixed.to(prefix.dtype.element_ty).to(tl.float32)
        output_inverse_rms = tl.rsqrt(
            tl.sum(tl.where(hidden_mask, mixed * mixed, 0.0), axis=0)
            * (1.0 / hidden_size)
            + output_norm_eps
        )
        normalized = (mixed * output_inverse_rms).to(prefix.dtype.element_ty)
        gamma = tl.load(
            output_norm_weight + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        mixed = normalized.to(tl.float32) * gamma
    store_group128(
        mixed, output, output_scales, token, token_count, hidden_size, block_d
    )


@torch.compiler.disable
def kimi_k3_attn_res_fp8(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
    output_norm_weight: Optional[torch.Tensor] = None,
    output_norm_eps: Optional[float] = None,
    delta: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    block_write_idx: int = -1,
) -> "Fp8Activation":
    from rtp_llm.models_py.distributed.collective_torch import Fp8Activation

    active_blocks = block_residual.shape[1] if num_blocks is None else int(num_blocks)
    if not is_fp8_attn_res_supported(
        prefix_sum,
        block_residual,
        norm_weight,
        projection_weight,
        output_norm_weight,
        delta,
        active_blocks,
        block_write_idx,
    ):
        raise ValueError("unsupported input for K3 fused AttnRes")
    if output_norm_weight is not None and output_norm_eps is None:
        raise ValueError("output_norm_eps is required when output RMSNorm is fused")

    m, k = prefix_sum.shape
    output = Fp8Activation(
        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=prefix_sum.device),
        torch.empty(
            ((k + 511) // 512, (m + 3) // 4 * 4),
            dtype=torch.int32,
            device=prefix_sum.device,
        ),
    )
    if prefix_sum.shape[0] >= 256 or active_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _multi_block_attn_res_fp8_kernel[(prefix_sum.shape[0],)](
        prefix_sum,
        delta,
        block_residual,
        norm_weight,
        projection_weight.reshape(-1),
        output_norm_weight,
        output.values,
        output.scale_wire,
        prefix_sum.shape[0],
        prefix_sum.stride(0),
        0 if delta is None else delta.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        output.values.stride(0),
        num_blocks=active_blocks,
        hidden_size=prefix_sum.shape[1],
        block_write_idx=block_write_idx,
        eps=float(eps),
        output_norm_eps=(0.0 if output_norm_eps is None else float(output_norm_eps)),
        has_delta=delta is not None,
        write_block=block_write_idx >= 0,
        apply_output_norm=output_norm_weight is not None,
        block_l=block_l,
        block_d=max(512, triton.next_power_of_2(prefix_sum.shape[1])),
        num_warps=num_warps,
        num_stages=2,
    )
    return output


# MLA NoPE/RoPE key concatenation.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from sglang/python/sglang/srt/layers/attention/utils.py

@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    head_block: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    # A long prefix can make row * stride exceed signed 32-bit element offsets.
    pid_loc = tl.program_id(0).to(tl.int64)
    head_range = tl.arange(0, head_block)
    valid_head = head_range[:, None] < head_cnt

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr, mask=valid_head, other=0)
    tl.store(dst_nope_ptr, src_nope, mask=valid_head)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope, mask=valid_head)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        triton.next_power_of_2(k.shape[1]),
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


def concat_and_cast_mha_k(k_nope, k_pe, num_heads, nope_dim, rope_dim):
    """Build [tokens, heads, nope + rope] keys with a shared RoPE head.

    Reuse the native BF16 binding for the CUDA 128-head/128+64 layout,
    the Triton implementation for other supported head layouts, and the
    existing tensor-copy path for remaining layouts and platforms.
    """
    k_shape = (
        k_nope.shape[0],
        num_heads,
        nope_dim + rope_dim,
    )
    if (
        is_cuda()
        and (num_heads == 128)
        and (nope_dim == 128)
        and (rope_dim == 64)
    ):
        k = k_nope.new_empty(*k_shape)
        # Keep the compiled binding dependency local to its native dispatch.
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        rtp_llm_ops.mla_k_merge(k, k_nope, k_pe)
    elif is_cuda() and _is_triton_compatible(num_heads, nope_dim, rope_dim):
        # Triton kernel requires dimensions to be power of 2
        attn_dtype = k_nope.dtype
        k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
        concat_and_cast_mha_k_triton(k, k_nope, k_pe)
    else:
        # Fallback to PyTorch native operations for non-power-of-2 dimensions
        k = k_nope.new_empty(*k_shape)
        k[..., : nope_dim] = k_nope
        k[..., nope_dim :] = k_pe
    return k


def _is_triton_compatible(num_heads, nope_dim, rope_dim):
    """Check if dimensions are compatible with Triton kernel (must be power of 2)."""

    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0

    compatible = (
        is_power_of_2(nope_dim)
        and is_power_of_2(rope_dim)
        and num_heads > 0
    )
    return compatible


# FP8 activation quantization and fused pointwise producers.

@triton.jit
def _store_group128(values, out, scales, row, rows, width: tl.constexpr, block: tl.constexpr):
    columns = tl.arange(0, block)
    # The existing quantizer observes the BF16 result of the pointwise op.
    values = values.to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(tl.where(columns < width, values, 0.0), (block // 128, 128))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.reshape(
        tl.minimum(tl.maximum(groups / scale[:, None], -448.0), 448.0), (block,)
    )
    tl.store(out + row * width + columns, quantized, columns < width)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    # Match the existing quantizer's legacy/v2 padding at its 4M-element switch.
    pad_exponent = tl.where(rows * width >= 4 * 1024 * 1024, 0, 0x3F)
    exponent = tl.where(
        tl.arange(0, block // 128) < width // 128, exponent, pad_exponent
    )
    packed = tl.sum(
        tl.reshape(exponent, (block // 512, 4)) << (tl.arange(0, 4)[None, :] * 8),
        axis=1,
    )
    packs = tl.arange(0, block // 512)
    aligned_rows = tl.cdiv(rows, 4) * 4
    tl.store(scales + packs * aligned_rows + row, packed, packs < triton.cdiv(width, 512))
    if row == 0:
        padding = rows + tl.arange(0, 4)
        tl.store(
            scales + packs[:, None] * aligned_rows + padding[None, :],
            0x7F7F7F7F,
            (packs[:, None] < triton.cdiv(width, 512))
            & (padding[None, :] < aligned_rows),
        )


@triton.jit(do_not_specialize=["rows"])
def _sigmoid_mul_group128_kernel(
    x,
    gate,
    out,
    scales,
    rows,
    width: tl.constexpr,
    x_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    a = tl.load(x + row * x_row_stride + columns, columns < width, other=0.0).to(
        tl.float32
    )
    b = tl.load(
        gate + row * gate_row_stride + columns, columns < width, other=0.0
    ).to(tl.float32)
    # Match BF16 eager gating: sigmoid rounds before the multiply, then the
    # grouped quantizer observes the BF16 product.
    sigmoid_bf16 = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    _store_group128(a * sigmoid_bf16, out, scales, row, rows, width, block)


def sigmoid_mul_per_token_group_quant_fp8(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grouped E4M3 values and packed UE8M0 scales for ``x*sigmoid(gate)``.

    The scale layout matches ``sgl_per_token_group_quant_fp8`` with group 128,
    column-major TMA-aligned scales, and UE8M0 packing.
    """
    if x.ndim != 2 or gate.shape != x.shape:
        raise ValueError("FP8 sigmoid-mul requires matching [rows, width] inputs")
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("FP8 sigmoid-mul requires BF16 inputs")
    if not x.is_cuda or gate.device != x.device:
        raise ValueError("FP8 sigmoid-mul inputs must share a CUDA device")
    if x.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("FP8 sigmoid-mul requires contiguous feature dimensions")
    rows, width = x.shape
    if width == 0 or width % 128:
        raise ValueError("FP8 sigmoid-mul width must be divisible by 128")

    out = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scale_wire = torch.empty(
        ((width + 511) // 512, (rows + 3) // 4 * 4),
        device=x.device,
        dtype=torch.int32,
    )
    if rows:
        _sigmoid_mul_group128_kernel[(rows,)](
            x,
            gate,
            out,
            scale_wire,
            rows,
            width,
            x.stride(0),
            gate.stride(0),
            max(512, triton.next_power_of_2(width)),
        )
    return out, scale_wire.T[:rows, :]


@triton.jit(do_not_specialize=["rows"])
def _rmsnorm_sigmoid_gate_group128_kernel(
    x,
    gate,
    weight,
    out,
    scales,
    rows,
    heads: tl.constexpr,
    x_row_stride: tl.constexpr,
    x_head_stride: tl.constexpr,
    x_dim_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    gate_head_stride: tl.constexpr,
    gate_dim_stride: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    head = columns // 128
    dim = columns % 128
    valid = columns < heads * 128
    a = tl.load(
        x + row * x_row_stride + head * x_head_stride + dim * x_dim_stride,
        valid,
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        gate
        + row * gate_row_stride
        + head * gate_head_stride
        + dim * gate_dim_stride,
        valid,
        other=0.0,
    ).to(tl.float32)
    gamma = tl.load(weight + dim).to(tl.float32)
    groups = tl.reshape(a, (block // 128, 128))
    variance = tl.sum(groups * groups, axis=1) / 128.0
    # Preserve the native gated-norm kernel's BF16 rounding at quantization input.
    inverse = 1 / tl.sqrt(variance + eps)
    normalized = tl.reshape(groups * inverse[:, None], (block,))
    _store_group128(
        normalized * gamma * tl.sigmoid(b),
        out,
        scales,
        row,
        rows,
        heads * 128,
        block,
    )


def rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head RMSNorm, sigmoid gate and group128 E4M3 production.

    Inputs are BF16 ``[tokens, heads, 128]``. Scales use the same packed UE8M0
    layout as ``sigmoid_mul_per_token_group_quant_fp8``.
    """
    if x.ndim != 3 or gate.shape != x.shape or x.shape[-1] != 128:
        raise ValueError("FP8 gated RMSNorm requires matching [tokens, heads, 128] inputs")
    if weight.shape != (128,):
        raise ValueError("FP8 gated RMSNorm requires a 128-element weight")
    if x.dtype != torch.bfloat16 or gate.dtype != x.dtype or weight.dtype != x.dtype:
        raise ValueError("FP8 gated RMSNorm requires BF16 inputs and weight")
    if not x.is_cuda or gate.device != x.device or weight.device != x.device:
        raise ValueError("FP8 gated RMSNorm tensors must share a CUDA device")
    rows, heads, _ = x.shape
    if heads <= 0:
        raise ValueError("FP8 gated RMSNorm requires at least one head")
    width = heads * 128
    out = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scale_wire = torch.empty(
        ((width + 511) // 512, (rows + 3) // 4 * 4),
        device=x.device,
        dtype=torch.int32,
    )
    if rows:
        _rmsnorm_sigmoid_gate_group128_kernel[(rows,)](
            x,
            gate,
            weight,
            out,
            scale_wire,
            rows,
            heads,
            *x.stride(),
            *gate.stride(),
            eps,
            max(512, triton.next_power_of_2(width)),
        )
    return out, scale_wire.T[:rows, :]


@triton.jit(do_not_specialize=["rows"])
def _quantize_strided_group128_kernel(
    x,
    out,
    scales,
    rows,
    row_stride: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    values = tl.load(
        x + row[:, None] * row_stride + column[None, :],
        row[:, None] < rows,
        other=0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.minimum(tl.maximum(values / scale[:, None], -448.0), 448.0)
    tl.store(out + row[:, None] * 128 + column[None, :], quantized, row[:, None] < rows)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    packed = tl.where(row < rows, exponent | 0x7F7F7F00, 0x7F7F7F7F)
    tl.store(scales + row, packed, row < ((rows + 3) // 4) * 4)


def quantize_strided_group128_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-strided BF16 [M,128] view without a staging copy."""
    if x.ndim != 2 or x.shape[1] != 128 or x.stride(1) != 1 or x.stride(0) < 128:
        raise ValueError("FP8 strided group128 input must be row-strided [M,128]")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("FP8 strided group128 input must be CUDA BF16")
    rows = x.shape[0]
    out = torch.empty((rows, 128), dtype=torch.float8_e4m3fn, device=x.device)
    scale_wire = torch.empty(
        (1, triton.cdiv(rows, 4) * 4), dtype=torch.int32, device=x.device
    ).T
    if rows:
        _quantize_strided_group128_kernel[(triton.cdiv(rows, 4),)](
            x, out, scale_wire, rows, x.stride(0), 4, num_warps=4
        )
    return out, scale_wire[:rows]


@triton.jit
def _convert_fp8(X, Y, source, offsets, N, INV_SCALE: tl.constexpr):
    values = tl.load(X + source, offsets < N, other=0.0).to(tl.float32) * INV_SCALE
    # Preserve NaNs and saturate infinities/overflow to E4M3's finite range.
    values = tl.where(
        values != values, values, tl.minimum(tl.maximum(values, -448.0), 448.0)
    )
    tl.store(Y + offsets, values, offsets < N)


@triton.jit(do_not_specialize=["N", "SHAPE", "STRIDES"])
def _quantize(
    X, Y, N, INV_SCALE: tl.constexpr, BLOCK: tl.constexpr,
    SHAPE, STRIDES, CONTIGUOUS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        src = offsets
    else:
        src = tl.full((BLOCK,), 0, tl.int64)
        remaining = offsets
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            src += (remaining % SHAPE[dim]).to(tl.int64) * STRIDES[dim]
            remaining = remaining // SHAPE[dim]
    _convert_fp8(X, Y, src, offsets, N, INV_SCALE)


def quantize_fp8(
    x: torch.Tensor, scale: float = 1.0, out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FP8 MLA scale must be finite and positive")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"FP8 MLA quantizer requires floating activations, got {x.dtype}")
    if out is None:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    if (out.shape != x.shape or out.dtype != torch.float8_e4m3fn
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("FP8 MLA quantization output buffer mismatch")
    if x.numel():
        # Runtime tuples avoid exact-shape specialization. Triton may still
        # keep a bounded set of tuple alignment variants for strided layouts.
        _quantize[(triton.cdiv(x.numel(), 1024),)](
            x, out, x.numel(), 1.0 / scale, 1024,
            () if x.is_contiguous() else tuple(x.shape),
            () if x.is_contiguous() else tuple(x.stride()), x.is_contiguous(),
        )
    return out


@triton.jit
def _quantize_operand(
    X,
    Y,
    block,
    N,
    SHAPE: tl.constexpr,
    STRIDES,
    CONTIGUOUS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    N = N.to(tl.int64)
    if (SHAPE[0] * SHAPE[1]) % 8 == 0:
        # N = tokens * heads * features. Preserve its proven alignment without
        # specializing the token count; otherwise the tail mask scalarizes IO.
        N = tl.multiple_of(N, 8)
    if CONTIGUOUS:
        source = offsets
    else:
        # Only head/feature geometry is specialized. Token counts and strides
        # remain runtime values, with wide products for large prefix views.
        token = offsets // (SHAPE[0] * SHAPE[1])
        head = (offsets // SHAPE[1]) % SHAPE[0]
        feature = offsets % SHAPE[1]
        source = token * STRIDES[0] + head * STRIDES[1] + feature * STRIDES[2]
    _convert_fp8(X, Y, source, offsets, N, INV_SCALE)


@triton.jit(
    do_not_specialize=[
        "Q_N",
        "K_N",
        "V_N",
        "Q_BLOCKS",
        "K_BLOCKS",
        "Q_STRIDES",
        "K_STRIDES",
        "V_STRIDES",
    ]
)
def _quantize_qkv(
    Q,
    K,
    V,
    OQ,
    OK,
    OV,
    Q_N,
    K_N,
    V_N,
    Q_BLOCKS,
    K_BLOCKS,
    Q_SHAPE: tl.constexpr,
    K_SHAPE: tl.constexpr,
    V_SHAPE: tl.constexpr,
    Q_STRIDES,
    K_STRIDES,
    V_STRIDES,
    Q_CONTIGUOUS: tl.constexpr,
    K_CONTIGUOUS: tl.constexpr,
    V_CONTIGUOUS: tl.constexpr,
    Q_INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    # Partition CTAs, not elements: each CTA has one operand and one uniform
    # branch. Unequal Q/KV lengths need neither concatenation nor padded work.
    if block < Q_BLOCKS:
        _quantize_operand(
            Q, OQ, block, Q_N, Q_SHAPE, Q_STRIDES, Q_CONTIGUOUS, Q_INV_SCALE, BLOCK
        )
    elif block < Q_BLOCKS + K_BLOCKS:
        _quantize_operand(
            K, OK, block - Q_BLOCKS, K_N, K_SHAPE, K_STRIDES, K_CONTIGUOUS, 1.0, BLOCK
        )
    else:
        _quantize_operand(
            V,
            OV,
            block - Q_BLOCKS - K_BLOCKS,
            V_N,
            V_SHAPE,
            V_STRIDES,
            V_CONTIGUOUS,
            1.0,
            BLOCK,
        )


def quantize_qkv_fp8(q, k, v, q_scale=1.0):
    """Quantize three [tokens, heads, features] operands without staging.

    Q uses q_scale; expanded K/V use the existing unit scale. Each operand may
    have independent lengths, strides and floating dtype. Return three dense
    E4M3 tensors; there is no change to compressed KV-cache quantization.
    """
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("FP8 MLA scale must be finite and positive")
    inputs = (q, k, v)
    for x in inputs:
        if not x.is_cuda or x.device != q.device:
            raise ValueError("MLA Q/K/V must share one CUDA device")
        if x.ndim != 3 or x.shape[1] <= 0 or x.shape[2] <= 0:
            raise ValueError(
                "MLA Q/K/V require [tokens, heads, features] with positive head/feature sizes"
            )
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("FP8 MLA quantizer requires floating activations")
    outputs = tuple(
        torch.empty_like(
            x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
        )
        for x in inputs
    )
    block = 1024
    sizes = tuple(x.numel() for x in inputs)
    blocks = tuple(triton.cdiv(n, block) for n in sizes)
    if sum(blocks):
        _quantize_qkv[(sum(blocks),)](
            *inputs,
            *outputs,
            *sizes,
            blocks[0],
            blocks[1],
            *(tuple(x.shape[1:]) for x in inputs),
            *(tuple(x.stride()) for x in inputs),
            *(x.is_contiguous() for x in inputs),
            1.0 / q_scale,
            block,
        )
    return outputs


def quantize_kv_fp8(k, v):
    """Quantize expanded MLA K/V in one launch without staging Q."""
    for x in (k, v):
        if not x.is_cuda or x.device != k.device:
            raise ValueError("MLA K/V must share one CUDA device")
        if x.ndim != 3 or x.shape[1] <= 0 or x.shape[2] <= 0:
            raise ValueError("MLA K/V require [tokens, heads, features]")
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("FP8 MLA quantizer requires floating activations")
    out_k, out_v = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn,
                         memory_format=torch.contiguous_format)
        for x in (k, v)
    )
    block = 1024
    k_blocks = triton.cdiv(k.numel(), block)
    v_blocks = triton.cdiv(v.numel(), block)
    if k_blocks + v_blocks:
        # The existing Q/K/V kernel accepts an empty Q partition. Reuse its
        # K and V branches so long historical prefixes need only one launch.
        _quantize_qkv[(k_blocks + v_blocks,)](
            k, k, v, out_k, out_k, out_v,
            0, k.numel(), v.numel(), 0, k_blocks,
            tuple(k.shape[1:]), tuple(k.shape[1:]), tuple(v.shape[1:]),
            tuple(k.stride()), tuple(k.stride()), tuple(v.stride()),
            k.is_contiguous(), k.is_contiguous(), v.is_contiguous(),
            1.0, block,
        )
    return out_k, out_v

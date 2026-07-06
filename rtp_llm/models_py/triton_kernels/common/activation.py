from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _ieee_rn_div_f32(x, y):
    """IEEE round-to-nearest-even fp32 division (matches sgl CUDA `/`)."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
        _absmax = tl.max(tl.abs(gate_up))
        output_s = (_absmax.to(tl.float64) / fp8_max).to(tl.float32)
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_s_inv = (1.0 / output_s.to(tl.float64)).to(tl.float32)
        output_q = tl.clamp(gate_up * output_s_inv, fp8_min, fp8_max).to(
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
            _absmax = tl.max(tl.abs(gate_up))
            output_s = (_absmax.to(tl.float64) / fp8_max).to(tl.float32)
            # Round to power of 2 (UE8M0 format requires this)
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))

            # Quantize to FP8
            output_s_inv = (1.0 / output_s.to(tl.float64)).to(tl.float32)
            output_q = tl.clamp(gate_up * output_s_inv, fp8_min, fp8_max).to(
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


# ---------------------------------------------------------------------------
# 2-D dense version of `silu_and_mul + per-token-group fp8 quant`.
# Strips the (expert, masked_m) axis from `_silu_and_mul_post_quant_packed_kernel`
# so it can be used by `DenseMLP` / `shared_expert` paths whose down_proj is fp8.
# Two output-scale layouts are supported:
#   - SCALE_UE8M0=True  : int32 packed UE8M0, 4 groups per int32, column-major
#                         (matches Blackwell deepgemm's required layout).
#   - SCALE_UE8M0=False : float32 unpacked, column-major TMA-aligned
#                         (matches H20/SM9.0 deepgemm's required layout).
# Both layouts come from `create_per_token_group_quant_fp8_output_scale`.
# ---------------------------------------------------------------------------


@triton.jit
def _silu_and_mul_post_quant_dense_packed_kernel(
    input_ptr,  # [T, 2*H_out]  bf16/fp16
    stride_input_t,
    output_ptr,  # [T, H_out]   fp8_e4m3fn
    stride_output_t,
    output_scale_ptr,
    stride_output_scale_t,  # stride along token dim
    stride_output_scale_g,  # stride along (packed) group dim
    size_n,  # H_out
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,  # group_size (typically 128)
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    ROUND_POW2: tl.constexpr = False,
    GEMM1_ALPHA: tl.constexpr = 0.0,
    GEMM1_CLAMP_LIMIT: tl.constexpr = 0.0,
):
    """Fused: SiLU-and-mul + per-token-group FP8 quant.

    When SCALE_UE8M0 is True, processes 4 groups at a time and writes a packed
    int32 (one byte per group exponent). When False, processes 1 group at a
    time and writes float32 scale. When ROUND_POW2 is True (and SCALE_UE8M0 is
    False), the float32 scale is rounded to the nearest power of two (MXFP8).
    """
    block_id = tl.program_id(axis=0)
    token_id = tl.program_id(axis=1)

    stride_input_t = tl.cast(stride_input_t, dtype=tl.int64)
    stride_output_t = tl.cast(stride_output_t, dtype=tl.int64)

    in_base = input_ptr + token_id * stride_input_t
    out_base = output_ptr + token_id * stride_output_t

    if SCALE_UE8M0:
        base_group_idx = block_id * 4
        scale_base = (
            output_scale_ptr
            + token_id * stride_output_scale_t
            + block_id * stride_output_scale_g
        )
        packed_scale: tl.int32 = 0
        for g in tl.static_range(4):
            group_idx = base_group_idx + g
            offs_in_d = group_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_in_d < size_n
            gate = tl.load(in_base + offs_in_d, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(in_base + offs_in_d + size_n, mask=mask, other=0.0).to(
                tl.float32
            )
            if GEMM1_ALPHA > 0:
                gate = tl.minimum(gate, GEMM1_CLAMP_LIMIT)
                up = tl.clamp(up, -GEMM1_CLAMP_LIMIT, GEMM1_CLAMP_LIMIT)
                gate_up_bf16 = (gate * tl.sigmoid(gate * GEMM1_ALPHA) * (up + 1)).to(
                    tl.bfloat16
                )
            else:
                silu_gate = gate / (1 + tl.exp(-gate))
                gate_up_bf16 = (silu_gate * up).to(tl.bfloat16)
            gate_up = gate_up_bf16.to(tl.float32)
            # Match sgl_per_token_group_quant_fp8 byte-exact: 1e-10 floor on
            # absmax, IEEE-RNE fp32 div for scale and per-element val/s
            # (fp64-promoted to escape Triton's ``div.approx.f32`` default).
            _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-4)
            # IEEE-RNE fp32 div via inline ``div.rn.f32`` (Triton default `/`
            # uses ``div.approx.f32`` which is ~1 ULP off sgl).
            output_s = _ieee_rn_div_f32(_absmax, fp8_max)
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
            output_q = tl.clamp(
                _ieee_rn_div_f32(gate_up, tl.full(gate_up.shape, output_s, tl.float32)),
                fp8_min,
                fp8_max,
            ).to(output_ptr.dtype.element_ty)
            tl.store(out_base + offs_in_d, output_q, mask=mask)
            scale_bits = output_s.to(tl.int32, bitcast=True)
            exp_bits = (scale_bits >> 23) & 0xFF
            packed_scale = packed_scale | (exp_bits << (g * 8))
        tl.store(scale_base, packed_scale)
    else:
        group_idx = block_id
        offs_in_d = group_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_in_d < size_n
        gate = tl.load(in_base + offs_in_d, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(in_base + offs_in_d + size_n, mask=mask, other=0.0).to(tl.float32)
        if GEMM1_ALPHA > 0:
            gate = tl.minimum(gate, GEMM1_CLAMP_LIMIT)
            up = tl.clamp(up, -GEMM1_CLAMP_LIMIT, GEMM1_CLAMP_LIMIT)
            gate_up_bf16 = (gate * tl.sigmoid(gate * GEMM1_ALPHA) * (up + 1)).to(
                tl.bfloat16
            )
        else:
            silu_gate = gate / (1 + tl.exp(-gate))
            gate_up_bf16 = (silu_gate * up).to(tl.bfloat16)
        gate_up = gate_up_bf16.to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-4)
        output_s = _ieee_rn_div_f32(_absmax, fp8_max)
        if ROUND_POW2:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(
            _ieee_rn_div_f32(gate_up, tl.full(gate_up.shape, output_s, tl.float32)),
            fp8_min,
            fp8_max,
        ).to(output_ptr.dtype.element_ty)
        tl.store(out_base + offs_in_d, output_q, mask=mask)
        scale_offset = (
            output_scale_ptr
            + token_id * stride_output_scale_t
            + group_idx * stride_output_scale_g
        )
        tl.store(scale_offset, output_s)


_SILU_MUL_FP8_QUANT_M_THRESHOLD = 1024


def silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    round_to_pow2: bool = False,
    gemm1_alpha: float = 0.0,
    gemm1_clamp_limit: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense 2-D fused activation + per-token-group FP8 quant.

    Supports both standard SiLU-and-mul (``gemm1_alpha == 0``) and SwiGLU-OAI
    (``gemm1_alpha > 0``): ``gate * sigmoid(gate*alpha) * (up+1)`` with
    one-sided gate clamp and two-sided up clamp at ``gemm1_clamp_limit``.

    Falls back to unfused path for large T where the fused Triton kernel is
    slower than baseline.

    Args:
        input:        [T, 2*H_out]  bf16/fp16, contiguous, layout `[gate | up]`.
        quant_group_size: must divide H_out, defaults to 128.
        scale_ue8m0:  True for Blackwell-style packed int32 UE8M0 scales,
                      False for H20-style fp32 scales.
        round_to_pow2: When True and scale_ue8m0 is False, round scales to the
                      nearest power of two (MXFP8 1×32 format). The returned
                      scale is a row-major fp32 ``[T, H_out // quant_group_size]``
                      tensor matching :func:`mxfp8_quant_act`'s contract.
        gemm1_alpha:  SwiGLU-OAI alpha (>0 enables OAI math, 0 for standard SiLU).
        gemm1_clamp_limit: SwiGLU-OAI clamp limit (used when gemm1_alpha > 0).

    Returns:
        (fp8_output, output_scale) tuple.
    """
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    assert input.is_contiguous(), "input must be contiguous"
    assert input.dim() == 2
    assert input.shape[-1] % 2 == 0
    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0
    num_groups = size_n // quant_group_size

    mxfp8_mode = round_to_pow2 and not scale_ue8m0

    T = input.shape[0]

    if T >= _SILU_MUL_FP8_QUANT_M_THRESHOLD:
        if gemm1_alpha > 0:
            from rtp_llm.models_py.triton_kernels.common.swiglu_oai import (
                swiglu_oai_torch,
            )

            activated = swiglu_oai_torch(
                input, gemm1_alpha, gemm1_clamp_limit, gate_first=True
            )
        else:
            from rtp_llm.models_py.modules.base import FusedSiluAndMul

            activated = FusedSiluAndMul()(input)
        if mxfp8_mode:
            from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

            return mxfp8_quant_act_packed(activated)
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        return sgl_per_token_group_quant_fp8(
            activated,
            group_size=quant_group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )

    output = torch.empty((T, size_n), dtype=torch.float8_e4m3fn, device=input.device)
    if mxfp8_mode:
        output_scale = torch.empty(
            (T, num_groups),
            dtype=torch.float32,
            device=input.device,
        )
    else:
        output_scale = create_per_token_group_quant_fp8_output_scale(
            x_shape=(T, size_n),
            device=input.device,
            group_size=quant_group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )

    if T == 0:
        return output, output_scale

    if scale_ue8m0:
        assert (
            num_groups % 4 == 0
        ), "Number of groups must be divisible by 4 for UE8M0 packing"
        num_blocks = num_groups // 4
    else:
        num_blocks = num_groups

    BLOCK_N = quant_group_size
    NUM_STAGE = 2
    grid = (num_blocks, T)

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_dense_packed_kernel[grid](
        input,
        input.stride(0),
        output,
        output.stride(0),
        output_scale,
        output_scale.stride(0),
        output_scale.stride(1),
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGE,
        SCALE_UE8M0=scale_ue8m0,
        ROUND_POW2=mxfp8_mode,
        GEMM1_ALPHA=gemm1_alpha,
        GEMM1_CLAMP_LIMIT=gemm1_clamp_limit,
        num_warps=1,
    )
    return output, output_scale


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


def _heuristic_params(expert_num: int, expected_m: int) -> Tuple[int, int, int]:
    """
    Heuristic parameter selection based on actual test data.
    Rules derived from parameter search analysis of 12,978 data points.

    Args:
        expert_num: Number of experts
        expected_m: Expected number of tokens

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
    # Most cases use 4 stages according to test data
    num_stages = 2 if expected_m < 64 else 4

    # num_warps heuristic rules (based on actual test data)
    # Overwhelming majority (8151/12978) use 1 warp
    num_warps = 1

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
        _absmax = tl.max(tl.abs(gate_up))
        output_s = (_absmax.to(tl.float64) / fp8_max).to(tl.float32)
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_s_inv = (1.0 / output_s.to(tl.float64)).to(tl.float32)
        output_q = tl.clamp(gate_up * output_s_inv, fp8_min, fp8_max).to(
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
    group_size: int = 256,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype bf16
    masked_m shape [expert_num],
    expected_m int,
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

import torch
import triton
import triton.language as tl


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
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
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

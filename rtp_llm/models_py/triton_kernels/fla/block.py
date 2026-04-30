import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["max_block_size"])
def load_initial_state_from_block_map_kernel(
    prefix_lengths: tl.tensor,
    block_map: tl.tensor,
    conv_states: tl.tensor,
    initial_states: tl.tensor,
    max_block_size: tl.int32,
    HEAD_NUM: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
    CONV_STRIDE_TOKEN: tl.constexpr,
):

    SSM_PER_HEAD = K * V
    SSM_PER_BATCH = HEAD_NUM * SSM_PER_HEAD

    i_b, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    prefix = tl.load(prefix_lengths + i_b)

    v_offset = i_v * BLOCK_V

    is_zero = prefix == 0
    block_offset = tl.where(is_zero, 0, (prefix - 1) // SEQ_SIZE_PER_BLOCK)

    block_idx = tl.where(
        is_zero, 0, tl.load(block_map + i_b * max_block_size + block_offset)
    ).to(tl.int64)

    p_out = tl.make_block_ptr(
        initial_states + i_b * SSM_PER_BATCH + i_h * SSM_PER_HEAD,
        (V, K),
        (K, 1),
        (v_offset, 0),
        (BLOCK_V, K),
        (1, 0),
    )

    p_in = tl.make_block_ptr(
        conv_states + block_idx * CONV_STRIDE_TOKEN + i_h * SSM_PER_HEAD,
        (V, K),
        (K, 1),
        (v_offset, 0),
        (BLOCK_V, K),
        (1, 0),
    )

    b_in = tl.where(
        is_zero,
        tl.zeros([BLOCK_V, K], dtype=initial_states.dtype.element_ty),
        tl.load(p_in, boundary_check=(0, 1)).to(initial_states.dtype.element_ty),
    )

    tl.store(p_out, b_in, boundary_check=(0, 1))


def load_initial_state_from_block_map(
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    conv_states: torch.Tensor,
    initial_states: torch.Tensor,
    seq_size_per_block: int,
    block_v: int = 64,
):
    batch, max_block_size = block_map.shape
    _, head_num, v, k = conv_states.shape
    assert prefix_lengths.shape[0] == batch

    # 增加V维度的并行度
    grid = (batch, head_num, triton.cdiv(v, block_v))
    token_stride_conv = conv_states.stride(0)

    load_initial_state_from_block_map_kernel[grid](
        prefix_lengths,
        block_map,
        conv_states,
        initial_states,
        max_block_size,
        HEAD_NUM=head_num,
        V=v,
        K=k,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        CONV_STRIDE_TOKEN=token_stride_conv,
        BLOCK_V=block_v,
    )


@triton.jit(do_not_specialize=["max_block_size"])
def store_ssm_state_to_block_map_kernel(
    chunk_indices: tl.tensor,
    h: tl.tensor,
    final_states: tl.tensor,
    prefix_lengths: tl.tensor,
    cu_seqlens: tl.tensor,
    block_map: tl.tensor,
    ssm_states: tl.tensor,
    max_block_size: tl.int32,
    HEAD_NUM: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CONV_STRIDE_TOKEN: tl.constexpr,
):
    i_c, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    batch = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    chunk = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)

    SSM_PER_HEAD = K * V
    SSM_PER_BATCH = SSM_PER_HEAD * HEAD_NUM
    v_offset = i_v * BLOCK_V

    prefix = tl.load(prefix_lengths + batch)
    bos = tl.load(cu_seqlens + batch).to(tl.int32)
    eos = tl.load(cu_seqlens + batch + 1).to(tl.int32)
    input_len = eos - bos

    should_write = False
    dest_block_pos = 0
    source_ptr = final_states

    # last chunk always record to final states
    if (chunk + 1) * CHUNK_SIZE >= input_len:
        source_ptr = final_states + batch * SSM_PER_BATCH + i_h * SSM_PER_HEAD
        dest_block_pos = (prefix + input_len - 1) // SEQ_SIZE_PER_BLOCK
        should_write = True
    elif chunk > 0 and (chunk + 1) * CHUNK_SIZE % SEQ_SIZE_PER_BLOCK == 0:
        dest_block_pos = (
            prefix + chunk * CHUNK_SIZE + CHUNK_SIZE - 1
        ) // SEQ_SIZE_PER_BLOCK
        source_ptr = h + (i_c + 1) * SSM_PER_BATCH + i_h * SSM_PER_HEAD
        should_write = True

    if not should_write:
        return

    block_idx = tl.load(block_map + batch * max_block_size + dest_block_pos).to(
        tl.int64
    )

    if block_idx <= 0:
        return

    dest_ptr = ssm_states + block_idx * CONV_STRIDE_TOKEN + i_h * SSM_PER_HEAD

    p_in = tl.make_block_ptr(
        source_ptr,
        (V, K),
        (K, 1),
        (v_offset, 0),
        (BLOCK_V, K),
        (1, 0),
    )
    p_out = tl.make_block_ptr(
        dest_ptr,
        (V, K),
        (K, 1),
        (v_offset, 0),
        (BLOCK_V, K),
        (1, 0),
    )

    tl.store(
        p_out,
        tl.load(p_in, boundary_check=(0, 1)).to(ssm_states.dtype.element_ty),
        boundary_check=(0, 1),
    )


def store_ssm_state_to_block_map(
    h: torch.Tensor,
    final_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_map: torch.Tensor,
    ssm_states: torch.Tensor,
    seq_size_per_block: int,
    chunk_size: int,
    block_v: int = 64,
):
    # fp32 required: the Triton kernel accumulates SSM state directly at the
    # loaded dtype; lower precision causes numerical drift across chunks.
    assert (
        h.dtype == torch.float32 and final_states.dtype == torch.float32
    ), "h and final_states must be float32"
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    _, head_num, v, k = ssm_states.shape
    chunk_num = chunk_indices.shape[0]
    max_block_size = block_map.shape[1]
    grid = (chunk_num, head_num, triton.cdiv(v, block_v))
    token_stride_ssm_state = ssm_states.stride(0)
    store_ssm_state_to_block_map_kernel[grid](
        chunk_indices,
        h,
        final_states,
        prefix_lengths,
        cu_seqlens,
        block_map,
        ssm_states,
        max_block_size,
        HEAD_NUM=head_num,
        V=v,
        K=k,
        BLOCK_V=block_v,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        CONV_STRIDE_TOKEN=token_stride_ssm_state,
        CHUNK_SIZE=chunk_size,
    )


@triton.jit
def store_conv_state_kernel(
    full_qkv,  # [DIM, total_padded] channel-first
    conv_states,  # [num_blocks, DIM, CTX_LEN] (transposed view)
    prefix_lengths_ptr,  # [batch] int32, abs prefix per seq
    input_lengths_ptr,  # [batch] int32, new-token count per seq (orig N)
    padded_cu_ptr,  # [batch+1] int32, full_qkv offsets per seq
    block_map_ptr,  # [batch, max_blocks] int32
    max_blocks: tl.int32,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # tokens_per_block
    CTX_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_qkv_dim: tl.constexpr,
    stride_qkv_token: tl.constexpr,
    stride_cs_block: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_ctx: tl.constexpr,
):
    """Each program handles one write for one (batch, write_idx_within_seq).

    Per seq with N new tokens, num_writes = ceil(N / BLOCK_SIZE).
    Write k (k = 0 .. num_writes-2) targets block boundary at abs P + (k+1)*B - 1.
    Last write (k = num_writes - 1) targets seq end at abs P + N - 1.
    Programs with i_write >= num_writes early-return.
    """
    i_write = tl.program_id(0)
    i_batch = tl.program_id(1)
    i_d = tl.program_id(2)

    P = tl.load(prefix_lengths_ptr + i_batch).to(tl.int32)
    N = tl.load(input_lengths_ptr + i_batch).to(tl.int32)
    if N <= 0:
        return

    num_writes = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    if i_write >= num_writes:
        return

    last_idx = num_writes - 1
    if i_write < last_idx:
        abs_pos = P + (i_write + 1) * BLOCK_SIZE - 1  # block boundary
    else:
        abs_pos = P + N - 1  # seq end (may equal last boundary)

    block_idx_in_seq = abs_pos // BLOCK_SIZE
    block_id = tl.load(block_map_ptr + i_batch * max_blocks + block_idx_in_seq).to(
        tl.int64
    )
    if block_id <= 0:
        return

    seq_padded_off = tl.load(padded_cu_ptr + i_batch).to(tl.int32)
    src_start = seq_padded_off + (abs_pos - P) - CTX_LEN + 1

    d_offset = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < DIM

    for c in tl.static_range(CTX_LEN):
        src_ptr = (
            full_qkv + d_offset * stride_qkv_dim + (src_start + c) * stride_qkv_token
        )
        dst_ptr = (
            conv_states
            + block_id * stride_cs_block
            + d_offset * stride_cs_dim
            + c * stride_cs_ctx
        )
        val = tl.load(src_ptr, mask=d_mask, other=0.0)
        tl.store(dst_ptr, val, mask=d_mask)


def store_conv_state_to_block_map(
    full_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    prefix_lengths_d: torch.Tensor,
    input_lengths_d: torch.Tensor,
    padded_cu_d: torch.Tensor,
    block_map_d: torch.Tensor,
    seq_size_per_block: int,
    max_writes_per_seq: int,
    block_d: int = 128,
):
    """Self-describing variant of store_conv_state_to_block_map.

    Args:
        full_qkv: [dim, total_padded] channel-first, post all_gather + restore.
                  Same data layout as single-card mixed_qkv (pre-conv).
        conv_states: [num_blocks, dim, ctx_len] block cache (transposed view).
        prefix_lengths_d: [batch] int32, prefix per seq.
        input_lengths_d: [batch] int32, new-token count per seq (orig N).
        padded_cu_d: [batch+1] int32, cumulative offsets in full_qkv per seq.
        block_map_d: [batch, max_blocks] int32, block_id table.
        seq_size_per_block: tokens_per_block.
        max_writes_per_seq: upper bound for grid dim 0; pass max(ceil(N_b / B)).
                            Programs beyond actual num_writes early-return.
    """
    batch_size = block_map_d.shape[0]
    max_blocks = block_map_d.shape[1]
    dim = full_qkv.shape[0]
    ctx_len = conv_states.shape[2]

    if max_writes_per_seq <= 0 or batch_size <= 0:
        return

    grid = (max_writes_per_seq, batch_size, triton.cdiv(dim, block_d))
    store_conv_state_kernel[grid](
        full_qkv,
        conv_states,
        prefix_lengths_d,
        input_lengths_d,
        padded_cu_d,
        block_map_d,
        max_blocks,
        DIM=dim,
        BLOCK_SIZE=seq_size_per_block,
        CTX_LEN=ctx_len,
        BLOCK_D=block_d,
        stride_qkv_dim=full_qkv.stride(0),
        stride_qkv_token=full_qkv.stride(1),
        stride_cs_block=conv_states.stride(0),
        stride_cs_dim=conv_states.stride(1),
        stride_cs_ctx=conv_states.stride(2),
    )

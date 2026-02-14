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
    )

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
        tl.load(p_in, boundary_check=(0, 1)),
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

    block_idx = tl.load(block_map + batch * max_block_size + dest_block_pos)

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

    tl.store(p_out, tl.load(p_in, boundary_check=(0, 1)), boundary_check=(0, 1))


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

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["block_map_stride_b"])
def load_initial_state_from_block_map_kernel(
    prefix_lengths: tl.tensor,
    block_map: tl.tensor,
    conv_states: tl.tensor,
    initial_states: tl.tensor,
    block_map_stride_b: tl.int64,
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
        is_zero, 0, tl.load(block_map + i_b * block_map_stride_b + block_offset)
    ).to(tl.int64)

    p_out_i0 = tl.arange(0, BLOCK_V).to(tl.int64) + (v_offset)
    p_out_m0 = (p_out_i0 >= 0) & (p_out_i0 < (V))
    p_out_i1 = tl.arange(0, K).to(tl.int64) + (0)
    p_out_m1 = (p_out_i1 >= 0) & (p_out_i1 < (K))
    p_out = (
        (initial_states + i_b * SSM_PER_BATCH + i_h * SSM_PER_HEAD)
        + p_out_i0[:, None] * (K)
        + p_out_i1[None, :] * (1)
    )

    p_in_i0 = tl.arange(0, BLOCK_V).to(tl.int64) + (v_offset)
    p_in_m0 = (p_in_i0 >= 0) & (p_in_i0 < (V))
    p_in_i1 = tl.arange(0, K).to(tl.int64) + (0)
    p_in_m1 = (p_in_i1 >= 0) & (p_in_i1 < (K))
    p_in = (
        (conv_states + block_idx * CONV_STRIDE_TOKEN + i_h * SSM_PER_HEAD)
        + p_in_i0[:, None] * (K)
        + p_in_i1[None, :] * (1)
    )

    b_in = tl.where(
        is_zero,
        tl.zeros([BLOCK_V, K], dtype=initial_states.dtype.element_ty),
        tl.load(p_in, mask=p_in_m0[:, None] & p_in_m1[None, :], other=0).to(
            initial_states.dtype.element_ty
        ),
    )

    tl.store(p_out, b_in, mask=p_out_m0[:, None] & p_out_m1[None, :])


def load_initial_state_from_block_map(
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    conv_states: torch.Tensor,
    initial_states: torch.Tensor,
    seq_size_per_block: int,
    block_v: int = 64,
):
    batch = block_map.shape[0]
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
        block_map.stride(0),
        HEAD_NUM=head_num,
        V=v,
        K=k,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        CONV_STRIDE_TOKEN=token_stride_conv,
        BLOCK_V=block_v,
    )


@triton.jit
def _store_ssm_state_block(
    source_ptr,
    dest_block_pos,
    batch,
    i_h,
    v_offset,
    block_map: tl.tensor,
    ssm_states: tl.tensor,
    block_map_stride_b: tl.int64,
    SSM_PER_HEAD: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CONV_STRIDE_TOKEN: tl.constexpr,
):
    block_idx = tl.load(block_map + batch * block_map_stride_b + dest_block_pos).to(
        tl.int64
    )

    if block_idx > 0:
        dest_ptr = (
            ssm_states
            + linear_offset_64(block_idx, CONV_STRIDE_TOKEN)
            + linear_offset_64(i_h, SSM_PER_HEAD)
        )

        p_in_i0 = tl.arange(0, BLOCK_V).to(tl.int64) + (v_offset)
        p_in_m0 = (p_in_i0 >= 0) & (p_in_i0 < (V))
        p_in_i1 = tl.arange(0, K).to(tl.int64) + (0)
        p_in_m1 = (p_in_i1 >= 0) & (p_in_i1 < (K))
        p_in = (source_ptr) + p_in_i0[:, None] * (K) + p_in_i1[None, :] * (1)
        p_out_i0 = tl.arange(0, BLOCK_V).to(tl.int64) + (v_offset)
        p_out_m0 = (p_out_i0 >= 0) & (p_out_i0 < (V))
        p_out_i1 = tl.arange(0, K).to(tl.int64) + (0)
        p_out_m1 = (p_out_i1 >= 0) & (p_out_i1 < (K))
        p_out = (dest_ptr) + p_out_i0[:, None] * (K) + p_out_i1[None, :] * (1)

        tl.store(
            p_out,
            tl.load(p_in, mask=p_in_m0[:, None] & p_in_m1[None, :], other=0).to(
                ssm_states.dtype.element_ty
            ),
            mask=p_out_m0[:, None] & p_out_m1[None, :],
        )


@triton.jit(do_not_specialize=["block_map_stride_b"])
def store_ssm_state_to_block_map_kernel(
    chunk_indices: tl.tensor,
    h: tl.tensor,
    final_states: tl.tensor,
    prefix_lengths: tl.tensor,
    cu_seqlens: tl.tensor,
    block_map: tl.tensor,
    ssm_states: tl.tensor,
    block_map_stride_b: tl.int64,
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

    # Keep source pointers branch-local. Triton 3.7 cannot canonicalize a
    # pointer phi whose branches use different offset widths.
    if (chunk + 1) * CHUNK_SIZE >= input_len:
        source_ptr = (
            final_states
            + linear_offset_64(batch, SSM_PER_BATCH)
            + linear_offset_64(i_h, SSM_PER_HEAD)
        )
        dest_block_pos = (prefix + input_len - 1) // SEQ_SIZE_PER_BLOCK
        _store_ssm_state_block(
            source_ptr,
            dest_block_pos,
            batch,
            i_h,
            v_offset,
            block_map,
            ssm_states,
            block_map_stride_b,
            SSM_PER_HEAD,
            V,
            K,
            BLOCK_V,
            CONV_STRIDE_TOKEN,
        )
    elif (
        prefix % SEQ_SIZE_PER_BLOCK == 0
        and (chunk + 1) * CHUNK_SIZE % SEQ_SIZE_PER_BLOCK == 0
    ):
        source_ptr = (
            h
            + linear_offset_64(i_c + 1, SSM_PER_BATCH)
            + linear_offset_64(i_h, SSM_PER_HEAD)
        )
        dest_block_pos = (
            prefix + chunk * CHUNK_SIZE + CHUNK_SIZE - 1
        ) // SEQ_SIZE_PER_BLOCK
        _store_ssm_state_block(
            source_ptr,
            dest_block_pos,
            batch,
            i_h,
            v_offset,
            block_map,
            ssm_states,
            block_map_stride_b,
            SSM_PER_HEAD,
            V,
            K,
            BLOCK_V,
            CONV_STRIDE_TOKEN,
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
    block_map_stride_b = block_map.stride(0)
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
        block_map_stride_b,
        HEAD_NUM=head_num,
        V=v,
        K=k,
        BLOCK_V=block_v,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        CONV_STRIDE_TOKEN=token_stride_ssm_state,
        CHUNK_SIZE=chunk_size,
    )


def _selected_save_device_tensor(
    tensor: torch.Tensor, device: torch.device, name: str
) -> torch.Tensor:
    result = tensor if tensor.device == device else tensor.to(device, non_blocking=True)
    if not result.is_contiguous():
        result = result.contiguous()
    return result


def prepare_selected_state_positions(
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_positions: torch.Tensor,
    save_block_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert absolute save endpoints to sequence-local snapshot positions.

    The returned ``[N, S]`` tensor is suitable for the selected-state kernels.
    Tail endpoint N is represented by ``-1`` so the model consumes final_state
    instead of requesting a duplicate selected snapshot. Padding, endpoints not
    covered by this forward, and rows without a destination are also disabled.
    """
    if state_positions.dim() != 2 or save_block_ids.shape != state_positions.shape:
        raise ValueError("state positions and destination IDs must have shape [N, S]")
    if prefix_lengths.dim() != 1 or prefix_lengths.numel() != state_positions.shape[0]:
        raise ValueError("prefix lengths must have one row per save request")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() != state_positions.shape[0] + 1:
        raise ValueError("cu_seqlens must have one entry per save request plus one")

    device_positions = _selected_save_device_tensor(
        state_positions, device, "state_positions"
    ).to(torch.int64)
    device_block_ids = _selected_save_device_tensor(
        save_block_ids, device, "save_block_ids"
    ).to(torch.int32)
    device_prefix = _selected_save_device_tensor(
        prefix_lengths, device, "prefix_lengths"
    ).to(torch.int64)
    device_cu = _selected_save_device_tensor(cu_seqlens, device, "cu_seqlens").to(
        torch.int64
    )

    input_lengths = device_cu.diff()
    endpoints = device_prefix + input_lengths
    valid = (
        (device_block_ids >= 0)
        & (device_positions > device_prefix[:, None])
        & (device_positions <= endpoints[:, None])
    )
    tail = device_positions == endpoints[:, None]
    relative = device_positions - device_prefix[:, None]
    disabled = torch.full_like(relative, -1)
    return torch.where(valid & ~tail, relative, disabled).to(torch.int32).contiguous()


def store_selected_ssm_states_to_block_map(
    selected_states: torch.Tensor,
    final_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_positions: torch.Tensor,
    save_block_ids: torch.Tensor,
    ssm_states: torch.Tensor,
) -> None:
    """Scatter selected/final SSM states to caller-owned sparse destinations.

    ``selected_states`` must use the producer's state-axis convention and be
    scatterable through the existing LINEAR store layout. GDN supplies
    ``[N, S, H, V, K]`` directly. KDA's current K==V model supplies its public
    ``[N, S, H, K, V]`` buffer directly to match the existing final-state store;
    callers must not transpose merely because K and V are equal.
    ``final_states`` must have the producer's final-state layout. Cache dtype
    conversion is performed only at this store boundary. Runtime writes are not
    suppressed.
    """
    if selected_states.dtype != torch.float32 or final_states.dtype != torch.float32:
        raise ValueError("selected and final SSM states must have FP32 dtype")
    if (
        selected_states.dim() != 5
        or selected_states.shape[1] != state_positions.shape[1]
    ):
        raise ValueError("selected_states must have shape [N, S, H, V, K]")
    if final_states.shape != (
        state_positions.shape[0],
        *ssm_states.shape[1:],
    ):
        raise ValueError("final_states must have the destination SSM cache layout")

    device = ssm_states.device
    positions = _selected_save_device_tensor(
        state_positions, device, "state_positions"
    ).to(torch.int64)
    block_ids = _selected_save_device_tensor(
        save_block_ids, device, "save_block_ids"
    ).to(torch.int32)
    prefix = _selected_save_device_tensor(prefix_lengths, device, "prefix_lengths").to(
        torch.int64
    )
    cu = _selected_save_device_tensor(cu_seqlens, device, "cu_seqlens").to(torch.int64)

    input_lengths = cu.diff()
    endpoints = prefix + input_lengths
    valid = (
        (block_ids >= 0)
        & (positions > prefix[:, None])
        & (positions <= endpoints[:, None])
    )
    tail = valid & (positions == endpoints[:, None])
    selected = valid & ~tail

    ssm_states.index_copy_(
        0,
        block_ids[selected].to(torch.int64),
        selected_states[selected].to(ssm_states.dtype),
    )
    tail_rows = torch.arange(positions.shape[0], device=device)[:, None].expand(
        positions.shape
    )[tail]
    ssm_states.index_copy_(
        0,
        block_ids[tail].to(torch.int64),
        final_states[tail_rows].to(ssm_states.dtype),
    )


def store_selected_conv_states_to_block_map(
    conv_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_positions: torch.Tensor,
    save_block_ids: torch.Tensor,
    runtime_block_map: torch.Tensor,
    seq_size_per_block: int,
    raw_inputs: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
) -> None:
    """Save conv history at exact global endpoints to sparse destinations.

    With raw inputs, form interior history from the input tokens and the
    immutable prefix history captured before the ordinary conv producer ran.
    Existing callers without raw inputs keep the boundary/final-state copy.
    An alias destination is left untouched.
    """
    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")
    if state_positions.dim() != 2 or save_block_ids.shape != state_positions.shape:
        raise ValueError("state positions and destination IDs must have shape [N, S]")
    if runtime_block_map.dim() != 2:
        raise ValueError("runtime block map must have shape [N, blocks]")

    device = conv_states.device
    positions = _selected_save_device_tensor(
        state_positions, device, "state_positions"
    ).to(torch.int64)
    block_ids = _selected_save_device_tensor(
        save_block_ids, device, "save_block_ids"
    ).to(torch.int32)
    prefix = _selected_save_device_tensor(prefix_lengths, device, "prefix_lengths").to(
        torch.int64
    )
    cu = _selected_save_device_tensor(cu_seqlens, device, "cu_seqlens").to(torch.int64)
    runtime_map = _selected_save_device_tensor(
        runtime_block_map, device, "runtime_block_map"
    ).to(torch.int32)

    input_lengths = cu.diff()
    endpoints = prefix + input_lengths
    boundary = (positions % seq_size_per_block == 0) | (positions == endpoints[:, None])
    valid = (
        (boundary if raw_inputs is None else torch.ones_like(boundary))
        & (block_ids >= 0)
        & (positions > prefix[:, None])
        & (positions <= endpoints[:, None])
    )
    source_columns = ((positions - 1) // seq_size_per_block).clamp(
        min=0, max=runtime_map.shape[1] - 1
    )
    source_block_ids = torch.gather(
        runtime_map[:, None, :].expand(-1, positions.shape[1], -1),
        2,
        source_columns.unsqueeze(-1),
    ).squeeze(-1)
    copy = valid & (source_block_ids >= 0) & (block_ids != source_block_ids)
    destination_ids = block_ids[copy].to(torch.int64)
    if destination_ids.numel() != 0:
        if raw_inputs is not None:
            if (
                initial_states is None
                or raw_inputs.ndim != 2
                or raw_inputs.shape[1] != conv_states.shape[1]
                or initial_states.shape
                != (prefix.numel(), conv_states.shape[1], conv_states.shape[2])
            ):
                raise ValueError(
                    "selected interior conv states require raw inputs and initial history"
                )
            history = conv_states.shape[2]
            offsets = (
                positions[:, :, None]
                - prefix[:, None, None]
                - history
                + torch.arange(history, device=device)
            )
            raw_indices = (cu[:-1, None, None] + offsets).clamp(
                0, raw_inputs.shape[0] - 1
            )
            fresh = (
                raw_inputs.index_select(0, raw_indices.flatten())
                .reshape(*positions.shape, history, conv_states.shape[1])
                .permute(0, 1, 3, 2)
            )
            prior_indices = (offsets + history).clamp(0, history - 1)
            prior = torch.gather(
                initial_states[:, None].expand(-1, positions.shape[1], -1, -1),
                3,
                prior_indices[:, :, None].expand(-1, -1, conv_states.shape[1], -1),
            )
            selected = torch.where(offsets[:, :, None] >= 0, fresh, prior)
            conv_states.index_copy_(
                0, destination_ids, selected[copy].to(conv_states.dtype)
            )
            return
        conv_states.index_copy_(
            0,
            destination_ids,
            conv_states.index_select(0, source_block_ids[copy].to(torch.int64)),
        )

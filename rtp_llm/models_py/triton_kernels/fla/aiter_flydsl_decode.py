"""AITER FlyDSL GDN decode adapter for RTP-LLM's ROCm block cache."""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def is_aiter_flydsl_gdn_decode_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> bool:
    """Return whether AITER's FlyDSL GDR decode kernel supports the inputs."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or state.ndim != 4:
        return False

    batch, query_length, key_heads, key_dim = q.shape
    value_heads = v.shape[2]
    value_dim = v.shape[3]
    return (
        torch.version.hip is not None
        and q.device.type == "cuda"
        and q.device == k.device == v.device == a.device == b.device == state.device
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and state.dtype in (torch.float32, torch.bfloat16)
        and state.data_ptr() % 16 == 0
        and k.shape == q.shape
        and v.shape[:2] == (batch, query_length)
        and value_heads % key_heads == 0
        and key_dim == 128
        and value_dim == 128
        and a.numel() == batch * query_length * value_heads
        and b.numel() == batch * query_length * value_heads
        and state.shape[1:] == (value_heads, value_dim, key_dim)
    )


@triton.jit
def _prepare_decode_state_indices_kernel(
    block_map,
    sequence_lengths_plus_1,
    read_indices,
    write_indices,
    block_map_row_stride: tl.constexpr,
    seq_size_per_block: tl.constexpr,
):
    batch = tl.program_id(0)
    sequence_length = tl.load(sequence_lengths_plus_1 + batch).to(tl.int64)
    read_pos = (sequence_length - 2) // seq_size_per_block
    write_pos = (sequence_length - 1) // seq_size_per_block
    row_start = batch * block_map_row_stride
    read_id = tl.load(block_map + row_start + read_pos)
    write_id = tl.load(block_map + row_start + write_pos)
    tl.store(read_indices + batch, read_id)
    tl.store(write_indices + batch, write_id)


def prepare_aiter_flydsl_gdn_decode_state_indices(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the source/current RTP block IDs for one decode step."""
    if block_map.ndim != 2:
        raise ValueError(f"block_map must be 2D, got {tuple(block_map.shape)}")
    if block_map.dtype != torch.int32:
        raise ValueError(f"block_map must be int32, got {block_map.dtype}")
    if sequence_lengths_plus_1.dtype != torch.int32:
        raise ValueError(
            "sequence_lengths_plus_1 must be int32, "
            f"got {sequence_lengths_plus_1.dtype}"
        )
    if block_map.device != sequence_lengths_plus_1.device:
        raise ValueError("block_map and sequence lengths must be on the same device")

    batch = block_map.shape[0]
    if sequence_lengths_plus_1.numel() != batch:
        raise ValueError(
            "sequence length count must equal block-map batch size, "
            f"got {sequence_lengths_plus_1.numel()} and {batch}"
        )

    read_indices = torch.empty(batch, device=block_map.device, dtype=torch.int32)
    write_indices = torch.empty_like(read_indices)
    _prepare_decode_state_indices_kernel[(batch,)](
        block_map,
        sequence_lengths_plus_1,
        read_indices,
        write_indices,
        block_map_row_stride=block_map.stride(0),
        seq_size_per_block=seq_size_per_block,
        num_warps=1,
    )
    return read_indices, write_indices


@triton.jit
def _copy_decode_state_at_block_boundary_kernel(
    state,
    read_indices,
    write_indices,
    state_stride: tl.constexpr,
    state_elements_per_head: tl.constexpr,
    block_elements: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    read_id = tl.load(read_indices + batch).to(tl.int64)
    write_id = tl.load(write_indices + batch).to(tl.int64)
    if read_id == write_id or read_id <= 0 or write_id <= 0:
        return

    offsets = block * block_elements + tl.arange(0, block_elements)
    mask = offsets < state_elements_per_head
    head_offset = head * state_elements_per_head
    values = tl.load(
        state + read_id * state_stride + head_offset + offsets,
        mask=mask,
    )
    tl.store(
        state + write_id * state_stride + head_offset + offsets,
        values,
        mask=mask,
    )


def copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
    state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
) -> None:
    """Copy the preceding block state before the first token of a new block."""
    _, value_heads, value_dim, key_dim = state.shape
    elements_per_head = value_dim * key_dim
    block_elements = 256
    grid = (
        read_indices.numel(),
        value_heads,
        triton.cdiv(elements_per_head, block_elements),
    )
    _copy_decode_state_at_block_boundary_kernel[grid](
        state,
        read_indices,
        write_indices,
        state_stride=state.stride(0),
        state_elements_per_head=elements_per_head,
        block_elements=block_elements,
        num_warps=4,
    )


@torch.compiler.disable
def aiter_flydsl_gdn_decode(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    copy_state: bool,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run AITER's fused FlyDSL GDR decode directly on RTP's VK state pool."""
    if not is_aiter_flydsl_gdn_decode_supported(q, k, v, a, b, state):
        raise ValueError(
            "AITER FlyDSL GDN decode requires ROCm BF16 tensors, K=V=128, "
            "FP32/BF16 VK state, and an integral GQA ratio; "
            f"got q={tuple(q.shape)}/{q.dtype}, v={tuple(v.shape)}/{v.dtype}, "
            f"a={a.dtype}, b={b.dtype}, state={tuple(state.shape)}/{state.dtype}"
        )
    if A_log.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"AITER FlyDSL GDN decode requires FP32/BF16 A_log, got {A_log.dtype}"
        )
    if A_log.device != q.device or A_log.numel() != v.shape[2]:
        raise ValueError(
            "A_log must contain one value per value head on the input device"
        )
    if dt_bias.dtype != q.dtype:
        raise ValueError(
            "AITER FlyDSL GDN decode requires dt_bias to match q.dtype, "
            f"got {dt_bias.dtype} and {q.dtype}"
        )
    if dt_bias.device != q.device or dt_bias.numel() != v.shape[2]:
        raise ValueError(
            "dt_bias must contain one value per value head on the input device"
        )
    if (
        read_indices.dtype != torch.int32
        or write_indices.dtype != torch.int32
        or read_indices.device != q.device
        or write_indices.device != q.device
        or read_indices.numel() != q.shape[0]
        or write_indices.numel() != q.shape[0]
    ):
        raise ValueError(
            "read_indices and write_indices must be batch-sized int32 tensors "
            "on the input device"
        )

    expected_scale = q.shape[-1] ** -0.5
    if scale is not None and not math.isclose(scale, expected_scale, rel_tol=1e-6):
        raise ValueError(
            "AITER FlyDSL GDN decode uses the fixed head-dimension scale "
            f"{expected_scale}, got {scale}"
        )

    if copy_state:
        copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
            state, read_indices, write_indices
        )

    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    batch, query_length = q.shape[:2]
    value_heads = v.shape[2]
    output = torch.empty_like(v)
    flydsl_gdr_decode(
        query=q,
        key=k,
        value=v,
        a=a.reshape(batch, query_length, value_heads),
        b=b.reshape(batch, query_length, value_heads),
        dt_bias=dt_bias,
        A_log=A_log,
        indices=write_indices,
        state=state,
        out=output,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        # RTP stores the persistent SSM cache in VK layout already.
        need_shuffle_state=False,
    )
    return output

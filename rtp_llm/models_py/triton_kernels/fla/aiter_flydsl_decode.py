"""AITER FlyDSL GDN decode adapter for RTP-LLM's ROCm block cache."""

import functools
import logging
import math
from collections.abc import Callable
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.utils import is_amd_cdna3

_LOGGER = logging.getLogger(__name__)

# Shapes validated against RTP's Triton decode implementation on MI308X.
# Add a shape only after both numerical and performance validation.
AITER_FLYDSL_GDN_DECODE_ENABLED_SHAPES = frozenset(
    {
        (2, 8, 128, 128),
        (16, 32, 128, 128),
    }
)


@functools.lru_cache(maxsize=None)
def _get_aiter_flydsl_gdn_decode() -> Optional[Callable]:
    """Resolve the optional AITER symbol once and fall back cleanly if absent."""
    try:
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
    except (AttributeError, ImportError) as error:
        _LOGGER.warning(
            "AITER FlyDSL GDN decode is unavailable; falling back to Triton: %s",
            error,
        )
        return None
    return flydsl_gdr_decode


def _state_inner_layout_is_contiguous(state: torch.Tensor) -> bool:
    _, _, value_dim, key_dim = state.shape
    return state.stride()[1:] == (value_dim * key_dim, key_dim, 1)


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
    shape = (key_heads, value_heads, key_dim, value_dim)
    return (
        is_amd_cdna3
        and q.device.type == "cuda"
        and q.device == k.device == v.device == a.device == b.device == state.device
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and state.dtype in (torch.float32, torch.bfloat16)
        and state.data_ptr() % 16 == 0
        and _state_inner_layout_is_contiguous(state)
        and query_length == 1
        and k.shape == q.shape
        and v.shape[:2] == (batch, query_length)
        and shape in AITER_FLYDSL_GDN_DECODE_ENABLED_SHAPES
        and a.numel() == batch * query_length * value_heads
        and b.numel() == batch * query_length * value_heads
        and state.shape[1:] == (value_heads, value_dim, key_dim)
        and _get_aiter_flydsl_gdn_decode() is not None
    )


@triton.jit
def _prepare_decode_state_indices_kernel(
    block_map,
    sequence_lengths_plus_1,
    read_indices,
    write_indices,
    block_map_row_stride: tl.constexpr,
    block_map_width: tl.constexpr,
    seq_size_per_block: tl.constexpr,
):
    batch = tl.program_id(0)
    sequence_length = tl.load(sequence_lengths_plus_1 + batch).to(tl.int64)
    read_pos = (sequence_length - 2) // seq_size_per_block
    write_pos = (sequence_length - 1) // seq_size_per_block
    valid = (
        (sequence_length >= 2)
        & (read_pos >= 0)
        & (write_pos >= 0)
        & (read_pos < block_map_width)
        & (write_pos < block_map_width)
    )
    safe_read_pos = tl.minimum(tl.maximum(read_pos, 0), block_map_width - 1)
    safe_write_pos = tl.minimum(tl.maximum(write_pos, 0), block_map_width - 1)
    row_start = batch * block_map_row_stride
    read_id = tl.load(block_map + row_start + safe_read_pos)
    write_id = tl.load(block_map + row_start + safe_write_pos)
    tl.store(read_indices + batch, tl.where(valid, read_id, 0))
    tl.store(write_indices + batch, tl.where(valid, write_id, 0))


def prepare_aiter_flydsl_gdn_decode_state_indices(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve source/current block IDs for one decode token per request.

    ``sequence_lengths_plus_1`` must contain the post-decode length and normally
    be at least 2. Invalid graph-padding rows, out-of-range positions, and block
    map entries with value 0 resolve to block 0. RTP reserves block 0 as a dummy
    state: the boundary-copy kernel skips it and AITER output for padding rows is
    ignored by the graph scheduler.
    """
    if block_map.ndim != 2:
        raise ValueError(f"block_map must be 2D, got {tuple(block_map.shape)}")
    if block_map.shape[1] == 0:
        raise ValueError("block_map must contain at least one block column")
    if block_map.dtype != torch.int32:
        raise ValueError(f"block_map must be int32, got {block_map.dtype}")
    if sequence_lengths_plus_1.ndim != 1:
        raise ValueError(
            "sequence_lengths_plus_1 must be 1D, "
            f"got {tuple(sequence_lengths_plus_1.shape)}"
        )
    if sequence_lengths_plus_1.dtype != torch.int32:
        raise ValueError(
            "sequence_lengths_plus_1 must be int32, "
            f"got {sequence_lengths_plus_1.dtype}"
        )
    if block_map.device != sequence_lengths_plus_1.device:
        raise ValueError("block_map and sequence lengths must be on the same device")
    if seq_size_per_block <= 0:
        raise ValueError(
            f"seq_size_per_block must be positive, got {seq_size_per_block}"
        )

    batch = block_map.shape[0]
    if sequence_lengths_plus_1.numel() != batch:
        raise ValueError(
            "sequence length count must equal block-map batch size, "
            f"got {sequence_lengths_plus_1.numel()} and {batch}"
        )

    read_indices = torch.empty(batch, device=block_map.device, dtype=torch.int32)
    write_indices = torch.empty_like(read_indices)
    if batch == 0:
        return read_indices, write_indices
    _prepare_decode_state_indices_kernel[(batch,)](
        block_map,
        sequence_lengths_plus_1,
        read_indices,
        write_indices,
        block_map_row_stride=block_map.stride(0),
        block_map_width=block_map.shape[1],
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
    state_pool_size: tl.constexpr,
    state_elements_per_head: tl.constexpr,
    block_elements: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    read_id = tl.load(read_indices + batch).to(tl.int64)
    write_id = tl.load(write_indices + batch).to(tl.int64)
    if (
        read_id == write_id
        or read_id <= 0
        or write_id <= 0
        or read_id >= state_pool_size
        or write_id >= state_pool_size
    ):
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
    """Copy preceding block states; non-boundary and dummy rows are no-ops."""
    if state.ndim != 4:
        raise ValueError(f"state must be 4D, got {tuple(state.shape)}")
    _, value_heads, value_dim, key_dim = state.shape
    expected_inner_stride = (value_dim * key_dim, key_dim, 1)
    if state.stride()[1:] != expected_inner_stride:
        raise ValueError(
            "state inner dimensions must be contiguous, "
            f"expected stride {expected_inner_stride}, got {state.stride()[1:]}"
        )
    if read_indices.shape != write_indices.shape:
        raise ValueError(
            "read_indices and write_indices must have the same shape, "
            f"got {tuple(read_indices.shape)} and {tuple(write_indices.shape)}"
        )
    if read_indices.numel() == 0:
        return

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
        state_pool_size=state.shape[0],
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
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run the fused kernel and always launch the graph-safe state copy first."""
    if not is_aiter_flydsl_gdn_decode_supported(q, k, v, a, b, state):
        raise ValueError(
            "AITER FlyDSL GDN decode requires an available MI300X/MI308X "
            "backend, a validated BF16 T=1 shape, contiguous FP32/BF16 VK "
            f"state; got q={tuple(q.shape)}/{q.dtype}, "
            f"v={tuple(v.shape)}/{v.dtype}, a={a.dtype}, b={b.dtype}, "
            f"state={tuple(state.shape)}/{state.dtype}/{state.stride()}"
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

    # Always record this launch in CUDA/HIP Graph. Its device-side per-row
    # guard makes ordinary non-boundary steps no-ops and keeps replay correct
    # when sequence lengths later cross a block boundary.
    copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
        state, read_indices, write_indices
    )

    flydsl_gdr_decode = _get_aiter_flydsl_gdn_decode()
    if flydsl_gdr_decode is None:
        raise RuntimeError("AITER FlyDSL GDN decode became unavailable after dispatch")

    batch, query_length = q.shape[:2]
    value_heads = v.shape[2]
    output = torch.empty_like(v)
    flydsl_gdr_decode(
        # torch.split in the production Qwen path returns cross-strided views.
        # Make the packing contract explicit rather than relying on AITER's
        # current wrapper implementation to perform these copies internally.
        query=q.contiguous(),
        key=k.contiguous(),
        value=v.contiguous(),
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

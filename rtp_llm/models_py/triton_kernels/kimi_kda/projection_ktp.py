"""Small CUDA BF16 layout kernels for KDA projection KTP."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.kimi_kda.cached_launch import CachedLaunch


@triton.jit(do_not_specialize=["M"])
def _pack_ktp_projection_payload_kernel(
    projected,
    raw_gate,
    output,
    M,
    PROJECTED_ROW_STRIDE: tl.constexpr,
    RAW_GATE_ROW_STRIDE: tl.constexpr,
    LOCAL_PROJECTION_SIZE: tl.constexpr,
    FORGET_LATENT_SIZE: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    BETA_BEGIN: tl.constexpr,
    PAYLOAD_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < M) & (columns < PAYLOAD_WIDTH)

    projected_section = columns < 4 * LOCAL_PROJECTION_SIZE
    projected_values = tl.load(
        projected + row * PROJECTED_ROW_STRIDE + columns,
        mask=valid & projected_section,
        other=0.0,
    )

    gate_columns = columns - 4 * LOCAL_PROJECTION_SIZE
    gate_section = (gate_columns >= 0) & (gate_columns < LOCAL_PROJECTION_SIZE)
    gate_values = tl.load(
        raw_gate + row * RAW_GATE_ROW_STRIDE + gate_columns,
        mask=valid & gate_section,
        other=0.0,
    )

    beta_columns = columns - 5 * LOCAL_PROJECTION_SIZE
    beta_section = (beta_columns >= 0) & (beta_columns < LOCAL_HEADS)
    beta_offset = (
        4 * LOCAL_PROJECTION_SIZE + FORGET_LATENT_SIZE + BETA_BEGIN + beta_columns
    )
    beta_values = tl.load(
        projected + row * PROJECTED_ROW_STRIDE + beta_offset,
        mask=valid & beta_section,
        other=0.0,
    )

    values = tl.where(
        projected_section,
        projected_values,
        tl.where(gate_section, gate_values, beta_values),
    )
    tl.store(output + row * PAYLOAD_WIDTH + columns, values, mask=valid)


@triton.jit(do_not_specialize=["M"])
def _reassemble_ktp_projection_payload_kernel(
    received,
    q,
    k,
    v,
    output_gate,
    raw_gate,
    raw_beta,
    M,
    KTP_SIZE: tl.constexpr,
    LOCAL_PROJECTION_SIZE: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAYLOAD_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    owner = tl.program_id(0)
    source = tl.program_id(1)
    columns = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    projection_mask = (owner < M) & (columns < LOCAL_PROJECTION_SIZE)
    source_row = source * M + owner
    input_base = source_row * PAYLOAD_WIDTH
    projection_output_base = (
        owner * KTP_SIZE * LOCAL_PROJECTION_SIZE + source * LOCAL_PROJECTION_SIZE
    )

    outputs = (q, k, v, output_gate, raw_gate)
    for section in tl.static_range(5):
        values = tl.load(
            received + input_base + section * LOCAL_PROJECTION_SIZE + columns,
            mask=projection_mask,
            other=0.0,
        )
        tl.store(
            outputs[section] + projection_output_base + columns,
            values,
            mask=projection_mask,
        )

    beta_mask = (owner < M) & (columns < LOCAL_HEADS)
    beta_values = tl.load(
        received + input_base + 5 * LOCAL_PROJECTION_SIZE + columns,
        mask=beta_mask,
        other=0.0,
    )
    beta_output_base = owner * KTP_SIZE * LOCAL_HEADS + source * LOCAL_HEADS
    tl.store(raw_beta + beta_output_base + columns, beta_values, mask=beta_mask)


_launch_pack = CachedLaunch(_pack_ktp_projection_payload_kernel, num_warps=4)


def pack_ktp_projection_payload_cuda(
    projected: torch.Tensor,
    raw_gate: torch.Tensor,
    output: torch.Tensor,
    *,
    local_projection_size: int,
    forget_latent_size: int,
    local_heads: int,
    ktp_rank: int,
) -> torch.Tensor:
    """Pack strided fused-projection sections into caller-owned storage."""

    rows = int(projected.shape[0])
    payload_width = 5 * local_projection_size + local_heads
    block = 256
    _launch_pack(
        (rows, (payload_width + block - 1) // block, 1),
        (projected, raw_gate, output),
        (
            rows,
            projected.stride(0),
            raw_gate.stride(0),
            local_projection_size,
            forget_latent_size,
            local_heads,
            ktp_rank * local_heads,
            payload_width,
            block,
        ),
    )
    return output


def reassemble_ktp_projection_payload_cuda(
    received: torch.Tensor,
    *,
    ktp_size: int,
    physical_batch: int,
    local_projection_size: int,
    local_heads: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Unpack source-major transport storage into independent owner-major tensors."""

    projection_shape = (physical_batch, ktp_size * local_projection_size)
    beta_shape = (physical_batch, ktp_size * local_heads)
    q = torch.empty(projection_shape, dtype=received.dtype, device=received.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    output_gate = torch.empty_like(q)
    raw_gate = torch.empty_like(q)
    raw_beta = torch.empty(beta_shape, dtype=received.dtype, device=received.device)
    payload_width = 5 * local_projection_size + local_heads
    block = 256
    _reassemble_ktp_projection_payload_kernel[
        (
            physical_batch,
            ktp_size,
            triton.cdiv(local_projection_size, block),
        )
    ](
        received,
        q,
        k,
        v,
        output_gate,
        raw_gate,
        raw_beta,
        physical_batch,
        ktp_size,
        local_projection_size,
        local_heads,
        payload_width,
        block,
        num_warps=4,
    )
    return q, k, v, output_gate, raw_gate, raw_beta


__all__ = [
    "pack_ktp_projection_payload_cuda",
    "reassemble_ktp_projection_payload_cuda",
]

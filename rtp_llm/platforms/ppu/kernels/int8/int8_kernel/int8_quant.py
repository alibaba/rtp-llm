"""PPU INT8 per-token activation quantization kernels."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale = absmax / 127.0
    x_q = tl.clamp(x / scale, -128.0, 127.0)
    x_q = tldevice.round(x_q).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale)


@triton.jit
def _per_token_quant_int8_moe_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    BLOCK: tl.constexpr,
):
    """Match vLLM's PPU DeepGEMM MoE per-token quantization order."""
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale = absmax / 127.0
    x_q = x * (127.0 / absmax)
    x_q = tldevice.round(x_q).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale)


@triton.jit
def _per_token_quant_int8_masked_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    masked_m_ptr,
    stride_x,
    stride_xq,
    rows_per_expert,
    n_cols,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    expert_id = row_id // rows_per_expert
    row_in_expert = row_id % rows_per_expert
    valid_row = row_in_expert < tl.load(masked_m_ptr + expert_id)
    cols = tl.arange(0, BLOCK)
    mask = (cols < n_cols) & valid_row
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale = absmax / 127.0
    x_q = tl.clamp(x / scale, -128.0, 127.0)
    x_q = tldevice.round(x_q).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=cols < n_cols)
    tl.store(scale_ptr + row_id, scale)


def per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize the last dimension to symmetric INT8 per token."""
    if x.dim() < 2:
        raise ValueError(f"per-token INT8 quant expects ndim >= 2, got {x.dim()}")
    original_shape = x.shape
    x_contiguous = x.contiguous().view(-1, original_shape[-1])
    x_q = torch.empty_like(x_contiguous, dtype=torch.int8)
    scales = torch.empty(
        (x_contiguous.shape[0], 1),
        device=x_contiguous.device,
        dtype=torch.float32,
    )
    if x_contiguous.numel() == 0:
        return x_q.view(original_shape), scales.view(*original_shape[:-1], 1)

    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)
    _per_token_quant_int8_kernel[(x_contiguous.shape[0],)](
        x_contiguous,
        x_q,
        scales,
        stride_x=x_contiguous.stride(-2),
        stride_xq=x_q.stride(-2),
        n_cols=n_cols,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q.view(original_shape), scales.view(*original_shape[:-1], 1)


def per_token_quant_int8_moe(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize like vLLM's PPU DeepGEMM MoE input and down activation."""
    if x.dim() < 2:
        raise ValueError(f"per-token INT8 quant expects ndim >= 2, got {x.dim()}")
    original_shape = x.shape
    x_contiguous = x.contiguous().view(-1, original_shape[-1])
    x_q = torch.empty_like(x_contiguous, dtype=torch.int8)
    scales = torch.empty(
        (x_contiguous.shape[0], 1),
        device=x_contiguous.device,
        dtype=torch.float32,
    )
    if x_contiguous.numel() == 0:
        return x_q.view(original_shape), scales.view(*original_shape[:-1], 1)

    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)
    _per_token_quant_int8_moe_kernel[(x_contiguous.shape[0],)](
        x_contiguous,
        x_q,
        scales,
        stride_x=x_contiguous.stride(-2),
        stride_xq=x_q.stride(-2),
        n_cols=n_cols,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q.view(original_shape), scales.view(*original_shape[:-1], 1)


def per_token_quant_int8_masked(
    x: torch.Tensor, masked_m: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 quantization for fixed-capacity ``[E, M, K]`` buffers."""
    if x.dim() != 3:
        raise ValueError(f"masked INT8 quant expects [E, M, K], got {x.shape}")
    if masked_m.dim() != 1 or masked_m.shape[0] != x.shape[0]:
        raise ValueError(
            f"masked_m must have shape ({x.shape[0]},), got {masked_m.shape}"
        )
    x_contiguous = x.contiguous()
    x_q = torch.empty_like(x_contiguous, dtype=torch.int8)
    scales = torch.empty(
        (*x_contiguous.shape[:-1], 1),
        device=x_contiguous.device,
        dtype=torch.float32,
    )
    if x_contiguous.numel() == 0:
        return x_q, scales

    experts, rows_per_expert, n_cols = x_contiguous.shape
    rows = experts * rows_per_expert
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)
    _per_token_quant_int8_masked_kernel[(rows,)](
        x_contiguous,
        x_q,
        scales,
        masked_m,
        stride_x=x_contiguous.stride(-2),
        stride_xq=x_q.stride(-2),
        rows_per_expert=rows_per_expert,
        n_cols=n_cols,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, scales

"""SiTU-GLU activation for Qwen 4.5-Flash (Qwen4_5ForCausalLM).

Exact mathematical specification (computed in FP32, cast back to input dtype):
    g = gate_beta * tanh(gate / gate_beta) * sigmoid(gate)
    u = up_beta * tanh(up / up_beta)
    out = g * u

Default betas for Qwen 4.5-Flash: ``gate_beta = 4.0``, ``up_beta = 25.0``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


def situ_and_mul_ref(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
) -> torch.Tensor:
    """Pure PyTorch FP32 reference for SiTU-GLU.

    Accepts either:
      - concatenated ``x`` of shape ``[..., 2 * D]`` (with ``up=None``), or
      - separate ``x`` (gate) and ``up`` each of shape ``[..., D]``.
    """
    if up is None:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected even last dimension for concatenated gate_up, got {x.shape[-1]}"
            )
        gate, up_t = x.chunk(2, dim=-1)
    else:
        if x.shape != up.shape:
            raise ValueError(f"gate shape {x.shape} != up shape {up.shape}")
        gate, up_t = x, up

    out_dtype = gate.dtype
    gate_f = gate.float()
    up_f = up_t.float()
    g = gate_beta * torch.tanh(gate_f / gate_beta) * torch.sigmoid(gate_f)
    u = up_beta * torch.tanh(up_f / up_beta)
    return (g * u).to(out_dtype)


@triton.jit
def _situ_and_mul_cat_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    D,
    stride_x_row: tl.int64,
    stride_out_row: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    x_row_ptr = x_ptr + pid_row * stride_x_row
    gate = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + D + offs, mask=mask, other=0.0).to(tl.float32)

    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(tl.div_rn(gate, gate_beta)) * sig_gate
    u = up_beta * libdevice.tanh(tl.div_rn(up, up_beta))
    out = g * u

    out_row_ptr = out_ptr + pid_row * stride_out_row
    tl.store(out_row_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _situ_and_mul_split_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    num_rows,
    D,
    stride_g_row: tl.int64,
    stride_u_row: tl.int64,
    stride_out_row: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    gate = tl.load(gate_ptr + pid_row * stride_g_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(up_ptr + pid_row * stride_u_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(tl.div_rn(gate, gate_beta)) * sig_gate
    u = up_beta * libdevice.tanh(tl.div_rn(up, up_beta))
    out = g * u

    tl.store(
        out_ptr + pid_row * stride_out_row + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def situ_and_mul_triton(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
) -> torch.Tensor:
    """Fused Triton kernel for SiTU-GLU activation.

    Automatically falls back to :func:`situ_and_mul_ref` for non-CUDA tensors.
    """
    if not x.is_cuda or (up is not None and not up.is_cuda):
        return situ_and_mul_ref(x, up, gate_beta=gate_beta, up_beta=up_beta)

    gate_beta_f = float(gate_beta)
    up_beta_f = float(up_beta)

    if up is None:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected even last dimension for concatenated gate_up, got {x.shape[-1]}"
            )
        orig_shape = x.shape[:-1]
        D = x.shape[-1] // 2
        if x.numel() == 0:
            return torch.empty((*orig_shape, D), dtype=x.dtype, device=x.device)
        x_2d = x.reshape(-1, 2 * D)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        num_rows = x_2d.shape[0]
        out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)

        BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(max(D, 64))
        num_warps = 8 if BLOCK_D >= 1024 else 4
        grid = (num_rows, triton.cdiv(D, BLOCK_D))
        _situ_and_mul_cat_kernel[grid](
            x_2d,
            out,
            num_rows,
            D,
            x_2d.stride(0),
            out.stride(0),
            gate_beta=gate_beta_f,
            up_beta=up_beta_f,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
        return out.reshape(*orig_shape, D)

    if x.shape != up.shape:
        raise ValueError(f"gate shape {x.shape} != up shape {up.shape}")
    orig_shape = x.shape[:-1]
    D = x.shape[-1]
    if x.numel() == 0:
        return torch.empty_like(x)
    g_2d = x.reshape(-1, D)
    u_2d = up.reshape(-1, D)
    if g_2d.stride(-1) != 1:
        g_2d = g_2d.contiguous()
    if u_2d.stride(-1) != 1:
        u_2d = u_2d.contiguous()
    num_rows = g_2d.shape[0]
    out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)

    BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(max(D, 64))
    num_warps = 8 if BLOCK_D >= 1024 else 4
    grid = (num_rows, triton.cdiv(D, BLOCK_D))
    _situ_and_mul_split_kernel[grid](
        g_2d,
        u_2d,
        out,
        num_rows,
        D,
        g_2d.stride(0),
        u_2d.stride(0),
        out.stride(0),
        gate_beta=gate_beta_f,
        up_beta=up_beta_f,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out.reshape(*orig_shape, D)


class SituAndMul(nn.Module):
    """SiTU-GLU activation module for Qwen 4.5-Flash Dense MLP & Latent MoE."""

    def __init__(
        self,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        use_triton: bool = True,
    ) -> None:
        super().__init__()
        self.gate_beta = float(gate_beta)
        self.up_beta = float(up_beta)
        self.use_triton = bool(use_triton)

    def forward(
        self, x: torch.Tensor, up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_triton and x.is_cuda and (up is None or up.is_cuda):
            return situ_and_mul_triton(
                x, up, gate_beta=self.gate_beta, up_beta=self.up_beta
            )
        return situ_and_mul_ref(
            x, up, gate_beta=self.gate_beta, up_beta=self.up_beta
        )

"""Qwen 4.5-Flash 4-stream Hyper-Connections with hc_prev_mix and GemmaRMSNorm+bias.

Implements:
  - Per-branch GemmaRMSNorm (1.0 + weight) with affine bias (Triton + FP32 PyTorch ref)
  - Fused down + block-inject projection and activation epilogue
  - hc_prev_mix gate routing (head / normal / non-emitting / final mixer modes)
  - Fused branch mix-reduce and residual stream combine (+ combine-and-norm fusion)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_gemma_rmsnorm_bias_kernel(
    X_PTR,
    W_PTR,
    B_PTR,
    OUT_PTR,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE
    col = group * GROUP_SIZE + offs
    x = tl.load(X_PTR + row * WIDTH + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_PTR + col, mask=mask, other=0.0).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / GROUP_SIZE + EPS)
    out = x * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + col, mask=mask, other=0.0).to(tl.float32)
        out = out + b
    tl.store(OUT_PTR + row * WIDTH + col, out, mask=mask)


@triton.jit
def _grouped_gemma_rmsnorm_bias_chunked_kernel(
    X_PTR,
    W_PTR,
    B_PTR,
    OUT_PTR,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    base = row * WIDTH + group * GROUP_SIZE
    w_base = group * GROUP_SIZE
    sum_sq = tl.zeros((), dtype=tl.float32)
    for start in range(0, GROUP_SIZE, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(X_PTR + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / GROUP_SIZE + EPS)
    for start in range(0, GROUP_SIZE, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(X_PTR + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_PTR + w_base + offs, mask=mask, other=0.0).to(tl.float32)
        out = x * inv_rms * (1.0 + w)
        if HAS_BIAS:
            b = tl.load(B_PTR + w_base + offs, mask=mask, other=0.0).to(tl.float32)
            out = out + b
        tl.store(OUT_PTR + base + offs, out, mask=mask)


@triton.jit
def _hc_combine_kernel(
    RES_PTR,
    OUT_PTR,
    INJ_PTR,
    DST_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    sub = tl.load(OUT_PTR + row * HIDDEN + col, mask=mask, other=0.0).to(tl.float32)
    row_base = row * (HC_COUNT * HIDDEN)
    for branch in tl.static_range(HC_COUNT):
        offset = row_base + branch * HIDDEN + col
        res = tl.load(RES_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
        comb = res + sub * w
        tl.store(DST_PTR + offset, comb, mask=mask)


@triton.jit
def _hc_combine_and_norm_kernel(
    RES_PTR,
    SUB_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    col = branch * HIDDEN + offs
    res_offset = row * (HC_COUNT * HIDDEN) + col

    res = tl.load(RES_PTR + res_offset, mask=mask, other=0.0).to(tl.float32)
    sub = tl.load(SUB_PTR + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
    inj = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
    comb = (res + sub * inj).to(RES_PTR.dtype.element_ty)
    tl.store(COMB_PTR + res_offset, comb, mask=mask)

    comb_f = tl.where(mask, comb.to(tl.float32), 0.0)
    inv_rms = 1.0 / tl.sqrt(tl.sum(comb_f * comb_f, axis=0) / HIDDEN + EPS)
    w = tl.load(W_PTR + col, mask=mask, other=0.0).to(tl.float32)
    normed = comb_f * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + col, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b
    tl.store(NORM_PTR + res_offset, normed, mask=mask)


@triton.jit
def _hc_mix_reduce_kernel(
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    row_base = row * (HC_COUNT * HIDDEN)
    for branch in tl.static_range(HC_COUNT):
        offset = row_base + branch * HIDDEN + col
        logits = tl.load(LOGITS_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(NORMED_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits)
        acc += gate * normed
    out = acc * (1.0 / HC_COUNT)
    tl.store(OUT_PTR + row * HIDDEN + col, out, mask=mask)


@triton.jit
def _hc_head_mix_kernel(
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    HIDDEN: tl.constexpr,
    HC_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    logits = tl.load(LOGITS_PTR + row * HIDDEN + col, mask=mask, other=0.0).to(tl.float32)
    normed0 = tl.load(NORMED_PTR + row * HC_WIDTH + col, mask=mask, other=0.0).to(tl.float32)
    out = tl.sigmoid(logits) * normed0
    tl.store(OUT_PTR + row * HIDDEN + col, out, mask=mask)


@triton.jit
def _hc_down_inject_epilogue_kernel(
    RAW_PTR,
    LATENT_PTR,
    INJ_PTR,
    ROWS,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r_offs = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < ROWS
    inv_hc: tl.constexpr = 1.0 / HC_COUNT

    l_offs = tl.arange(0, BLOCK_L)
    l_mask = r_mask[:, None] & (l_offs[None, :] < LOWRANK)
    raw_l = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + l_offs[None, :],
        mask=l_mask,
        other=0.0,
    ).to(tl.float32)
    scaled_l = raw_l * inv_hc
    latent = scaled_l * tl.sigmoid(scaled_l)
    tl.store(
        LATENT_PTR + r_offs[:, None] * LOWRANK + l_offs[None, :],
        latent,
        mask=l_mask,
    )

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = r_mask[:, None] & (h_offs[None, :] < HC_COUNT)
    raw_h = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + LOWRANK + h_offs[None, :],
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    inj = 2.0 * tl.sigmoid(raw_h * inv_hc)
    tl.store(
        INJ_PTR + r_offs[:, None] * HC_COUNT + h_offs[None, :],
        inj,
        mask=h_mask,
    )


# ---------------------------------------------------------------------------
# Functional Triton & Reference Wrappers
# ---------------------------------------------------------------------------


def grouped_gemma_rmsnorm_bias_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    hc_count: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact FP32 PyTorch reference for per-branch GemmaRMSNorm (1 + w) + bias."""
    width = x.shape[-1]
    if width % hc_count != 0:
        raise ValueError(f"Last dim {width} is not divisible by hc_count={hc_count}")
    hidden_size = width // hc_count
    streams = x.unflatten(-1, (hc_count, hidden_size)).float()
    normed = streams * torch.rsqrt(streams.square().mean(dim=-1, keepdim=True) + eps)
    out = normed.flatten(-2) * (1.0 + weight.float())
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def grouped_gemma_rmsnorm_bias_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    hc_count: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused Triton per-branch GemmaRMSNorm (1.0 + weight) + optional affine bias."""
    if not x.is_cuda:
        return grouped_gemma_rmsnorm_bias_ref(x, weight, bias, hc_count, eps)
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else w_c
    width = int(x_c.shape[-1])
    if width % hc_count != 0:
        raise ValueError(f"Last dim {width} is not divisible by hc_count={hc_count}")
    group_size = width // hc_count
    rows = x_c.numel() // width
    out = torch.empty_like(x_c)
    if rows == 0:
        return out
    has_bias = bias is not None
    if group_size <= 8192:
        block = max(128, triton.next_power_of_2(group_size))
        num_warps = 8 if block >= 4096 else 4
        _grouped_gemma_rmsnorm_bias_kernel[(rows, hc_count)](
            x_c,
            w_c,
            b_c,
            out,
            WIDTH=width,
            GROUP_SIZE=group_size,
            EPS=eps,
            HAS_BIAS=has_bias,
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        _grouped_gemma_rmsnorm_bias_chunked_kernel[(rows, hc_count)](
            x_c,
            w_c,
            b_c,
            out,
            WIDTH=width,
            GROUP_SIZE=group_size,
            EPS=eps,
            HAS_BIAS=has_bias,
            BLOCK=4096,
            num_warps=8,
        )
    return out


def hc_combine_ref(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_w: torch.Tensor,
) -> torch.Tensor:
    """Exact FP32 PyTorch reference for injecting block_output into all HC streams."""
    hc_count = inject_w.shape[-1]
    hidden_size = block_output.shape[-1]
    res_f = residual.unflatten(-1, (hc_count, hidden_size)).float()
    out_f = block_output.unsqueeze(-2).float()
    inj_f = inject_w.unsqueeze(-1).float()
    return (res_f + out_f * inj_f).flatten(-2).to(residual.dtype)


def hc_combine_triton(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_w: torch.Tensor,
) -> torch.Tensor:
    """Fused Triton injection of block_output into hc_count residual streams."""
    if not residual.is_cuda:
        return hc_combine_ref(block_output, residual, inject_w)
    res_c = residual.contiguous()
    out_c = block_output.contiguous()
    inj_c = inject_w.contiguous()
    hc_count = int(inj_c.shape[-1])
    hidden_size = int(out_c.shape[-1])
    width = int(res_c.shape[-1])
    if width != hc_count * hidden_size:
        raise ValueError(
            f"residual width {width} != hc_count ({hc_count}) * hidden_size ({hidden_size})"
        )
    rows = res_c.numel() // width
    dst = torch.empty_like(res_c)
    if rows == 0:
        return dst
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    _hc_combine_kernel[(rows, triton.cdiv(hidden_size, block))](
        res_c,
        out_c,
        inj_c,
        dst,
        HIDDEN=hidden_size,
        HC_COUNT=hc_count,
        BLOCK=block,
        num_warps=4,
    )
    return dst


def hc_combine_and_norm_triton(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    prev_inject_w: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused combine + per-branch GemmaRMSNorm+bias."""
    hc_count = int(prev_inject_w.shape[-1])
    hidden_size = int(block_output.shape[-1])
    if not residual.is_cuda or hidden_size > 8192:
        combined = hc_combine_triton(block_output, residual, prev_inject_w)
        normed = grouped_gemma_rmsnorm_bias_triton(
            combined, norm_weight, norm_bias, hc_count, eps
        )
        return combined, normed

    res_c = residual.contiguous()
    sub_c = block_output.contiguous()
    inj_c = prev_inject_w.contiguous()
    w_c = norm_weight.contiguous()
    b_c = norm_bias.contiguous() if norm_bias is not None else w_c
    width = int(res_c.shape[-1])
    rows = res_c.numel() // width
    combined = torch.empty_like(res_c)
    normed = torch.empty_like(res_c)
    if rows == 0:
        return combined, normed

    block = max(128, triton.next_power_of_2(hidden_size))
    num_warps = 8 if block >= 4096 else 4
    _hc_combine_and_norm_kernel[(rows, hc_count)](
        res_c,
        sub_c,
        inj_c,
        w_c,
        b_c,
        combined,
        normed,
        HIDDEN=hidden_size,
        HC_COUNT=hc_count,
        EPS=eps,
        HAS_BIAS=norm_bias is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return combined, normed


def hc_mix_reduce_triton(
    gate_logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Fused (sigmoid(gate_logits) * normed).mean(dim=-2) across HC branches."""
    if not gate_logits.is_cuda:
        hidden_size = gate_logits.shape[-1] // hc_count
        gate = torch.sigmoid(gate_logits.float()).unflatten(-1, (hc_count, hidden_size))
        n_f = normed.float().unflatten(-1, (hc_count, hidden_size))
        return (gate * n_f).mean(dim=-2).to(normed.dtype)

    logits_c = gate_logits.contiguous()
    normed_c = normed.contiguous()
    width = int(logits_c.shape[-1])
    hidden_size = width // hc_count
    rows = logits_c.numel() // width
    out = torch.empty(
        (*logits_c.shape[:-1], hidden_size),
        device=logits_c.device,
        dtype=normed_c.dtype,
    )
    if rows == 0:
        return out
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    _hc_mix_reduce_kernel[(rows, triton.cdiv(hidden_size, block))](
        logits_c,
        normed_c,
        out,
        HIDDEN=hidden_size,
        HC_COUNT=hc_count,
        BLOCK=block,
        num_warps=4,
    )
    return out


def hc_head_mix_triton(
    gate_logits: torch.Tensor,
    normed: torch.Tensor,
) -> torch.Tensor:
    """Fused Layer-0 head mix: sigmoid(gate_logits) * normed[..., :hidden_size]."""
    hidden_size = int(gate_logits.shape[-1])
    if not gate_logits.is_cuda:
        return (
            torch.sigmoid(gate_logits.float()) * normed[..., :hidden_size].float()
        ).to(normed.dtype)

    logits_c = gate_logits.contiguous()
    normed_c = normed.contiguous()
    hc_width = int(normed_c.shape[-1])
    rows = logits_c.numel() // hidden_size
    out = torch.empty_like(logits_c)
    if rows == 0:
        return out
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    _hc_head_mix_kernel[(rows, triton.cdiv(hidden_size, block))](
        logits_c,
        normed_c,
        out,
        HIDDEN=hidden_size,
        HC_WIDTH=hc_width,
        BLOCK=block,
        num_warps=4,
    )
    return out


def hc_down_inject_epilogue_triton(
    down_inject_raw: torch.Tensor,
    hc_lowrank: int,
    hc_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused [..., lowrank + hc_count] GEMM output into (latent, inject_w)."""
    prefix = down_inject_raw.shape[:-1]
    if not down_inject_raw.is_cuda:
        raw_f = down_inject_raw.float()
        latent = F.silu(raw_f[..., :hc_lowrank] / hc_count).to(down_inject_raw.dtype)
        inject_w = (2.0 * torch.sigmoid(raw_f[..., hc_lowrank:] / hc_count)).to(
            down_inject_raw.dtype
        )
        return latent, inject_w

    raw_c = down_inject_raw.contiguous()
    raw_stride = int(raw_c.shape[-1])
    rows = raw_c.numel() // raw_stride
    latent = torch.empty((*prefix, hc_lowrank), device=raw_c.device, dtype=raw_c.dtype)
    inject_w = torch.empty((*prefix, hc_count), device=raw_c.device, dtype=raw_c.dtype)
    if rows == 0:
        return latent, inject_w

    block_r = 1 if rows <= 8 else (8 if rows <= 128 else 16)
    block_l = max(16, triton.next_power_of_2(hc_lowrank))
    block_h = max(16, triton.next_power_of_2(hc_count))
    _hc_down_inject_epilogue_kernel[(triton.cdiv(rows, block_r),)](
        raw_c,
        latent,
        inject_w,
        ROWS=rows,
        LOWRANK=hc_lowrank,
        HC_COUNT=hc_count,
        RAW_STRIDE=raw_stride,
        BLOCK_R=block_r,
        BLOCK_L=block_l,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return latent, inject_w


# ---------------------------------------------------------------------------
# Module: Qwen4_5HyperConnection
# ---------------------------------------------------------------------------


class Qwen4_5HyperConnection(nn.Module):
    """4-stream Hyper-Connection block for Qwen 4.5-Flash (Qwen4_5ForCausalLM).

    Supports all four architectural sites:
      - Layer 0 attn_hc:        is_head=True,  prev_mix=True,  emits_latent=True,  use_combine=True
      - Layers 0..39 standard:  is_head=False, prev_mix=True,  emits_latent=True,  use_combine=True
      - Layer 39 mlp_hc:        is_head=False, prev_mix=True,  emits_latent=False, use_combine=True
      - Final mixer:            is_head=False, prev_mix=False, emits_latent=True,  use_combine=False
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        hc_count: int = 4,
        hc_lowrank: int = 128,
        eps: float = 1e-6,
        use_combine: bool = True,
        prev_mix: bool = True,
        is_head: bool = False,
        emits_latent: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if not emits_latent and (not prev_mix or is_head):
            raise ValueError(
                "emits_latent=False requires prev_mix=True and is_head=False "
                "so gate_latent can come from prev_latent."
            )

        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.hc_hidden_size = hc_count * hidden_size
        self.eps = eps
        self.use_combine = use_combine
        self.prev_mix = prev_mix
        self.is_head = is_head
        self.emits_latent = emits_latent

        # Checkpoint-matching parameters:
        # hc_norm_weight is zero-centered (applied as 1.0 + weight in GemmaRMSNorm)
        self.hc_norm_weight = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )
        self.hc_norm_bias = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )

        if emits_latent:
            self.input_mix_weight_down: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(hc_lowrank, self.hc_hidden_size, dtype=dtype, device=device)
            )
            nn.init.normal_(self.input_mix_weight_down, mean=0.0, std=1.0 / math.sqrt(self.hc_hidden_size))
        else:
            self.register_parameter("input_mix_weight_down", None)

        up_out_dim = hidden_size if is_head else self.hc_hidden_size
        self.input_mix_weight_up = nn.Parameter(
            torch.empty(up_out_dim, hc_lowrank, dtype=dtype, device=device)
        )
        nn.init.normal_(self.input_mix_weight_up, mean=0.0, std=0.02)

        if use_combine:
            self.block_inject_weight: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(hc_count, self.hc_hidden_size, dtype=dtype, device=device)
            )
            nn.init.normal_(self.block_inject_weight, mean=0.0, std=1.0 / math.sqrt(self.hc_hidden_size))
        else:
            self.register_parameter("block_inject_weight", None)

        # Inference cache for fused [input_mix_weight_down; block_inject_weight]
        self._fused_down_inject_key: Optional[Tuple[int, int, int, int]] = None
        self._fused_down_inject_weight: Optional[torch.Tensor] = None

    def _get_fused_down_inject_weight(self) -> torch.Tensor:
        w_down = self.input_mix_weight_down
        w_inj = self.block_inject_weight
        assert w_down is not None and w_inj is not None
        if torch.is_grad_enabled() or w_down.requires_grad or w_inj.requires_grad:
            return torch.cat([w_down, w_inj], dim=0)
        key = (w_down.data_ptr(), w_down._version, w_inj.data_ptr(), w_inj._version)
        if self._fused_down_inject_key != key or self._fused_down_inject_weight is None:
            self._fused_down_inject_weight = torch.cat([w_down, w_inj], dim=0).contiguous()
            self._fused_down_inject_key = key
        return self._fused_down_inject_weight

    def _project_and_mix_from_normed(
        self,
        normed: torch.Tensor,
        residual: torch.Tensor,
        prev_latent: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 1. Down & inject projections (fused single GEMM when both exist)
        if self.emits_latent and self.use_combine:
            w_fused = self._get_fused_down_inject_weight()
            raw = F.linear(normed, w_fused)
            latent, inject_w = hc_down_inject_epilogue_triton(
                raw, self.hc_lowrank, self.hc_count
            )
        else:
            latent = None
            inject_w = None
            if self.emits_latent:
                assert self.input_mix_weight_down is not None
                raw_down = F.linear(normed, self.input_mix_weight_down)
                latent = F.silu(raw_down.float() / self.hc_count).to(normed.dtype)
            if self.use_combine:
                assert self.block_inject_weight is not None
                raw_inj = F.linear(normed, self.block_inject_weight)
                inject_w = (2.0 * torch.sigmoid(raw_inj.float() / self.hc_count)).to(
                    normed.dtype
                )

        # 2. hc_prev_mix gate latent selection
        if not self.prev_mix or self.is_head:
            gate_latent = latent
        else:
            gate_latent = prev_latent
        if gate_latent is None:
            raise ValueError(
                "prev_latent must be provided when prev_mix=True and is_head=False"
            )

        # 3. Up projection + sigmoid gate mix
        gate_logits = F.linear(gate_latent, self.input_mix_weight_up)
        if self.is_head:
            mixed = hc_head_mix_triton(gate_logits, normed)
        else:
            mixed = hc_mix_reduce_triton(gate_logits, normed, self.hc_count)

        return mixed, residual, inject_w, latent

    def mix_with_latent(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read-side hyper-connection mix returning (mixed, residual, inject_w, next_latent)."""
        normed = grouped_gemma_rmsnorm_bias_triton(
            hyper_input,
            self.hc_norm_weight,
            self.hc_norm_bias,
            self.hc_count,
            self.eps,
        )
        return self._project_and_mix_from_normed(normed, hyper_input, prev_latent)

    def combine(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        inject_w: torch.Tensor,
    ) -> torch.Tensor:
        """Write-side hyper-connection injection back into the 4-stream residual."""
        return hc_combine_triton(block_output, residual, inject_w)

    def combine_and_mix_with_latent(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        prev_inject_w: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fused previous-block combine + current-block GemmaRMSNorm+bias + mix_with_latent."""
        combined, normed = hc_combine_and_norm_triton(
            block_output,
            residual,
            prev_inject_w,
            self.hc_norm_weight,
            self.hc_norm_bias,
            self.eps,
        )
        return self._project_and_mix_from_normed(normed, combined, prev_latent)

    def forward(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.mix_with_latent(hyper_input, prev_latent=prev_latent)

    # ------------------------------------------------------------------
    # Exact FP32 PyTorch Reference Methods (for numerical verification)
    # ------------------------------------------------------------------

    def mix_with_latent_ref(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Exact pure-PyTorch FP32 reference matching the Qwen 4.5-Flash HC specification."""
        out_dtype = hyper_input.dtype
        streams = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size)).float()
        normed_f = streams * torch.rsqrt(
            streams.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normed = (
            normed_f.flatten(-2) * (1.0 + self.hc_norm_weight.float())
            + self.hc_norm_bias.float()
        ).to(out_dtype)
        normed_f = normed.float()

        latent_f: Optional[torch.Tensor] = None
        if self.emits_latent:
            assert self.input_mix_weight_down is not None
            latent_f = F.silu(
                F.linear(normed_f, self.input_mix_weight_down.float()) / self.hc_count
            )

        inject_f: Optional[torch.Tensor] = None
        if self.use_combine:
            assert self.block_inject_weight is not None
            inject_f = 2.0 * torch.sigmoid(
                F.linear(normed_f, self.block_inject_weight.float()) / self.hc_count
            )

        if not self.prev_mix or self.is_head:
            gate_latent_f = latent_f
        else:
            gate_latent_f = prev_latent.float() if prev_latent is not None else None
        if gate_latent_f is None:
            raise ValueError(
                "prev_latent must be provided when prev_mix=True and is_head=False"
            )

        gate_logits_f = F.linear(gate_latent_f, self.input_mix_weight_up.float())
        if self.is_head:
            mixed_f = torch.sigmoid(gate_logits_f) * normed_f[..., : self.hidden_size]
        else:
            gate_f = torch.sigmoid(gate_logits_f).unflatten(
                -1, (self.hc_count, self.hidden_size)
            )
            mixed_f = (
                gate_f * normed_f.unflatten(-1, (self.hc_count, self.hidden_size))
            ).mean(dim=-2)

        return (
            mixed_f.to(out_dtype),
            hyper_input,
            inject_f.to(out_dtype) if inject_f is not None else None,
            latent_f.to(out_dtype) if latent_f is not None else None,
        )

    def combine_ref(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        inject_w: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine_ref(block_output, residual, inject_w)

    def combine_and_mix_with_latent_ref(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        prev_inject_w: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        combined = self.combine_ref(block_output, residual, prev_inject_w)
        return self.mix_with_latent_ref(combined, prev_latent=prev_latent)

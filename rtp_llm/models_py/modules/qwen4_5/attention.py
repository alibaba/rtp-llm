"""Qwen 4.5-Flash Full Attention (MQA-1 + output gate + KV short-conv + partial RoPE + YoCo KV sharing).

Architecture (1109-bf16-re-mxfp4-ple-fp8, dense MQA-1, NO QSA indexer):
  - hidden_size = 5120, num_heads = 32, num_kv_heads = 1, head_dim = 256
  - attn_output_gate = True (q_proj outputs [q, gate] interleaved per head: [32, 512])
  - partial_rotary_factor = 0.25 (NeoX RoPE on first 64 dims), rope_theta = 10000.0
  - GemmaRMSNorm (1.0 + weight in FP32) on q_norm [256] and k_norm [256]
  - Residual depthwise causal short-conv (k=4, no activation) on K and V (producer/normal only)
  - YoCo KV sharing: Layer 19 produces KV cache; consumer layers (23, 27, 31, 35, 39)
    have ONLY (q_proj, o_proj, q_norm) and read Layer 19 KV cache without mutating it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.activation import (
    can_sm10x_bf16_tgv,
    get_sm10x_bf16_tgv_backend,
    run_sm10x_bf16_tgv_gemm,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    FullAttnKVCache,
    Qwen4_5ArchConfig,
    ShortConvState,
)
from rtp_llm.models_py.modules.qwen4_5.yoco import YoCoSharingPlan
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer, exact_head_rms_norm
from rtp_llm.models_py.modules.qwen4_5.short_conv import _triton_int_spec, triton_fast_launch

_ATTN_TRITON_CACHE: dict = {}
_TGV_WARMED_DEVS: set[int] = set()


def _is_bf16_tgv_enabled() -> bool:
    raw = os.environ.get("QWEN4_5_BF16_TGV", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _get_flashinfer_tgv_backend(device: torch.device) -> Tuple[Any, int]:
    """Return cached SM10x ``(tgv_gemm_fn, tactic_num)`` via shared ``activation.py`` backend."""
    if not _is_bf16_tgv_enabled() or device.type != "cuda":
        return None, 0
    dev_idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    if dev_idx not in _TGV_WARMED_DEVS and torch.cuda.is_current_stream_capturing():
        return None, 0
    with torch.cuda.device(device):
        _, tgv_fn, tactic_num = get_sm10x_bf16_tgv_backend(device)
    if tgv_fn is not None and tactic_num > 0:
        _TGV_WARMED_DEVS.add(dev_idx)
        return tgv_fn, int(tactic_num)
    return None, 0


def _select_bf16_tgv_tactic(
    m_rows: int, k_dim: int, n_dim: int, tactic_num: int = 11
) -> Optional[int]:
    """Select strict-win FlashInfer SM10x ``tgv_gemm`` tactic on L2-cold CG (SM103)."""
    if (
        tactic_num <= 0
        or m_rows <= 0
        or m_rows > 48
        or k_dim < 512
        or n_dim < 1280
        or n_dim > 12288
        or (k_dim % 16) != 0
        or (n_dim % 8) != 0
        or (n_dim * k_dim) >= (1 << 30)
    ):
        return None

    tactic: Optional[int] = None
    if k_dim < 2048:
        if n_dim < 2048 or n_dim > 6144 or m_rows > 32:
            return None
        if n_dim <= 3072:
            if m_rows <= 8:
                tactic = 0
            elif m_rows <= 24:
                tactic = 5
        elif m_rows <= 8:
            tactic = 2
        elif m_rows <= 16:
            tactic = 5
        else:
            tactic = 8
    elif n_dim < 2048:
        if m_rows >= 9:
            tactic = 3
    elif n_dim <= 6144:
        if m_rows <= 8:
            tactic = 2 if k_dim < 4096 else 3
        elif m_rows <= 16:
            tactic = 3 if (k_dim >= 5120 and n_dim <= 4096) else 6
        elif m_rows <= 32:
            if n_dim <= 3072 and m_rows <= 24:
                tactic = 3
            elif n_dim <= 4096:
                tactic = 6
            else:
                tactic = 8
        elif 4096 <= k_dim <= 6144:
            if n_dim <= 3072:
                tactic = 6
            elif 4096 < n_dim <= 5376:
                tactic = 9
    elif n_dim <= 8192:
        if m_rows <= 8:
            tactic = 2
        elif m_rows <= 16:
            tactic = 6
    else:
        if m_rows <= 16:
            tactic = 10

    if tactic is None or tactic < 0 or tactic >= tactic_num:
        return None
    return tactic


def _should_use_legacy_mm_bf16_tgv(m_rows: int, k_dim: int, n_dim: int) -> bool:
    """Conservative fallback buckets if only ``flashinfer.mm_bf16(..., backend='tgv')`` exists."""
    if (
        m_rows <= 0
        or k_dim < 2048
        or n_dim < 2048
        or n_dim > 5760
        or (k_dim % 16) != 0
        or (n_dim % 8) != 0
        or (n_dim * k_dim) >= (1 << 30)
    ):
        return False
    if n_dim <= 3072:
        return m_rows <= 24
    if n_dim <= 4096:
        return m_rows <= 16
    return m_rows <= 8


def _should_use_bf16_tgv(m_rows: int, k_dim: int, n_dim: int) -> bool:
    """Return True iff ``(M, K, N)`` selects a strict-win SM10x ``tgv_gemm`` tactic."""
    return _select_bf16_tgv_tactic(m_rows, k_dim, n_dim) is not None


def _is_bf16_weight_ready(w: Any, x: torch.Tensor) -> bool:
    """Validate that ``w`` and ``x`` can safely execute a bias-free BF16 CUDA GEMM."""
    if not (isinstance(w, torch.Tensor) and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim >= 2):
        return False
    if (
        not w.is_cuda
        or w.dtype != torch.bfloat16
        or w.ndim != 2
        or not w.is_contiguous()
        or w.device != x.device
    ):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return False
    k_dim = int(x.shape[-1])
    return k_dim > 0 and int(w.shape[1]) == k_dim and int(x.numel() // k_dim) > 0


def _extract_bf16_linear_weight(mod: Any, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Return contiguous BF16 weight if ``mod(x)`` resolves to a bias-free BF16 linear."""
    if getattr(mod, "bias", None) is not None:
        return None
    w = getattr(mod, "weight", None)
    if not _is_bf16_weight_ready(w, x):
        return None
    resolve_fn = getattr(mod, "_resolve_backend", None)
    if callable(resolve_fn):
        m_rows = int(x.numel() // int(x.shape[-1]))
        try:
            if resolve_fn(m_rows, x.device) != "bf16":
                return None
        except Exception:
            return None
    return w


def _try_bf16_tgv(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    if not _is_bf16_tgv_enabled() or not x.is_cuda:
        return None
    k_dim = int(x.shape[-1])
    m_rows = int(x.numel() // k_dim)
    n_dim = int(w.shape[0])
    if _select_bf16_tgv_tactic(m_rows, k_dim, n_dim) is None:
        return None
    dev_idx = int(x.device.index) if x.device.index is not None else int(torch.cuda.current_device())
    if dev_idx not in _TGV_WARMED_DEVS and torch.cuda.is_current_stream_capturing():
        return None
    with torch.cuda.device(x.device):
        mm_fn, tgv_fn, tactic_num = get_sm10x_bf16_tgv_backend(x.device)
    if tgv_fn is not None:
        _TGV_WARMED_DEVS.add(dev_idx)
        tactic = _select_bf16_tgv_tactic(m_rows, k_dim, n_dim, tactic_num=tactic_num)
        if tactic is None:
            return None
        spec = can_sm10x_bf16_tgv(x, w, tactic=tactic)
        if spec is not None:
            return run_sm10x_bf16_tgv_gemm(x, w, spec, pdl=True)
        return None
    if mm_fn is not None and _should_use_legacy_mm_bf16_tgv(m_rows, k_dim, n_dim):
        _TGV_WARMED_DEVS.add(dev_idx)
        spec = can_sm10x_bf16_tgv(x, w, tactic=None)
        if spec is not None:
            return run_sm10x_bf16_tgv_gemm(x, w, spec, pdl=True)
    return None


def dense_weight_bf16_or_tgv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run ``F.linear(x, weight)`` or FlashInfer ``tgv`` on winning ``(M, K, N)`` buckets."""
    if _is_bf16_weight_ready(weight, x):
        tgv_out = _try_bf16_tgv(x, weight)
        if tgv_out is not None:
            return tgv_out
    return F.linear(x, weight)


def dense_linear_bf16_or_tgv(mod: Any, x: torch.Tensor) -> torch.Tensor:
    """Run ``mod(x)`` via FlashInfer BF16 ``tgv`` when in BF16 fallback on winning buckets."""
    if _is_bf16_tgv_enabled():
        w = _extract_bf16_linear_weight(mod, x)
        if w is not None:
            tgv_out = _try_bf16_tgv(x, w)
            if tgv_out is not None:
                return tgv_out
    return mod(x)


@triton.jit
def _causal_dwconv1d_step_k4_kernel(
    X_ptr,
    State_ptr,
    W_ptr,
    Y_ptr,
    stride_xb,
    stride_xc,
    stride_sb,
    stride_sc,
    stride_sk,
    stride_wc,
    stride_wk,
    stride_yb,
    stride_yc,
    C,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused single-token (T=1) causal depthwise conv1d (K=4) with in-place state update."""
    b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x = tl.load(X_ptr + b * stride_xb + offs_c * stride_xc, mask=mask, other=0.0).to(tl.float32)

    s_base = State_ptr + b * stride_sb + offs_c * stride_sc
    s0 = tl.load(s_base + 0 * stride_sk, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(s_base + 1 * stride_sk, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(s_base + 2 * stride_sk, mask=mask, other=0.0).to(tl.float32)

    # Shift state in-place: [s0, s1, s2] -> [s1, s2, x]
    tl.store(s_base + 0 * stride_sk, s1.to(State_ptr.dtype.element_ty), mask=mask)
    tl.store(s_base + 1 * stride_sk, s2.to(State_ptr.dtype.element_ty), mask=mask)
    tl.store(s_base + 2 * stride_sk, x.to(State_ptr.dtype.element_ty), mask=mask)

    w_base = W_ptr + offs_c * stride_wc
    w0 = tl.load(w_base + 0 * stride_wk, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_base + 1 * stride_wk, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_base + 2 * stride_wk, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w_base + 3 * stride_wk, mask=mask, other=0.0).to(tl.float32)

    y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x
    # Match F.conv1d BF16 output rounding before activation / residual so T=1 decode aligns with T>1 prefill
    y = y.to(Y_ptr.dtype.element_ty).to(tl.float32)
    if USE_SILU:
        y = y * tl.sigmoid(y)
    if RESIDUAL:
        y = y + x

    tl.store(Y_ptr + b * stride_yb + offs_c * stride_yc, y.to(Y_ptr.dtype.element_ty), mask=mask)


def causal_dwconv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: Optional[ShortConvState] = None,
    *,
    residual: bool = False,
    use_silu: bool = False,
) -> torch.Tensor:
    """Causal depthwise 1D convolution with optional rolling state, SiLU, and residual.

    Args:
        x: Input tensor ``[B, T, C]``.
        weight: Depthwise conv weight ``[C, 1, K]`` (e.g. ``[256, 1, 4]``).
        state: Optional ``ShortConvState`` with ``conv_state`` of shape ``[B, C, K - 1]``
            storing the last ``K - 1`` pre-conv input steps (updated in-place).
        residual: If True, returns ``x + conv(x)`` (``conv_state`` stores pre-conv ``x``).
        use_silu: If True, applies SiLU to the conv output before optional residual.
    """
    bsz, seq_len, channels = x.shape
    kernel_size = weight.shape[-1]
    state_len = kernel_size - 1

    # Fast single-step Triton path for decode (T == 1, K == 4)
    if (
        seq_len == 1
        and kernel_size == 4
        and state is not None
        and x.is_cuda
        and weight.is_cuda
        and state.conv_state.is_cuda
    ):
        x_2d = x[:, 0, :]
        y_2d = torch.empty_like(x_2d)
        w_2d = weight[:, 0, :]
        block_c = 256 if channels >= 256 else triton.next_power_of_2(channels)
        grid = (bsz, triton.cdiv(channels, block_c))
        _causal_dwconv1d_step_k4_kernel[grid](
            x_2d,
            state.conv_state,
            w_2d,
            y_2d,
            x_2d.stride(0),
            x_2d.stride(1),
            state.conv_state.stride(0),
            state.conv_state.stride(1),
            state.conv_state.stride(2),
            w_2d.stride(0),
            w_2d.stride(1),
            y_2d.stride(0),
            y_2d.stride(1),
            channels,
            RESIDUAL=residual,
            USE_SILU=use_silu,
            BLOCK_C=block_c,
            num_warps=4,
        )
        return y_2d.unsqueeze(1)

    x_ct = x.transpose(1, 2)  # [B, C, T]
    if state is not None:
        conv_state = state.conv_state
        x_padded = torch.cat([conv_state.to(x_ct.dtype), x_ct], dim=-1)
        conv_state.copy_(x_padded[:, :, -state_len:].to(conv_state.dtype))
    else:
        x_padded = F.pad(x_ct, (state_len, 0))

    out_ct = F.conv1d(
        x_padded.to(weight.dtype),
        weight,
        bias=None,
        padding=0,
        groups=channels,
    )
    if use_silu:
        out_ct = F.silu(out_ct)
    out = out_ct.transpose(1, 2).to(x.dtype)
    if residual:
        out = out + x
    return out


@triton.jit
def _qsa_head_rmsnorm_d128_kernel(
    X_ptr,
    G_ptr,
    OUT_ptr,
    TOK_STRIDE: tl.int64,
    HEAD_STRIDE: tl.int64,
    OUT_STRIDE: tl.int64,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr,
):
    """SM80+ FP32 1+gamma QSA head RMSNorm matching exact_head_rms_norm."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, D)
    off = (row // HEADS) * TOK_STRIDE + (row % HEADS) * HEAD_STRIDE + cols
    x = tl.load(X_ptr + off).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + EPS)
    g = 1.0 + tl.load(G_ptr + cols).to(tl.float32)
    out = (x * rstd) * g
    tl.store(OUT_ptr + row * OUT_STRIDE + cols, out.to(OUT_ptr.dtype.element_ty))


def _fast_qsa_head_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    if (
        x.is_cuda
        and gamma.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim in (2, 3, 4)
        and int(x.shape[-1]) == 128
        and x.stride(-1) == 1
        and all(int(s) > 0 for s in x.stride())
        and (x.ndim != 4 or int(x.stride(0)) == int(x.shape[1]) * int(x.stride(1)))
        and gamma.is_contiguous()
        and not (x.requires_grad or gamma.requires_grad)
    ):
        heads = int(x.shape[-2]) if x.ndim > 2 else 1
        tok_stride = int(x.stride(-3)) if x.ndim > 2 else int(x.stride(0))
        head_stride = int(x.stride(-2)) if x.ndim > 2 else 128
        rows = x.numel() // 128
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if rows == 0:
            return out
        ckey = (
            x.device.index,
            0,
            x.dtype,
            gamma.dtype,
            heads,
            _triton_int_spec(tok_stride),
            _triton_int_spec(head_stride),
            _triton_int_spec(128),
            float(eps),
        )
        triton_fast_launch(
            _ATTN_TRITON_CACHE,
            ckey,
            _qsa_head_rmsnorm_d128_kernel,
            rows,
            1,
            1,
            x.data_ptr() | gamma.data_ptr() | out.data_ptr(),
            (x, gamma, out, tok_stride, head_stride, 128),
            {"HEADS": heads, "EPS": float(eps), "D": 128},
            num_warps=4,
        )
        return out
    return exact_head_rms_norm(x, gamma, eps)


@triton.jit
def _gemma_rmsnorm_partial_rope_d256_r64_kernel(
    X_ptr,
    W_ptr,
    Pos_ptr,
    InvFreq_ptr,
    Out_ptr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_pb,
    stride_pt,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    T,
    eps,
):
    """Fused GemmaRMSNorm (1 + w in FP32) + partial NeoX RoPE (rotary_dim=64) for head_dim=256."""
    bt = tl.program_id(0)
    h = tl.program_id(1)
    b = bt // T
    t = bt % T

    offs_32 = tl.arange(0, 32)
    offs_64 = tl.arange(0, 64)
    offs_128 = tl.arange(0, 128)

    x_base = X_ptr + b * stride_xb + t * stride_xt + h * stride_xh
    x0 = tl.load(x_base + offs_32 * stride_xd).to(tl.float32)
    x1 = tl.load(x_base + (32 + offs_32) * stride_xd).to(tl.float32)
    x2 = tl.load(x_base + (64 + offs_64) * stride_xd).to(tl.float32)
    x3 = tl.load(x_base + (128 + offs_128) * stride_xd).to(tl.float32)

    sum_sq = (
        tl.sum(x0 * x0, axis=0)
        + tl.sum(x1 * x1, axis=0)
        + tl.sum(x2 * x2, axis=0)
        + tl.sum(x3 * x3, axis=0)
    )
    rstd = tl.rsqrt(sum_sq * (1.0 / 256.0) + eps)

    w0 = tl.load(W_ptr + offs_32).to(tl.float32)
    w1 = tl.load(W_ptr + 32 + offs_32).to(tl.float32)
    w2 = tl.load(W_ptr + 64 + offs_64).to(tl.float32)
    w3 = tl.load(W_ptr + 128 + offs_128).to(tl.float32)

    y0 = (x0 * rstd) * (1.0 + w0)
    y1 = (x1 * rstd) * (1.0 + w1)
    y2 = (x2 * rstd) * (1.0 + w2)
    y3 = (x3 * rstd) * (1.0 + w3)

    pos = tl.load(Pos_ptr + b * stride_pb + t * stride_pt).to(tl.float32)
    inv_freq = tl.load(InvFreq_ptr + offs_32).to(tl.float32)
    freqs = pos * inv_freq
    c = tl.cos(freqs)
    s = tl.sin(freqs)

    r0 = y0 * c - y1 * s
    r1 = y1 * c + y0 * s

    out_base = Out_ptr + b * stride_ob + t * stride_ot + h * stride_oh
    out_dtype = Out_ptr.dtype.element_ty
    tl.store(out_base + offs_32 * stride_od, r0.to(out_dtype))
    tl.store(out_base + (32 + offs_32) * stride_od, r1.to(out_dtype))
    tl.store(out_base + (64 + offs_64) * stride_od, y2.to(out_dtype))
    tl.store(out_base + (128 + offs_128) * stride_od, y3.to(out_dtype))


class GemmaRMSNorm(nn.Module):
    """Qwen/Gemma-style RMSNorm: ``rms_norm(x.float()) * (1.0 + weight.float())``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = normed * (1.0 + self.weight.float())
        return out.to(x.dtype)


def apply_partial_neox_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply NeoX-style partial RoPE on the first ``rotary_dim`` channels of ``x``.

    Args:
        x: Tensor of shape ``[B, T, H, D]``.
        positions: Position indices ``[B, T]``.
        inv_freq: Inverse frequencies ``[rotary_dim // 2]`` (FP32).
        rotary_dim: Number of leading channels to rotate (64 for Qwen 4.5-Flash).
    """
    freqs = positions.float().unsqueeze(-1) * inv_freq.float()  # [B, T, rotary_dim // 2]
    cos = freqs.cos().unsqueeze(-2)  # [B, T, 1, rotary_dim // 2]
    sin = freqs.sin().unsqueeze(-2)  # [B, T, 1, rotary_dim // 2]

    half = rotary_dim // 2
    x_rope = x[..., :rotary_dim].float()
    x0 = x_rope[..., :half]
    x1 = x_rope[..., half:]
    r0 = x0 * cos - x1 * sin
    r1 = x1 * cos + x0 * sin
    rotated = torch.cat([r0, r1], dim=-1).to(x.dtype)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def gemma_rmsnorm_and_partial_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply GemmaRMSNorm followed by partial NeoX RoPE (fused Triton fast path for D=256, R=64)."""
    if (
        x.is_cuda
        and weight.is_cuda
        and positions.is_cuda
        and inv_freq.is_cuda
        and x.ndim == 4
        and x.shape[-1] == 256
        and rotary_dim == 64
        and weight.is_contiguous()
        and inv_freq.is_contiguous()
    ):
        bsz, seq_len, num_heads, head_dim = x.shape
        out = torch.empty((bsz, seq_len, num_heads, head_dim), dtype=x.dtype, device=x.device)
        rt_args = (
            x,
            weight,
            positions,
            inv_freq,
            out,
            int(x.stride(0)),
            int(x.stride(1)),
            int(x.stride(2)),
            int(x.stride(3)),
            int(positions.stride(0)),
            int(positions.stride(1)),
            int(out.stride(0)),
            int(out.stride(1)),
            int(out.stride(2)),
            int(out.stride(3)),
            seq_len,
            float(eps),
        )
        ckey = (
            x.device.index,
            1,
            x.dtype,
            weight.dtype,
            positions.dtype,
            inv_freq.dtype,
            tuple(_triton_int_spec(v) for v in rt_args[5:16]),
        )
        triton_fast_launch(
            _ATTN_TRITON_CACHE,
            ckey,
            _gemma_rmsnorm_partial_rope_d256_r64_kernel,
            bsz * seq_len,
            num_heads,
            1,
            x.data_ptr() | weight.data_ptr() | positions.data_ptr() | inv_freq.data_ptr() | out.data_ptr(),
            rt_args,
            {},
            num_warps=4,
        )
        return out

    x_f = x.float()
    normed = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)) * (1.0 + weight.float())
    return apply_partial_neox_rope(normed.to(x.dtype), positions, inv_freq, rotary_dim)


@triton.jit
def _fused_attn_output_gate_kernel(
    ATTN_ptr,
    GATE_ptr,
    OUT_ptr,
    stride_ab: tl.int64,
    stride_ah: tl.int64,
    stride_at: tl.int64,
    stride_ad: tl.int64,
    stride_gb: tl.int64,
    stride_gt: tl.int64,
    stride_gh: tl.int64,
    stride_gd: tl.int64,
    stride_ob: tl.int64,
    stride_ot: tl.int64,
    stride_oh: tl.int64,
    stride_od: tl.int64,
    T,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused [B, H_q, T, D] transpose + sigmoid(strided gate) + [B, T, H_q*D] flatten."""
    bt = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    b = bt // T
    t = bt % T
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    a = tl.load(
        ATTN_ptr + b * stride_ab + h * stride_ah + t * stride_at + offs * stride_ad,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g = tl.load(
        GATE_ptr + b * stride_gb + t * stride_gt + h * stride_gh + offs * stride_gd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = a * tl.sigmoid(g)
    tl.store(
        OUT_ptr + b * stride_ob + t * stride_ot + h * stride_oh + offs * stride_od,
        out.to(OUT_ptr.dtype.element_ty),
        mask=mask,
    )


def apply_attn_output_gate_flatten(
    attn_bhtd: torch.Tensor,
    gate_bthd: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply optional sigmoid output gate and return contiguous [B, T, H_q * D]."""
    bsz, num_heads, seq_len, head_dim = attn_bhtd.shape
    if gate_bthd is None:
        return attn_bhtd.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    if (
        attn_bhtd.is_cuda
        and gate_bthd.is_cuda
        and head_dim <= 512
        and bsz * seq_len > 0
    ):
        out = torch.empty(
            (bsz, seq_len, num_heads * head_dim),
            dtype=attn_bhtd.dtype,
            device=attn_bhtd.device,
        )
        block_d = max(16, triton.next_power_of_2(head_dim))
        rt_args = (
            attn_bhtd,
            gate_bthd,
            out,
            int(attn_bhtd.stride(0)),
            int(attn_bhtd.stride(1)),
            int(attn_bhtd.stride(2)),
            int(attn_bhtd.stride(3)),
            int(gate_bthd.stride(0)),
            int(gate_bthd.stride(1)),
            int(gate_bthd.stride(2)),
            int(gate_bthd.stride(3)),
            int(out.stride(0)),
            int(out.stride(1)),
            head_dim,
            1,
            seq_len,
        )
        ckey = (
            attn_bhtd.device.index,
            2,
            attn_bhtd.dtype,
            gate_bthd.dtype,
            tuple(_triton_int_spec(v) for v in rt_args[3:]),
            head_dim,
            block_d,
        )
        triton_fast_launch(
            _ATTN_TRITON_CACHE,
            ckey,
            _fused_attn_output_gate_kernel,
            bsz * seq_len,
            num_heads,
            1,
            attn_bhtd.data_ptr() | gate_bthd.data_ptr() | out.data_ptr(),
            rt_args,
            {"D": head_dim, "BLOCK_D": block_d},
            num_warps=4,
        )
        return out
    attn_out = attn_bhtd.transpose(1, 2) * torch.sigmoid(gate_bthd)
    return attn_out.reshape(bsz, seq_len, num_heads * head_dim)


class Qwen4_5QSAIndexer(Qwen4ExpQSAIndexer):
    """``nn.Module`` owner wrapping ``Qwen4ExpQSAIndexer`` with registered BF16 parameters."""

    def __init__(self, config: Qwen4_5ArchConfig) -> None:
        idx_d = int(config.indexer_head_dim)
        qk_out = (int(config.indexer_n_heads) + int(config.indexer_kv_heads)) * idx_d
        index_qk_proj = nn.Linear(config.hidden_size, qk_out, bias=False)
        q_layernorm = GemmaRMSNorm(idx_d, eps=config.rms_norm_eps)
        k_layernorm = GemmaRMSNorm(idx_d, eps=config.rms_norm_eps)
        super().__init__(
            index_qk_proj.weight,
            q_layernorm.weight,
            k_layernorm.weight,
            n_heads=int(config.indexer_n_heads),
            kv_heads=int(config.indexer_kv_heads),
            head_dim=idx_d,
            token_budget=int(config.indexer_budget),
            compress_ratio=int(config.indexer_compress_ratio),
            norm_eps=float(config.rms_norm_eps),
        )
        self.index_qk_proj = index_qk_proj
        self.q_layernorm = q_layernorm
        self.k_layernorm = k_layernorm

    def _apply(self, fn):  # type: ignore[override]
        ret = super()._apply(fn)
        self.qk_proj = self.index_qk_proj.weight
        self.q_norm_gamma = self.q_layernorm.weight
        self.k_norm_gamma = self.k_layernorm.weight
        return ret


class Qwen4_5Attention(nn.Module):
    """Qwen 4.5-Flash Dense MQA-1 Full Attention with output gate, KV short-conv, and YoCo sharing.

    Checkpoint parameter names match ``model.layers.{i}.self_attn.*``:
      - Producer / normal (``is_consumer=False``):
        ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``q_norm``, ``k_norm``,
        ``k_sconv``, ``v_sconv``.
      - YoCo consumer (``is_consumer=True``, layers 23/27/31/35/39):
        ONLY ``q_proj``, ``o_proj``, ``q_norm``.
    """

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        is_consumer: bool = False,
        source_layer_idx: Optional[int] = None,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
        fp8_indexer_kv: bool = False,
        kv_cache_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_consumer = is_consumer
        self.source_layer_idx = source_layer_idx

        self.tp_size = max(1, int(tp_size))
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce
        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)
        self.fp8_indexer_kv = bool(fp8_indexer_kv)
        self.kv_cache_dtype = str(kv_cache_dtype)
        if config.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({config.num_attention_heads}) not divisible by tp_size ({self.tp_size})"
            )

        self.hidden_size = config.hidden_size
        self.global_num_heads = config.num_attention_heads
        self.num_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attn_output_gate = config.attn_output_gate
        self.rms_norm_eps = config.rms_norm_eps
        self.scaling = self.head_dim ** -0.5

        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope_theta = float(config.rope_theta)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.use_kv_sconv = config.use_full_attn_kv_sconv
        self.kv_sconv_residual = config.full_attn_kv_sconv_residual
        self.kv_sconv_use_activation = config.full_attn_kv_sconv_use_activation
        self.kv_sconv_kernel_size = config.linear_conv_kernel_dim  # 4

        q_out_dim = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, q_out_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if not self.is_consumer:
            self.k_proj: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, kv_dim, bias=config.attention_bias
            )
            self.v_proj: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, kv_dim, bias=config.attention_bias
            )
            self.k_norm: Optional[GemmaRMSNorm] = GemmaRMSNorm(
                self.head_dim, eps=self.rms_norm_eps
            )
            if self.use_kv_sconv:
                self.k_sconv: Optional[nn.Conv1d] = nn.Conv1d(
                    kv_dim,
                    kv_dim,
                    kernel_size=self.kv_sconv_kernel_size,
                    groups=kv_dim,
                    bias=False,
                )
                self.v_sconv: Optional[nn.Conv1d] = nn.Conv1d(
                    kv_dim,
                    kv_dim,
                    kernel_size=self.kv_sconv_kernel_size,
                    groups=kv_dim,
                    bias=False,
                )
            else:
                self.k_sconv = None
                self.v_sconv = None
        else:
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None
            self.k_sconv = None
            self.v_sconv = None

        if config.has_qsa:
            self.indexer_rotary_dim = min(self.rotary_dim, int(config.indexer_head_dim))
            self.indexer: Optional[Qwen4ExpQSAIndexer] = Qwen4_5QSAIndexer(config)
        else:
            self.indexer_rotary_dim = 0
            self.indexer = None

        self.qkv_idx_proj: Optional[nn.Module] = None
        self._qkv_idx_ptrs: Optional[Tuple[int, ...]] = None
        self._qkv_idx_splits: Tuple[int, ...] = ()
        self._qkv_idx_has_kv: bool = False
        self._qkv_idx_has_idx: bool = False

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        sharing_plan: Optional[YoCoSharingPlan] = None,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
        fp8_indexer_kv: bool = False,
        kv_cache_dtype: str = "bf16",
    ) -> "Qwen4_5Attention":
        plan = sharing_plan if sharing_plan is not None else YoCoSharingPlan.from_config(config)
        is_consumer = plan.is_full_attn_consumer(layer_idx)
        source_idx = plan.source_for(layer_idx)
        return cls(
            config=config,
            layer_idx=layer_idx,
            is_consumer=is_consumer,
            source_layer_idx=source_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            dp_attn=dp_attn,
            fp8_indexer_kv=fp8_indexer_kv,
            kv_cache_dtype=kv_cache_dtype,
        )

    def _resolve_cp_indexer_dist(self) -> Tuple[int, int, Any]:
        if (
            self.dp_attn
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return 1, 0, None
        if self.sp_size > 1 and self.sp_group is not None:
            return self.sp_size, self.sp_rank, self.sp_group
        if self.tp_size > 1 and self.tp_group is not None:
            return self.tp_size, self.tp_rank, self.tp_group
        return 1, 0, None

    def _maybe_all_reduce(
        self, out: torch.Tensor, *, skip_allreduce: bool = False
    ) -> torch.Tensor:
        if skip_allreduce:
            return out
        if self.fast_all_reduce is not None:
            return self.fast_all_reduce(out)
        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(
                out, op=torch.distributed.ReduceOp.SUM, group=self.tp_group
            )
        return out

    @property
    def k_sconv_weight(self) -> Optional[torch.Tensor]:
        return self.k_sconv.weight if self.k_sconv is not None else None

    @property
    def v_sconv_weight(self) -> Optional[torch.Tensor]:
        return self.v_sconv.weight if self.v_sconv is not None else None

    def create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Optional[FullAttnKVCache]:
        """Allocate a KV cache for normal/producer layers; returns ``None`` for consumers."""
        if self.is_consumer:
            return None
        shape = (batch_size, max_seq_len, self.num_kv_heads, self.head_dim)
        return FullAttnKVCache(
            k_cache=torch.zeros(shape, device=device, dtype=dtype),
            v_cache=torch.zeros(shape, device=device, dtype=dtype),
            seq_len=0,
        )

    def create_sconv_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[Optional[ShortConvState], Optional[ShortConvState]]:
        """Allocate rolling K/V short-conv states; returns ``(None, None)`` for consumers."""
        if self.is_consumer or not self.use_kv_sconv:
            return None, None
        kv_dim = self.num_kv_heads * self.head_dim
        state_len = self.kv_sconv_kernel_size - 1
        return (
            ShortConvState(
                conv_state=torch.zeros((batch_size, kv_dim, state_len), device=device, dtype=dtype)
            ),
            ShortConvState(
                conv_state=torch.zeros((batch_size, kv_dim, state_len), device=device, dtype=dtype)
            ),
        )

    def _collect_fusable_projections(
        self,
    ) -> Optional[Tuple[Tuple[nn.Linear, ...], bool, bool]]:
        has_kv = not self.is_consumer
        mods: list[nn.Linear] = []
        if not isinstance(self.q_proj, nn.Linear) or self.q_proj.bias is not None:
            return None
        mods.append(self.q_proj)
        if has_kv:
            if (
                not isinstance(self.k_proj, nn.Linear)
                or not isinstance(self.v_proj, nn.Linear)
                or self.k_proj.bias is not None
                or self.v_proj.bias is not None
            ):
                return None
            mods.extend((self.k_proj, self.v_proj))

        has_idx = False
        if self.indexer is not None:
            idx_lin = getattr(self.indexer, "index_qk_proj", None)
            if not isinstance(idx_lin, nn.Linear) or idx_lin.bias is not None:
                return None
            mods.append(idx_lin)
            has_idx = True

        if len(mods) < 2:
            return None
        w0 = mods[0].weight
        for m in mods[1:]:
            w = m.weight
            if (
                w.dtype != w0.dtype
                or w.device != w0.device
                or w.shape[1] != w0.shape[1]
            ):
                return None
        return tuple(mods), has_kv, has_idx

    def _fused_qkv_idx_ptrs_valid(self) -> bool:
        ptrs = self._qkv_idx_ptrs
        if self.qkv_idx_proj is None or ptrs is None:
            return False
        if int(self.q_proj.weight.data_ptr()) != ptrs[0]:
            return False
        pos = 1
        if self._qkv_idx_has_kv:
            if (
                self.k_proj is None
                or self.v_proj is None
                or int(self.k_proj.weight.data_ptr()) != ptrs[pos]
                or int(self.v_proj.weight.data_ptr()) != ptrs[pos + 1]
            ):
                return False
            pos += 2
        if self._qkv_idx_has_idx:
            if self.indexer is None:
                return False
            idx_lin = getattr(self.indexer, "index_qk_proj", None)
            if (
                idx_lin is None
                or int(idx_lin.weight.data_ptr()) != ptrs[pos]
                or int(self.indexer.qk_proj.data_ptr()) != ptrs[pos]
            ):
                return False
            pos += 1
        return pos == len(ptrs)

    def prepare_fused_projections(
        self,
        wrap_linear_fn: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> Optional[nn.Module]:
        """Fuse ``q_proj [+ k_proj + v_proj] [+ indexer.index_qk_proj]`` with zero BF16 duplication."""
        spec = self._collect_fusable_projections()
        if spec is None:
            return None
        mods, has_kv, has_idx = spec

        if self._fused_qkv_idx_ptrs_valid():
            if wrap_linear_fn is not None and isinstance(self.qkv_idx_proj, nn.Linear):
                self.qkv_idx_proj = wrap_linear_fn(self.qkv_idx_proj)
            return self.qkv_idx_proj

        splits = tuple(int(m.weight.shape[0]) for m in mods)
        in_features = int(mods[0].weight.shape[1])
        dtype = mods[0].weight.dtype
        device = mods[0].weight.device
        total_out = sum(splits)

        fused = nn.Linear(in_features, total_out, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            off = 0
            for m, dim in zip(mods, splits):
                fused.weight.data[off : off + dim].copy_(m.weight.data)
                off += dim

        base_w = fused.weight
        wrapped = wrap_linear_fn(fused) if wrap_linear_fn is not None else fused
        alias_owner = getattr(wrapped, "weight", None)
        alias_base = (
            alias_owner.data
            if (isinstance(alias_owner, torch.Tensor) and tuple(alias_owner.shape) == tuple(base_w.shape))
            else base_w.data
        )
        with torch.no_grad():
            off = 0
            for m, dim in zip(mods, splits):
                m.weight.data = alias_base[off : off + dim]
                off += dim
            if has_idx and self.indexer is not None:
                self.indexer.qk_proj = mods[-1].weight

        self.qkv_idx_proj = wrapped
        self._qkv_idx_splits = splits
        self._qkv_idx_has_kv = has_kv
        self._qkv_idx_has_idx = has_idx
        self._qkv_idx_ptrs = tuple(int(m.weight.data_ptr()) for m in mods)
        return wrapped

    def project_qkv_and_indexer(
        self,
        hidden_states: torch.Tensor,
        *,
        need_idx: bool = True,
        need_idx_norm: bool = True,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Project ``(q_packed, k_raw, v_raw, q_idx, raw_k_idx)`` in one fused GEMM when valid."""
        if self._fused_qkv_idx_ptrs_valid():
            assert self.qkv_idx_proj is not None
            packed = dense_linear_bf16_or_tgv(self.qkv_idx_proj, hidden_states)
            chunks = torch.split(packed, self._qkv_idx_splits, dim=-1)
            q_packed = chunks[0]
            if self._qkv_idx_has_kv:
                k_raw: Optional[torch.Tensor] = chunks[1]
                v_raw: Optional[torch.Tensor] = chunks[2]
                idx_pos = 3
            else:
                k_raw = None
                v_raw = None
                idx_pos = 1

            q_idx: Optional[torch.Tensor] = None
            raw_k_idx: Optional[torch.Tensor] = None
            if self._qkv_idx_has_idx and need_idx and self.indexer is not None:
                q_idx_raw, raw_k_idx = self.indexer.split_projected(chunks[idx_pos])
                q_idx = (
                    _fast_qsa_head_rmsnorm(
                        q_idx_raw, self.indexer.q_norm_gamma, self.indexer.norm_eps
                    )
                    if need_idx_norm
                    else q_idx_raw
                )
            return q_packed, k_raw, v_raw, q_idx, raw_k_idx

        q_packed = dense_linear_bf16_or_tgv(self.q_proj, hidden_states)
        k_raw = (
            dense_linear_bf16_or_tgv(self.k_proj, hidden_states)
            if (not self.is_consumer and self.k_proj is not None)
            else None
        )
        v_raw = (
            dense_linear_bf16_or_tgv(self.v_proj, hidden_states)
            if (not self.is_consumer and self.v_proj is not None)
            else None
        )
        if self.indexer is not None and need_idx:
            if need_idx_norm:
                q_idx, raw_k_idx = self.indexer.project(hidden_states)
            else:
                q_idx, raw_k_idx = self.indexer.project_raw(hidden_states)
        else:
            q_idx = None
            raw_k_idx = None
        return q_packed, k_raw, v_raw, q_idx, raw_k_idx

    def _resolve_positions(
        self,
        positions: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        seq_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        if positions is None:
            pos = torch.arange(seq_offset, seq_offset + seq_len, device=device, dtype=torch.int64)
            return pos.unsqueeze(0).expand(bsz, seq_len).contiguous()
        if positions.ndim == 1:
            return positions.unsqueeze(0).expand(bsz, seq_len).contiguous()
        return positions.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[FullAttnKVCache] = None,
        k_sconv_state: Optional[ShortConvState] = None,
        v_sconv_state: Optional[ShortConvState] = None,
        seq_offset: Optional[int] = None,
        k_sconv_fn: Optional[Callable[[torch.Tensor, Optional[ShortConvState]], torch.Tensor]] = None,
        v_sconv_fn: Optional[Callable[[torch.Tensor, Optional[ShortConvState]], torch.Tensor]] = None,
        qsa_selected: Optional[torch.Tensor] = None,
        qsa_past_raw_keys: Optional[torch.Tensor] = None,
        qsa_source_raw_keys: Optional[torch.Tensor] = None,
        qsa_source_k_norm_gamma: Optional[torch.Tensor] = None,
        qsa_on_raw_keys: Optional[Callable[[torch.Tensor], None]] = None,
        preprojected: Optional[
            Tuple[
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ]
        ] = None,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """Forward pass for dense MQA-1 Full Attention (producer/normal or YoCo consumer).

        Args:
            hidden_states: Input hidden states ``[B, T, hidden_size]``.
            positions: Token position ids ``[B, T]`` or ``[T]``. If omitted, inferred from
                ``seq_offset`` (or ``kv_cache.seq_len``).
            kv_cache: For producer/normal layers, the layer's own ``FullAttnKVCache`` (updated
                in-place). For consumer layers, the source layer's ``FullAttnKVCache`` (read-only;
                assumed already updated by the source layer for the current step).
            k_sconv_state: Rolling short-conv state for K (producer/normal only).
            v_sconv_state: Rolling short-conv state for V (producer/normal only).
            seq_offset: Explicit starting position offset in the KV cache.
            k_sconv_fn: Optional external K short-conv callable ``(k, state) -> k``.
            v_sconv_fn: Optional external V short-conv callable ``(v, state) -> v``.

        Returns:
            Output hidden states ``[B, T, hidden_size]``.
        """
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Q (+ optional KV & QSA indexer) projection
        if preprojected is not None:
            q_packed, pre_k_raw, pre_v_raw, pre_q_idx, pre_raw_k_idx = preprojected
        else:
            need_idx_in_fwd = self.indexer is not None and qsa_selected is None
            q_packed, pre_k_raw, pre_v_raw, pre_q_idx, pre_raw_k_idx = (
                self.project_qkv_and_indexer(hidden_states, need_idx=need_idx_in_fwd)
            )
        if self.attn_output_gate:
            q, gate = q_packed.unflatten(-1, (self.num_heads, self.head_dim * 2)).chunk(2, dim=-1)
        else:
            q = q_packed.unflatten(-1, (self.num_heads, self.head_dim))
            gate = None

        # 2. Resolve cache offset and positions
        if self.is_consumer:
            if kv_cache is None:
                raise ValueError(
                    f"Consumer FullAttn layer {self.layer_idx} requires source layer's kv_cache"
                )
            if seq_offset is None:
                if kv_cache.seq_len < seq_len:
                    raise ValueError(
                        f"Source kv_cache.seq_len ({kv_cache.seq_len}) < current T ({seq_len})"
                    )
                cur_offset = kv_cache.seq_len - seq_len
            else:
                cur_offset = seq_offset
            total_seq = cur_offset + seq_len
            if total_seq > kv_cache.seq_len:
                raise ValueError(
                    f"Consumer requested total_seq={total_seq} > source kv_cache.seq_len={kv_cache.seq_len}"
                )
        else:
            cur_offset = seq_offset if seq_offset is not None else (
                kv_cache.seq_len if kv_cache is not None else 0
            )
            total_seq = cur_offset + seq_len

        pos = self._resolve_positions(positions, bsz, seq_len, cur_offset, device)

        # 3. Q GemmaRMSNorm + Partial NeoX RoPE
        q = gemma_rmsnorm_and_partial_rope(
            q,
            self.q_norm.weight,
            pos,
            self.inv_freq,
            self.rotary_dim,
            eps=self.rms_norm_eps,
        )

        # 4. K, V projection / short-conv / norm / RoPE / KV cache read-write
        if self.is_consumer:
            assert kv_cache is not None
            k_full = kv_cache.k_cache[:, :total_seq]  # [B, total_seq, 1, 256]
            v_full = kv_cache.v_cache[:, :total_seq]  # [B, total_seq, 1, 256]
        else:
            assert self.k_proj is not None and self.v_proj is not None and self.k_norm is not None
            k_raw = pre_k_raw if pre_k_raw is not None else dense_linear_bf16_or_tgv(self.k_proj, hidden_states)  # [B, T, 256]
            v_raw = pre_v_raw if pre_v_raw is not None else dense_linear_bf16_or_tgv(self.v_proj, hidden_states)  # [B, T, 256]

            if self.use_kv_sconv:
                if k_sconv_fn is not None:
                    k_raw = k_sconv_fn(k_raw, k_sconv_state)
                elif self.k_sconv is not None:
                    k_raw = causal_dwconv1d(
                        k_raw,
                        self.k_sconv.weight,
                        state=k_sconv_state,
                        residual=self.kv_sconv_residual,
                        use_silu=self.kv_sconv_use_activation,
                    )
                if v_sconv_fn is not None:
                    v_raw = v_sconv_fn(v_raw, v_sconv_state)
                elif self.v_sconv is not None:
                    v_raw = causal_dwconv1d(
                        v_raw,
                        self.v_sconv.weight,
                        state=v_sconv_state,
                        residual=self.kv_sconv_residual,
                        use_silu=self.kv_sconv_use_activation,
                    )

            k = k_raw.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, 1, 256]
            v = v_raw.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, 1, 256]

            k = gemma_rmsnorm_and_partial_rope(
                k,
                self.k_norm.weight,
                pos,
                self.inv_freq,
                self.rotary_dim,
                eps=self.rms_norm_eps,
            )

            if kv_cache is not None:
                if total_seq > kv_cache.k_cache.shape[1]:
                    raise ValueError(
                        f"KV cache overflow: total_seq={total_seq} > max_seq={kv_cache.k_cache.shape[1]}"
                    )
                kv_cache.k_cache[:, cur_offset:total_seq].copy_(k)
                kv_cache.v_cache[:, cur_offset:total_seq].copy_(v)
                kv_cache.seq_len = total_seq
                k_full = kv_cache.k_cache[:, :total_seq]
                v_full = kv_cache.v_cache[:, :total_seq]
            else:
                k_full = k
                v_full = v

        # 5. Attention core: QSA Sparse FMHA when indexer enabled, else Dense SDPA
        q_bhtd = q.transpose(1, 2)       # [B, 32, T, 256]
        k_bhtd = k_full.transpose(1, 2)  # [B, 1, total_seq, 256]
        v_bhtd = v_full.transpose(1, 2)  # [B, 1, total_seq, 256]

        if self.indexer is not None:
            from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
                build_qsa_neox_cos_sin,
                qsa_select_causal_from_projected,
                qsa_sparse_prefill_attention,
            )

            qsa_selected_is_q4_valid = (
                qsa_selected is None and int(getattr(self.indexer, "compress_ratio", 0)) == 4
            )
            if qsa_selected is None:
                if not self.is_consumer:
                    if pre_q_idx is not None and pre_raw_k_idx is not None:
                        q_idx, raw_k_idx = pre_q_idx, pre_raw_k_idx
                    else:
                        q_idx, raw_k_idx = self.indexer.project(hidden_states)
                    if qsa_on_raw_keys is not None:
                        qsa_on_raw_keys(raw_k_idx)
                    cur_raw = raw_k_idx
                    past_raw = qsa_past_raw_keys
                    k_gamma = self.indexer.k_norm_gamma
                else:
                    if pre_q_idx is not None:
                        q_idx = pre_q_idx
                    else:
                        q_idx, _ = self.indexer.project(hidden_states)
                    if qsa_source_raw_keys is None:
                        raise ValueError(
                            f"YoCo consumer FullAttn layer {self.layer_idx} QSA forward "
                            "requires qsa_source_raw_keys or precomputed qsa_selected"
                        )
                    src_slice = qsa_source_raw_keys[:, :total_seq]
                    past_raw = src_slice[:, :cur_offset] if cur_offset > 0 else None
                    cur_raw = src_slice[:, cur_offset:total_seq]
                    k_gamma = (
                        qsa_source_k_norm_gamma
                        if qsa_source_k_norm_gamma is not None
                        else self.indexer.k_norm_gamma
                    )
                if cur_offset == 0 and past_raw is None:
                    pos_all = pos
                else:
                    past_len_resolved = (
                        int(past_raw.shape[1]) if past_raw is not None else cur_offset
                    )
                    prefix_pos = (
                        torch.arange(past_len_resolved, device=device, dtype=pos.dtype)
                        .unsqueeze(0)
                        .expand(bsz, past_len_resolved)
                    )
                    pos_all = torch.cat([prefix_pos, pos], dim=1)
                cos_all, sin_all = build_qsa_neox_cos_sin(
                    pos_all,
                    self.indexer_rotary_dim,
                    self.rope_theta,
                    hidden_states.dtype,
                )
                cp_s, cp_r, cp_g = self._resolve_cp_indexer_dist()
                qsa_selected = qsa_select_causal_from_projected(
                    self.indexer,
                    q_idx,
                    cur_raw,
                    cos_all,
                    sin_all,
                    past_raw_keys=past_raw,
                    k_norm_gamma=k_gamma,
                    sp_size=cp_s,
                    sp_rank=cp_r,
                    sp_group=cp_g,
                    fast_all_reduce=self.fast_all_reduce,
                )
            attn_out = qsa_sparse_prefill_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                qsa_selected,
                validate_selected=False,
                assume_unique_selected=qsa_selected_is_q4_valid,
            )
        elif seq_len == 1:
            attn_out = F.scaled_dot_product_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                attn_mask=None,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=True,
            )
        elif cur_offset == 0:
            attn_out = F.scaled_dot_product_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                attn_mask=None,
                is_causal=True,
                scale=self.scaling,
                enable_gqa=True,
            )
        else:
            q_idx = torch.arange(cur_offset, total_seq, device=device).unsqueeze(1)
            k_idx = torch.arange(total_seq, device=device).unsqueeze(0)
            causal_mask = q_idx >= k_idx  # [T, total_seq] bool
            attn_out = F.scaled_dot_product_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                attn_mask=causal_mask,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=True,
            )

        # 6. Output gate & output projection
        out_flat = apply_attn_output_gate_flatten(attn_out, gate)
        return self._maybe_all_reduce(
            dense_linear_bf16_or_tgv(self.o_proj, out_flat), skip_allreduce=skip_allreduce
        )

    def forward_paged_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache_layer: Any,
        decode_attn_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        *,
        block_table: Optional[torch.Tensor] = None,
        seq_before_dev: Optional[torch.Tensor] = None,
        k_sconv_state: Optional[ShortConvState] = None,
        v_sconv_state: Optional[ShortConvState] = None,
        write_kv_fn: Optional[
            Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
        ] = None,
        preprojected: Optional[
            Tuple[
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ]
        ] = None,
        preconv_kv: bool = False,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """CUDA-graph-safe B*1 paged decode path for producer and YoCo consumer layers."""
        bsz, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError(f"forward_paged_decode expects seq_len=1, got {seq_len}")
        pos = positions.unsqueeze(1) if positions.ndim == 1 else positions

        if preprojected is not None:
            q_packed, pre_k_raw, pre_v_raw, _, _ = preprojected
        else:
            q_packed, pre_k_raw, pre_v_raw, _, _ = self.project_qkv_and_indexer(
                hidden_states, need_idx=False
            )
        if self.attn_output_gate:
            q, gate = q_packed.unflatten(-1, (self.num_heads, self.head_dim * 2)).chunk(
                2, dim=-1
            )
        else:
            q = q_packed.unflatten(-1, (self.num_heads, self.head_dim))
            gate = None

        q = gemma_rmsnorm_and_partial_rope(
            q,
            self.q_norm.weight,
            pos,
            self.inv_freq,
            self.rotary_dim,
            eps=self.rms_norm_eps,
        )

        if not self.is_consumer:
            assert (
                self.k_proj is not None
                and self.v_proj is not None
                and self.k_norm is not None
                and write_kv_fn is not None
                and block_table is not None
                and seq_before_dev is not None
            )
            k_raw = pre_k_raw if pre_k_raw is not None else dense_linear_bf16_or_tgv(self.k_proj, hidden_states)
            v_raw = pre_v_raw if pre_v_raw is not None else dense_linear_bf16_or_tgv(self.v_proj, hidden_states)
            if self.use_kv_sconv and not preconv_kv:
                if self.k_sconv is not None:
                    k_raw = causal_dwconv1d(
                        k_raw,
                        self.k_sconv.weight,
                        state=k_sconv_state,
                        residual=self.kv_sconv_residual,
                        use_silu=self.kv_sconv_use_activation,
                    )
                if self.v_sconv is not None:
                    v_raw = causal_dwconv1d(
                        v_raw,
                        self.v_sconv.weight,
                        state=v_sconv_state,
                        residual=self.kv_sconv_residual,
                        use_silu=self.kv_sconv_use_activation,
                    )

            k = k_raw.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = v_raw.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = gemma_rmsnorm_and_partial_rope(
                k,
                self.k_norm.weight,
                pos,
                self.inv_freq,
                self.rotary_dim,
                eps=self.rms_norm_eps,
            )
            write_kv_fn(kv_cache_layer, block_table, seq_before_dev, k, v)

        attn_out = decode_attn_fn(q.squeeze(1), kv_cache_layer)
        if gate is not None:
            out_flat = apply_attn_output_gate_flatten(attn_out.unsqueeze(2), gate)
        else:
            out_flat = attn_out.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self._maybe_all_reduce(
            dense_linear_bf16_or_tgv(self.o_proj, out_flat), skip_allreduce=skip_allreduce
        )


__all__ = [
    "GemmaRMSNorm",
    "Qwen4_5Attention",
    "Qwen4_5QSAIndexer",
    "apply_attn_output_gate_flatten",
    "apply_partial_neox_rope",
    "causal_dwconv1d",
    "dense_linear_bf16_or_tgv",
    "dense_weight_bf16_or_tgv",
    "gemma_rmsnorm_and_partial_rope",
]

"""Qwen 4.5-Flash Causal Depthwise Short-Conv (k=4) with Triton Prefill & Decode kernels.

Supports:
  - KV short-conv (channels=256, residual=True, use_activation=False)
  - Output short-conv (channels=5120, residual=True/False, use_activation=False/True)
  - PLE short-conv (channels=20480, residual=False, use_activation=False)
  - Stateful Prefill, Chunked Prefill, Single-Step Decode (T=1), and Multi-Step Decode (T>1)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .contracts import PLELayerState, ShortConvState

ConvStateLike = Union[ShortConvState, PLELayerState, torch.Tensor]


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _causal_dwconv1d_prefill_kernel(
    X_PTR,
    W_PTR,
    STATE_PTR,
    OUT_PTR,
    T,
    C: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_STATE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    t_start = tl.program_id(1).to(tl.int64) * BLOCK_T
    c_start = tl.program_id(2).to(tl.int64) * BLOCK_C

    t_offs = t_start + tl.arange(0, BLOCK_T).to(tl.int64)
    c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
    t_mask = t_offs < T
    c_mask = c_offs < C
    tc_mask = t_mask[:, None] & c_mask[None, :]

    x_batch_base = batch * T * C
    s_batch_base = batch * C * STATE_LEN
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    x_cur = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for k in tl.static_range(KERNEL_SIZE):
        w_k = tl.load(
            W_PTR + c_offs * KERNEL_SIZE + k, mask=c_mask, other=0.0
        ).to(tl.float32)
        src_t = t_offs + (k - STATE_LEN)
        in_x = tc_mask & (src_t[:, None] >= 0)
        val = tl.load(
            X_PTR + x_batch_base + src_t[:, None] * C + c_offs[None, :],
            mask=in_x,
            other=0.0,
        ).to(tl.float32)
        if HAS_STATE:
            if t_start < STATE_LEN:
                in_s = tc_mask & (src_t[:, None] < 0)
                s_idx = src_t + STATE_LEN
                s_val = tl.load(
                    STATE_PTR
                    + s_batch_base
                    + c_offs[None, :] * STATE_LEN
                    + s_idx[:, None],
                    mask=in_s,
                    other=0.0,
                ).to(tl.float32)
                val = tl.where(src_t[:, None] < 0, s_val, val)
        if RESIDUAL:
            if k == KERNEL_SIZE - 1:
                x_cur = val
        acc += val * w_k[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if RESIDUAL:
        acc = x_cur + acc

    tl.store(
        OUT_PTR + x_batch_base + t_offs[:, None] * C + c_offs[None, :],
        acc,
        mask=tc_mask,
    )


@triton.jit(do_not_specialize=["T"])
def _update_conv_state_kernel(
    X_PTR,
    STATE_PTR,
    T,
    C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C
    x_batch_base = batch * T * C
    s_base = batch * C * STATE_LEN + c_offs * STATE_LEN

    for j in tl.static_range(STATE_LEN):
        src_t = T - STATE_LEN + j
        if src_t >= 0:
            val = tl.load(X_PTR + x_batch_base + src_t * C + c_offs, mask=c_mask)
        else:
            val = tl.load(STATE_PTR + s_base + (j + T), mask=c_mask)
        tl.store(STATE_PTR + s_base + j, val, mask=c_mask)


@triton.jit
def _causal_dwconv1d_decode_k4_kernel(
    X_PTR,
    W_PTR,
    STATE_PTR,
    OUT_PTR,
    C: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C

    x_off = batch * C + c_offs
    s_base = batch * C * 3 + c_offs * 3
    w_base = c_offs * 4

    s0_raw = tl.load(STATE_PTR + s_base + 0, mask=c_mask, other=0.0)
    s1_raw = tl.load(STATE_PTR + s_base + 1, mask=c_mask, other=0.0)
    s2_raw = tl.load(STATE_PTR + s_base + 2, mask=c_mask, other=0.0)
    x0_raw = tl.load(X_PTR + x_off, mask=c_mask, other=0.0)

    w0 = tl.load(W_PTR + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W_PTR + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W_PTR + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(W_PTR + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)

    x0_f = x0_raw.to(tl.float32)
    acc = (
        s0_raw.to(tl.float32) * w0
        + s1_raw.to(tl.float32) * w1
        + s2_raw.to(tl.float32) * w2
        + x0_f * w3
    )

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if RESIDUAL:
        acc = x0_f + acc

    tl.store(OUT_PTR + x_off, acc, mask=c_mask)

    # Roll state in-place: [s0, s1, s2] -> [s1, s2, x0]
    tl.store(STATE_PTR + s_base + 0, s1_raw, mask=c_mask)
    tl.store(STATE_PTR + s_base + 1, s2_raw, mask=c_mask)
    tl.store(STATE_PTR + s_base + 2, x0_raw, mask=c_mask)


# ---------------------------------------------------------------------------
# Helper / Functional Wrappers
# ---------------------------------------------------------------------------


def _extract_conv_state(state: Optional[ConvStateLike]) -> Optional[torch.Tensor]:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state
    return state.conv_state


def causal_dwconv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
) -> torch.Tensor:
    """Exact pure-PyTorch FP32 reference for causal depthwise conv1d + state update."""
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")
    B, T, C = x.shape
    kernel_size = int(weight.shape[-1])
    state_len = kernel_size - 1
    if T == 0:
        return torch.empty_like(x)

    x_t = x.transpose(1, 2)  # [B, C, T]
    if conv_state is not None:
        if tuple(conv_state.shape) != (B, C, state_len):
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)} != {(B, C, state_len)}"
            )
        x_seq = torch.cat([conv_state, x_t], dim=-1)  # [B, C, state_len + T]
        if state_len > 0:
            conv_state.copy_(x_seq[:, :, -state_len:])
    else:
        x_seq = F.pad(x_t, (state_len, 0))

    y = F.conv1d(x_seq.float(), weight.float(), bias=None, stride=1, padding=0, groups=C)
    if use_activation:
        y = F.silu(y)
    if residual:
        y = x_t.float() + y
    return y.transpose(1, 2).contiguous().to(x.dtype)


def causal_dwconv1d_prefill_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
) -> torch.Tensor:
    """Fused Triton causal depthwise conv1d for Prefill / multi-step input [B, T, C].

    If ``conv_state`` ([B, C, K-1]) is provided, it is consumed as left-context
    and updated in-place with the last ``K-1`` tokens of ``[conv_state, x]``.
    """
    if not x.is_cuda:
        return causal_dwconv1d_ref(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")

    x_c = x.contiguous()
    w_c = weight.contiguous()
    B, T, C = x_c.shape
    kernel_size = int(w_c.shape[-1])
    state_len = kernel_size - 1
    out = torch.empty_like(x_c)
    if B == 0 or T == 0:
        return out

    if conv_state is not None:
        if tuple(conv_state.shape) != (B, C, state_len):
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)} != {(B, C, state_len)}"
            )
        if not conv_state.is_contiguous():
            raise ValueError("conv_state must be contiguous for in-place Triton update")

    block_c = 64 if C <= 256 else 128
    block_t = 16 if (T <= 32 or C >= 4096) else 32
    num_warps = 4

    _causal_dwconv1d_prefill_kernel[
        (B, triton.cdiv(T, block_t), triton.cdiv(C, block_c))
    ](
        x_c,
        w_c,
        conv_state if conv_state is not None else x_c,
        out,
        T=T,
        C=C,
        KERNEL_SIZE=kernel_size,
        STATE_LEN=state_len,
        HAS_STATE=conv_state is not None and state_len > 0,
        RESIDUAL=residual,
        USE_SILU=use_activation,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )

    if conv_state is not None and state_len > 0:
        upd_block_c = 256 if C <= 512 else 512
        _update_conv_state_kernel[(B, triton.cdiv(C, upd_block_c))](
            x_c,
            conv_state,
            T=T,
            C=C,
            STATE_LEN=state_len,
            BLOCK_C=upd_block_c,
            num_warps=4,
        )
    return out


def causal_dwconv1d_decode_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    *,
    residual: bool = False,
    use_activation: bool = False,
) -> torch.Tensor:
    """Single-launch fused Triton decode conv1d + optional SiLU/residual + in-place state roll.

    Specialized for ``T == 1, kernel_size == 4``; seamlessly handles multi-step
    decode (``T > 1``) or other kernel sizes via ``causal_dwconv1d_prefill_triton``.
    """
    if not x.is_cuda:
        return causal_dwconv1d_ref(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")

    B, T, C = x.shape
    kernel_size = int(weight.shape[-1])
    if T != 1 or kernel_size != 4:
        return causal_dwconv1d_prefill_triton(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )

    if tuple(conv_state.shape) != (B, C, 3):
        raise ValueError(
            f"conv_state shape {tuple(conv_state.shape)} != {(B, C, 3)}"
        )
    if not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous for in-place Triton update")

    x_c = x.contiguous()
    w_c = weight.contiguous()
    out = torch.empty_like(x_c)
    if B == 0:
        return out

    block_c = 256 if C <= 512 else 512
    num_warps = 4 if block_c <= 256 else 8
    _causal_dwconv1d_decode_k4_kernel[(B, triton.cdiv(C, block_c))](
        x_c,
        w_c,
        conv_state,
        out,
        C=C,
        RESIDUAL=residual,
        USE_SILU=use_activation,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Module: Qwen4_5ShortConv
# ---------------------------------------------------------------------------


class Qwen4_5ShortConv(nn.Module):
    """Causal depthwise 1D short convolution (default k=4) for Qwen 4.5-Flash."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        residual: bool = False,
        use_activation: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.residual = residual
        self.use_activation = use_activation
        self.activation = "silu" if use_activation else None

        self.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, dtype=dtype, device=device)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def alloc_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ShortConvState:
        dev = device if device is not None else self.weight.device
        dt = dtype if dtype is not None else self.weight.dtype
        return ShortConvState(
            conv_state=torch.zeros(
                (batch_size, self.channels, self.kernel_size - 1),
                device=dev,
                dtype=dt,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
    ) -> torch.Tensor:
        squeeze_batch = False
        if x.ndim == 2:
            if state is not None:
                raise ValueError("Stateful ShortConv requires 3D input [B, T, C]")
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")

        conv_state = _extract_conv_state(state)
        if x.is_cuda:
            if conv_state is not None and x.shape[1] == 1 and self.kernel_size == 4:
                y = causal_dwconv1d_decode_triton(
                    x,
                    self.weight,
                    conv_state,
                    residual=self.residual,
                    use_activation=self.use_activation,
                )
            else:
                y = causal_dwconv1d_prefill_triton(
                    x,
                    self.weight,
                    conv_state,
                    residual=self.residual,
                    use_activation=self.use_activation,
                )
        else:
            y = causal_dwconv1d_ref(
                x,
                self.weight,
                conv_state,
                residual=self.residual,
                use_activation=self.use_activation,
            )

        return y.squeeze(0) if squeeze_batch else y

    def forward_ref(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
    ) -> torch.Tensor:
        """Pure-PyTorch FP32 reference forward (updates state in-place if provided)."""
        squeeze_batch = False
        if x.ndim == 2:
            if state is not None:
                raise ValueError("Stateful ShortConv requires 3D input [B, T, C]")
            x = x.unsqueeze(0)
            squeeze_batch = True
        conv_state = _extract_conv_state(state)
        y = causal_dwconv1d_ref(
            x,
            self.weight,
            conv_state,
            residual=self.residual,
            use_activation=self.use_activation,
        )
        return y.squeeze(0) if squeeze_batch else y

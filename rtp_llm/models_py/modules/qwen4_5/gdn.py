"""Qwen 4.5-Flash Gated DeltaNet (GDN) Linear Attention with Triton Chunk Prefill, Recurrent Decode, and YoCo Sharing.

Architecture (1109-bf16-re-mxfp4-ple-fp8):
  - hidden_size = 5120
  - num_k_heads = 32, head_k_dim = 64  (key_dim = 2048)
  - num_v_heads = 128, head_v_dim = 64 (value_dim = 8192, 4:1 GQA ratio)
  - conv_kernel_size = 4, gdn_conv_residual = False, activation = SiLU
  - RMSNormGated: weight [64] (ones-init, standard RMSNorm, NOT 1+w), norm_before_gate = True,
    output_gate_type = "sigmoid"
  - SSM state layout (matching ``contracts.py`` and ``yoco.py``):
        ``[B, H_v, D_k, D_v]`` in ``torch.float32``
  - YoCo sharing:
      * Source layer (layer 20, ``is_consumer=False, publishes_shared=True``): publishes
        ``SharedGDNInputs(key, value, g, beta, ssm_state)`` post-update.
      * Consumer layers (layers 21,22,24..26,28..30,32..34,36..38, ``is_consumer=True``):
        have ONLY ``in_proj_qkvz [10240, 5120]``, ``conv1d [2048, 1, 4]``, ``norm [64]``,
        ``out_proj [5120, 8192]`` (NO ``in_proj_ba``, ``A_log``, ``dt_bias``).
        During Decode (``T == 1`` and ``enable_yoco_decode=True``), consumers execute
        ``yoco_gdn_decode_readout(q, shared.ssm_state, 64)`` directly without storing or
        updating a separate SSM state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.attention import causal_dwconv1d
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    GDNLayerState,
    Qwen4_5ArchConfig,
    SharedGDNInputs,
    ShortConvState,
)
from rtp_llm.models_py.modules.qwen4_5.yoco import (
    SSM_STATE_LAYOUT,
    YoCoSharingPlan,
    yoco_gdn_decode_readout,
)


# ---------------------------------------------------------------------------
# Fused GDN gating + interleaved BA unpack kernel
# ---------------------------------------------------------------------------


@triton.jit
def _gdn_gating_ba_interleaved_kernel(
    BA_ptr,
    A_log_ptr,
    Dt_bias_ptr,
    G_ptr,
    Beta_ptr,
    stride_bab,
    stride_bat,
    stride_bac,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_bb,
    stride_bt,
    stride_bh,
    T,
    H_V: tl.constexpr,
    GROUP_V: tl.constexpr,
):
    """Unpack interleaved in_proj_ba [B, T, H_k, 2*GROUP_V] and compute (g, beta) in FP32."""
    bt = tl.program_id(0)
    b = bt // T
    t = bt % T

    offs_h = tl.arange(0, H_V)
    h_k = offs_h // GROUP_V
    sub = offs_h % GROUP_V
    b_idx = h_k * (2 * GROUP_V) + sub
    a_idx = b_idx + GROUP_V

    ba_base = BA_ptr + b * stride_bab + t * stride_bat
    b_val = tl.load(ba_base + b_idx * stride_bac).to(tl.float32)
    a_val = tl.load(ba_base + a_idx * stride_bac).to(tl.float32)

    a_log = tl.load(A_log_ptr + offs_h).to(tl.float32)
    dt_bias = tl.load(Dt_bias_ptr + offs_h).to(tl.float32)

    x = a_val + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = -tl.exp(a_log) * sp
    beta = tl.sigmoid(b_val)

    tl.store(G_ptr + b * stride_gb + t * stride_gt + offs_h * stride_gh, g)
    tl.store(Beta_ptr + b * stride_bb + t * stride_bt + offs_h * stride_bh, beta)


def compute_gdn_gating(
    ba_interleaved: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack interleaved ``in_proj_ba`` output and compute ``(g, beta)`` in FP32.

    Args:
        ba_interleaved: Output of ``in_proj_ba``, shape ``[B, T, 2 * H_v]``, organized as
            ``H_k`` groups of ``[b (group_v), a (group_v)]``.
        a_log: Parameter ``A_log`` of shape ``[H_v]``.
        dt_bias: Parameter ``dt_bias`` of shape ``[H_v]``.

    Returns:
        ``(g, beta)``, each FP32 of shape ``[B, T, H_v]``.
    """
    bsz, seq_len, _ = ba_interleaved.shape
    group_v = num_v_heads // num_k_heads

    if (
        ba_interleaved.is_cuda
        and a_log.is_cuda
        and dt_bias.is_cuda
        and (num_v_heads & (num_v_heads - 1)) == 0
        and num_v_heads <= 1024
        and a_log.is_contiguous()
        and dt_bias.is_contiguous()
    ):
        g = torch.empty((bsz, seq_len, num_v_heads), dtype=torch.float32, device=ba_interleaved.device)
        beta = torch.empty((bsz, seq_len, num_v_heads), dtype=torch.float32, device=ba_interleaved.device)
        _gdn_gating_ba_interleaved_kernel[(bsz * seq_len,)](
            ba_interleaved,
            a_log,
            dt_bias,
            g,
            beta,
            ba_interleaved.stride(0),
            ba_interleaved.stride(1),
            ba_interleaved.stride(2),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            seq_len,
            H_V=num_v_heads,
            GROUP_V=group_v,
            num_warps=4,
        )
        return g, beta

    b_g, a_g = ba_interleaved.unflatten(-1, (num_k_heads, 2 * group_v)).split([group_v, group_v], dim=-1)
    b = b_g.reshape(bsz, seq_len, num_v_heads).float()
    a = a_g.reshape(bsz, seq_len, num_v_heads).float()
    g = -torch.exp(a_log.float()) * F.softplus(a + dt_bias.float())
    beta = torch.sigmoid(b)
    return g, beta


# ---------------------------------------------------------------------------
# Fused RMSNormGated (norm_before_gate=True, sigmoid or silu gate)
# ---------------------------------------------------------------------------


@triton.jit
def _rms_norm_gated_kernel(
    X_ptr,
    Z_ptr,
    W_ptr,
    Y_ptr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_zb,
    stride_zt,
    stride_zh,
    stride_zd,
    stride_yb,
    stride_yt,
    stride_yh,
    stride_yd,
    T,
    eps,
    D: tl.constexpr,
    USE_SIGMOID: tl.constexpr,
):
    """Per-head RMSNorm (standard weight, NOT 1+w) followed by sigmoid/silu gating."""
    bt = tl.program_id(0)
    h = tl.program_id(1)
    b = bt // T
    t = bt % T

    offs = tl.arange(0, D)
    x = tl.load(X_ptr + b * stride_xb + t * stride_xt + h * stride_xh + offs * stride_xd).to(tl.float32)
    z = tl.load(Z_ptr + b * stride_zb + t * stride_zt + h * stride_zh + offs * stride_zd).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    y = (x * rstd) * w
    gate = tl.sigmoid(z) if USE_SIGMOID else (z * tl.sigmoid(z))
    y = y * gate
    tl.store(
        Y_ptr + b * stride_yb + t * stride_yt + h * stride_yh + offs * stride_yd,
        y.to(Y_ptr.dtype.element_ty),
    )


class RMSNormGated(nn.Module):
    """Gated per-head RMSNorm matching Qwen3Next/Qwen4.5 ``linear_attn.norm``.

    Computes ``rms_norm(x, weight, eps) * act(z)`` where ``weight`` is standard
    ones-initialized multiplicative scale (NOT ``1 + weight``).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        activation: str = "sigmoid",
        norm_before_gate: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.activation = activation
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if (
            self.norm_before_gate
            and self.activation in ("sigmoid", "silu")
            and x.is_cuda
            and gate.is_cuda
            and self.weight.is_cuda
            and x.ndim == 4
            and x.shape[-1] == self.dim
            and (self.dim & (self.dim - 1)) == 0
            and self.dim <= 512
            and self.weight.is_contiguous()
        ):
            bsz, seq_len, num_heads, dim = x.shape
            out = torch.empty((bsz, seq_len, num_heads, dim), dtype=x.dtype, device=x.device)
            _rms_norm_gated_kernel[(bsz * seq_len, num_heads)](
                x,
                gate,
                self.weight,
                out,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                gate.stride(0),
                gate.stride(1),
                gate.stride(2),
                gate.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                seq_len,
                float(self.eps),
                D=dim,
                USE_SIGMOID=(self.activation == "sigmoid"),
                num_warps=2 if dim <= 64 else 4,
            )
            return out

        x_f = x.float()
        g_f = gate.float()
        gate_val = torch.sigmoid(g_f) if self.activation == "sigmoid" else F.silu(g_f)
        if not self.norm_before_gate:
            x_f = x_f * gate_val
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x_f * rstd) * self.weight.float()
        if self.norm_before_gate:
            y = y * gate_val
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Reference PyTorch GDN Recurrent (exact FP32 golden)
# ---------------------------------------------------------------------------


def gdn_recurrent_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    l2_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Exact PyTorch FP32 reference for Gated Delta Rule with 4:1 GQA and state [B, H_v, D_k, D_v].

    Args:
        q: Post-conv, pre-L2 query ``[B, T, H_k, D_k]``.
        k: Post-conv, pre-L2 key ``[B, T, H_k, D_k]``.
        v: Post-conv value ``[B, T, H_v, D_v]``.
        g: Negative log-decay ``[B, T, H_v]`` (FP32).
        beta: Update gate ``[B, T, H_v]`` in (0, 1) (FP32).
        initial_state: Optional initial FP32 SSM state ``[B, H_v, D_k, D_v]``.
        output_final_state: Whether to return the final FP32 SSM state.
        l2_eps: Epsilon for Q/K L2 normalization.

    Returns:
        ``(out, final_state)`` where ``out`` is ``[B, T, H_v, D_v]`` in ``v.dtype``,
        and ``final_state`` is ``[B, H_v, D_k, D_v]`` FP32 (or ``None``).
    """
    bsz, seq_len, num_k_heads, d_k = q.shape
    num_v_heads, d_v = v.shape[-2], v.shape[-1]
    gqa_ratio = num_v_heads // num_k_heads

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()

    q_f = q_f * torch.rsqrt(q_f.square().sum(dim=-1, keepdim=True) + l2_eps) * (d_k ** -0.5)
    k_f = k_f * torch.rsqrt(k_f.square().sum(dim=-1, keepdim=True) + l2_eps)

    if gqa_ratio > 1:
        q_f = q_f.repeat_interleave(gqa_ratio, dim=-2)
        k_f = k_f.repeat_interleave(gqa_ratio, dim=-2)

    if initial_state is None:
        state = torch.zeros((bsz, num_v_heads, d_k, d_v), dtype=torch.float32, device=v.device)
    else:
        state = initial_state.float().clone()

    out = torch.empty((bsz, seq_len, num_v_heads, d_v), dtype=torch.float32, device=v.device)
    for t in range(seq_len):
        q_t = q_f[:, t]          # [B, H_v, D_k]
        k_t = k_f[:, t]          # [B, H_v, D_k]
        v_t = v_f[:, t]          # [B, H_v, D_v]
        g_t = g_f[:, t]          # [B, H_v]
        beta_t = beta_f[:, t]    # [B, H_v]

        state = state * torch.exp(g_t)[..., None, None]
        v_pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)       # [B, H_v, D_v]
        delta = (v_t - v_pred) * beta_t.unsqueeze(-1)          # [B, H_v, D_v]
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.to(v.dtype), (state if output_final_state else None)


# ---------------------------------------------------------------------------
# Fused Triton Recurrent GDN Kernel (Decode T=1 & short sequences)
# ---------------------------------------------------------------------------


@triton.jit
def _gdn_recurrent_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    InitState_ptr,
    FinalState_ptr,
    Out_ptr,
    stride_qb,
    stride_qt,
    stride_qhk,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_khk,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vhv,
    stride_vd,
    stride_gb,
    stride_gt,
    stride_ghv,
    stride_bb,
    stride_bt,
    stride_bhv,
    stride_sb,
    stride_shv,
    stride_sk,
    stride_sv,
    stride_fsb,
    stride_fshv,
    stride_fsk,
    stride_fsv,
    stride_ob,
    stride_ot,
    stride_ohv,
    stride_od,
    T,
    scale,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Fused FP32 recurrent Gated Delta Rule kernel holding [D_K, D_V] state in registers."""
    b = tl.program_id(0)
    h_v = tl.program_id(1)
    h_k = h_v // GQA_RATIO

    offs_k = tl.arange(0, D_K)
    offs_v = tl.arange(0, D_V)

    if HAS_INIT_STATE:
        s_ptrs = (
            InitState_ptr
            + b * stride_sb
            + h_v * stride_shv
            + offs_k[:, None] * stride_sk
            + offs_v[None, :] * stride_sv
        )
        h = tl.load(s_ptrs).to(tl.float32)
    else:
        h = tl.zeros((D_K, D_V), dtype=tl.float32)

    q_head_base = Q_ptr + b * stride_qb + h_k * stride_qhk + offs_k * stride_qd
    k_head_base = K_ptr + b * stride_kb + h_k * stride_khk + offs_k * stride_kd
    v_head_base = V_ptr + b * stride_vb + h_v * stride_vhv + offs_v * stride_vd
    g_head_base = G_ptr + b * stride_gb + h_v * stride_ghv
    beta_head_base = Beta_ptr + b * stride_bb + h_v * stride_bhv
    out_head_base = Out_ptr + b * stride_ob + h_v * stride_ohv + offs_v * stride_od

    out_dtype = Out_ptr.dtype.element_ty

    for t in range(T):
        q_t = tl.load(q_head_base + t * stride_qt).to(tl.float32)
        k_t = tl.load(k_head_base + t * stride_kt).to(tl.float32)
        v_t = tl.load(v_head_base + t * stride_vt).to(tl.float32)
        g_t = tl.load(g_head_base + t * stride_gt).to(tl.float32)
        beta_t = tl.load(beta_head_base + t * stride_bt).to(tl.float32)

        q_t = q_t * (tl.rsqrt(tl.sum(q_t * q_t, axis=0) + l2_eps) * scale)
        k_t = k_t * tl.rsqrt(tl.sum(k_t * k_t, axis=0) + l2_eps)

        h = h * tl.exp(g_t)
        v_pred = tl.sum(h * k_t[:, None], axis=0)
        delta = (v_t - v_pred) * beta_t
        h = h + k_t[:, None] * delta[None, :]
        o_t = tl.sum(h * q_t[:, None], axis=0)

        tl.store(out_head_base + t * stride_ot, o_t.to(out_dtype))

    if STORE_FINAL_STATE:
        fs_ptrs = (
            FinalState_ptr
            + b * stride_fsb
            + h_v * stride_fshv
            + offs_k[:, None] * stride_fsk
            + offs_v[None, :] * stride_fsv
        )
        tl.store(fs_ptrs, h)


def gdn_recurrent_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    inplace_state: bool = False,
    l2_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused Triton recurrent Gated Delta Rule (1 CTA per (batch, v_head)).

    State layout: ``[B, H_v, D_k, D_v]`` FP32.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda):
        out, final_state = gdn_recurrent_ref(
            q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state, l2_eps=l2_eps
        )
        if inplace_state and initial_state is not None and final_state is not None:
            initial_state.copy_(final_state)
            return out, initial_state
        return out, final_state

    bsz, seq_len, num_k_heads, d_k = q.shape
    num_v_heads, d_v = v.shape[-2], v.shape[-1]
    gqa_ratio = num_v_heads // num_k_heads

    out = torch.empty((bsz, seq_len, num_v_heads, d_v), dtype=v.dtype, device=v.device)
    has_init = initial_state is not None

    if output_final_state:
        if inplace_state and initial_state is not None:
            final_state = initial_state
        else:
            final_state = torch.empty(
                (bsz, num_v_heads, d_k, d_v), dtype=torch.float32, device=v.device
            )
    else:
        final_state = None

    init_ptr = initial_state if has_init else out
    fs_ptr = final_state if output_final_state else out

    _gdn_recurrent_kernel[(bsz, num_v_heads)](
        q,
        k,
        v,
        g,
        beta,
        init_ptr,
        fs_ptr,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        initial_state.stride(0) if has_init else 0,
        initial_state.stride(1) if has_init else 0,
        initial_state.stride(2) if has_init else 0,
        initial_state.stride(3) if has_init else 0,
        final_state.stride(0) if output_final_state else 0,
        final_state.stride(1) if output_final_state else 0,
        final_state.stride(2) if output_final_state else 0,
        final_state.stride(3) if output_final_state else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        seq_len,
        float(d_k ** -0.5),
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=d_k,
        D_V=d_v,
        HAS_INIT_STATE=has_init,
        STORE_FINAL_STATE=output_final_state,
        num_warps=4,
    )
    return out, final_state


# ---------------------------------------------------------------------------
# Fast 3-Kernel Triton Chunked Parallel Prefill (CHUNK_SIZE = 64, D_K = D_V = 64)
# ---------------------------------------------------------------------------


@triton.jit
def _invert_unit_tril_16(A):
    """Exact IEEE FP32 forward-substitution inverse of (I + A) for strictly lower-triangular [16, 16] A."""
    o = tl.arange(0, 16)
    M = tl.where(o[:, None] == o[None, :], 1.0, 0.0).to(tl.float32)
    for i in range(1, 16):
        a_row = tl.sum(tl.where(o[:, None] == i, A, 0.0), axis=0)
        m_row = -tl.sum(a_row[:, None] * M, axis=0)
        m_row = tl.where(o == i, 1.0, m_row)
        M = tl.where(o[:, None] == i, m_row[None, :], M)
    return M


@triton.jit
def _gdn_chunk_prep_wy_kernel(
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    GCum_ptr,
    W_ptr,
    U_ptr,
    stride_kb,
    stride_kt,
    stride_khk,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vhv,
    stride_vd,
    stride_gb,
    stride_gt,
    stride_ghv,
    stride_bb,
    stride_bt,
    stride_bhv,
    stride_gcb,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_wt,
    stride_wd,
    stride_ub,
    stride_uh,
    stride_uc,
    stride_ut,
    stride_ud,
    T,
    H_V,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """Stage 1 (parallel over B*H_v*N_c): compute chunk cumsum G and exact block-16 WY factors (W, U)."""
    bh = tl.program_id(0)
    c = tl.program_id(1)
    b = bh // H_V
    h_v = bh % H_V
    h_k = h_v // GQA_RATIO

    t0 = c * 64
    o16 = tl.arange(0, 16)
    ok = tl.arange(0, D_K)
    ov = tl.arange(0, D_V)
    tril_mask = o16[:, None] > o16[None, :]

    k_base = K_ptr + b * stride_kb + h_k * stride_khk
    v_base = V_ptr + b * stride_vb + h_v * stride_vhv
    g_base = G_ptr + b * stride_gb + h_v * stride_ghv
    b_base = Beta_ptr + b * stride_bb + h_v * stride_bhv

    # Load 4 sub-blocks of 16 tokens each
    t_0 = t0 + o16
    t_1 = t0 + 16 + o16
    t_2 = t0 + 32 + o16
    t_3 = t0 + 48 + o16
    m_0 = t_0 < T
    m_1 = t_1 < T
    m_2 = t_2 < T
    m_3 = t_3 < T

    g_0 = tl.load(g_base + t_0 * stride_gt, mask=m_0, other=0.0).to(tl.float32)
    g_1 = tl.load(g_base + t_1 * stride_gt, mask=m_1, other=0.0).to(tl.float32)
    g_2 = tl.load(g_base + t_2 * stride_gt, mask=m_2, other=0.0).to(tl.float32)
    g_3 = tl.load(g_base + t_3 * stride_gt, mask=m_3, other=0.0).to(tl.float32)

    G_0 = tl.cumsum(g_0, axis=0)
    s0 = tl.sum(g_0, axis=0)
    G_1 = tl.cumsum(g_1, axis=0) + s0
    s1 = s0 + tl.sum(g_1, axis=0)
    G_2 = tl.cumsum(g_2, axis=0) + s1
    s2 = s1 + tl.sum(g_2, axis=0)
    G_3 = tl.cumsum(g_3, axis=0) + s2

    gc_base = GCum_ptr + b * stride_gcb + h_v * stride_gch + c * stride_gcc
    tl.store(gc_base + o16 * stride_gct, G_0)
    tl.store(gc_base + (16 + o16) * stride_gct, G_1)
    tl.store(gc_base + (32 + o16) * stride_gct, G_2)
    tl.store(gc_base + (48 + o16) * stride_gct, G_3)

    beta_0 = tl.load(b_base + t_0 * stride_bt, mask=m_0, other=0.0).to(tl.float32)
    beta_1 = tl.load(b_base + t_1 * stride_bt, mask=m_1, other=0.0).to(tl.float32)
    beta_2 = tl.load(b_base + t_2 * stride_bt, mask=m_2, other=0.0).to(tl.float32)
    beta_3 = tl.load(b_base + t_3 * stride_bt, mask=m_3, other=0.0).to(tl.float32)

    k_0 = tl.load(k_base + t_0[:, None] * stride_kt + ok[None, :] * stride_kd, mask=m_0[:, None], other=0.0).to(tl.float32)
    k_1 = tl.load(k_base + t_1[:, None] * stride_kt + ok[None, :] * stride_kd, mask=m_1[:, None], other=0.0).to(tl.float32)
    k_2 = tl.load(k_base + t_2[:, None] * stride_kt + ok[None, :] * stride_kd, mask=m_2[:, None], other=0.0).to(tl.float32)
    k_3 = tl.load(k_base + t_3[:, None] * stride_kt + ok[None, :] * stride_kd, mask=m_3[:, None], other=0.0).to(tl.float32)

    k_0 = k_0 * tl.rsqrt(tl.sum(k_0 * k_0, axis=1) + l2_eps)[:, None]
    k_1 = k_1 * tl.rsqrt(tl.sum(k_1 * k_1, axis=1) + l2_eps)[:, None]
    k_2 = k_2 * tl.rsqrt(tl.sum(k_2 * k_2, axis=1) + l2_eps)[:, None]
    k_3 = k_3 * tl.rsqrt(tl.sum(k_3 * k_3, axis=1) + l2_eps)[:, None]

    v_0 = tl.load(v_base + t_0[:, None] * stride_vt + ov[None, :] * stride_vd, mask=m_0[:, None], other=0.0).to(tl.float32)
    v_1 = tl.load(v_base + t_1[:, None] * stride_vt + ov[None, :] * stride_vd, mask=m_1[:, None], other=0.0).to(tl.float32)
    v_2 = tl.load(v_base + t_2[:, None] * stride_vt + ov[None, :] * stride_vd, mask=m_2[:, None], other=0.0).to(tl.float32)
    v_3 = tl.load(v_base + t_3[:, None] * stride_vt + ov[None, :] * stride_vd, mask=m_3[:, None], other=0.0).to(tl.float32)

    # Scaled RHS for W and U
    ks_0 = k_0 * (beta_0 * tl.exp(G_0))[:, None]
    ks_1 = k_1 * (beta_1 * tl.exp(G_1))[:, None]
    ks_2 = k_2 * (beta_2 * tl.exp(G_2))[:, None]
    ks_3 = k_3 * (beta_3 * tl.exp(G_3))[:, None]

    vs_0 = v_0 * beta_0[:, None]
    vs_1 = v_1 * beta_1[:, None]
    vs_2 = v_2 * beta_2[:, None]
    vs_3 = v_3 * beta_3[:, None]

    # Block 0: (I + A_00) W_0 = ks_0
    diff_00 = tl.where(tril_mask, G_0[:, None] - G_0[None, :], -1e9)
    A_00 = tl.where(tril_mask, beta_0[:, None] * tl.exp(diff_00) * tl.dot(k_0, tl.trans(k_0), input_precision="tf32"), 0.0)
    M_00 = _invert_unit_tril_16(A_00)
    W_0 = tl.dot(M_00, ks_0, input_precision="tf32")
    U_0 = tl.dot(M_00, vs_0, input_precision="tf32")

    # Block 1: (I + A_11) W_1 = ks_1 - A_10 W_0
    A_10 = beta_1[:, None] * tl.exp(G_1[:, None] - G_0[None, :]) * tl.dot(k_1, tl.trans(k_0), input_precision="tf32")
    diff_11 = tl.where(tril_mask, G_1[:, None] - G_1[None, :], -1e9)
    A_11 = tl.where(tril_mask, beta_1[:, None] * tl.exp(diff_11) * tl.dot(k_1, tl.trans(k_1), input_precision="tf32"), 0.0)
    M_11 = _invert_unit_tril_16(A_11)
    W_1 = tl.dot(M_11, ks_1 - tl.dot(A_10, W_0, input_precision="tf32"), input_precision="tf32")
    U_1 = tl.dot(M_11, vs_1 - tl.dot(A_10, U_0, input_precision="tf32"), input_precision="tf32")

    # Block 2: (I + A_22) W_2 = ks_2 - A_20 W_0 - A_21 W_1
    A_20 = beta_2[:, None] * tl.exp(G_2[:, None] - G_0[None, :]) * tl.dot(k_2, tl.trans(k_0), input_precision="tf32")
    A_21 = beta_2[:, None] * tl.exp(G_2[:, None] - G_1[None, :]) * tl.dot(k_2, tl.trans(k_1), input_precision="tf32")
    diff_22 = tl.where(tril_mask, G_2[:, None] - G_2[None, :], -1e9)
    A_22 = tl.where(tril_mask, beta_2[:, None] * tl.exp(diff_22) * tl.dot(k_2, tl.trans(k_2), input_precision="tf32"), 0.0)
    M_22 = _invert_unit_tril_16(A_22)
    rhs_w2 = ks_2 - tl.dot(A_20, W_0, input_precision="tf32") - tl.dot(A_21, W_1, input_precision="tf32")
    rhs_u2 = vs_2 - tl.dot(A_20, U_0, input_precision="tf32") - tl.dot(A_21, U_1, input_precision="tf32")
    W_2 = tl.dot(M_22, rhs_w2, input_precision="tf32")
    U_2 = tl.dot(M_22, rhs_u2, input_precision="tf32")

    # Block 3: (I + A_33) W_3 = ks_3 - A_30 W_0 - A_31 W_1 - A_32 W_2
    A_30 = beta_3[:, None] * tl.exp(G_3[:, None] - G_0[None, :]) * tl.dot(k_3, tl.trans(k_0), input_precision="tf32")
    A_31 = beta_3[:, None] * tl.exp(G_3[:, None] - G_1[None, :]) * tl.dot(k_3, tl.trans(k_1), input_precision="tf32")
    A_32 = beta_3[:, None] * tl.exp(G_3[:, None] - G_2[None, :]) * tl.dot(k_3, tl.trans(k_2), input_precision="tf32")
    diff_33 = tl.where(tril_mask, G_3[:, None] - G_3[None, :], -1e9)
    A_33 = tl.where(tril_mask, beta_3[:, None] * tl.exp(diff_33) * tl.dot(k_3, tl.trans(k_3), input_precision="tf32"), 0.0)
    M_33 = _invert_unit_tril_16(A_33)
    rhs_w3 = (
        ks_3
        - tl.dot(A_30, W_0, input_precision="tf32")
        - tl.dot(A_31, W_1, input_precision="tf32")
        - tl.dot(A_32, W_2, input_precision="tf32")
    )
    rhs_u3 = (
        vs_3
        - tl.dot(A_30, U_0, input_precision="tf32")
        - tl.dot(A_31, U_1, input_precision="tf32")
        - tl.dot(A_32, U_2, input_precision="tf32")
    )
    W_3 = tl.dot(M_33, rhs_w3, input_precision="tf32")
    U_3 = tl.dot(M_33, rhs_u3, input_precision="tf32")

    w_base = W_ptr + b * stride_wb + h_v * stride_wh + c * stride_wc
    u_base = U_ptr + b * stride_ub + h_v * stride_uh + c * stride_uc
    w_dtype = W_ptr.dtype.element_ty
    u_dtype = U_ptr.dtype.element_ty

    tl.store(w_base + o16[:, None] * stride_wt + ok[None, :] * stride_wd, W_0.to(w_dtype))
    tl.store(w_base + (16 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_1.to(w_dtype))
    tl.store(w_base + (32 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_2.to(w_dtype))
    tl.store(w_base + (48 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_3.to(w_dtype))

    tl.store(u_base + o16[:, None] * stride_ut + ov[None, :] * stride_ud, U_0.to(u_dtype))
    tl.store(u_base + (16 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_1.to(u_dtype))
    tl.store(u_base + (32 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_2.to(u_dtype))
    tl.store(u_base + (48 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_3.to(u_dtype))


@triton.jit
def _gdn_chunk_state_recurrence_kernel(
    K_ptr,
    GCum_ptr,
    W_ptr,
    U_ptr,
    InitState_ptr,
    HAll_ptr,
    VNewAll_ptr,
    FinalState_ptr,
    stride_kb,
    stride_kt,
    stride_khk,
    stride_kd,
    stride_gcb,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_wb,
    stride_wh,
    stride_wc,
    stride_wt,
    stride_wd,
    stride_ub,
    stride_uh,
    stride_uc,
    stride_ut,
    stride_ud,
    stride_sb,
    stride_shv,
    stride_sk,
    stride_sv,
    stride_hab,
    stride_hah,
    stride_hac,
    stride_hak,
    stride_hav,
    stride_vnb,
    stride_vnh,
    stride_vnc,
    stride_vnt,
    stride_vnd,
    stride_fsb,
    stride_fshv,
    stride_fsk,
    stride_fsv,
    T,
    num_chunks,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Stage 2 (grid (B, H_v)): inter-chunk FP32 state recurrence keeping [64, 64] state in registers."""
    b = tl.program_id(0)
    h_v = tl.program_id(1)
    h_k = h_v // GQA_RATIO

    oc = tl.arange(0, 64)
    ok = tl.arange(0, D_K)
    ov = tl.arange(0, D_V)

    if HAS_INIT_STATE:
        h = tl.load(
            InitState_ptr
            + b * stride_sb
            + h_v * stride_shv
            + ok[:, None] * stride_sk
            + ov[None, :] * stride_sv
        ).to(tl.float32)
    else:
        h = tl.zeros((D_K, D_V), dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + h_k * stride_khk
    gc_head = GCum_ptr + b * stride_gcb + h_v * stride_gch
    w_head = W_ptr + b * stride_wb + h_v * stride_wh
    u_head = U_ptr + b * stride_ub + h_v * stride_uh
    ha_head = HAll_ptr + b * stride_hab + h_v * stride_hah
    vn_head = VNewAll_ptr + b * stride_vnb + h_v * stride_vnh

    ha_dtype = HAll_ptr.dtype.element_ty
    vn_dtype = VNewAll_ptr.dtype.element_ty

    for c in range(num_chunks):
        # Save chunk-entry state h_c
        tl.store(
            ha_head + c * stride_hac + ok[:, None] * stride_hak + ov[None, :] * stride_hav,
            h.to(ha_dtype),
        )

        # v_new = U_c - W_c @ h
        b_w = tl.load(w_head + c * stride_wc + oc[:, None] * stride_wt + ok[None, :] * stride_wd)
        b_u = tl.load(u_head + c * stride_uc + oc[:, None] * stride_ut + ov[None, :] * stride_ud).to(tl.float32)
        b_v_new = b_u - tl.dot(b_w.to(tl.bfloat16), h.to(tl.bfloat16), out_dtype=tl.float32)
        tl.store(
            vn_head + c * stride_vnc + oc[:, None] * stride_vnt + ov[None, :] * stride_vnd,
            b_v_new.to(vn_dtype),
        )

        # Load cumulative decay & normalized K for this chunk
        b_gc = tl.load(gc_head + c * stride_gcc + oc * stride_gct).to(tl.float32)
        g_last = tl.sum(tl.where(oc == 63, b_gc, 0.0), axis=0)

        t_offs = c * 64 + oc
        m_t = t_offs < T
        b_k = tl.load(
            k_base + t_offs[:, None] * stride_kt + ok[None, :] * stride_kd,
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1) + l2_eps)[:, None]
        b_k_decay = b_k * tl.exp(g_last - b_gc)[:, None]

        # h_next = exp(g_last) * h + K_decay^T @ v_new
        h = h * tl.exp(g_last) + tl.dot(
            tl.trans(b_k_decay.to(tl.bfloat16)),
            b_v_new.to(tl.bfloat16),
            out_dtype=tl.float32,
        )

    if STORE_FINAL_STATE:
        tl.store(
            FinalState_ptr
            + b * stride_fsb
            + h_v * stride_fshv
            + ok[:, None] * stride_fsk
            + ov[None, :] * stride_fsv,
            h,
        )


@triton.jit
def _gdn_chunk_output_kernel(
    Q_ptr,
    K_ptr,
    GCum_ptr,
    HAll_ptr,
    VNewAll_ptr,
    Out_ptr,
    stride_qb,
    stride_qt,
    stride_qhk,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_khk,
    stride_kd,
    stride_gcb,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_hab,
    stride_hah,
    stride_hac,
    stride_hak,
    stride_hav,
    stride_vnb,
    stride_vnh,
    stride_vnc,
    stride_vnt,
    stride_vnd,
    stride_ob,
    stride_ot,
    stride_ohv,
    stride_od,
    T,
    H_V,
    scale,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """Stage 3 (parallel over B*H_v*N_c): combine inter-chunk and causal intra-chunk outputs."""
    bh = tl.program_id(0)
    c = tl.program_id(1)
    b = bh // H_V
    h_v = bh % H_V
    h_k = h_v // GQA_RATIO

    oc = tl.arange(0, 64)
    ok = tl.arange(0, D_K)
    ov = tl.arange(0, D_V)

    t_offs = c * 64 + oc
    m_t = t_offs < T

    b_q = tl.load(
        Q_ptr + b * stride_qb + h_k * stride_qhk + t_offs[:, None] * stride_qt + ok[None, :] * stride_qd,
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)
    b_k = tl.load(
        K_ptr + b * stride_kb + h_k * stride_khk + t_offs[:, None] * stride_kt + ok[None, :] * stride_kd,
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)

    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, axis=1) + l2_eps)[:, None]
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1) + l2_eps)[:, None]

    b_gc = tl.load(
        GCum_ptr + b * stride_gcb + h_v * stride_gch + c * stride_gcc + oc * stride_gct
    ).to(tl.float32)
    b_h = tl.load(
        HAll_ptr
        + b * stride_hab
        + h_v * stride_hah
        + c * stride_hac
        + ok[:, None] * stride_hak
        + ov[None, :] * stride_hav
    )
    b_v_new = tl.load(
        VNewAll_ptr
        + b * stride_vnb
        + h_v * stride_vnh
        + c * stride_vnc
        + oc[:, None] * stride_vnt
        + ov[None, :] * stride_vnd
    )

    b_q_decay = b_q * tl.exp(b_gc)[:, None]
    o_inter = tl.dot(b_q_decay.to(tl.bfloat16), b_h.to(tl.bfloat16), out_dtype=tl.float32)

    qk = tl.dot(b_q.to(tl.bfloat16), tl.trans(b_k.to(tl.bfloat16)), out_dtype=tl.float32)
    causal = oc[:, None] >= oc[None, :]
    g_diff = tl.where(causal, b_gc[:, None] - b_gc[None, :], -1e9)
    attn = tl.where(causal, qk * tl.exp(g_diff), 0.0)
    o_intra = tl.dot(attn.to(tl.bfloat16), b_v_new.to(tl.bfloat16), out_dtype=tl.float32)

    o = (o_inter + o_intra) * scale
    tl.store(
        Out_ptr + b * stride_ob + h_v * stride_ohv + t_offs[:, None] * stride_ot + ov[None, :] * stride_od,
        o.to(Out_ptr.dtype.element_ty),
        mask=m_t[:, None],
    )


def gdn_chunk_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    inplace_state: bool = False,
    l2_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fast 3-stage Triton chunked parallel prefill for Gated Delta Rule (chunk_size=64).

    Requires ``D_k == 64`` and ``D_v == 64`` on CUDA; falls back to ``gdn_recurrent_triton`` otherwise.
    State layout: ``[B, H_v, D_k, D_v]`` FP32.
    """
    bsz, seq_len, num_k_heads, d_k = q.shape
    num_v_heads, d_v = v.shape[-2], v.shape[-1]
    if (
        not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda)
        or d_k != 64
        or d_v != 64
        or num_v_heads % num_k_heads != 0
    ):
        return gdn_recurrent_triton(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            inplace_state=inplace_state,
            l2_eps=l2_eps,
        )

    device = v.device
    gqa_ratio = num_v_heads // num_k_heads
    num_chunks = (seq_len + 63) // 64

    g_cum = torch.empty((bsz, num_v_heads, num_chunks, 64), dtype=torch.float32, device=device)
    w_buf = torch.empty((bsz, num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    u_buf = torch.empty((bsz, num_v_heads, num_chunks, 64, 64), dtype=torch.float32, device=device)
    h_all = torch.empty((bsz, num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    v_new_all = torch.empty((bsz, num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty((bsz, seq_len, num_v_heads, 64), dtype=v.dtype, device=device)

    has_init = initial_state is not None
    if output_final_state:
        if inplace_state and initial_state is not None:
            final_state = initial_state
        else:
            final_state = torch.empty((bsz, num_v_heads, 64, 64), dtype=torch.float32, device=device)
    else:
        final_state = None

    # Stage 1: Parallel chunk cumsum + exact block-16 WY prep
    _gdn_chunk_prep_wy_kernel[(bsz * num_v_heads, num_chunks)](
        k,
        v,
        g,
        beta,
        g_cum,
        w_buf,
        u_buf,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        g_cum.stride(3),
        w_buf.stride(0),
        w_buf.stride(1),
        w_buf.stride(2),
        w_buf.stride(3),
        w_buf.stride(4),
        u_buf.stride(0),
        u_buf.stride(1),
        u_buf.stride(2),
        u_buf.stride(3),
        u_buf.stride(4),
        seq_len,
        num_v_heads,
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        num_warps=4,
    )

    # Stage 2: Inter-chunk FP32 state recurrence
    init_ptr = initial_state if has_init else out
    fs_ptr = final_state if output_final_state else out
    _gdn_chunk_state_recurrence_kernel[(bsz, num_v_heads)](
        k,
        g_cum,
        w_buf,
        u_buf,
        init_ptr,
        h_all,
        v_new_all,
        fs_ptr,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        g_cum.stride(3),
        w_buf.stride(0),
        w_buf.stride(1),
        w_buf.stride(2),
        w_buf.stride(3),
        w_buf.stride(4),
        u_buf.stride(0),
        u_buf.stride(1),
        u_buf.stride(2),
        u_buf.stride(3),
        u_buf.stride(4),
        initial_state.stride(0) if has_init else 0,
        initial_state.stride(1) if has_init else 0,
        initial_state.stride(2) if has_init else 0,
        initial_state.stride(3) if has_init else 0,
        h_all.stride(0),
        h_all.stride(1),
        h_all.stride(2),
        h_all.stride(3),
        h_all.stride(4),
        v_new_all.stride(0),
        v_new_all.stride(1),
        v_new_all.stride(2),
        v_new_all.stride(3),
        v_new_all.stride(4),
        final_state.stride(0) if output_final_state else 0,
        final_state.stride(1) if output_final_state else 0,
        final_state.stride(2) if output_final_state else 0,
        final_state.stride(3) if output_final_state else 0,
        seq_len,
        num_chunks,
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        HAS_INIT_STATE=has_init,
        STORE_FINAL_STATE=output_final_state,
        num_warps=4,
    )

    # Stage 3: Parallel intra + inter chunk output
    _gdn_chunk_output_kernel[(bsz * num_v_heads, num_chunks)](
        q,
        k,
        g_cum,
        h_all,
        v_new_all,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        g_cum.stride(3),
        h_all.stride(0),
        h_all.stride(1),
        h_all.stride(2),
        h_all.stride(3),
        h_all.stride(4),
        v_new_all.stride(0),
        v_new_all.stride(1),
        v_new_all.stride(2),
        v_new_all.stride(3),
        v_new_all.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        seq_len,
        num_v_heads,
        float(64 ** -0.5),
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        num_warps=4,
    )

    return out, final_state


# ---------------------------------------------------------------------------
# Qwen4_5GatedDeltaNet Module
# ---------------------------------------------------------------------------


class Qwen4_5GatedDeltaNet(nn.Module):
    """Qwen 4.5-Flash Gated DeltaNet Linear Attention (normal, YoCo source, or YoCo consumer).

    Checkpoint parameter names match ``model.layers.{i}.linear_attn.*``:
      - Normal / Source (``is_consumer=False``):
        ``in_proj_qkvz [20480, 5120]``, ``in_proj_ba [256, 5120]``, ``conv1d [12288, 1, 4]``,
        ``A_log [128]``, ``dt_bias [128]``, ``norm [64]``, ``out_proj [5120, 8192]``.
      - YoCo Consumer (``is_consumer=True``):
        ONLY ``in_proj_qkvz [10240, 5120]``, ``conv1d [2048, 1, 4]``, ``norm [64]``,
        ``out_proj [5120, 8192]``.
    """

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        is_consumer: bool = False,
        publishes_shared: bool = False,
        source_layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_consumer = is_consumer
        self.publishes_shared = publishes_shared
        self.source_layer_idx = source_layer_idx
        self.enable_yoco_decode = config.enable_yoco_decode

        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.num_v_heads = config.linear_num_value_heads
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gqa_ratio = self.num_v_heads // self.num_k_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.gdn_conv_residual = config.gdn_conv_residual
        self.gdn_no_q_conv = config.gdn_no_q_conv
        self.rms_norm_eps = config.rms_norm_eps

        if self.is_consumer:
            # 32 groups x [Q(64), Z(256)] = 10240
            self.qkvz_out_dim = self.key_dim + self.value_dim
            self.conv_dim = self.key_dim  # 2048
            self.in_proj_qkvz = nn.Linear(self.hidden_size, self.qkvz_out_dim, bias=False)
            self.in_proj_ba: Optional[nn.Linear] = None
            self.conv1d = nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                bias=False,
            )
            self.A_log: Optional[nn.Parameter] = None
            self.dt_bias: Optional[nn.Parameter] = None
        else:
            # 32 groups x [Q(64), K(64), V(256), Z(256)] = 20480
            self.qkvz_out_dim = self.key_dim * 2 + self.value_dim * 2
            self.conv_dim = self.key_dim * 2 + self.value_dim  # 12288
            self.in_proj_qkvz = nn.Linear(self.hidden_size, self.qkvz_out_dim, bias=False)
            self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)
            self.conv1d = nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                bias=False,
            )
            self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
            self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.rms_norm_eps,
            activation=config.output_gate_type,
            norm_before_gate=config.norm_before_gate,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        sharing_plan: Optional[YoCoSharingPlan] = None,
    ) -> "Qwen4_5GatedDeltaNet":
        plan = sharing_plan if sharing_plan is not None else YoCoSharingPlan.from_config(config)
        return cls(
            config=config,
            layer_idx=layer_idx,
            is_consumer=plan.is_gdn_consumer(layer_idx),
            publishes_shared=plan.is_gdn_source(layer_idx),
            source_layer_idx=plan.source_for(layer_idx),
        )

    @property
    def conv1d_weight(self) -> torch.Tensor:
        return self.conv1d.weight

    @property
    def norm_weight(self) -> torch.Tensor:
        return self.norm.weight

    def create_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        allocate_consumer_ssm: Optional[bool] = None,
    ) -> GDNLayerState:
        """Allocate per-layer conv + SSM state.

        For YoCo consumers with ``enable_yoco_decode=True`` (default ``allocate_consumer_ssm=False``),
        ``ssm_state`` is ``None`` to save 14x SSM state memory.
        """
        conv_state = torch.zeros(
            (batch_size, self.conv_dim, self.conv_kernel_size - 1),
            device=device,
            dtype=dtype,
        )
        if allocate_consumer_ssm is None:
            need_ssm = not (self.is_consumer and self.enable_yoco_decode)
        else:
            need_ssm = (not self.is_consumer) or allocate_consumer_ssm

        ssm_state = (
            torch.zeros(
                (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                device=device,
                dtype=torch.float32,
            )
            if need_ssm
            else None
        )
        return GDNLayerState(conv_state=conv_state, ssm_state=ssm_state)

    def _unpack_qkvz_normal(
        self, projected: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack interleaved [B, T, 32, 640] -> (mixed_qkv [B, T, 12288], z [B, T, 128, 64])."""
        bsz, seq_len, _ = projected.shape
        v_group = self.gqa_ratio * self.head_v_dim
        grouped = projected.unflatten(
            -1, (self.num_k_heads, self.head_k_dim * 2 + v_group * 2)
        )
        q_g, k_g, v_g, z_g = grouped.split(
            [self.head_k_dim, self.head_k_dim, v_group, v_group], dim=-1
        )
        mixed_qkv = torch.cat(
            [
                q_g.reshape(bsz, seq_len, self.key_dim),
                k_g.reshape(bsz, seq_len, self.key_dim),
                v_g.reshape(bsz, seq_len, self.value_dim),
            ],
            dim=-1,
        )
        z = z_g.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        return mixed_qkv, z

    def _unpack_qz_consumer(
        self, projected: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack interleaved [B, T, 32, 320] -> (q_flat [B, T, 2048], z [B, T, 128, 64])."""
        bsz, seq_len, _ = projected.shape
        v_group = self.gqa_ratio * self.head_v_dim
        grouped = projected.unflatten(-1, (self.num_k_heads, self.head_k_dim + v_group))
        q_g, z_g = grouped.split([self.head_k_dim, v_group], dim=-1)
        q_flat = q_g.reshape(bsz, seq_len, self.key_dim).contiguous()
        z = z_g.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        return q_flat, z

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[GDNLayerState] = None,
        shared: Optional[SharedGDNInputs] = None,
        *,
        use_chunk_threshold: int = 16,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[SharedGDNInputs]]:
        """Run GDN forward pass for normal, YoCo source, or YoCo consumer layer.

        Args:
            hidden_states: Input tensor ``[B, T, hidden_size]``.
            state: Optional per-layer ``GDNLayerState`` (updated in-place).
            shared: Required when ``is_consumer=True``; provides ``SharedGDNInputs``
                published by the YoCo source layer (layer 20).
            use_chunk_threshold: Sequence lengths ``> use_chunk_threshold`` dispatch to
                ``gdn_chunk_prefill_triton``; shorter sequences dispatch to ``gdn_recurrent_triton``.
            enable_yoco_decode: Optional per-call override for YoCo decode fast path.

        Returns:
            ``(output, published_shared)`` where ``output`` has shape ``[B, T, hidden_size]``
            and ``published_shared`` is ``SharedGDNInputs`` if ``publishes_shared=True`` else ``None``.
        """
        bsz, seq_len, _ = hidden_states.shape
        conv_state_wrap = ShortConvState(state.conv_state) if state is not None else None
        yoco_decode_on = (
            self.enable_yoco_decode if enable_yoco_decode is None else enable_yoco_decode
        )

        if self.is_consumer:
            if shared is None:
                raise ValueError(
                    f"Consumer GDN layer {self.layer_idx} requires SharedGDNInputs from source layer"
                )
            qz = self.in_proj_qkvz(hidden_states)
            q_flat, z = self._unpack_qz_consumer(qz)
            if not self.gdn_no_q_conv:
                q_flat = causal_dwconv1d(
                    q_flat,
                    self.conv1d.weight,
                    state=conv_state_wrap,
                    residual=self.gdn_conv_residual,
                    use_silu=True,
                )
            q = q_flat.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)

            # YoCo Decode fast path: readout directly from source layer's post-update FP32 SSM state
            if seq_len == 1 and yoco_decode_on and shared.ssm_state is not None:
                core_out = yoco_gdn_decode_readout(
                    q,
                    shared.ssm_state,
                    head_k_dim=self.head_k_dim,
                )
            else:
                init_ssm = state.ssm_state if state is not None else None
                need_final = init_ssm is not None
                if seq_len > use_chunk_threshold:
                    core_out, _ = gdn_chunk_prefill_triton(
                        q,
                        shared.key,
                        shared.value,
                        shared.g,
                        shared.beta,
                        initial_state=init_ssm,
                        output_final_state=need_final,
                        inplace_state=need_final,
                    )
                else:
                    core_out, _ = gdn_recurrent_triton(
                        q,
                        shared.key,
                        shared.value,
                        shared.g,
                        shared.beta,
                        initial_state=init_ssm,
                        output_final_state=need_final,
                        inplace_state=need_final,
                    )

            out = self.norm(core_out, z)
            out = self.out_proj(out.reshape(bsz, seq_len, self.value_dim))
            return out, None

        # Normal / YoCo Source layer
        assert self.in_proj_ba is not None and self.A_log is not None and self.dt_bias is not None
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)

        mixed_qkv, z = self._unpack_qkvz_normal(qkvz)
        mixed_qkv = causal_dwconv1d(
            mixed_qkv,
            self.conv1d.weight,
            state=conv_state_wrap,
            residual=self.gdn_conv_residual,
            use_silu=True,
        )
        q_flat, k_flat, v_flat = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q_flat.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        k = k_flat.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        v = v_flat.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        g, beta = compute_gdn_gating(
            ba,
            self.A_log,
            self.dt_bias,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )

        if state is not None:
            if state.ssm_state is None:
                state.ssm_state = torch.zeros(
                    (bsz, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            init_ssm = state.ssm_state
            inplace_ssm = True
            need_final = True
        else:
            init_ssm = None
            inplace_ssm = False
            need_final = self.publishes_shared

        if seq_len > use_chunk_threshold:
            core_out, final_ssm = gdn_chunk_prefill_triton(
                q,
                k,
                v,
                g,
                beta,
                initial_state=init_ssm,
                output_final_state=need_final,
                inplace_state=inplace_ssm,
            )
        else:
            core_out, final_ssm = gdn_recurrent_triton(
                q,
                k,
                v,
                g,
                beta,
                initial_state=init_ssm,
                output_final_state=need_final,
                inplace_state=inplace_ssm,
            )

        out = self.norm(core_out, z)
        out = self.out_proj(out.reshape(bsz, seq_len, self.value_dim))

        published: Optional[SharedGDNInputs] = None
        if self.publishes_shared:
            published = SharedGDNInputs(
                key=k,
                value=v,
                g=g,
                beta=beta,
                ssm_state=final_ssm,
            )
        return out, published


__all__ = [
    "Qwen4_5GatedDeltaNet",
    "RMSNormGated",
    "SSM_STATE_LAYOUT",
    "compute_gdn_gating",
    "gdn_chunk_prefill_triton",
    "gdn_recurrent_ref",
    "gdn_recurrent_triton",
]

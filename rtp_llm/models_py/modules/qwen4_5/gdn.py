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

from typing import Callable, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.attention import causal_dwconv1d
from rtp_llm.models_py.modules.qwen4_5.short_conv import _triton_int_spec, triton_fast_launch

_GDN_TRITON_CACHE: dict = {}
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
    *,
    out_g: Optional[torch.Tensor] = None,
    out_beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack interleaved ``in_proj_ba`` output and compute ``(g, beta)`` in FP32.

    Args:
        ba_interleaved: Output of ``in_proj_ba``, shape ``[B, T, 2 * H_v]``, organized as
            ``H_k`` groups of ``[b (group_v), a (group_v)]``.
        a_log: Parameter ``A_log`` of shape ``[H_v]``.
        dt_bias: Parameter ``dt_bias`` of shape ``[H_v]``.
        out_g: Optional preallocated output buffer for ``g`` ``[B, T, H_v]`` fp32.
        out_beta: Optional preallocated output buffer for ``beta`` ``[B, T, H_v]`` fp32.

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
        g = (
            out_g
            if (out_g is not None and tuple(out_g.shape) == (bsz, seq_len, num_v_heads) and out_g.dtype == torch.float32)
            else torch.empty((bsz, seq_len, num_v_heads), dtype=torch.float32, device=ba_interleaved.device)
        )
        beta = (
            out_beta
            if (out_beta is not None and tuple(out_beta.shape) == (bsz, seq_len, num_v_heads) and out_beta.dtype == torch.float32)
            else torch.empty((bsz, seq_len, num_v_heads), dtype=torch.float32, device=ba_interleaved.device)
        )
        if bsz * seq_len > 0:
            rt_args = (
                ba_interleaved, a_log, dt_bias, g, beta,
                ba_interleaved.stride(0), ba_interleaved.stride(1), ba_interleaved.stride(2),
                g.stride(0), g.stride(1), g.stride(2),
                beta.stride(0), beta.stride(1), beta.stride(2),
                seq_len,
            )
            ckey = (
                ba_interleaved.device.index, 0, ba_interleaved.dtype, a_log.dtype, dt_bias.dtype,
                tuple(_triton_int_spec(v) for v in rt_args[5:]),
                num_v_heads, group_v,
            )
            ptr_or = ba_interleaved.data_ptr() | a_log.data_ptr() | dt_bias.data_ptr() | g.data_ptr() | beta.data_ptr()
            triton_fast_launch(
                _GDN_TRITON_CACHE, ckey, _gdn_gating_ba_interleaved_kernel,
                bsz * seq_len, 1, 1, ptr_or, rt_args,
                {"H_V": num_v_heads, "GROUP_V": group_v}, num_warps=4,
            )
        return g, beta

    b_g, a_g = ba_interleaved.unflatten(-1, (num_k_heads, 2 * group_v)).split([group_v, group_v], dim=-1)
    b = b_g.reshape(bsz, seq_len, num_v_heads).float()
    a = a_g.reshape(bsz, seq_len, num_v_heads).float()
    g = -torch.exp(a_log.float()) * F.softplus(a + dt_bias.float())
    beta = torch.sigmoid(b)
    if out_g is not None:
        out_g.copy_(g)
        g = out_g
    if out_beta is not None:
        out_beta.copy_(beta)
        beta = out_beta
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
    stride_xb: tl.int64,
    stride_xt: tl.int64,
    stride_xh: tl.int64,
    stride_xd: tl.int64,
    stride_zb: tl.int64,
    stride_zt: tl.int64,
    stride_zhk: tl.int64,
    stride_zsub: tl.int64,
    stride_zd: tl.int64,
    stride_yb: tl.int64,
    stride_yt: tl.int64,
    stride_yh: tl.int64,
    stride_yd: tl.int64,
    T,
    eps,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    USE_SIGMOID: tl.constexpr,
    H_V: tl.constexpr = 1,
    BLOCK_H: tl.constexpr = 1,
):
    """Per-head RMSNorm (standard weight, NOT 1+w) followed by sigmoid/silu gating."""
    bt = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1).to(tl.int64)
    b = bt // T
    t = bt % T

    offs = tl.arange(0, D)
    if BLOCK_H == 1:
        h = hb
        x = tl.load(X_ptr + b * stride_xb + t * stride_xt + h * stride_xh + offs * stride_xd).to(tl.float32)
        if GQA_RATIO > 1:
            h_k = h // GQA_RATIO
            h_sub = h % GQA_RATIO
            z = tl.load(
                Z_ptr
                + b * stride_zb
                + t * stride_zt
                + h_k * stride_zhk
                + h_sub * stride_zsub
                + offs * stride_zd
            ).to(tl.float32)
        else:
            z = tl.load(
                Z_ptr + b * stride_zb + t * stride_zt + h * stride_zhk + offs * stride_zd
            ).to(tl.float32)
        w = tl.load(W_ptr + offs).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        y = (x * rstd) * w
        gate = tl.sigmoid(z) if USE_SIGMOID else (z * tl.sigmoid(z))
        y = y * gate
        tl.store(
            Y_ptr + b * stride_yb + t * stride_yt + h * stride_yh + offs * stride_yd,
            y.to(Y_ptr.dtype.element_ty),
        )
    else:
        hs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = hs < H_V
        x = tl.load(
            X_ptr + b * stride_xb + t * stride_xt + hs[:, None] * stride_xh + offs[None, :] * stride_xd,
            mask=h_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        if GQA_RATIO > 1:
            h_k = hs // GQA_RATIO
            h_sub = hs % GQA_RATIO
            z = tl.load(
                Z_ptr
                + b * stride_zb
                + t * stride_zt
                + h_k[:, None] * stride_zhk
                + h_sub[:, None] * stride_zsub
                + offs[None, :] * stride_zd,
                mask=h_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            z = tl.load(
                Z_ptr + b * stride_zb + t * stride_zt + hs[:, None] * stride_zhk + offs[None, :] * stride_zd,
                mask=h_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        w = tl.load(W_ptr + offs).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)[:, None]
        y = (x * rstd) * w[None, :]
        gate = tl.sigmoid(z) if USE_SIGMOID else (z * tl.sigmoid(z))
        y = y * gate
        tl.store(
            Y_ptr + b * stride_yb + t * stride_yt + hs[:, None] * stride_yh + offs[None, :] * stride_yd,
            y.to(Y_ptr.dtype.element_ty),
            mask=h_mask[:, None],
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
        is_5d_gate = (
            gate.ndim == 5
            and x.ndim == 4
            and gate.shape[:2] == x.shape[:2]
            and gate.shape[2] * gate.shape[3] == x.shape[2]
            and gate.shape[4] == self.dim
        )
        if (
            self.norm_before_gate
            and self.activation in ("sigmoid", "silu")
            and x.is_cuda
            and gate.is_cuda
            and self.weight.is_cuda
            and x.ndim == 4
            and (gate.ndim == 4 or is_5d_gate)
            and x.shape[-1] == self.dim
            and (self.dim & (self.dim - 1)) == 0
            and self.dim <= 512
            and self.weight.is_contiguous()
        ):
            bsz, seq_len, num_heads, dim = x.shape
            out = torch.empty((bsz, seq_len, num_heads, dim), dtype=x.dtype, device=x.device)
            if is_5d_gate:
                gqa_ratio = int(gate.shape[3])
                z_sb, z_st, z_shk, z_zsub, z_sd = (
                    gate.stride(0),
                    gate.stride(1),
                    gate.stride(2),
                    gate.stride(3),
                    gate.stride(4),
                )
            else:
                gqa_ratio = 1
                z_sb, z_st, z_shk, z_zsub, z_sd = (
                    gate.stride(0),
                    gate.stride(1),
                    gate.stride(2),
                    0,
                    gate.stride(3),
                )
            block_h = 4 if (num_heads % 4 == 0 and dim <= 128) else 1
            nw_rng = 2 if (block_h > 1 or dim <= 64) else 4
            if bsz * seq_len > 0:
                rt_args = (
                    x, gate, self.weight, out,
                    x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                    z_sb, z_st, z_shk, z_zsub, z_sd,
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    seq_len, float(self.eps),
                )
                use_sig = self.activation == "sigmoid"
                ckey = (
                    x.device.index, 1, x.dtype, gate.dtype, self.weight.dtype,
                    tuple(_triton_int_spec(v) for v in rt_args[4:18]),
                    gqa_ratio, dim, use_sig, num_heads, block_h, nw_rng,
                )
                ptr_or = x.data_ptr() | gate.data_ptr() | self.weight.data_ptr() | out.data_ptr()
                triton_fast_launch(
                    _GDN_TRITON_CACHE, ckey, _rms_norm_gated_kernel,
                    bsz * seq_len, triton.cdiv(num_heads, block_h), 1, ptr_or, rt_args,
                    {"GQA_RATIO": gqa_ratio, "D": dim, "USE_SIGMOID": use_sig, "H_V": num_heads, "BLOCK_H": block_h},
                    num_warps=nw_rng,
                )
            return out

        x_f = x.float()
        g_t = gate.reshape(x.shape) if is_5d_gate else gate
        g_f = g_t.float()
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


@triton.jit(do_not_specialize=["TPB", "MAX_BLOCKS", "NUM_BLOCKS"])
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
    BT_ptr,
    SEQ_BEFORE_ptr,
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    INIT_COL_OFF: tl.int64,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_DV: tl.constexpr = 64,
    IS_CONTIG_INIT: tl.constexpr = False,
    IS_CONTIG_FS: tl.constexpr = False,
    IS_OPAQUE_SSM: tl.constexpr = False,
):
    """Fused FP32 recurrent Gated Delta Rule kernel holding [D_K, BLOCK_DV] state in registers."""
    b = tl.program_id(0).to(tl.int64)
    h_v = tl.program_id(1).to(tl.int64)
    dv_blk = tl.program_id(2).to(tl.int64)
    h_k = h_v // GQA_RATIO

    offs_k = tl.arange(0, D_K)
    offs_v = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)

    if IS_OPAQUE_SSM:
        seq_b = tl.load(SEQ_BEFORE_ptr + b).to(tl.int32)
        has_pfx = seq_b > 0
        in_blk = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
        ok_in_b = has_pfx & (in_blk < MAX_BLOCKS)
        in_phys = tl.load(
            BT_ptr + b * stride_bt_b + tl.minimum(in_blk, MAX_BLOCKS - 1) * stride_bt_p,
            mask=ok_in_b,
            other=0,
        ).to(tl.int64)
        ok_in = ok_in_b & (in_phys > 0) & (in_phys < NUM_BLOCKS)
        sk_op: tl.constexpr = D_V
        s_ptrs = (
            InitState_ptr
            + in_phys * stride_sb
            + INIT_COL_OFF
            + h_v * (D_K * D_V)
            + offs_k[:, None] * sk_op
            + offs_v[None, :]
        )
        h = tl.load(s_ptrs, mask=ok_in, other=0.0, eviction_policy="evict_first").to(tl.float32)
    elif HAS_INIT_STATE:
        if IS_CONTIG_INIT:
            sk_init: tl.constexpr = D_V
            s_ptrs = (
                InitState_ptr
                + b * stride_sb
                + h_v * stride_shv
                + offs_k[:, None] * sk_init
                + offs_v[None, :]
            )
            h = tl.load(s_ptrs, eviction_policy="evict_first").to(tl.float32)
        else:
            s_ptrs = (
                InitState_ptr
                + b * stride_sb
                + h_v * stride_shv
                + offs_k[:, None] * stride_sk
                + offs_v[None, :] * stride_sv
            )
            h = tl.load(s_ptrs).to(tl.float32)
    else:
        h = tl.zeros((D_K, BLOCK_DV), dtype=tl.float32)

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

    if IS_OPAQUE_SSM:
        seq_b2 = tl.load(SEQ_BEFORE_ptr + b).to(tl.int32)
        out_blk = seq_b2 // TPB
        ok_out_b = (seq_b2 >= 0) & (out_blk < MAX_BLOCKS)
        out_phys = tl.load(
            BT_ptr + b * stride_bt_b + tl.minimum(tl.maximum(out_blk, 0), MAX_BLOCKS - 1) * stride_bt_p,
            mask=ok_out_b,
            other=0,
        ).to(tl.int64)
        ok_out = ok_out_b & (out_phys > 0) & (out_phys < NUM_BLOCKS)
        sk_op2: tl.constexpr = D_V
        op_ptrs = (
            InitState_ptr
            + out_phys * stride_sb
            + INIT_COL_OFF
            + h_v * (D_K * D_V)
            + offs_k[:, None] * sk_op2
            + offs_v[None, :]
        )
        tl.store(op_ptrs, h, mask=ok_out, eviction_policy="evict_first")

    if STORE_FINAL_STATE:
        if IS_CONTIG_FS:
            sk_fs: tl.constexpr = D_V
            fs_ptrs = (
                FinalState_ptr
                + b * stride_fsb
                + h_v * stride_fshv
                + offs_k[:, None] * sk_fs
                + offs_v[None, :]
            )
            tl.store(fs_ptrs, h, eviction_policy="evict_first")
        else:
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

    is_contig_init = (not has_init) or (
        initial_state.stride(-1) == 1 and initial_state.stride(-2) == d_v
    )
    is_contig_fs = (not output_final_state) or (
        final_state.stride(-1) == 1 and final_state.stride(-2) == d_v
    )
    # Keep (BLOCK_DV=d_v, num_warps=4) matching _yoco_gdn_decode_readout_kernel so
    # tl.sum(h * q_t[:, None], axis=0) stays bitwise identical between YoCo ON and OFF.
    block_dv = d_v

    if bsz > 0 and seq_len > 0:
        rt_args = (
            q, k, v, g, beta, init_ptr, fs_ptr, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            initial_state.stride(0) if has_init else 0,
            initial_state.stride(1) if has_init else 0,
            initial_state.stride(2) if has_init else 0,
            initial_state.stride(3) if has_init else 0,
            final_state.stride(0) if output_final_state else 0,
            final_state.stride(1) if output_final_state else 0,
            final_state.stride(2) if output_final_state else 0,
            final_state.stride(3) if output_final_state else 0,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len, float(d_k ** -0.5), float(l2_eps),
            q, q, 0, 0, 1, 1, 1, 0,
        )
        ckey = (
            q.device.index, 2, q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, init_ptr.dtype, fs_ptr.dtype,
            tuple(_triton_int_spec(x) for x in rt_args[8:39]),
            gqa_ratio, d_k, d_v, has_init, output_final_state, block_dv, is_contig_init, is_contig_fs, False,
        )
        ptr_or = q.data_ptr() | k.data_ptr() | v.data_ptr() | g.data_ptr() | beta.data_ptr() | init_ptr.data_ptr() | fs_ptr.data_ptr() | out.data_ptr()
        triton_fast_launch(
            _GDN_TRITON_CACHE, ckey, _gdn_recurrent_kernel,
            bsz, num_v_heads, 1, ptr_or, rt_args,
            {
                "GQA_RATIO": gqa_ratio, "D_K": d_k, "D_V": d_v,
                "HAS_INIT_STATE": has_init, "STORE_FINAL_STATE": output_final_state,
                "BLOCK_DV": block_dv, "IS_CONTIG_INIT": is_contig_init,
                "IS_CONTIG_FS": is_contig_fs, "IS_OPAQUE_SSM": False,
            },
            num_warps=4,
        )
    return out, final_state


def gdn_recurrent_decode_opaque_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_pool_2d: torch.Tensor,
    block_table: torch.Tensor,
    seq_before: torch.Tensor,
    *,
    page_size: int,
    pool_col_offset: int = 0,
    output_final_state: bool = False,
    l2_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Single-launch T=1 GDN recurrent decode reading & updating paged FP32 SSM pool directly."""
    bsz, seq_len, num_k_heads, d_k = q.shape
    num_v_heads, d_v = v.shape[-2], v.shape[-1]
    gqa_ratio = num_v_heads // num_k_heads
    out = torch.empty((bsz, seq_len, num_v_heads, d_v), dtype=v.dtype, device=v.device)
    final_state = (
        torch.empty((bsz, num_v_heads, d_k, d_v), dtype=torch.float32, device=v.device)
        if output_final_state
        else None
    )
    if bsz == 0 or seq_len == 0:
        return out, final_state
    fs_ptr = final_state if output_final_state else out
    spb = int(ssm_pool_2d.stride(0))
    sbt0 = int(block_table.stride(0))
    sbt1 = int(block_table.stride(1))
    pco = int(pool_col_offset)
    rt_args = (
        q, k, v, g, beta, ssm_pool_2d, fs_ptr, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        spb, 0, 0, 0,
        final_state.stride(0) if output_final_state else 0,
        final_state.stride(1) if output_final_state else 0,
        final_state.stride(2) if output_final_state else 0,
        final_state.stride(3) if output_final_state else 0,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len, float(d_k ** -0.5), float(l2_eps),
        block_table, seq_before, sbt0, sbt1,
        int(page_size), int(block_table.shape[1]), int(ssm_pool_2d.shape[0]), pco,
    )
    ckey = (
        q.device.index, 3, q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype,
        ssm_pool_2d.dtype, fs_ptr.dtype, block_table.dtype, seq_before.dtype,
        tuple(_triton_int_spec(x) for x in (rt_args[8:39] + (sbt0, sbt1, pco))),
        gqa_ratio, d_k, d_v, output_final_state,
    )
    ptr_or = (
        q.data_ptr() | k.data_ptr() | v.data_ptr() | g.data_ptr() | beta.data_ptr()
        | ssm_pool_2d.data_ptr() | fs_ptr.data_ptr() | out.data_ptr() | block_table.data_ptr() | seq_before.data_ptr()
    )
    triton_fast_launch(
        _GDN_TRITON_CACHE, ckey, _gdn_recurrent_kernel,
        bsz, num_v_heads, 1, ptr_or, rt_args,
        {
            "GQA_RATIO": gqa_ratio, "D_K": d_k, "D_V": d_v,
            "HAS_INIT_STATE": False, "STORE_FINAL_STATE": output_final_state,
            "BLOCK_DV": d_v, "IS_CONTIG_INIT": True,
            "IS_CONTIG_FS": True, "IS_OPAQUE_SSM": True,
        },
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
# Fused Unpack + Conv1d Kernels (Decode, Target Verify, Packed Ragged Prefill)
# ---------------------------------------------------------------------------


@triton.jit
def _gdn_map_conv_tile(
    c_start,
    offs,
    KEY_DIM: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    IS_CONSUMER: tl.constexpr,
):
    """Map a 64-wide conv channel tile to (qkvz_col, dst_kind 0=Q/1=K/2=V, dst_offset)."""
    if IS_CONSUMER:
        h_k = c_start // D_K
        qkvz_col = h_k * GROUP_STRIDE + offs
        return qkvz_col, 0, c_start + offs
    if c_start < KEY_DIM:
        h_k = c_start // D_K
        qkvz_col = h_k * GROUP_STRIDE + offs
        return qkvz_col, 0, c_start + offs
    if c_start < 2 * KEY_DIM:
        c2 = c_start - KEY_DIM
        h_k = c2 // D_K
        qkvz_col = h_k * GROUP_STRIDE + D_K + offs
        return qkvz_col, 1, c2 + offs
    c3 = c_start - 2 * KEY_DIM
    v_group: tl.constexpr = GQA_RATIO * D_V
    h_k = c3 // v_group
    in_v = c3 % v_group
    qkvz_col = h_k * GROUP_STRIDE + 2 * D_K + in_v + offs
    return qkvz_col, 2, c3 + offs


@triton.jit(do_not_specialize=["TPB", "MAX_BLOCKS", "NUM_BLOCKS"])
def _gdn_fused_unpack_conv1d_decode_k4_kernel(
    QKVZ_ptr,
    W_ptr,
    STATE_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    stride_qb: tl.int64,
    stride_qc: tl.int64,
    stride_sb: tl.int64,
    stride_sc: tl.int64,
    stride_sk: tl.int64,
    stride_out_qb: tl.int64,
    stride_out_kb: tl.int64,
    stride_out_vb: tl.int64,
    BT_ptr,
    SEQ_BEFORE_ptr,
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    POOL_COL_OFF: tl.int64,
    KEY_DIM: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    IS_CONSUMER: tl.constexpr,
    RESIDUAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_OPAQUE_STATE: tl.constexpr = False,
):
    """Fused T=1 unpack qkvz + K=4 causal dwconv1d + BF16 round + SiLU + state roll."""
    b = tl.program_id(0).to(tl.int64)
    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C
    offs = tl.arange(0, BLOCK_C).to(tl.int64)
    c_offs = c_start + offs

    qkvz_col, kind, dst_off = _gdn_map_conv_tile(
        c_start, offs, KEY_DIM, D_K, D_V, GQA_RATIO, GROUP_STRIDE, IS_CONSUMER
    )
    x0_raw = tl.load(QKVZ_ptr + b * stride_qb + qkvz_col * stride_qc)

    if IS_OPAQUE_STATE:
        seq_b = tl.load(SEQ_BEFORE_ptr + b).to(tl.int32)
        has_pfx = seq_b > 0
        in_blk = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
        ok_in_b = has_pfx & (in_blk < MAX_BLOCKS)
        in_phys = tl.load(
            BT_ptr + b * stride_bt_b + tl.minimum(in_blk, MAX_BLOCKS - 1) * stride_bt_p,
            mask=ok_in_b,
            other=0,
        ).to(tl.int64)
        ok_in = ok_in_b & (in_phys > 0) & (in_phys < NUM_BLOCKS)
        s_in_base = STATE_ptr + in_phys * stride_sb + POOL_COL_OFF + c_offs * 3
        s0_raw = tl.load(s_in_base + 0, mask=ok_in, other=0.0)
        s1_raw = tl.load(s_in_base + 1, mask=ok_in, other=0.0)
        s2_raw = tl.load(s_in_base + 2, mask=ok_in, other=0.0)
    else:
        s_base = STATE_ptr + b * stride_sb + c_offs * stride_sc
        s0_raw = tl.load(s_base + 0 * stride_sk)
        s1_raw = tl.load(s_base + 1 * stride_sk)
        s2_raw = tl.load(s_base + 2 * stride_sk)

    w_base = W_ptr + c_offs * 4
    w0 = tl.load(w_base + 0).to(tl.float32)
    w1 = tl.load(w_base + 1).to(tl.float32)
    w2 = tl.load(w_base + 2).to(tl.float32)
    w3 = tl.load(w_base + 3).to(tl.float32)

    x0_f = x0_raw.to(tl.float32)
    y = s0_raw.to(tl.float32) * w0 + s1_raw.to(tl.float32) * w1 + s2_raw.to(tl.float32) * w2 + x0_f * w3
    out_ty = Q_ptr.dtype.element_ty
    y = y.to(out_ty).to(tl.float32)
    y = y * tl.sigmoid(y)
    if RESIDUAL:
        y = y + x0_f

    if kind == 0:
        tl.store(Q_ptr + b * stride_out_qb + dst_off, y.to(out_ty))
    elif kind == 1:
        tl.store(K_ptr + b * stride_out_kb + dst_off, y.to(out_ty))
    else:
        tl.store(V_ptr + b * stride_out_vb + dst_off, y.to(out_ty))

    st_ty = STATE_ptr.dtype.element_ty
    if IS_OPAQUE_STATE:
        seq_b2 = tl.load(SEQ_BEFORE_ptr + b).to(tl.int32)
        out_blk = seq_b2 // TPB
        ok_out_b = (seq_b2 >= 0) & (out_blk < MAX_BLOCKS)
        out_phys = tl.load(
            BT_ptr + b * stride_bt_b + tl.minimum(tl.maximum(out_blk, 0), MAX_BLOCKS - 1) * stride_bt_p,
            mask=ok_out_b,
            other=0,
        ).to(tl.int64)
        ok_out = ok_out_b & (out_phys > 0) & (out_phys < NUM_BLOCKS)
        s_out_base = STATE_ptr + out_phys * stride_sb + POOL_COL_OFF + c_offs * 3
        tl.store(s_out_base + 0, s1_raw.to(st_ty), mask=ok_out)
        tl.store(s_out_base + 1, s2_raw.to(st_ty), mask=ok_out)
        tl.store(s_out_base + 2, x0_raw.to(st_ty), mask=ok_out)
    else:
        s_base = STATE_ptr + b * stride_sb + c_offs * stride_sc
        tl.store(s_base + 0 * stride_sk, s1_raw.to(st_ty))
        tl.store(s_base + 1 * stride_sk, s2_raw.to(st_ty))
        tl.store(s_base + 2 * stride_sk, x0_raw.to(st_ty))


@triton.jit
def _gdn_fused_unpack_conv1d_verify_k4_kernel(
    QKVZ_ptr,
    W_ptr,
    INIT_STATE_ptr,
    STAGE_CONV_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    stride_qb: tl.int64,
    stride_qv: tl.int64,
    stride_qc: tl.int64,
    stride_sb: tl.int64,
    stride_sc: tl.int64,
    stride_sk: tl.int64,
    stride_st_b: tl.int64,
    stride_st_v: tl.int64,
    stride_st_c: tl.int64,
    stride_st_k: tl.int64,
    stride_out_qb: tl.int64,
    stride_out_qv: tl.int64,
    stride_out_kb: tl.int64,
    stride_out_kv: tl.int64,
    stride_out_vb: tl.int64,
    stride_out_vv: tl.int64,
    V_STEPS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    IS_CONSUMER: tl.constexpr,
    RESIDUAL: tl.constexpr,
    HAS_STAGE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused V-step verify unpack qkvz + K=4 causal dwconv1d + SiLU + rollback stage write."""
    b = tl.program_id(0).to(tl.int64)
    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C
    offs = tl.arange(0, BLOCK_C).to(tl.int64)
    c_offs = c_start + offs

    qkvz_col, kind, dst_off = _gdn_map_conv_tile(
        c_start, offs, KEY_DIM, D_K, D_V, GQA_RATIO, GROUP_STRIDE, IS_CONSUMER
    )

    s_base = INIT_STATE_ptr + b * stride_sb + c_offs * stride_sc
    s0_raw = tl.load(s_base + 0 * stride_sk)
    s1_raw = tl.load(s_base + 1 * stride_sk)
    s2_raw = tl.load(s_base + 2 * stride_sk)

    w_base = W_ptr + c_offs * 4
    w0 = tl.load(w_base + 0).to(tl.float32)
    w1 = tl.load(w_base + 1).to(tl.float32)
    w2 = tl.load(w_base + 2).to(tl.float32)
    w3 = tl.load(w_base + 3).to(tl.float32)

    out_ty = Q_ptr.dtype.element_ty
    st_ty = STAGE_CONV_ptr.dtype.element_ty

    for v in tl.static_range(V_STEPS):
        x_raw = tl.load(QKVZ_ptr + b * stride_qb + v * stride_qv + qkvz_col * stride_qc)
        x_f = x_raw.to(tl.float32)
        y = s0_raw.to(tl.float32) * w0 + s1_raw.to(tl.float32) * w1 + s2_raw.to(tl.float32) * w2 + x_f * w3
        y = y.to(out_ty).to(tl.float32)
        y = y * tl.sigmoid(y)
        if RESIDUAL:
            y = y + x_f

        if kind == 0:
            tl.store(Q_ptr + b * stride_out_qb + v * stride_out_qv + dst_off, y.to(out_ty))
        elif kind == 1:
            tl.store(K_ptr + b * stride_out_kb + v * stride_out_kv + dst_off, y.to(out_ty))
        else:
            tl.store(V_ptr + b * stride_out_vb + v * stride_out_vv + dst_off, y.to(out_ty))

        s0_raw = s1_raw
        s1_raw = s2_raw
        s2_raw = x_raw
        if HAS_STAGE:
            st_v = STAGE_CONV_ptr + b * stride_st_b + v * stride_st_v + c_offs * stride_st_c
            tl.store(st_v + 0 * stride_st_k, s0_raw.to(st_ty))
            tl.store(st_v + 1 * stride_st_k, s1_raw.to(st_ty))
            tl.store(st_v + 2 * stride_st_k, s2_raw.to(st_ty))


@triton.jit
def _gdn_fused_unpack_conv1d_ragged_kernel(
    QKVZ_ptr,
    W_ptr,
    INIT_STATE_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    CU_SEQLENS_ptr,
    TILE_REQ_ptr,
    TILE_T0_ptr,
    stride_qn: tl.int64,
    stride_qc: tl.int64,
    stride_ib: tl.int64,
    stride_ic: tl.int64,
    stride_ik: tl.int64,
    stride_out_qn: tl.int64,
    stride_out_kn: tl.int64,
    stride_out_vn: tl.int64,
    KEY_DIM: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    IS_CONSUMER: tl.constexpr,
    HAS_INIT: tl.constexpr,
    RESIDUAL: tl.constexpr,
    SUB_T: tl.constexpr,
    SUB_NUM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """1D register sliding-window fused unpack qkvz + K=4 causal dwconv1d + SiLU."""
    pid0 = tl.program_id(0).to(tl.int64)
    tile_id = pid0 // SUB_NUM
    sub_id = pid0 % SUB_NUM
    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C
    b = tl.load(TILE_REQ_ptr + tile_id).to(tl.int64)
    t_start = tl.load(TILE_T0_ptr + tile_id).to(tl.int64) + sub_id * SUB_T
    req_start = tl.load(CU_SEQLENS_ptr + b).to(tl.int64)
    req_end = tl.load(CU_SEQLENS_ptr + b + 1).to(tl.int64)
    req_len = req_end - req_start
    if t_start >= req_len:
        return
    tile_len = tl.minimum(SUB_T, req_len - t_start)

    offs = tl.arange(0, BLOCK_C).to(tl.int64)
    c_offs = c_start + offs
    qkvz_col, kind, dst_off = _gdn_map_conv_tile(
        c_start, offs, KEY_DIM, D_K, D_V, GQA_RATIO, GROUP_STRIDE, IS_CONSUMER
    )

    w_base = W_ptr + c_offs * 4
    w0 = tl.load(w_base + 0).to(tl.float32)
    w1 = tl.load(w_base + 1).to(tl.float32)
    w2 = tl.load(w_base + 2).to(tl.float32)
    w3 = tl.load(w_base + 3).to(tl.float32)

    q_col_off = qkvz_col * stride_qc
    if t_start >= 3:
        s0 = tl.load(QKVZ_ptr + (req_start + t_start - 3) * stride_qn + q_col_off).to(tl.float32)
        s1 = tl.load(QKVZ_ptr + (req_start + t_start - 2) * stride_qn + q_col_off).to(tl.float32)
        s2 = tl.load(QKVZ_ptr + (req_start + t_start - 1) * stride_qn + q_col_off).to(tl.float32)
    else:
        t0 = t_start - 3
        t1 = t_start - 2
        t2 = t_start - 1
        s0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        s1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        s2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if t0 >= 0:
            s0 = tl.load(QKVZ_ptr + (req_start + t0) * stride_qn + q_col_off).to(tl.float32)
        elif HAS_INIT:
            s0 = tl.load(INIT_STATE_ptr + b * stride_ib + c_offs * stride_ic + (t0 + 3) * stride_ik).to(tl.float32)
        if t1 >= 0:
            s1 = tl.load(QKVZ_ptr + (req_start + t1) * stride_qn + q_col_off).to(tl.float32)
        elif HAS_INIT:
            s1 = tl.load(INIT_STATE_ptr + b * stride_ib + c_offs * stride_ic + (t1 + 3) * stride_ik).to(tl.float32)
        if t2 >= 0:
            s2 = tl.load(QKVZ_ptr + (req_start + t2) * stride_qn + q_col_off).to(tl.float32)
        elif HAS_INIT:
            s2 = tl.load(INIT_STATE_ptr + b * stride_ib + c_offs * stride_ic + (t2 + 3) * stride_ik).to(tl.float32)

    out_ty = Q_ptr.dtype.element_ty
    in_ptr = QKVZ_ptr + (req_start + t_start) * stride_qn + q_col_off
    if kind == 0:
        out_ptr = Q_ptr + (req_start + t_start) * stride_out_qn + dst_off
        stride_out = stride_out_qn
    elif kind == 1:
        out_ptr = K_ptr + (req_start + t_start) * stride_out_kn + dst_off
        stride_out = stride_out_kn
    else:
        out_ptr = V_ptr + (req_start + t_start) * stride_out_vn + dst_off
        stride_out = stride_out_vn

    for _dt in range(tile_len):
        xf = tl.load(in_ptr).to(tl.float32)
        acc = s0 * w0 + s1 * w1 + s2 * w2 + xf * w3
        acc = acc.to(out_ty).to(tl.float32)
        acc = acc * tl.sigmoid(acc)
        if RESIDUAL:
            acc = acc + xf
        tl.store(out_ptr, acc.to(out_ty))
        s0 = s1
        s1 = s2
        s2 = xf
        in_ptr += stride_qn
        out_ptr += stride_out


@triton.jit
def _gdn_scatter_qkvz_conv_snapshots_ragged_kernel(
    QKVZ_ptr,
    INIT_STATE_ptr,
    CU_SEQLENS_ptr,
    SNAP_TOK_ptr,
    SNAP_REQ_ptr,
    SNAP_DST_ptr,
    DST_ptr,
    stride_qn: tl.int64,
    stride_qc: tl.int64,
    stride_ib: tl.int64,
    stride_ic: tl.int64,
    stride_ik: tl.int64,
    dst_stride_row: tl.int64,
    dst_col_offset: tl.int64,
    dst_rows: tl.int64,
    KEY_DIM: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    IS_CONSUMER: tl.constexpr,
    HAS_INIT: tl.constexpr,
    SKIP_INVALID_PHYS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Scatter 3-token pre-conv boundary states directly from packed qkvz without intermediate buffer."""
    s_id = tl.program_id(0).to(tl.int64)
    dst_idx = tl.load(SNAP_DST_ptr + s_id).to(tl.int64)
    if SKIP_INVALID_PHYS:
        if dst_idx <= 0 or dst_idx >= dst_rows:
            return
    g_tok = tl.load(SNAP_TOK_ptr + s_id).to(tl.int64)
    b = tl.load(SNAP_REQ_ptr + s_id).to(tl.int64)
    req_start = tl.load(CU_SEQLENS_ptr + b).to(tl.int64)
    local_t = g_tok - req_start

    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C
    offs = tl.arange(0, BLOCK_C).to(tl.int64)
    c_offs = c_start + offs
    qkvz_col, _kind, _dst_off = _gdn_map_conv_tile(
        c_start, offs, KEY_DIM, D_K, D_V, GQA_RATIO, GROUP_STRIDE, IS_CONSUMER
    )

    dst_base = DST_ptr + dst_idx * dst_stride_row + dst_col_offset + c_offs * 3
    dst_ty = DST_ptr.dtype.element_ty
    for j in tl.static_range(3):
        src_local = local_t - 2 + j
        if src_local >= 0:
            val = tl.load(
                QKVZ_ptr + (req_start + src_local) * stride_qn + qkvz_col * stride_qc
            )
        elif HAS_INIT:
            val = tl.load(
                INIT_STATE_ptr
                + b * stride_ib
                + c_offs * stride_ic
                + (src_local + 3) * stride_ik
            )
        else:
            val = tl.zeros((BLOCK_C,), dtype=dst_ty)
        tl.store(dst_base + j, val.to(dst_ty))


@triton.jit
def _gdn_verify_scan_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    InitState_ptr,
    StageState_ptr,
    Out_ptr,
    stride_qb: tl.int64,
    stride_qv: tl.int64,
    stride_qhk: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kv: tl.int64,
    stride_khk: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vv: tl.int64,
    stride_vhv: tl.int64,
    stride_vd: tl.int64,
    stride_gb: tl.int64,
    stride_gv: tl.int64,
    stride_ghv: tl.int64,
    stride_bb: tl.int64,
    stride_bv: tl.int64,
    stride_bhv: tl.int64,
    stride_sb: tl.int64,
    stride_shv: tl.int64,
    stride_sk: tl.int64,
    stride_sv: tl.int64,
    stride_ssb: tl.int64,
    stride_ssv: tl.int64,
    stride_sshv: tl.int64,
    stride_ssk: tl.int64,
    stride_ssvv: tl.int64,
    stride_ob: tl.int64,
    stride_ov: tl.int64,
    stride_ohv: tl.int64,
    stride_od: tl.int64,
    scale,
    l2_eps,
    V_STEPS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_STAGE_STATE: tl.constexpr,
    IS_CONTIG_INIT: tl.constexpr = False,
    IS_CONTIG_STAGE: tl.constexpr = False,
    EVICT_ST: tl.constexpr = False,
):
    """Single-launch GDN verify scan keeping [D_K, BLOCK_DV] FP32 SSM state in registers over V steps."""
    b = tl.program_id(0).to(tl.int64)
    h_v = tl.program_id(1).to(tl.int64)
    dv_blk = tl.program_id(2).to(tl.int64)
    h_k = h_v // GQA_RATIO

    offs_k = tl.arange(0, D_K)
    offs_v = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)

    if HAS_INIT_STATE:
        if IS_CONTIG_INIT:
            sk_init: tl.constexpr = D_V
            init_ptrs = (
                InitState_ptr
                + b * stride_sb
                + h_v * stride_shv
                + offs_k[:, None] * sk_init
                + offs_v[None, :]
            )
        else:
            init_ptrs = (
                InitState_ptr
                + b * stride_sb
                + h_v * stride_shv
                + offs_k[:, None] * stride_sk
                + offs_v[None, :] * stride_sv
            )
        if EVICT_ST:
            h = tl.load(init_ptrs, eviction_policy="evict_first").to(tl.float32)
        else:
            h = tl.load(init_ptrs).to(tl.float32)
    else:
        h = tl.zeros((D_K, BLOCK_DV), dtype=tl.float32)

    q_base = Q_ptr + b * stride_qb + h_k * stride_qhk + offs_k * stride_qd
    k_base = K_ptr + b * stride_kb + h_k * stride_khk + offs_k * stride_kd
    v_base = V_ptr + b * stride_vb + h_v * stride_vhv + offs_v * stride_vd
    g_base = G_ptr + b * stride_gb + h_v * stride_ghv
    b_base = Beta_ptr + b * stride_bb + h_v * stride_bhv
    o_base = Out_ptr + b * stride_ob + h_v * stride_ohv + offs_v * stride_od
    if IS_CONTIG_STAGE:
        sk_stg: tl.constexpr = D_V
        st_base = (
            StageState_ptr
            + b * stride_ssb
            + h_v * stride_sshv
            + offs_k[:, None] * sk_stg
            + offs_v[None, :]
        )
    else:
        st_base = (
            StageState_ptr
            + b * stride_ssb
            + h_v * stride_sshv
            + offs_k[:, None] * stride_ssk
            + offs_v[None, :] * stride_ssvv
        )
    out_dtype = Out_ptr.dtype.element_ty

    for t in tl.static_range(V_STEPS):
        q_t = tl.load(q_base + t * stride_qv).to(tl.float32)
        k_t = tl.load(k_base + t * stride_kv).to(tl.float32)
        v_t = tl.load(v_base + t * stride_vv).to(tl.float32)
        g_t = tl.load(g_base + t * stride_gv).to(tl.float32)
        beta_t = tl.load(b_base + t * stride_bv).to(tl.float32)

        q_t = q_t * (tl.rsqrt(tl.sum(q_t * q_t, axis=0) + l2_eps) * scale)
        k_t = k_t * tl.rsqrt(tl.sum(k_t * k_t, axis=0) + l2_eps)

        h = h * tl.exp(g_t)
        v_pred = tl.sum(h * k_t[:, None], axis=0)
        delta = (v_t - v_pred) * beta_t
        h = h + k_t[:, None] * delta[None, :]
        o_t = tl.sum(h * q_t[:, None], axis=0)

        tl.store(o_base + t * stride_ov, o_t.to(out_dtype))
        if HAS_STAGE_STATE:
            if EVICT_ST:
                tl.store(st_base + t * stride_ssv, h, eviction_policy="evict_first")
            else:
                tl.store(st_base + t * stride_ssv, h)


@triton.jit
def _yoco_gdn_verify_readout_kernel(
    Q_ptr,
    S_ptr,
    Out_ptr,
    stride_qb: tl.int64,
    stride_qv: tl.int64,
    stride_qhk: tl.int64,
    stride_qd: tl.int64,
    stride_sb: tl.int64,
    stride_sv: tl.int64,
    stride_shv: tl.int64,
    stride_sk: tl.int64,
    stride_svv: tl.int64,
    stride_ob: tl.int64,
    stride_ov: tl.int64,
    stride_ohv: tl.int64,
    stride_od: tl.int64,
    V_STEPS: tl.constexpr,
    scale,
    eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_DV: tl.constexpr = 64,
    BK: tl.constexpr = 0,
    IS_S_CONTIG: tl.constexpr = False,
):
    """Fused batched V-step YoCo GDN verify readout against source per-step SSM states."""
    bv = tl.program_id(0).to(tl.int64)
    h_v = tl.program_id(1).to(tl.int64)
    dv = tl.program_id(2).to(tl.int64)
    b = bv // V_STEPS
    t = bv % V_STEPS
    h_k = h_v // GQA_RATIO

    offs_k = tl.arange(0, D_K)
    offs_v = dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    q_row = Q_ptr + b * stride_qb + t * stride_qv + h_k * stride_qhk
    q_full = tl.load(q_row + offs_k * stride_qd).to(tl.float32)
    q_inv = tl.rsqrt(tl.sum(q_full * q_full, axis=0) + eps) * scale

    if BK > 0 and IS_S_CONTIG:
        s_base = S_ptr + b * stride_sb + t * stride_sv + h_v * stride_shv
        out = tl.zeros((BLOCK_DV,), dtype=tl.float32)
        sk_s: tl.constexpr = D_V
        for k0 in tl.static_range(0, D_K, BK):
            ok = k0 + tl.arange(0, BK)
            q_k = tl.load(q_row + ok * stride_qd).to(tl.float32) * q_inv
            s_k = tl.load(
                s_base + ok[:, None] * sk_s + offs_v[None, :],
                eviction_policy="evict_first",
            ).to(tl.float32)
            out += tl.sum(s_k * q_k[:, None], axis=0)
    else:
        q_scaled = q_full * q_inv
        s = tl.load(
            S_ptr
            + b * stride_sb
            + t * stride_sv
            + h_v * stride_shv
            + offs_k[:, None] * stride_sk
            + offs_v[None, :] * stride_svv,
            eviction_policy="evict_first",
        ).to(tl.float32)
        out = tl.sum(s * q_scaled[:, None], axis=0)

    tl.store(
        Out_ptr + b * stride_ob + t * stride_ov + h_v * stride_ohv + offs_v * stride_od,
        out.to(Out_ptr.dtype.element_ty),
    )


# ---------------------------------------------------------------------------
# Packed Ragged GDN Chunked Parallel Prefill + Block-Boundary Snapshots
# ---------------------------------------------------------------------------


@triton.jit
def _gdn_chunk_prep_wy_ragged_kernel(
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    CU_SEQLENS_ptr,
    CHUNK_REQ_ptr,
    CHUNK_T0_ptr,
    GCum_ptr,
    W_ptr,
    U_ptr,
    stride_kn,
    stride_khk,
    stride_kd,
    stride_vn,
    stride_vhv,
    stride_vd,
    stride_gn,
    stride_ghv,
    stride_bn,
    stride_bhv,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_wh,
    stride_wc,
    stride_wt,
    stride_wd,
    stride_uh,
    stride_uc,
    stride_ut,
    stride_ud,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """Ragged Stage 1: compute chunk cumsum G and exact block-16 WY factors (W, U)."""
    h_v = tl.program_id(0)
    c = tl.program_id(1)
    h_k = h_v // GQA_RATIO

    b = tl.load(CHUNK_REQ_ptr + c)
    t0 = tl.load(CHUNK_T0_ptr + c)
    req_start = tl.load(CU_SEQLENS_ptr + b)
    req_end = tl.load(CU_SEQLENS_ptr + b + 1)
    req_len = req_end - req_start

    o16 = tl.arange(0, 16)
    ok = tl.arange(0, D_K)
    ov = tl.arange(0, D_V)
    tril_mask = o16[:, None] > o16[None, :]

    rs64 = req_start.to(tl.int64)
    k_base = K_ptr + rs64 * stride_kn + h_k * stride_khk
    v_base = V_ptr + rs64 * stride_vn + h_v * stride_vhv
    g_base = G_ptr + rs64 * stride_gn + h_v * stride_ghv
    b_base = Beta_ptr + rs64 * stride_bn + h_v * stride_bhv

    # Load 4 sub-blocks of 16 tokens each
    t_0 = t0 + o16
    t_1 = t0 + 16 + o16
    t_2 = t0 + 32 + o16
    t_3 = t0 + 48 + o16
    m_0 = t_0 < req_len
    m_1 = t_1 < req_len
    m_2 = t_2 < req_len
    m_3 = t_3 < req_len

    g_0 = tl.load(g_base + t_0 * stride_gn, mask=m_0, other=0.0).to(tl.float32)
    g_1 = tl.load(g_base + t_1 * stride_gn, mask=m_1, other=0.0).to(tl.float32)
    g_2 = tl.load(g_base + t_2 * stride_gn, mask=m_2, other=0.0).to(tl.float32)
    g_3 = tl.load(g_base + t_3 * stride_gn, mask=m_3, other=0.0).to(tl.float32)

    G_0 = tl.cumsum(g_0, axis=0)
    s0 = tl.sum(g_0, axis=0)
    G_1 = tl.cumsum(g_1, axis=0) + s0
    s1 = s0 + tl.sum(g_1, axis=0)
    G_2 = tl.cumsum(g_2, axis=0) + s1
    s2 = s1 + tl.sum(g_2, axis=0)
    G_3 = tl.cumsum(g_3, axis=0) + s2

    gc_base = GCum_ptr + h_v * stride_gch + c * stride_gcc
    tl.store(gc_base + o16 * stride_gct, G_0)
    tl.store(gc_base + (16 + o16) * stride_gct, G_1)
    tl.store(gc_base + (32 + o16) * stride_gct, G_2)
    tl.store(gc_base + (48 + o16) * stride_gct, G_3)

    beta_0 = tl.load(b_base + t_0 * stride_bn, mask=m_0, other=0.0).to(tl.float32)
    beta_1 = tl.load(b_base + t_1 * stride_bn, mask=m_1, other=0.0).to(tl.float32)
    beta_2 = tl.load(b_base + t_2 * stride_bn, mask=m_2, other=0.0).to(tl.float32)
    beta_3 = tl.load(b_base + t_3 * stride_bn, mask=m_3, other=0.0).to(tl.float32)

    k_0 = tl.load(k_base + t_0[:, None] * stride_kn + ok[None, :] * stride_kd, mask=m_0[:, None], other=0.0).to(tl.float32)
    k_1 = tl.load(k_base + t_1[:, None] * stride_kn + ok[None, :] * stride_kd, mask=m_1[:, None], other=0.0).to(tl.float32)
    k_2 = tl.load(k_base + t_2[:, None] * stride_kn + ok[None, :] * stride_kd, mask=m_2[:, None], other=0.0).to(tl.float32)
    k_3 = tl.load(k_base + t_3[:, None] * stride_kn + ok[None, :] * stride_kd, mask=m_3[:, None], other=0.0).to(tl.float32)

    k_0 = k_0 * tl.rsqrt(tl.sum(k_0 * k_0, axis=1) + l2_eps)[:, None]
    k_1 = k_1 * tl.rsqrt(tl.sum(k_1 * k_1, axis=1) + l2_eps)[:, None]
    k_2 = k_2 * tl.rsqrt(tl.sum(k_2 * k_2, axis=1) + l2_eps)[:, None]
    k_3 = k_3 * tl.rsqrt(tl.sum(k_3 * k_3, axis=1) + l2_eps)[:, None]

    # Phase A: lower-triangular coupling blocks A_ij and diagonal inverses M_ii (no V live yet)
    diff_00 = tl.where(tril_mask, G_0[:, None] - G_0[None, :], -1e9)
    A_00 = tl.where(tril_mask, beta_0[:, None] * tl.exp(diff_00) * tl.dot(k_0, tl.trans(k_0), input_precision="tf32"), 0.0)
    M_00 = _invert_unit_tril_16(A_00)

    A_10 = beta_1[:, None] * tl.exp(G_1[:, None] - G_0[None, :]) * tl.dot(k_1, tl.trans(k_0), input_precision="tf32")
    diff_11 = tl.where(tril_mask, G_1[:, None] - G_1[None, :], -1e9)
    A_11 = tl.where(tril_mask, beta_1[:, None] * tl.exp(diff_11) * tl.dot(k_1, tl.trans(k_1), input_precision="tf32"), 0.0)
    M_11 = _invert_unit_tril_16(A_11)

    A_20 = beta_2[:, None] * tl.exp(G_2[:, None] - G_0[None, :]) * tl.dot(k_2, tl.trans(k_0), input_precision="tf32")
    A_21 = beta_2[:, None] * tl.exp(G_2[:, None] - G_1[None, :]) * tl.dot(k_2, tl.trans(k_1), input_precision="tf32")
    diff_22 = tl.where(tril_mask, G_2[:, None] - G_2[None, :], -1e9)
    A_22 = tl.where(tril_mask, beta_2[:, None] * tl.exp(diff_22) * tl.dot(k_2, tl.trans(k_2), input_precision="tf32"), 0.0)
    M_22 = _invert_unit_tril_16(A_22)

    A_30 = beta_3[:, None] * tl.exp(G_3[:, None] - G_0[None, :]) * tl.dot(k_3, tl.trans(k_0), input_precision="tf32")
    A_31 = beta_3[:, None] * tl.exp(G_3[:, None] - G_1[None, :]) * tl.dot(k_3, tl.trans(k_1), input_precision="tf32")
    A_32 = beta_3[:, None] * tl.exp(G_3[:, None] - G_2[None, :]) * tl.dot(k_3, tl.trans(k_2), input_precision="tf32")
    diff_33 = tl.where(tril_mask, G_3[:, None] - G_3[None, :], -1e9)
    A_33 = tl.where(tril_mask, beta_3[:, None] * tl.exp(diff_33) * tl.dot(k_3, tl.trans(k_3), input_precision="tf32"), 0.0)
    M_33 = _invert_unit_tril_16(A_33)

    # Phase B: solve W_0..W_3 and store immediately to free K/W register pressure
    w_base = W_ptr + h_v * stride_wh + c * stride_wc
    w_dtype = W_ptr.dtype.element_ty

    ks_0 = k_0 * (beta_0 * tl.exp(G_0))[:, None]
    W_0 = tl.dot(M_00, ks_0, input_precision="tf32")
    tl.store(w_base + o16[:, None] * stride_wt + ok[None, :] * stride_wd, W_0.to(w_dtype))

    ks_1 = k_1 * (beta_1 * tl.exp(G_1))[:, None]
    W_1 = tl.dot(M_11, ks_1 - tl.dot(A_10, W_0, input_precision="tf32"), input_precision="tf32")
    tl.store(w_base + (16 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_1.to(w_dtype))

    ks_2 = k_2 * (beta_2 * tl.exp(G_2))[:, None]
    rhs_w2 = ks_2 - tl.dot(A_20, W_0, input_precision="tf32") - tl.dot(A_21, W_1, input_precision="tf32")
    W_2 = tl.dot(M_22, rhs_w2, input_precision="tf32")
    tl.store(w_base + (32 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_2.to(w_dtype))

    ks_3 = k_3 * (beta_3 * tl.exp(G_3))[:, None]
    rhs_w3 = (
        ks_3
        - tl.dot(A_30, W_0, input_precision="tf32")
        - tl.dot(A_31, W_1, input_precision="tf32")
        - tl.dot(A_32, W_2, input_precision="tf32")
    )
    W_3 = tl.dot(M_33, rhs_w3, input_precision="tf32")
    tl.store(w_base + (48 + o16)[:, None] * stride_wt + ok[None, :] * stride_wd, W_3.to(w_dtype))

    # Phase C: stream v_0..v_3 one tile at a time and store U_0..U_3 immediately
    u_base = U_ptr + h_v * stride_uh + c * stride_uc
    u_dtype = U_ptr.dtype.element_ty

    v_0 = tl.load(v_base + t_0[:, None] * stride_vn + ov[None, :] * stride_vd, mask=m_0[:, None], other=0.0).to(tl.float32)
    U_0 = tl.dot(M_00, v_0 * beta_0[:, None], input_precision="tf32")
    tl.store(u_base + o16[:, None] * stride_ut + ov[None, :] * stride_ud, U_0.to(u_dtype))

    v_1 = tl.load(v_base + t_1[:, None] * stride_vn + ov[None, :] * stride_vd, mask=m_1[:, None], other=0.0).to(tl.float32)
    U_1 = tl.dot(M_11, v_1 * beta_1[:, None] - tl.dot(A_10, U_0, input_precision="tf32"), input_precision="tf32")
    tl.store(u_base + (16 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_1.to(u_dtype))

    v_2 = tl.load(v_base + t_2[:, None] * stride_vn + ov[None, :] * stride_vd, mask=m_2[:, None], other=0.0).to(tl.float32)
    rhs_u2 = v_2 * beta_2[:, None] - tl.dot(A_20, U_0, input_precision="tf32") - tl.dot(A_21, U_1, input_precision="tf32")
    U_2 = tl.dot(M_22, rhs_u2, input_precision="tf32")
    tl.store(u_base + (32 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_2.to(u_dtype))

    v_3 = tl.load(v_base + t_3[:, None] * stride_vn + ov[None, :] * stride_vd, mask=m_3[:, None], other=0.0).to(tl.float32)
    rhs_u3 = (
        v_3 * beta_3[:, None]
        - tl.dot(A_30, U_0, input_precision="tf32")
        - tl.dot(A_31, U_1, input_precision="tf32")
        - tl.dot(A_32, U_2, input_precision="tf32")
    )
    U_3 = tl.dot(M_33, rhs_u3, input_precision="tf32")
    tl.store(u_base + (48 + o16)[:, None] * stride_ut + ov[None, :] * stride_ud, U_3.to(u_dtype))


@triton.jit
def _gdn_chunk_state_recurrence_ragged_kernel(
    K_ptr,
    GCum_ptr,
    W_ptr,
    U_ptr,
    CU_SEQLENS_ptr,
    REQ_CHUNK_START_ptr,
    InitState_ptr,
    HAll_ptr,
    VNewAll_ptr,
    FinalState_ptr,
    REQ_SNAP_START_ptr,
    SNAP_LOCAL_END_ptr,
    SNAP_PHYS_ptr,
    DST_SSM_ptr,
    stride_kn,
    stride_khk,
    stride_kd,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_wh,
    stride_wc,
    stride_wt,
    stride_wd,
    stride_uh,
    stride_uc,
    stride_ut,
    stride_ud,
    stride_sb,
    stride_shv,
    stride_sk,
    stride_sv,
    stride_hah,
    stride_hac,
    stride_hak,
    stride_hav,
    stride_vnh,
    stride_vnc,
    stride_vnt,
    stride_vnd,
    stride_fsb,
    stride_fshv,
    stride_fsk,
    stride_fsv,
    dst_ssm_stride_blk: tl.int64,
    dst_ssm_rows: tl.int64,
    H_V: tl.constexpr,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_SNAPS: tl.constexpr,
    SKIP_INVALID_PHYS: tl.constexpr,
    DIRECT_CHUNK_SNAP: tl.constexpr = False,
    BLOCK_DV: tl.constexpr = 64,
):
    """Ragged Stage 2: inter-chunk FP32 state recurrence + direct block-boundary FP32 SSM scatter."""
    b = tl.program_id(0)
    h_v = tl.program_id(1)
    dv_blk = tl.program_id(2)
    h_k = h_v // GQA_RATIO

    oc = tl.arange(0, 64)
    ok = tl.arange(0, D_K)
    ov = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)

    if HAS_INIT_STATE:
        h = tl.load(
            InitState_ptr
            + b * stride_sb
            + h_v * stride_shv
            + ok[:, None] * stride_sk
            + ov[None, :] * stride_sv
        ).to(tl.float32)
    else:
        h = tl.zeros((D_K, BLOCK_DV), dtype=tl.float32)

    req_start = tl.load(CU_SEQLENS_ptr + b)
    req_end = tl.load(CU_SEQLENS_ptr + b + 1)
    req_len = req_end - req_start
    c_start = tl.load(REQ_CHUNK_START_ptr + b)
    c_end = tl.load(REQ_CHUNK_START_ptr + b + 1)

    snap_ptr = tl.load(REQ_SNAP_START_ptr + b) if HAS_SNAPS else 0
    snap_end = tl.load(REQ_SNAP_START_ptr + b + 1) if HAS_SNAPS else 0

    k_base = K_ptr + req_start.to(tl.int64) * stride_kn + h_k * stride_khk
    gc_head = GCum_ptr + h_v * stride_gch
    w_head = W_ptr + h_v * stride_wh
    u_head = U_ptr + h_v * stride_uh
    ha_head = HAll_ptr + h_v * stride_hah
    vn_head = VNewAll_ptr + h_v * stride_vnh
    ha_dtype = HAll_ptr.dtype.element_ty
    vn_dtype = VNewAll_ptr.dtype.element_ty

    for c in range(c_start, c_end):
        t0 = (c - c_start) * 64
        tl.store(
            ha_head + c * stride_hac + ok[:, None] * stride_hak + ov[None, :] * stride_hav,
            h.to(ha_dtype),
        )

        b_w = tl.load(w_head + c * stride_wc + oc[:, None] * stride_wt + ok[None, :] * stride_wd)
        b_u = tl.load(u_head + c * stride_uc + oc[:, None] * stride_ut + ov[None, :] * stride_ud).to(tl.float32)
        b_v_new = b_u - tl.dot(b_w.to(tl.bfloat16), h.to(tl.bfloat16), out_dtype=tl.float32)
        tl.store(
            vn_head + c * stride_vnc + oc[:, None] * stride_vnt + ov[None, :] * stride_vnd,
            b_v_new.to(vn_dtype),
        )

        b_gc = tl.load(gc_head + c * stride_gcc + oc * stride_gct).to(tl.float32)
        g_last = tl.sum(tl.where(oc == 63, b_gc, 0.0), axis=0)

        t_offs = t0 + oc
        m_t = t_offs < req_len
        b_k = tl.load(
            k_base + t_offs[:, None] * stride_kn + ok[None, :] * stride_kd,
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1) + l2_eps)[:, None]
        b_k_decay = b_k * tl.exp(g_last - b_gc)[:, None]

        h_next = h * tl.exp(g_last) + tl.dot(
            tl.trans(b_k_decay.to(tl.bfloat16)),
            b_v_new.to(tl.bfloat16),
            out_dtype=tl.float32,
        )

        if HAS_SNAPS:
            if DIRECT_CHUNK_SNAP:
                sc_idx = snap_ptr + (c - c_start)
                if sc_idx < snap_end:
                    phys = tl.load(SNAP_PHYS_ptr + sc_idx).to(tl.int64)
                    valid_phys = True
                    if SKIP_INVALID_PHYS:
                        valid_phys = (phys > 0) & (phys < dst_ssm_rows)
                    if valid_phys:
                        tl.store(
                            DST_SSM_ptr
                            + phys * dst_ssm_stride_blk
                            + h_v * (D_K * D_V)
                            + ok[:, None] * D_V
                            + ov[None, :],
                            h_next,
                        )
            else:
                chunk_end = tl.minimum(req_len, t0 + 64)
                done_snaps = 0
                while snap_ptr < snap_end and done_snaps == 0:
                    s_end = tl.load(SNAP_LOCAL_END_ptr + snap_ptr)
                    if s_end > chunk_end:
                        done_snaps = 1
                    else:
                        phys = tl.load(SNAP_PHYS_ptr + snap_ptr).to(tl.int64)
                        valid_phys = True
                        if SKIP_INVALID_PHYS:
                            valid_phys = (phys > 0) & (phys < dst_ssm_rows)
                        if valid_phys:
                            m = s_end - t0
                            if s_end == chunk_end:
                                h_snap = h_next
                            else:
                                g_m = tl.sum(tl.where(oc == (m - 1), b_gc, 0.0), axis=0)
                                sub_m = oc < m
                                k_sub = tl.where(
                                    sub_m[:, None], b_k * tl.exp(g_m - b_gc)[:, None], 0.0
                                )
                                v_sub = tl.where(sub_m[:, None], b_v_new, 0.0)
                                h_snap = h * tl.exp(g_m) + tl.dot(
                                    tl.trans(k_sub.to(tl.bfloat16)),
                                    v_sub.to(tl.bfloat16),
                                    out_dtype=tl.float32,
                                )
                            tl.store(
                                DST_SSM_ptr
                                + phys * dst_ssm_stride_blk
                                + h_v * (D_K * D_V)
                                + ok[:, None] * D_V
                                + ov[None, :],
                                h_snap,
                            )
                        snap_ptr += 1

        h = h_next

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
def _gdn_chunk_output_ragged_kernel(
    Q_ptr,
    K_ptr,
    GCum_ptr,
    HAll_ptr,
    VNewAll_ptr,
    CU_SEQLENS_ptr,
    CHUNK_REQ_ptr,
    CHUNK_T0_ptr,
    Out_ptr,
    stride_qn,
    stride_qhk,
    stride_qd,
    stride_kn,
    stride_khk,
    stride_kd,
    stride_gch,
    stride_gcc,
    stride_gct,
    stride_hah,
    stride_hac,
    stride_hak,
    stride_hav,
    stride_vnh,
    stride_vnc,
    stride_vnt,
    stride_vnd,
    stride_on,
    stride_ohv,
    stride_od,
    scale,
    l2_eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """Ragged Stage 3: combine inter-chunk and causal intra-chunk outputs."""
    h_v = tl.program_id(0)
    c = tl.program_id(1)
    h_k = h_v // GQA_RATIO

    b = tl.load(CHUNK_REQ_ptr + c)
    t0 = tl.load(CHUNK_T0_ptr + c)
    req_start = tl.load(CU_SEQLENS_ptr + b)
    req_end = tl.load(CU_SEQLENS_ptr + b + 1)
    req_len = req_end - req_start

    oc = tl.arange(0, 64)
    ok = tl.arange(0, D_K)
    ov = tl.arange(0, D_V)

    t_offs = t0 + oc
    m_t = t_offs < req_len
    rs64 = req_start.to(tl.int64)
    q_base = Q_ptr + rs64 * stride_qn + h_k * stride_qhk
    k_base = K_ptr + rs64 * stride_kn + h_k * stride_khk
    out_base = Out_ptr + rs64 * stride_on + h_v * stride_ohv

    b_q = tl.load(
        q_base + t_offs[:, None] * stride_qn + ok[None, :] * stride_qd,
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)
    b_k = tl.load(
        k_base + t_offs[:, None] * stride_kn + ok[None, :] * stride_kd,
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)

    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, axis=1) + l2_eps)[:, None]
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1) + l2_eps)[:, None]

    b_gc = tl.load(
        GCum_ptr + h_v * stride_gch + c * stride_gcc + oc * stride_gct
    ).to(tl.float32)
    b_h = tl.load(
        HAll_ptr
        + h_v * stride_hah
        + c * stride_hac
        + ok[:, None] * stride_hak
        + ov[None, :] * stride_hav
    )
    b_v_new = tl.load(
        VNewAll_ptr
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
        out_base + t_offs[:, None] * stride_on + ov[None, :] * stride_od,
        o.to(Out_ptr.dtype.element_ty),
        mask=m_t[:, None],
    )


def gdn_chunk_prefill_ragged_with_snapshots_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_req: torch.Tensor,
    chunk_t0: torch.Tensor,
    req_chunk_start: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    req_snap_start: Optional[torch.Tensor] = None,
    snap_local_end: Optional[torch.Tensor] = None,
    snap_phys: Optional[torch.Tensor] = None,
    ssm_dst_pool: Optional[torch.Tensor] = None,
    skip_invalid_phys: bool = True,
    snaps_direct_chunk64: bool = False,
    output_final_states: bool = False,
    l2_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Packed ragged 3-stage Triton chunked prefill (CHUNK_SIZE=64) with exact FP32 SSM snapshots."""
    N, num_k_heads, d_k = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    num_v_heads, d_v = int(v.shape[1]), int(v.shape[2])
    bsz = int(cu_seqlens.numel()) - 1
    device = v.device

    if N == 0 or bsz == 0:
        out_empty = torch.empty((N, num_v_heads, d_v), dtype=v.dtype, device=device)
        fs_empty = (
            torch.zeros((bsz, num_v_heads, d_k, d_v), dtype=torch.float32, device=device)
            if output_final_states
            else None
        )
        return out_empty, fs_empty

    if (
        not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda)
        or d_k != 64
        or d_v != 64
        or num_v_heads % num_k_heads != 0
    ):
        # Fallback for CPU / non-64 head dims
        out = torch.empty((N, num_v_heads, d_v), dtype=v.dtype, device=device)
        cu_list = cu_seqlens.tolist()
        fs_list = []
        has_snaps = (
            req_snap_start is not None
            and snap_local_end is not None
            and snap_phys is not None
            and ssm_dst_pool is not None
        )
        req_ss_list = req_snap_start.tolist() if has_snaps and req_snap_start is not None else None
        s_end_list = snap_local_end.tolist() if has_snaps and snap_local_end is not None else None
        s_phys_list = snap_phys.tolist() if has_snaps and snap_phys is not None else None
        for b in range(bsz):
            s, e = int(cu_list[b]), int(cu_list[b + 1])
            init_b = initial_state[b : b + 1].clone() if initial_state is not None else None
            o_b, fs_b = gdn_recurrent_ref(
                q[s:e].unsqueeze(0),
                k[s:e].unsqueeze(0),
                v[s:e].unsqueeze(0),
                g[s:e].unsqueeze(0),
                beta[s:e].unsqueeze(0),
                initial_state=init_b,
                output_final_state=True,
                l2_eps=l2_eps,
            )
            out[s:e].copy_(o_b.squeeze(0))
            if output_final_states and fs_b is not None:
                fs_list.append(fs_b)
            if has_snaps and req_ss_list is not None and s_end_list is not None and s_phys_list is not None and ssm_dst_pool is not None:
                ss0, ss1 = int(req_ss_list[b]), int(req_ss_list[b + 1])
                if ss1 > ss0:
                    prev_e = 0
                    cur_st = init_b
                    for idx_s in range(ss0, ss1):
                        end_loc = int(s_end_list[idx_s])
                        phys = int(s_phys_list[idx_s])
                        if end_loc > prev_e:
                            _, cur_st = gdn_recurrent_ref(
                                q[s + prev_e : s + end_loc].unsqueeze(0),
                                k[s + prev_e : s + end_loc].unsqueeze(0),
                                v[s + prev_e : s + end_loc].unsqueeze(0),
                                g[s + prev_e : s + end_loc].unsqueeze(0),
                                beta[s + prev_e : s + end_loc].unsqueeze(0),
                                initial_state=cur_st,
                                output_final_state=True,
                                l2_eps=l2_eps,
                            )
                            prev_e = end_loc
                        if cur_st is not None and (not skip_invalid_phys or (0 < phys < ssm_dst_pool.shape[0])):
                            ssm_dst_pool[phys].view(-1)[: num_v_heads * d_k * d_v].copy_(cur_st.reshape(-1))
        fs_out = torch.cat(fs_list, dim=0) if output_final_states and fs_list else None
        return out, fs_out

    gqa_ratio = num_v_heads // num_k_heads
    num_chunks = int(chunk_req.numel())
    g_cum = torch.empty((num_v_heads, num_chunks, 64), dtype=torch.float32, device=device)
    w_buf = torch.empty((num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    u_buf = torch.empty((num_v_heads, num_chunks, 64, 64), dtype=torch.float32, device=device)
    h_all = torch.empty((num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    v_new_all = torch.empty((num_v_heads, num_chunks, 64, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty((N, num_v_heads, 64), dtype=v.dtype, device=device)

    final_states = (
        torch.empty((bsz, num_v_heads, 64, 64), dtype=torch.float32, device=device)
        if output_final_states
        else None
    )

    _gdn_chunk_prep_wy_ragged_kernel[(num_v_heads, num_chunks)](
        k,
        v,
        g,
        beta,
        cu_seqlens,
        chunk_req,
        chunk_t0,
        g_cum,
        w_buf,
        u_buf,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        beta.stride(0),
        beta.stride(1),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        w_buf.stride(0),
        w_buf.stride(1),
        w_buf.stride(2),
        w_buf.stride(3),
        u_buf.stride(0),
        u_buf.stride(1),
        u_buf.stride(2),
        u_buf.stride(3),
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        num_warps=(1 if num_v_heads * num_chunks >= 1024 else (2 if num_chunks >= 8 else 4)),
        num_stages=(1 if (num_v_heads * num_chunks >= 1024 or (bsz >= 4 and num_chunks >= 128)) else 2),
    )

    has_init = initial_state is not None
    has_snaps = (
        req_snap_start is not None
        and snap_local_end is not None
        and snap_phys is not None
        and ssm_dst_pool is not None
        and int(snap_local_end.numel()) > 0
    )
    init_arg = initial_state if has_init else out
    fs_arg = final_states if output_final_states else out
    rss_arg = req_snap_start if has_snaps else cu_seqlens
    sle_arg = snap_local_end if has_snaps else cu_seqlens
    sp_arg = snap_phys if has_snaps else cu_seqlens
    dst_arg = ssm_dst_pool if has_snaps else out
    dst_stride_blk = (
        int(ssm_dst_pool.view(-1).shape[0] // ssm_dst_pool.shape[0])
        if has_snaps and ssm_dst_pool is not None
        else 0
    )
    if has_snaps and ssm_dst_pool is not None:
        elem_scale = ssm_dst_pool.element_size() // 4
        if elem_scale <= 0:
            raise ValueError(f"ssm_dst_pool dtype {ssm_dst_pool.dtype} must be >= 4 bytes")
        dst_stride_blk = int(ssm_dst_pool.stride(0)) * elem_scale

    direct_chunk_snap = bool(has_snaps and snaps_direct_chunk64)
    if has_snaps and not direct_chunk_snap:
        block_dv_s2 = (
            16
            if (bsz * num_v_heads <= 128 and num_chunks >= 32)
            else (32 if (bsz * num_v_heads <= 512 and num_chunks >= 32) else 64)
        )
        ns_s2 = 2
    else:
        block_dv_s2 = (
            16
            if (bsz * num_v_heads <= 64 and num_chunks >= 32)
            else (32 if (bsz * num_v_heads <= 512 and num_chunks >= 32) else 64)
        )
        ns_s2 = 3 if (not has_snaps or bsz * num_v_heads <= 64) else 2
    _gdn_chunk_state_recurrence_ragged_kernel[(bsz, num_v_heads, 64 // block_dv_s2)](
        k,
        g_cum,
        w_buf,
        u_buf,
        cu_seqlens,
        req_chunk_start,
        init_arg,
        h_all,
        v_new_all,
        fs_arg,
        rss_arg,
        sle_arg,
        sp_arg,
        dst_arg,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        w_buf.stride(0),
        w_buf.stride(1),
        w_buf.stride(2),
        w_buf.stride(3),
        u_buf.stride(0),
        u_buf.stride(1),
        u_buf.stride(2),
        u_buf.stride(3),
        initial_state.stride(0) if has_init else 0,
        initial_state.stride(1) if has_init else 0,
        initial_state.stride(2) if has_init else 0,
        initial_state.stride(3) if has_init else 0,
        h_all.stride(0),
        h_all.stride(1),
        h_all.stride(2),
        h_all.stride(3),
        v_new_all.stride(0),
        v_new_all.stride(1),
        v_new_all.stride(2),
        v_new_all.stride(3),
        final_states.stride(0) if output_final_states else 0,
        final_states.stride(1) if output_final_states else 0,
        final_states.stride(2) if output_final_states else 0,
        final_states.stride(3) if output_final_states else 0,
        dst_stride_blk,
        int(ssm_dst_pool.shape[0]) if has_snaps and ssm_dst_pool is not None else 0,
        num_v_heads,
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        HAS_INIT_STATE=has_init,
        STORE_FINAL_STATE=output_final_states,
        HAS_SNAPS=has_snaps,
        SKIP_INVALID_PHYS=skip_invalid_phys,
        DIRECT_CHUNK_SNAP=direct_chunk_snap,
        BLOCK_DV=block_dv_s2,
        num_warps=2 if (block_dv_s2 < 64 or bsz >= 4) else 4,
        num_stages=ns_s2,
    )

    _gdn_chunk_output_ragged_kernel[(num_v_heads, num_chunks)](
        q,
        k,
        g_cum,
        h_all,
        v_new_all,
        cu_seqlens,
        chunk_req,
        chunk_t0,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        g_cum.stride(0),
        g_cum.stride(1),
        g_cum.stride(2),
        h_all.stride(0),
        h_all.stride(1),
        h_all.stride(2),
        h_all.stride(3),
        v_new_all.stride(0),
        v_new_all.stride(1),
        v_new_all.stride(2),
        v_new_all.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        float(64 ** -0.5),
        float(l2_eps),
        GQA_RATIO=gqa_ratio,
        D_K=64,
        D_V=64,
        num_warps=8 if num_chunks >= 64 else 4,
        num_stages=1 if (bsz >= 4 and num_chunks >= 128) else 2,
    )

    return out, final_states


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
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_consumer = is_consumer
        self.publishes_shared = publishes_shared
        self.source_layer_idx = source_layer_idx
        self.enable_yoco_decode = config.enable_yoco_decode

        self.tp_size = max(1, int(tp_size))
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce
        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)
        if config.linear_num_key_heads % self.tp_size != 0:
            raise ValueError(
                f"linear_num_key_heads ({config.linear_num_key_heads}) not divisible by tp_size ({self.tp_size})"
            )
        if config.linear_num_value_heads % self.tp_size != 0:
            raise ValueError(
                f"linear_num_value_heads ({config.linear_num_value_heads}) not divisible by tp_size ({self.tp_size})"
            )

        self.hidden_size = config.hidden_size
        self.global_num_k_heads = config.linear_num_key_heads
        self.global_num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads // self.tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.num_v_heads = config.linear_num_value_heads // self.tp_size
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
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
    ) -> "Qwen4_5GatedDeltaNet":
        plan = sharing_plan if sharing_plan is not None else YoCoSharingPlan.from_config(config)
        return cls(
            config=config,
            layer_idx=layer_idx,
            is_consumer=plan.is_gdn_consumer(layer_idx),
            publishes_shared=plan.is_gdn_source(layer_idx),
            source_layer_idx=plan.source_for(layer_idx),
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            dp_attn=dp_attn,
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
        skip_allreduce: bool = False,
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
            z_view = self._view_z_strided(qz, is_consumer=True)
            if (
                not self.gdn_no_q_conv
                and seq_len == 1
                and state is not None
                and self._can_use_fused_unpack_conv(qz)
            ):
                q = torch.empty(
                    (bsz, 1, self.num_k_heads, self.head_k_dim),
                    dtype=qz.dtype,
                    device=qz.device,
                )
                self._launch_fused_unpack_conv_decode(qz, state.conv_state, q, q, q, is_consumer=True)
            elif (
                not self.gdn_no_q_conv
                and seq_len > 1
                and bsz > 0
                and self._can_use_fused_unpack_conv(qz)
                and self._can_use_conv_state_pool(state.conv_state if state is not None else None)
            ):
                q, _, _ = self._launch_fused_unpack_conv_seq(
                    qz, state.conv_state if state is not None else None, is_consumer=True
                )
            else:
                q_flat, z_view = self._unpack_qz_consumer(qz)
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

            out = self.norm(core_out, z_view)
            out = self.out_proj(out.reshape(bsz, seq_len, self.value_dim))
            out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)
            return out, None

        # Normal / YoCo Source layer
        assert self.in_proj_ba is not None and self.A_log is not None and self.dt_bias is not None
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)

        if (
            seq_len == 1
            and state is not None
            and self._can_use_fused_unpack_conv(qkvz)
        ):
            q = torch.empty((bsz, 1, self.num_k_heads, self.head_k_dim), dtype=qkvz.dtype, device=qkvz.device)
            k = torch.empty((bsz, 1, self.num_k_heads, self.head_k_dim), dtype=qkvz.dtype, device=qkvz.device)
            v = torch.empty((bsz, 1, self.num_v_heads, self.head_v_dim), dtype=qkvz.dtype, device=qkvz.device)
            self._launch_fused_unpack_conv_decode(qkvz, state.conv_state, q, k, v, is_consumer=False)
            z_view = self._view_z_strided(qkvz, is_consumer=False)
        elif (
            seq_len > 1
            and bsz > 0
            and self._can_use_fused_unpack_conv(qkvz)
            and self._can_use_conv_state_pool(state.conv_state if state is not None else None)
        ):
            q, k, v = self._launch_fused_unpack_conv_seq(
                qkvz, state.conv_state if state is not None else None, is_consumer=False
            )
            assert k is not None and v is not None
            z_view = self._view_z_strided(qkvz, is_consumer=False)
        else:
            mixed_qkv, z_view = self._unpack_qkvz_normal(qkvz)
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

        out = self.norm(core_out, z_view)
        out = self.out_proj(out.reshape(bsz, seq_len, self.value_dim))
        out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)

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

    def _view_z_strided(self, projected: torch.Tensor, *, is_consumer: bool) -> torch.Tensor:
        v_group = self.gqa_ratio * self.head_v_dim
        if is_consumer:
            grouped = projected.unflatten(-1, (self.num_k_heads, self.head_k_dim + v_group))
            z_g = grouped[..., self.head_k_dim :]
        else:
            grouped = projected.unflatten(
                -1, (self.num_k_heads, self.head_k_dim * 2 + v_group * 2)
            )
            z_g = grouped[..., self.head_k_dim * 2 + v_group :]
        return z_g.unflatten(-1, (self.gqa_ratio, self.head_v_dim))

    def _can_use_fused_unpack_conv(self, projected: torch.Tensor) -> bool:
        return (
            projected.is_cuda
            and self.conv1d.weight.is_cuda
            and self.conv_kernel_size == 4
            and self.head_k_dim == 64
            and self.head_v_dim == 64
            and projected.stride(-1) == 1
            and self.conv1d.weight.is_contiguous()
        )

    def _launch_fused_unpack_conv_decode(
        self,
        projected: torch.Tensor,
        conv_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_consumer: bool,
    ) -> None:
        bsz = int(projected.shape[0])
        if bsz == 0:
            return
        v_group = self.gqa_ratio * self.head_v_dim
        group_stride = (self.head_k_dim + v_group) if is_consumer else (2 * self.head_k_dim + 2 * v_group)
        grid = (bsz, self.conv_dim // 64)
        rt_args = (
            projected, self.conv1d.weight, conv_state, q, k, v,
            projected.stride(0), projected.stride(-1),
            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            q.stride(0), k.stride(0), v.stride(0),
            projected, projected, 0, 0, 1, 1, 1, 0,
        )
        ckey = (
            projected.device.index, 4, projected.dtype, self.conv1d.weight.dtype, conv_state.dtype, q.dtype, k.dtype, v.dtype,
            tuple(_triton_int_spec(x) for x in rt_args[6:14]),
            self.key_dim, self.gqa_ratio, group_stride, is_consumer, self.gdn_conv_residual, False,
        )
        ptr_or = projected.data_ptr() | self.conv1d.weight.data_ptr() | conv_state.data_ptr() | q.data_ptr() | k.data_ptr() | v.data_ptr()
        triton_fast_launch(
            _GDN_TRITON_CACHE, ckey, _gdn_fused_unpack_conv1d_decode_k4_kernel,
            bsz, self.conv_dim // 64, 1, ptr_or, rt_args,
            {
                "KEY_DIM": self.key_dim, "D_K": 64, "D_V": 64, "GQA_RATIO": self.gqa_ratio,
                "GROUP_STRIDE": group_stride, "IS_CONSUMER": is_consumer,
                "RESIDUAL": self.gdn_conv_residual, "BLOCK_C": 64, "IS_OPAQUE_STATE": False,
            },
            num_warps=2,
        )

    def _launch_fused_unpack_conv_decode_opaque(
        self,
        projected: torch.Tensor,
        pool_2d: torch.Tensor,
        block_table: torch.Tensor,
        seq_before: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        page_size: int,
        pool_col_offset: int,
        is_consumer: bool,
    ) -> None:
        bsz = int(projected.shape[0])
        if bsz == 0:
            return
        v_group = self.gqa_ratio * self.head_v_dim
        group_stride = (self.head_k_dim + v_group) if is_consumer else (2 * self.head_k_dim + 2 * v_group)
        spb = int(pool_2d.stride(0))
        sbt0 = int(block_table.stride(0))
        sbt1 = int(block_table.stride(1))
        pco = int(pool_col_offset)
        rt_args = (
            projected, self.conv1d.weight, pool_2d, q, k, v,
            projected.stride(0), projected.stride(-1),
            spb, 3, 1, q.stride(0), k.stride(0), v.stride(0),
            block_table, seq_before, sbt0, sbt1,
            int(page_size), int(block_table.shape[1]), int(pool_2d.shape[0]), pco,
        )
        ckey = (
            projected.device.index, 5, projected.dtype, self.conv1d.weight.dtype, pool_2d.dtype, q.dtype, k.dtype, v.dtype,
            block_table.dtype, seq_before.dtype,
            tuple(_triton_int_spec(x) for x in (rt_args[6:14] + (sbt0, sbt1, pco))),
            self.key_dim, self.gqa_ratio, group_stride, is_consumer, self.gdn_conv_residual, True,
        )
        ptr_or = (
            projected.data_ptr() | self.conv1d.weight.data_ptr() | pool_2d.data_ptr()
            | q.data_ptr() | k.data_ptr() | v.data_ptr() | block_table.data_ptr() | seq_before.data_ptr()
        )
        triton_fast_launch(
            _GDN_TRITON_CACHE, ckey, _gdn_fused_unpack_conv1d_decode_k4_kernel,
            bsz, self.conv_dim // 64, 1, ptr_or, rt_args,
            {
                "KEY_DIM": self.key_dim, "D_K": 64, "D_V": 64, "GQA_RATIO": self.gqa_ratio,
                "GROUP_STRIDE": group_stride, "IS_CONSUMER": is_consumer,
                "RESIDUAL": self.gdn_conv_residual, "BLOCK_C": 64, "IS_OPAQUE_STATE": True,
            },
            num_warps=2,
        )

    def forward_decode_opaque(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_pool_2d: Optional[torch.Tensor],
        bt_conv: Optional[torch.Tensor],
        tpb_conv: int,
        conv_col_offset: int,
        ssm_pool_2d: Optional[torch.Tensor],
        bt_ssm: Optional[torch.Tensor],
        tpb_ssm: int,
        ssm_col_offset: int,
        seq_before: torch.Tensor,
        shared: Optional[SharedGDNInputs] = None,
        enable_yoco_decode: Optional[bool] = None,
        skip_allreduce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SharedGDNInputs]]:
        """Zero-copy T=1 paged decode path reading & updating opaque pools directly."""
        bsz, seq_len, _ = hidden_states.shape
        yoco_decode_on = self.enable_yoco_decode if enable_yoco_decode is None else enable_yoco_decode
        if self.is_consumer:
            if shared is None:
                raise ValueError(f"Consumer GDN layer {self.layer_idx} requires SharedGDNInputs")
            qz = self.in_proj_qkvz(hidden_states)
            z_view = self._view_z_strided(qz, is_consumer=True)
            if (
                not self.gdn_no_q_conv
                and conv_pool_2d is not None
                and bt_conv is not None
                and self._can_use_fused_unpack_conv(qz)
            ):
                q = torch.empty((bsz, 1, self.num_k_heads, self.head_k_dim), dtype=qz.dtype, device=qz.device)
                self._launch_fused_unpack_conv_decode_opaque(
                    qz, conv_pool_2d, bt_conv, seq_before, q, q, q,
                    page_size=tpb_conv, pool_col_offset=conv_col_offset, is_consumer=True,
                )
            else:
                q_flat, z_view = self._unpack_qz_consumer(qz)
                q = q_flat.reshape(bsz, 1, self.num_k_heads, self.head_k_dim)
            if yoco_decode_on and shared.ssm_state is not None:
                core_out = yoco_gdn_decode_readout(q, shared.ssm_state, head_k_dim=self.head_k_dim)
            elif ssm_pool_2d is not None and bt_ssm is not None:
                core_out, _ = gdn_recurrent_decode_opaque_triton(
                    q, shared.key, shared.value, shared.g, shared.beta,
                    ssm_pool_2d, bt_ssm, seq_before, page_size=tpb_ssm, pool_col_offset=ssm_col_offset,
                    output_final_state=False,
                )
            else:
                core_out, _ = gdn_recurrent_triton(
                    q, shared.key, shared.value, shared.g, shared.beta,
                    initial_state=None, output_final_state=False, inplace_state=False,
                )
            out = self.norm(core_out, z_view)
            out = self.out_proj(out.reshape(bsz, 1, self.value_dim))
            return self._maybe_all_reduce(out, skip_allreduce=skip_allreduce), None

        assert self.in_proj_ba is not None and self.A_log is not None and self.dt_bias is not None
        assert conv_pool_2d is not None and bt_conv is not None and ssm_pool_2d is not None and bt_ssm is not None
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        q = torch.empty((bsz, 1, self.num_k_heads, self.head_k_dim), dtype=qkvz.dtype, device=qkvz.device)
        k = torch.empty((bsz, 1, self.num_k_heads, self.head_k_dim), dtype=qkvz.dtype, device=qkvz.device)
        v = torch.empty((bsz, 1, self.num_v_heads, self.head_v_dim), dtype=qkvz.dtype, device=qkvz.device)
        self._launch_fused_unpack_conv_decode_opaque(
            qkvz, conv_pool_2d, bt_conv, seq_before, q, k, v,
            page_size=tpb_conv, pool_col_offset=conv_col_offset, is_consumer=False,
        )
        z_view = self._view_z_strided(qkvz, is_consumer=False)
        g, beta = compute_gdn_gating(ba, self.A_log, self.dt_bias, num_k_heads=self.num_k_heads, num_v_heads=self.num_v_heads)
        core_out, final_ssm = gdn_recurrent_decode_opaque_triton(
            q, k, v, g, beta, ssm_pool_2d, bt_ssm, seq_before,
            page_size=tpb_ssm, pool_col_offset=ssm_col_offset,
            output_final_state=self.publishes_shared,
        )
        out = self.norm(core_out, z_view)
        out = self.out_proj(out.reshape(bsz, 1, self.value_dim))
        out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)
        published = SharedGDNInputs(key=k, value=v, g=g, beta=beta, ssm_state=final_ssm) if self.publishes_shared else None
        return out, published

    @staticmethod
    def _can_use_conv_state_pool(conv_state: Optional[torch.Tensor]) -> bool:
        return conv_state is None or (
            conv_state.is_cuda
            and conv_state.ndim == 3
            and conv_state.shape[-1] == 3
            and conv_state.stride(-1) == 1
            and conv_state.stride(-2) == 3
        )

    def _launch_fused_unpack_conv_ragged(
        self,
        projected_2d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tile_req: torch.Tensor,
        tile_t0: torch.Tensor,
        *,
        is_consumer: bool,
        init_conv_state: Optional[torch.Tensor] = None,
        snap_tok_idx: Optional[torch.Tensor] = None,
        snap_req_idx: Optional[torch.Tensor] = None,
        conv_dst_pool: Optional[torch.Tensor] = None,
        conv_snap_phys: Optional[torch.Tensor] = None,
        conv_dst_stride_row: int = 0,
        conv_dst_col_offset: int = 0,
        skip_invalid_phys: bool = True,
        block_t: int = 32,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run fused unpack + K=4 causal dwconv1d (+ optional 3-token state scatter) on packed [N, P]."""
        N = int(projected_2d.shape[0])
        dtype = projected_2d.dtype
        device = projected_2d.device
        q = torch.empty((N, self.num_k_heads, self.head_k_dim), dtype=dtype, device=device)
        if is_consumer:
            k = None
            v = None
            k_arg, v_arg = q, q
        else:
            k = torch.empty((N, self.num_k_heads, self.head_k_dim), dtype=dtype, device=device)
            v = torch.empty((N, self.num_v_heads, self.head_v_dim), dtype=dtype, device=device)
            k_arg, v_arg = k, v

        has_init_c = init_conv_state is not None
        init_c_arg = init_conv_state if has_init_c else q
        v_group = self.gqa_ratio * self.head_v_dim
        group_stride = (
            (self.head_k_dim + v_group)
            if is_consumer
            else (2 * self.head_k_dim + 2 * v_group)
        )
        num_tiles = int(tile_req.numel())
        if num_tiles > 0:
            sub_num = 2 if block_t % 2 == 0 else 1
            sub_t = block_t // sub_num
            rt_args = (
                projected_2d, self.conv1d.weight, init_c_arg, q, k_arg, v_arg,
                cu_seqlens, tile_req, tile_t0,
                projected_2d.stride(0), projected_2d.stride(1),
                init_conv_state.stride(0) if has_init_c else 0,
                init_conv_state.stride(1) if has_init_c else 0,
                init_conv_state.stride(2) if has_init_c else 0,
                q.stride(0), k_arg.stride(0), v_arg.stride(0),
            )
            ckey = (
                projected_2d.device.index, 7, projected_2d.dtype, self.conv1d.weight.dtype, init_c_arg.dtype,
                q.dtype, k_arg.dtype, v_arg.dtype, cu_seqlens.dtype, tile_req.dtype, tile_t0.dtype,
                tuple(_triton_int_spec(x) for x in rt_args[9:]),
                self.key_dim, self.gqa_ratio, group_stride, is_consumer, has_init_c, self.gdn_conv_residual, sub_t, sub_num,
            )
            ptr_or = (
                projected_2d.data_ptr() | self.conv1d.weight.data_ptr() | init_c_arg.data_ptr()
                | q.data_ptr() | k_arg.data_ptr() | v_arg.data_ptr() | cu_seqlens.data_ptr() | tile_req.data_ptr() | tile_t0.data_ptr()
            )
            triton_fast_launch(
                _GDN_TRITON_CACHE, ckey, _gdn_fused_unpack_conv1d_ragged_kernel,
                num_tiles * sub_num, self.conv_dim // 64, 1, ptr_or, rt_args,
                {
                    "KEY_DIM": self.key_dim, "D_K": 64, "D_V": 64, "GQA_RATIO": self.gqa_ratio,
                    "GROUP_STRIDE": group_stride, "IS_CONSUMER": is_consumer, "HAS_INIT": has_init_c,
                    "RESIDUAL": self.gdn_conv_residual, "SUB_T": sub_t, "SUB_NUM": sub_num, "BLOCK_C": 64,
                },
                num_warps=4,
            )
        if (
            snap_tok_idx is not None
            and snap_req_idx is not None
            and conv_dst_pool is not None
            and conv_snap_phys is not None
            and int(snap_tok_idx.numel()) > 0
        ):
            row_st = (
                int(conv_dst_stride_row)
                if conv_dst_stride_row > 0
                else int(conv_dst_pool.stride(0))
            )
            _gdn_scatter_qkvz_conv_snapshots_ragged_kernel[
                (int(snap_tok_idx.numel()), self.conv_dim // 64)
            ](
                projected_2d,
                init_c_arg,
                cu_seqlens,
                snap_tok_idx,
                snap_req_idx,
                conv_snap_phys,
                conv_dst_pool,
                projected_2d.stride(0),
                projected_2d.stride(1),
                init_conv_state.stride(0) if has_init_c else 0,
                init_conv_state.stride(1) if has_init_c else 0,
                init_conv_state.stride(2) if has_init_c else 0,
                row_st,
                int(conv_dst_col_offset),
                int(conv_dst_pool.shape[0]),
                KEY_DIM=self.key_dim,
                D_K=64,
                D_V=64,
                GQA_RATIO=self.gqa_ratio,
                GROUP_STRIDE=group_stride,
                IS_CONSUMER=is_consumer,
                HAS_INIT=has_init_c,
                SKIP_INVALID_PHYS=skip_invalid_phys,
                BLOCK_C=64,
                num_warps=2,
            )
        return q, k, v

    def _launch_fused_unpack_conv_seq(
        self,
        projected: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        *,
        is_consumer: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run fused unpack + K=4 causal dwconv1d (+ in-place conv_state update) on [B, T, P]."""
        bsz, seq_len, _ = projected.shape
        device = projected.device
        if (
            projected.stride(-1) == 1
            and (bsz == 1 or projected.stride(0) == seq_len * projected.stride(1))
        ):
            proj_2d = projected.view(bsz * seq_len, projected.shape[-1])
        else:
            proj_2d = projected.contiguous().view(bsz * seq_len, -1)
        block_t = 32
        nt_per_req = (seq_len + block_t - 1) // block_t
        cu_seqlens = torch.arange(
            0, (bsz + 1) * seq_len, seq_len, device=device, dtype=torch.int32
        )
        tile_req = torch.arange(bsz, device=device, dtype=torch.int32).repeat_interleave(
            nt_per_req
        )
        tile_t0 = (
            torch.arange(nt_per_req, device=device, dtype=torch.int32) * block_t
        ).repeat(bsz)
        init_c = conv_state.clone() if (conv_state is not None and seq_len < 3) else conv_state
        snap_tok = (cu_seqlens[1:] - 1) if conv_state is not None else None
        req_idx = (
            torch.arange(bsz, device=device, dtype=torch.int32)
            if conv_state is not None
            else None
        )
        q, k, v = self._launch_fused_unpack_conv_ragged(
            proj_2d,
            cu_seqlens,
            tile_req,
            tile_t0,
            is_consumer=is_consumer,
            init_conv_state=init_c,
            snap_tok_idx=snap_tok,
            snap_req_idx=req_idx,
            conv_dst_pool=conv_state,
            conv_snap_phys=req_idx,
            conv_dst_stride_row=int(conv_state.stride(0)) if conv_state is not None else 0,
            conv_dst_col_offset=0,
            skip_invalid_phys=False,
            block_t=block_t,
        )
        q = q.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        if k is not None:
            k = k.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        if v is not None:
            v = v.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        return q, k, v

    def _launch_gdn_verify_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        init_ssm_state: Optional[torch.Tensor],
        stage_ssm_state: Optional[torch.Tensor],
        *,
        verify_w: int,
    ) -> torch.Tensor:
        """Run split-D_V Triton GDN verify scan across all speculative steps."""
        bsz = int(q.shape[0])
        core_out = torch.empty(
            (bsz, verify_w, self.num_v_heads, self.head_v_dim),
            dtype=q.dtype,
            device=q.device,
        )
        has_init_s = init_ssm_state is not None
        has_stage_s = stage_ssm_state is not None
        is_contig_init = (not has_init_s) or (
            init_ssm_state.stride(-1) == 1 and init_ssm_state.stride(-2) == self.head_v_dim
        )
        is_contig_stage = (not has_stage_s) or (
            stage_ssm_state.stride(-1) == 1 and stage_ssm_state.stride(-2) == self.head_v_dim
        )
        use_fast_scan = (
            bsz >= 2 and is_contig_init and is_contig_stage and self.head_v_dim % 32 == 0
        )
        block_dv = (
            32
            if (use_fast_scan or (bsz == 1 and self.head_v_dim % 32 == 0))
            else (16 if self.head_v_dim % 16 == 0 else self.head_v_dim)
        )
        init_arg = init_ssm_state if has_init_s else core_out
        stg_arg = stage_ssm_state if has_stage_s else core_out
        nw_vs = 1 if bsz >= 2 else (8 if block_dv <= 16 else 2)
        rt_args = (
            q, k, v, g, beta, init_arg, stg_arg, core_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            init_ssm_state.stride(0) if has_init_s else 0,
            init_ssm_state.stride(1) if has_init_s else 0,
            init_ssm_state.stride(2) if has_init_s else 0,
            init_ssm_state.stride(3) if has_init_s else 0,
            stage_ssm_state.stride(0) if has_stage_s else 0,
            stage_ssm_state.stride(1) if has_stage_s else 0,
            stage_ssm_state.stride(2) if has_stage_s else 0,
            stage_ssm_state.stride(3) if has_stage_s else 0,
            stage_ssm_state.stride(4) if has_stage_s else 0,
            core_out.stride(0), core_out.stride(1), core_out.stride(2), core_out.stride(3),
            float(self.head_k_dim ** -0.5), 1e-6,
        )
        ckey = (
            q.device.index, 6, q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, init_arg.dtype, stg_arg.dtype,
            tuple(_triton_int_spec(x) for x in rt_args[8:39]),
            verify_w, self.gqa_ratio, self.head_k_dim, self.head_v_dim, block_dv,
            has_init_s, has_stage_s, is_contig_init, is_contig_stage, use_fast_scan, nw_vs,
        )
        ptr_or = q.data_ptr() | k.data_ptr() | v.data_ptr() | g.data_ptr() | beta.data_ptr() | init_arg.data_ptr() | stg_arg.data_ptr() | core_out.data_ptr()
        triton_fast_launch(
            _GDN_TRITON_CACHE, ckey, _gdn_verify_scan_kernel,
            bsz, self.num_v_heads, self.head_v_dim // block_dv, ptr_or, rt_args,
            {
                "V_STEPS": verify_w, "GQA_RATIO": self.gqa_ratio, "D_K": self.head_k_dim, "D_V": self.head_v_dim,
                "BLOCK_DV": block_dv, "HAS_INIT_STATE": has_init_s, "HAS_STAGE_STATE": has_stage_s,
                "IS_CONTIG_INIT": is_contig_init, "IS_CONTIG_STAGE": is_contig_stage, "EVICT_ST": use_fast_scan,
            },
            num_warps=nw_vs,
        )
        return core_out

    def forward_verify(
        self,
        hidden_states: torch.Tensor,
        *,
        init_conv_state: Optional[torch.Tensor] = None,
        init_ssm_state: Optional[torch.Tensor] = None,
        stage_conv_state: Optional[torch.Tensor] = None,
        stage_ssm_state: Optional[torch.Tensor] = None,
        shared: Optional[SharedGDNInputs] = None,
        enable_yoco_decode: Optional[bool] = None,
        lazy_kvgb_buffers: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        skip_allreduce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SharedGDNInputs]]:
        """Single-pass batched V-step verify forward keeping SSM & conv state in registers."""
        bsz, verify_w, _ = int(hidden_states.shape[0]), int(hidden_states.shape[1]), int(hidden_states.shape[2])
        if bsz == 0 or verify_w == 0:
            return torch.empty_like(hidden_states), None
        yoco_decode_on = (
            self.enable_yoco_decode if enable_yoco_decode is None else bool(enable_yoco_decode)
        )
        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.is_consumer:
            if shared is None:
                raise ValueError(
                    f"Consumer GDN layer {self.layer_idx} requires SharedGDNInputs from source layer"
                )
            qz = self.in_proj_qkvz(hidden_states)
            z_view = self._view_z_strided(qz, is_consumer=True)
            if self.gdn_no_q_conv:
                q_g = qz.unflatten(-1, (self.num_k_heads, self.head_k_dim + self.gqa_ratio * self.head_v_dim))[
                    ..., : self.head_k_dim
                ]
                q = q_g.contiguous()
            elif self._can_use_fused_unpack_conv(qz) and init_conv_state is not None:
                q = torch.empty((bsz, verify_w, self.num_k_heads, self.head_k_dim), dtype=dtype, device=device)
                has_stage_c = stage_conv_state is not None
                sc_arg = stage_conv_state if has_stage_c else q
                v_group = self.gqa_ratio * self.head_v_dim
                _gdn_fused_unpack_conv1d_verify_k4_kernel[(bsz, self.conv_dim // 64)](
                    qz,
                    self.conv1d.weight,
                    init_conv_state,
                    sc_arg,
                    q,
                    q,
                    q,
                    qz.stride(0),
                    qz.stride(1),
                    qz.stride(2),
                    init_conv_state.stride(0),
                    init_conv_state.stride(1),
                    init_conv_state.stride(2),
                    stage_conv_state.stride(0) if has_stage_c else 0,
                    stage_conv_state.stride(1) if has_stage_c else 0,
                    stage_conv_state.stride(2) if has_stage_c else 0,
                    stage_conv_state.stride(3) if has_stage_c else 0,
                    q.stride(0),
                    q.stride(1),
                    q.stride(0),
                    q.stride(1),
                    q.stride(0),
                    q.stride(1),
                    V_STEPS=verify_w,
                    KEY_DIM=self.key_dim,
                    D_K=64,
                    D_V=64,
                    GQA_RATIO=self.gqa_ratio,
                    GROUP_STRIDE=self.head_k_dim + v_group,
                    IS_CONSUMER=True,
                    RESIDUAL=self.gdn_conv_residual,
                    HAS_STAGE=has_stage_c,
                    BLOCK_C=64,
                    num_warps=2,
                )
            else:
                q_flat, z_view = self._unpack_qz_consumer(qz)
                st_c = ShortConvState(init_conv_state.clone()) if init_conv_state is not None else None
                steps_q = []
                for s in range(verify_w):
                    qs = causal_dwconv1d(
                        q_flat[:, s : s + 1],
                        self.conv1d.weight,
                        state=st_c,
                        residual=self.gdn_conv_residual,
                        use_silu=True,
                    )
                    if stage_conv_state is not None and st_c is not None:
                        stage_conv_state[:, s].copy_(st_c.conv_state)
                    steps_q.append(qs)
                q = torch.cat(steps_q, dim=1).reshape(bsz, verify_w, self.num_k_heads, self.head_k_dim)

            if yoco_decode_on and shared.ssm_state is not None:
                # shared.ssm_state has shape [B, V, H_v, D_k, D_v] (possibly strided inside packed linear stage)
                core_out = torch.empty(
                    (bsz, verify_w, self.num_v_heads, self.head_v_dim),
                    dtype=dtype,
                    device=device,
                )
                s_st = shared.ssm_state
                is_s_contig = s_st.stride(-1) == 1 and s_st.stride(-2) == self.head_v_dim
                use_bk8_ro = (
                    is_s_contig
                    and self.head_k_dim % 8 == 0
                    and bsz * verify_w * self.num_v_heads >= 64
                )
                if use_bk8_ro:
                    block_dv_ro = self.head_v_dim
                    bk_ro = 8
                    nw_ro = 1
                    ro_kw = {"num_stages": 2}
                else:
                    block_dv_ro = (
                        32
                        if (bsz * verify_w * self.num_v_heads >= 512 and self.head_v_dim % 32 == 0)
                        else self.head_v_dim
                    )
                    bk_ro = 0
                    nw_ro = 1 if (block_dv_ro < 64 or bsz >= 2) else 2
                    ro_kw = {}
                _yoco_gdn_verify_readout_kernel[
                    (bsz * verify_w, self.num_v_heads, self.head_v_dim // block_dv_ro)
                ](
                    q,
                    s_st,
                    core_out,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    s_st.stride(0),
                    s_st.stride(1),
                    s_st.stride(2),
                    s_st.stride(3),
                    s_st.stride(4),
                    core_out.stride(0),
                    core_out.stride(1),
                    core_out.stride(2),
                    core_out.stride(3),
                    V_STEPS=verify_w,
                    scale=float(self.head_k_dim ** -0.5),
                    eps=1e-6,
                    GQA_RATIO=self.gqa_ratio,
                    D_K=self.head_k_dim,
                    D_V=self.head_v_dim,
                    BLOCK_DV=block_dv_ro,
                    BK=bk_ro,
                    IS_S_CONTIG=is_s_contig,
                    num_warps=nw_ro,
                    **ro_kw,
                )
            else:
                core_out = self._launch_gdn_verify_scan(
                    q,
                    shared.key,
                    shared.value,
                    shared.g,
                    shared.beta,
                    init_ssm_state,
                    stage_ssm_state,
                    verify_w=verify_w,
                )

            out = self.norm(core_out, z_view)
            out = self.out_proj(out.reshape(bsz, verify_w, self.value_dim))
            out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)
            return out, None

        # Normal / YoCo Source verify
        assert self.in_proj_ba is not None and self.A_log is not None and self.dt_bias is not None
        assert init_conv_state is not None
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        g, beta = compute_gdn_gating(
            ba,
            self.A_log,
            self.dt_bias,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            out_g=lazy_kvgb_buffers[2][:bsz] if lazy_kvgb_buffers is not None else None,
            out_beta=lazy_kvgb_buffers[3][:bsz] if lazy_kvgb_buffers is not None else None,
        )

        if self._can_use_fused_unpack_conv(qkvz):
            q = torch.empty((bsz, verify_w, self.num_k_heads, self.head_k_dim), dtype=dtype, device=device)
            if lazy_kvgb_buffers is not None:
                k = lazy_kvgb_buffers[0][:bsz]
                v = lazy_kvgb_buffers[1][:bsz]
            else:
                k = torch.empty((bsz, verify_w, self.num_k_heads, self.head_k_dim), dtype=dtype, device=device)
                v = torch.empty((bsz, verify_w, self.num_v_heads, self.head_v_dim), dtype=dtype, device=device)
            has_stage_c = stage_conv_state is not None
            sc_arg = stage_conv_state if has_stage_c else q
            v_group = self.gqa_ratio * self.head_v_dim
            _gdn_fused_unpack_conv1d_verify_k4_kernel[(bsz, self.conv_dim // 64)](
                qkvz,
                self.conv1d.weight,
                init_conv_state,
                sc_arg,
                q,
                k,
                v,
                qkvz.stride(0),
                qkvz.stride(1),
                qkvz.stride(2),
                init_conv_state.stride(0),
                init_conv_state.stride(1),
                init_conv_state.stride(2),
                stage_conv_state.stride(0) if has_stage_c else 0,
                stage_conv_state.stride(1) if has_stage_c else 0,
                stage_conv_state.stride(2) if has_stage_c else 0,
                stage_conv_state.stride(3) if has_stage_c else 0,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                V_STEPS=verify_w,
                KEY_DIM=self.key_dim,
                D_K=64,
                D_V=64,
                GQA_RATIO=self.gqa_ratio,
                GROUP_STRIDE=2 * self.head_k_dim + 2 * v_group,
                IS_CONSUMER=False,
                RESIDUAL=self.gdn_conv_residual,
                HAS_STAGE=has_stage_c,
                BLOCK_C=64,
                num_warps=2,
            )
            z_view = self._view_z_strided(qkvz, is_consumer=False)
        else:
            mixed_qkv, z_view = self._unpack_qkvz_normal(qkvz)
            st_c = ShortConvState(init_conv_state.clone())
            steps_qkv = []
            for s in range(verify_w):
                ms = causal_dwconv1d(
                    mixed_qkv[:, s : s + 1],
                    self.conv1d.weight,
                    state=st_c,
                    residual=self.gdn_conv_residual,
                    use_silu=True,
                )
                if stage_conv_state is not None:
                    stage_conv_state[:, s].copy_(st_c.conv_state)
                steps_qkv.append(ms)
            mixed_all = torch.cat(steps_qkv, dim=1)
            qf, kf, vf = mixed_all.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(bsz, verify_w, self.num_k_heads, self.head_k_dim)
            k = kf.reshape(bsz, verify_w, self.num_k_heads, self.head_k_dim)
            v = vf.reshape(bsz, verify_w, self.num_v_heads, self.head_v_dim)
            if lazy_kvgb_buffers is not None:
                lazy_kvgb_buffers[0][:bsz].copy_(k)
                lazy_kvgb_buffers[1][:bsz].copy_(v)

        core_out = self._launch_gdn_verify_scan(
            q,
            k,
            v,
            g,
            beta,
            init_ssm_state,
            stage_ssm_state,
            verify_w=verify_w,
        )

        out = self.norm(core_out, z_view)
        out = self.out_proj(out.reshape(bsz, verify_w, self.value_dim))
        out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)

        published: Optional[SharedGDNInputs] = None
        if self.publishes_shared:
            published = SharedGDNInputs(
                key=k,
                value=v,
                g=g,
                beta=beta,
                ssm_state=stage_ssm_state,
            )
        return out, published

    def forward_packed_prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tile_req: torch.Tensor,
        tile_t0: torch.Tensor,
        chunk_req: torch.Tensor,
        chunk_t0: torch.Tensor,
        req_chunk_start: torch.Tensor,
        *,
        init_conv_state: Optional[torch.Tensor] = None,
        init_ssm_state: Optional[torch.Tensor] = None,
        shared: Optional[SharedGDNInputs] = None,
        snap_tok_idx: Optional[torch.Tensor] = None,
        snap_req_idx: Optional[torch.Tensor] = None,
        conv_dst_pool: Optional[torch.Tensor] = None,
        conv_snap_phys: Optional[torch.Tensor] = None,
        conv_dst_stride_row: int = 0,
        conv_dst_col_offset: int = 0,
        req_snap_start: Optional[torch.Tensor] = None,
        snap_local_end: Optional[torch.Tensor] = None,
        ssm_dst_pool: Optional[torch.Tensor] = None,
        ssm_snap_phys: Optional[torch.Tensor] = None,
        skip_invalid_phys: bool = True,
        snaps_direct_chunk64: bool = False,
        block_t: int = 32,
        enable_yoco_decode: Optional[bool] = None,
        skip_allreduce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SharedGDNInputs]]:
        """Whole-batch packed ragged prefill [N, H] with 64-token SSM & 3-token conv snapshots."""
        del enable_yoco_decode
        N = int(hidden_states.shape[0])
        if N == 0:
            return torch.empty_like(hidden_states), None
        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.is_consumer:
            if shared is None:
                raise ValueError(
                    f"Consumer GDN layer {self.layer_idx} requires SharedGDNInputs from source layer"
                )
            qz = self.in_proj_qkvz(hidden_states)
            z_view = self._view_z_strided(qz.unsqueeze(0), is_consumer=True)
            if self.gdn_no_q_conv:
                q_g = qz.unflatten(-1, (self.num_k_heads, self.head_k_dim + self.gqa_ratio * self.head_v_dim))[
                    ..., : self.head_k_dim
                ]
                q = q_g.contiguous()
            elif self._can_use_fused_unpack_conv(qz):
                q, _, _ = self._launch_fused_unpack_conv_ragged(
                    qz,
                    cu_seqlens,
                    tile_req,
                    tile_t0,
                    is_consumer=True,
                    init_conv_state=init_conv_state,
                    snap_tok_idx=snap_tok_idx,
                    snap_req_idx=snap_req_idx,
                    conv_dst_pool=conv_dst_pool,
                    conv_snap_phys=conv_snap_phys,
                    conv_dst_stride_row=conv_dst_stride_row,
                    conv_dst_col_offset=conv_dst_col_offset,
                    skip_invalid_phys=skip_invalid_phys,
                    block_t=block_t,
                )
            else:
                from rtp_llm.models_py.modules.qwen4_5.short_conv import (
                    causal_dwconv1d_ragged_prefill_with_snapshots_triton,
                )
                q_flat, z_v = self._unpack_qz_consumer(qz.unsqueeze(0))
                z_view = z_v
                q_conv = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                    q_flat.squeeze(0),
                    self.conv1d.weight,
                    cu_seqlens,
                    tile_req,
                    tile_t0,
                    init_state=init_conv_state,
                    snap_tok_idx=snap_tok_idx,
                    snap_req_idx=snap_req_idx,
                    dst_pool=conv_dst_pool,
                    snap_dst_idx=conv_snap_phys,
                    dst_stride_row=conv_dst_stride_row,
                    dst_col_offset=conv_dst_col_offset,
                    skip_invalid_phys=skip_invalid_phys,
                    residual=self.gdn_conv_residual,
                    use_activation=True,
                    round_pre_act=True,
                    block_t=block_t,
                )
                q = q_conv.view(N, self.num_k_heads, self.head_k_dim)

            core_out, _ = gdn_chunk_prefill_ragged_with_snapshots_triton(
                q,
                shared.key,
                shared.value,
                shared.g,
                shared.beta,
                cu_seqlens,
                chunk_req,
                chunk_t0,
                req_chunk_start,
                initial_state=init_ssm_state,
                req_snap_start=req_snap_start,
                snap_local_end=snap_local_end,
                snap_phys=ssm_snap_phys,
                ssm_dst_pool=ssm_dst_pool,
                skip_invalid_phys=skip_invalid_phys,
                snaps_direct_chunk64=snaps_direct_chunk64,
                output_final_states=False,
            )
            out = self.norm(core_out.unsqueeze(0), z_view).squeeze(0)
            out = self.out_proj(out.reshape(N, self.value_dim))
            out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)
            return out, None

        # Normal / YoCo Source packed prefill
        assert self.in_proj_ba is not None and self.A_log is not None and self.dt_bias is not None
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        g, beta = compute_gdn_gating(
            ba.unsqueeze(0),
            self.A_log,
            self.dt_bias,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )
        g_2d = g.squeeze(0)
        beta_2d = beta.squeeze(0)

        if self._can_use_fused_unpack_conv(qkvz):
            q, k, v = self._launch_fused_unpack_conv_ragged(
                qkvz,
                cu_seqlens,
                tile_req,
                tile_t0,
                is_consumer=False,
                init_conv_state=init_conv_state,
                snap_tok_idx=snap_tok_idx,
                snap_req_idx=snap_req_idx,
                conv_dst_pool=conv_dst_pool,
                conv_snap_phys=conv_snap_phys,
                conv_dst_stride_row=conv_dst_stride_row,
                conv_dst_col_offset=conv_dst_col_offset,
                skip_invalid_phys=skip_invalid_phys,
                block_t=block_t,
            )
            assert k is not None and v is not None
            z_view = self._view_z_strided(qkvz.unsqueeze(0), is_consumer=False)
        else:
            from rtp_llm.models_py.modules.qwen4_5.short_conv import (
                causal_dwconv1d_ragged_prefill_with_snapshots_triton,
            )
            mixed_qkv, z_view = self._unpack_qkvz_normal(qkvz.unsqueeze(0))
            mixed_conv = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                mixed_qkv.squeeze(0),
                self.conv1d.weight,
                cu_seqlens,
                tile_req,
                tile_t0,
                init_state=init_conv_state,
                snap_tok_idx=snap_tok_idx,
                snap_req_idx=snap_req_idx,
                dst_pool=conv_dst_pool,
                snap_dst_idx=conv_snap_phys,
                dst_stride_row=conv_dst_stride_row,
                dst_col_offset=conv_dst_col_offset,
                skip_invalid_phys=skip_invalid_phys,
                residual=self.gdn_conv_residual,
                use_activation=True,
                round_pre_act=True,
                block_t=block_t,
            )
            qf, kf, vf = mixed_conv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.view(N, self.num_k_heads, self.head_k_dim)
            k = kf.view(N, self.num_k_heads, self.head_k_dim)
            v = vf.view(N, self.num_v_heads, self.head_v_dim)

        core_out, final_ssm = gdn_chunk_prefill_ragged_with_snapshots_triton(
            q,
            k,
            v,
            g_2d,
            beta_2d,
            cu_seqlens,
            chunk_req,
            chunk_t0,
            req_chunk_start,
            initial_state=init_ssm_state,
            req_snap_start=req_snap_start,
            snap_local_end=snap_local_end,
            snap_phys=ssm_snap_phys,
            ssm_dst_pool=ssm_dst_pool,
            skip_invalid_phys=skip_invalid_phys,
            snaps_direct_chunk64=snaps_direct_chunk64,
            output_final_states=False,
        )

        out = self.norm(core_out.unsqueeze(0), z_view).squeeze(0)
        out = self.out_proj(out.reshape(N, self.value_dim))
        out = self._maybe_all_reduce(out, skip_allreduce=skip_allreduce)

        published: Optional[SharedGDNInputs] = None
        if self.publishes_shared:
            published = SharedGDNInputs(
                key=k,
                value=v,
                g=g_2d,
                beta=beta_2d,
                ssm_state=final_ssm,
            )
        return out, published


__all__ = [
    "Qwen4_5GatedDeltaNet",
    "RMSNormGated",
    "SSM_STATE_LAYOUT",
    "compute_gdn_gating",
    "gdn_chunk_prefill_ragged_with_snapshots_triton",
    "gdn_chunk_prefill_triton",
    "gdn_recurrent_ref",
    "gdn_recurrent_triton",
]

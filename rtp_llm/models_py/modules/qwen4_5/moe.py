"""Layer 0 Dense MLP and Layers 1..39 Latent MoE + Shared Expert for Qwen 4.5-Flash.

Exact Qwen 4.5-Flash (Qwen4_5ForCausalLM, 1109-bf16-re-mxfp4-ple-fp8) specification:
  - Layer 0: ``Qwen4_5DenseMLP`` (H=5120, I=12800, BF16, SiTU-GLU(4.0, 25.0)).
  - Layers 1..39: ``Qwen4_5LatentMoE``:
      1. ``routed = F.linear(x, latent_down_proj)``                 (5120 -> 2560)
      2. ``routed = gemma_rmsnorm(routed, pre_moe_latent_ln, eps)`` ((1.0 + weight) in FP32)
      3. ``routed = routed * sigmoid(F.linear(x, latent_gate_proj))`` (5120 -> 2560)
      4. Router on ``routed`` (``router_precision="fp32_logits"``):
           ``logits = F.linear(routed, gate_weight).float()``       ([M, 640] FP32)
           ``scores = sigmoid(logits)``
           ``selection = scores + expert_bias.float()``             (FP32 [640])
           ``topk_ids = selection.topk(top_k, dim=-1).indices``
           ``topk_weights = scores.gather(-1, topk_ids)``
           ``topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-8)``
           ``topk_weights = topk_weights.to(x.dtype).float()``
      5. Routed MXFP4 experts on ``routed`` (2560 -> 2560 -> 2560, E=640, top_k=10)
      6. ``routed = F.linear(routed, latent_up_proj)``              (2560 -> 5120)
      7. ``routed = routed * sigmoid(F.linear(x, latent_scale_gate))`` ([M, 1])
      8. Shared expert on original ``x``:
           ``shared = shared_expert(x) * sigmoid(shared_expert_gate(x))``
      9. Return ``routed + shared``.

NOTE: Post-FFN short-conv (``ffn_output_sconv``) and output clamping
(``clamp(-clip_mlp_output, clip_mlp_output)``, i.e. ``clamp(-1000.0, 1000.0)``)
are applied immediately after ``Qwen4_5DenseMLP`` / ``Qwen4_5LatentMoE`` inside
``DecoderLayer`` and are intentionally not baked into these modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rtp_llm.models_py.modules.qwen4_5.activation import SituAndMul
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.fp8_linear import SM100FP8Linear, _cuda_device_ctx
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    MXFP4_BLOCK_SIZE,
    MXFP4MoEWeights,
    mxfp4_moe_forward,
    mxfp4_moe_forward_ref,
    prepare_mxfp4_moe_weights,
    resolve_moe_backend_for_shape,
    supports_triton_mxfp4,
)


# ---------------------------------------------------------------------------
# Gemma-style RMSNorm: x_fp32 * rsqrt(mean(x_fp32^2) + eps) * (1.0 + weight)
# ---------------------------------------------------------------------------


def gemma_rmsnorm_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Pure PyTorch FP32 reference for Gemma-style RMSNorm ``(1.0 + weight)``."""
    x_f32 = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_f32 * inv_rms * (1.0 + weight.float())
    return out.to(x.dtype)


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D,
    stride_x_row: tl.int64,
    stride_out_row: tl.int64,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + pid_row * stride_x_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.div_rn(tl.sum(x * x, axis=0), D.to(tl.float32))
    inv_rms = libdevice.rsqrt(mean_sq + eps)
    y = x * inv_rms * (1.0 + w)

    tl.store(
        out_ptr + pid_row * stride_out_row + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Gemma-style RMSNorm computing ``(1.0 + weight)`` in FP32."""
    D = x.shape[-1]
    if not use_triton or not x.is_cuda or D > 16384:
        return gemma_rmsnorm_ref(x, weight, eps=eps)
    if x.numel() == 0:
        return torch.empty_like(x)

    orig_shape = x.shape
    x_2d = x.reshape(-1, D)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    w = weight.contiguous()
    num_rows = x_2d.shape[0]
    out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 8 if BLOCK_D >= 2048 else 4
    with _cuda_device_ctx(x_2d.device):
        _gemma_rmsnorm_kernel[(num_rows,)](
            x_2d,
            w,
            out,
            D,
            x_2d.stride(0),
            out.stride(0),
            eps=float(eps),
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
    return out.reshape(orig_shape)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with zero-initialized ``weight`` (scale = ``1.0 + weight``)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        use_triton: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.use_triton = bool(use_triton)
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device)
        )

    def forward(
        self, x: torch.Tensor, *, use_triton: Optional[bool] = None
    ) -> torch.Tensor:
        flag = self.use_triton if use_triton is None else use_triton
        return gemma_rmsnorm(x, self.weight, eps=self.eps, use_triton=flag)


@triton.jit(do_not_specialize=["M", "stride_dg_row", "stride_out_row"])
def _latent_norm_mul_sigmoid_kernel(
    dg_ptr,
    ln_w_ptr,
    out_ptr,
    M,
    L: tl.constexpr,
    stride_dg_row: tl.int64,
    stride_out_row: tl.int64,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused 2-pass GemmaRMSNorm(down) * sigmoid(gate) from [M, 2*L] avoiding register spill."""
    row = tl.program_id(0).to(tl.int64)
    if row >= M:
        return
    dg_base = dg_ptr + row * stride_dg_row
    out_base = out_ptr + row * stride_out_row
    ssq = 0.0
    for off in range(0, L, BLOCK_K):
        offs = off + tl.arange(0, BLOCK_K)
        mask = offs < L
        x_down = tl.load(dg_base + offs, mask=mask, other=0.0).to(tl.float32)
        ssq += tl.sum(x_down * x_down, axis=0)
    inv_rms = tl.rsqrt(ssq / L + eps)
    for off in range(0, L, BLOCK_K):
        offs = off + tl.arange(0, BLOCK_K)
        mask = offs < L
        x_down = tl.load(dg_base + offs, mask=mask, other=0.0).to(tl.float32)
        x_gate = tl.load(dg_base + L + offs, mask=mask, other=0.0).to(tl.float32)
        w_ln = tl.load(ln_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        normed = (x_down * inv_rms * (1.0 + w_ln)).to(out_ptr.dtype.element_ty)
        sig_g = tl.div_rn(1.0, 1.0 + libdevice.exp(-x_gate)).to(out_ptr.dtype.element_ty)
        out = (normed.to(tl.float32) * sig_g.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_base + offs, out, mask=mask)


@triton.jit(do_not_specialize=["stride_r", "stride_s", "stride_g", "stride_o"])
def _combine_latent_shared_out_kernel(
    routed_ptr,
    shared_ptr,
    gates_ptr,
    out_ptr,
    stride_r: tl.int64,
    stride_s: tl.int64,
    stride_g: tl.int64,
    stride_o: tl.int64,
    H: tl.constexpr,
    HAS_SHARED_GATE: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse routed * sigmoid(g_r) + shared (* sigmoid(g_s)) into one HBM pass."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1)
    offs = col * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    g_base = gates_ptr + row * stride_g
    gr = tl.load(g_base).to(tl.float32)
    if IS_BF16:
        # Preserve PyTorch separate bf16 mul + add rounding (avoid LLVM fma.rn.bf16 contraction).
        sr = tl.div_rn(1.0, 1.0 + libdevice.exp(-gr)).to(tl.bfloat16)
        r_v = tl.load(routed_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.bfloat16)
        s_v = tl.load(shared_ptr + row * stride_s + offs, mask=mask, other=0.0).to(tl.bfloat16)
        rs = r_v * sr
        if HAS_SHARED_GATE:
            gs = tl.load(g_base + 1).to(tl.float32)
            ss = tl.div_rn(1.0, 1.0 + libdevice.exp(-gs)).to(tl.bfloat16)
            ss_val = s_v * ss
        else:
            ss_val = s_v
        y = tl.inline_asm_elementwise(
            "add.rn.bf16 $0, $1, $2;",
            "=h,h,h",
            [rs, ss_val],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )
    else:
        sr = tl.div_rn(1.0, 1.0 + libdevice.exp(-gr)).to(out_ptr.dtype.element_ty).to(tl.float32)
        r = tl.load(routed_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(shared_ptr + row * stride_s + offs, mask=mask, other=0.0).to(tl.float32)
        rs = (r * sr).to(out_ptr.dtype.element_ty).to(tl.float32)
        if HAS_SHARED_GATE:
            gs = tl.load(g_base + 1).to(tl.float32)
            ss = tl.div_rn(1.0, 1.0 + libdevice.exp(-gs)).to(out_ptr.dtype.element_ty).to(tl.float32)
            ss_val = (s * ss).to(out_ptr.dtype.element_ty).to(tl.float32)
            y = (rs + ss_val).to(out_ptr.dtype.element_ty)
        else:
            y = (rs + s).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_o + offs, y, mask=mask)



@triton.jit
def _qwen45_sigmoid_topk_router_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    weights_ptr,
    E,
    stride_lm: tl.int64,
    E_PAD: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, E_PAD)
    mask_e = offs_e < E

    raw = tl.load(
        logits_ptr + row * stride_lm + offs_e, mask=mask_e, other=0.0
    ).to(tl.float32)
    scores = tl.where(mask_e, tl.div_rn(1.0, 1.0 + libdevice.exp(-raw)), 0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = tl.where(mask_e, scores + bias, -3.4e38)
    else:
        sel = tl.where(mask_e, scores, -3.4e38)

    slots = tl.arange(0, TOP_K_PAD)
    chosen_w = tl.zeros((TOP_K_PAD,), dtype=tl.float32)
    ids_base = ids_ptr + row * TOP_K
    for i in tl.static_range(TOP_K):
        best_val = tl.max(sel, axis=0)
        best_idx = tl.min(tl.where(sel == best_val, offs_e, 2147483647), axis=0)
        best_score = tl.sum(tl.where(offs_e == best_idx, scores, 0.0), axis=0)
        chosen_w = tl.where(slots == i, best_score, chosen_w)
        tl.store(ids_base + i, best_idx)
        sel = tl.where(offs_e == best_idx, -3.4e38, sel)

    if NORMALIZE:
        denom = tl.sum(chosen_w, axis=0) + 1e-8
        chosen_w = tl.div_rn(chosen_w, denom)
    if ROUND_BF16:
        chosen_w = chosen_w.to(tl.bfloat16).to(tl.float32)

    tl.store(weights_ptr + row * TOP_K + slots, chosen_w, mask=slots < TOP_K)


# ---------------------------------------------------------------------------
# Layer 0 Dense MLP & Shared Expert (BF16 + SiTU-GLU)
# ---------------------------------------------------------------------------


class Qwen4_5DenseMLP(nn.Module):
    """Dense gated MLP with SiTU-GLU activation and optional TP=4 sharding.

    Used for:
      - Layer 0 Dense MLP (``hidden_size=5120, intermediate_size=12800 -> 3200/rank at TP=4``)
      - Layers 1..39 Shared Expert (``hidden_size=5120, intermediate_size=1280``)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 12800,
        *,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        use_triton: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        dense_linear_mode: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.full_intermediate_size = int(intermediate_size)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce
        self.dense_linear_mode = (
            str(dense_linear_mode).lower()
            if dense_linear_mode is not None
            else os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "bf16").strip().lower()
        )
        self._fp8_gate_up: Optional[SM100FP8Linear] = None
        self._fp8_gate: Optional[SM100FP8Linear] = None
        self._fp8_up: Optional[SM100FP8Linear] = None
        self._fp8_down: Optional[SM100FP8Linear] = None
        if self.tp_size < 1 or self.full_intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size={self.full_intermediate_size} must be divisible by tp_size={self.tp_size}"
            )
        if not (0 <= self.tp_rank < self.tp_size):
            raise ValueError(f"tp_rank={self.tp_rank} out of range for tp_size={self.tp_size}")
        self.local_intermediate_size = self.full_intermediate_size // self.tp_size
        self.intermediate_size = self.local_intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.local_intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.local_intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = nn.Linear(
            self.local_intermediate_size,
            self.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.act_fn = SituAndMul(
            gate_beta=gate_beta, up_beta=up_beta, use_triton=use_triton
        )
        self.register_buffer("_gate_up_weight", None, persistent=False)

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        *,
        is_shared_expert: bool = False,
        use_triton: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        dense_linear_mode: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> "Qwen4_5DenseMLP":
        inter = (
            config.shared_expert_intermediate_size
            if is_shared_expert
            else config.intermediate_size
        )
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=inter,
            gate_beta=config.hidden_act_gate_beta,
            up_beta=config.hidden_act_up_beta,
            use_triton=use_triton,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
            dense_linear_mode=dense_linear_mode,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def load_tp_weights(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        *,
        already_sharded: bool = False,
    ) -> None:
        """Load full or pre-sharded gate/up/down weights onto the local TP rank."""
        if already_sharded or self.tp_size == 1:
            g_local, u_local, d_local = gate_weight, up_weight, down_weight
        else:
            start = self.tp_rank * self.local_intermediate_size
            end = start + self.local_intermediate_size
            g_local = gate_weight[start:end, :]
            u_local = up_weight[start:end, :]
            d_local = down_weight[:, start:end]

        if tuple(g_local.shape) != tuple(self.gate_proj.weight.shape):
            raise ValueError(
                f"gate_weight shard shape {tuple(g_local.shape)} != {tuple(self.gate_proj.weight.shape)}"
            )
        if tuple(u_local.shape) != tuple(self.up_proj.weight.shape):
            raise ValueError(
                f"up_weight shard shape {tuple(u_local.shape)} != {tuple(self.up_proj.weight.shape)}"
            )
        if tuple(d_local.shape) != tuple(self.down_proj.weight.shape):
            raise ValueError(
                f"down_weight shard shape {tuple(d_local.shape)} != {tuple(self.down_proj.weight.shape)}"
            )

        self.gate_proj.weight.data.copy_(g_local)
        self.up_proj.weight.data.copy_(u_local)
        self.down_proj.weight.data.copy_(d_local)
        if self._gate_up_weight is not None:
            self.prepare_weights(fuse_gate_up=True)

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        fuse_gate_up: bool = True,
        dense_linear_mode: Optional[str] = None,
        allow_mxfp8: bool = False,
    ) -> None:
        """Cache concatenated ``[gate_proj; up_proj]`` and optional SM100 FP8 weights."""
        if dense_linear_mode is not None:
            self.dense_linear_mode = str(dense_linear_mode).lower()
        if fuse_gate_up:
            self._gate_up_weight = torch.cat(
                [self.gate_proj.weight.data, self.up_proj.weight.data], dim=0
            ).contiguous()
        else:
            self._gate_up_weight = None

        eff_mode = (
            self.dense_linear_mode
            or os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "bf16").strip().lower()
        )
        if eff_mode != "bf16" and self.gate_proj.weight.is_cuda:
            # SiTU-GLU multiplies two noisy inputs; keep auto in BF16 unless allow_mxfp8 or explicit mxfp8.
            use_mxfp8 = bool(allow_mxfp8 or eff_mode == "mxfp8")
            lin_kw = dict(
                mode=eff_mode,
                keep_bf16=True,
                prepare_trtllm_ll=False,
                prepare_mxfp8=use_mxfp8,
                prepare_w4a8=(eff_mode == "w4a8"),
                allow_decode_scaled_mm=False,
            )
            if self._gate_up_weight is not None:
                self._fp8_gate_up = SM100FP8Linear(
                    self.hidden_size,
                    2 * self.local_intermediate_size,
                    **lin_kw,
                    dtype=self.gate_proj.weight.dtype,
                    device=self.gate_proj.weight.device,
                )
                self._fp8_gate_up.prepare_weights(self._gate_up_weight)
                self._fp8_gate = None
                self._fp8_up = None
            else:
                self._fp8_gate_up = None
                self._fp8_gate = SM100FP8Linear.from_linear(self.gate_proj, **lin_kw)
                self._fp8_up = SM100FP8Linear.from_linear(self.up_proj, **lin_kw)
            self._fp8_down = SM100FP8Linear.from_linear(self.down_proj, **lin_kw)
        else:
            self._fp8_gate_up = None
            self._fp8_gate = None
            self._fp8_up = None
            self._fp8_down = None

    def _finish_down_and_reduce(
        self, act: torch.Tensor, *, skip_allreduce: bool
    ) -> torch.Tensor:
        out = self._fp8_down(act) if self._fp8_down is not None else self.down_proj(act)
        if self.tp_size > 1 and not skip_allreduce:
            if self.fast_all_reduce is not None:
                orig_shape = out.shape
                out_2d = out.reshape(-1, self.hidden_size)
                if not out_2d.is_contiguous():
                    out_2d = out_2d.contiguous()
                out = self.fast_all_reduce(out_2d).reshape(orig_shape)
            elif self.tp_group is not None:
                if not out.is_contiguous():
                    out = out.contiguous()
                torch.distributed.all_reduce(
                    out, op=torch.distributed.ReduceOp.SUM, group=self.tp_group
                )
        return out

    def forward_from_gate_up(
        self,
        gate_up: torch.Tensor,
        *,
        use_triton: Optional[bool] = None,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """Run SiTU-GLU + down projection from precomputed ``gate_up`` [..., 2*D]."""
        act = self.act_fn(gate_up, use_triton=use_triton)
        return self._finish_down_and_reduce(act, skip_allreduce=skip_allreduce)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton: Optional[bool] = None,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return torch.empty_like(hidden_states)

        if self._gate_up_weight is not None and (use_triton is None or use_triton):
            gate_up = (
                self._fp8_gate_up(hidden_states)
                if self._fp8_gate_up is not None
                else F.linear(hidden_states, self._gate_up_weight)
            )
            return self.forward_from_gate_up(
                gate_up, use_triton=use_triton, skip_allreduce=skip_allreduce
            )

        gate = (
            self._fp8_gate(hidden_states)
            if self._fp8_gate is not None
            else self.gate_proj(hidden_states)
        )
        up = (
            self._fp8_up(hidden_states)
            if self._fp8_up is not None
            else self.up_proj(hidden_states)
        )
        act = self.act_fn(gate, up, use_triton=use_triton)
        return self._finish_down_and_reduce(act, skip_allreduce=skip_allreduce)


# ---------------------------------------------------------------------------
# Layers 1..39 Latent MoE + Shared Expert (MXFP4 Routed Experts)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _shared_stream_for_device(device_index: int) -> torch.cuda.Stream:
    """Reuse one auxiliary CUDA stream per device for shared-expert overlap."""
    return torch.cuda.Stream(device=torch.device("cuda", int(device_index)))


@dataclass
class LatentMoETBOContext:
    """Per-microbatch pipeline state for 3-stage LatentMoE TBO overlap."""

    orig_shape: Tuple[int, ...]
    x: torch.Tensor
    run_triton: bool
    mode: str
    backend: str
    weights: MXFP4MoEWeights
    has_a2a: bool
    topk_ids: Optional[torch.Tensor] = None
    topk_weights: Optional[torch.Tensor] = None
    routed: Optional[torch.Tensor] = None
    dispatch_handle: Optional[Any] = None
    combine_handle: Optional[Any] = None
    shared: Optional[torch.Tensor] = None
    early_return: Optional[torch.Tensor] = None


class Qwen4_5LatentMoE(nn.Module):
    """Qwen 4.5-Flash Latent MoE + Shared Expert with B300 SM103 MXFP4 Routed Experts.

    NOTE: ``ffn_output_sconv`` and ``clamp(-1000.0, 1000.0)`` are applied right after
    this module in ``DecoderLayer``.
    """

    def __init__(
        self,
        config: Optional[Qwen4_5ArchConfig] = None,
        *,
        hidden_size: int = 5120,
        routed_expert_hidden_size: int = 2560,
        moe_intermediate_size: int = 2560,
        shared_expert_intermediate_size: int = 1280,
        num_experts: int = 640,
        num_experts_per_tok: int = 10,
        rms_norm_eps: float = 1e-6,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        shared_expert_gate: bool = True,
        router_precision: str = "fp32_logits",
        router_normalize_topk_scores: bool = True,
        moe_router_enable_expert_bias: bool = True,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        all_to_all_comm: Any = None,
        dp_equal_tokens: Optional[bool] = None,
        use_symm_p2p_combine: Optional[bool] = None,
        dense_linear_mode: Optional[str] = None,
        enable_shared_overlap: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            routed_expert_hidden_size = config.routed_expert_hidden_size
            moe_intermediate_size = config.moe_intermediate_size
            shared_expert_intermediate_size = config.shared_expert_intermediate_size
            num_experts = config.num_experts
            num_experts_per_tok = config.num_experts_per_tok
            rms_norm_eps = config.rms_norm_eps
            gate_beta = config.hidden_act_gate_beta
            up_beta = config.hidden_act_up_beta
            shared_expert_gate = config.shared_expert_gate
            router_precision = config.router_precision
            router_normalize_topk_scores = config.router_normalize_topk_scores
            moe_router_enable_expert_bias = config.moe_router_enable_expert_bias

        if routed_expert_hidden_size % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"routed_expert_hidden_size={routed_expert_hidden_size} must be divisible by {MXFP4_BLOCK_SIZE}"
            )
        if moe_intermediate_size % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"moe_intermediate_size={moe_intermediate_size} must be divisible by {MXFP4_BLOCK_SIZE}"
            )

        self.hidden_size = int(hidden_size)
        self.latent_dim = int(routed_expert_hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
        self.num_experts = int(num_experts)
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.ep_group = ep_group
        if self.ep_size < 1 or self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by ep_size={self.ep_size}"
            )
        if not (0 <= self.ep_rank < self.ep_size):
            raise ValueError(f"ep_rank={self.ep_rank} out of range for ep_size={self.ep_size}")
        self.local_num_experts = self.num_experts // self.ep_size
        self.ep_start = self.ep_rank * self.local_num_experts
        self.ep_end = self.ep_start + self.local_num_experts
        self.top_k = int(num_experts_per_tok)
        self.rms_norm_eps = float(rms_norm_eps)
        self.gate_beta = float(gate_beta)
        self.up_beta = float(up_beta)
        self.router_precision = str(router_precision)
        self.normalize_topk_scores = bool(router_normalize_topk_scores)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce
        self.all_to_all_comm = all_to_all_comm
        self.dp_equal_tokens = dp_equal_tokens
        self.use_symm_p2p_combine = use_symm_p2p_combine
        self.dense_linear_mode = (
            str(dense_linear_mode).lower()
            if dense_linear_mode is not None
            else os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "bf16").strip().lower()
        )
        self.register_buffer("_latent_down_gate_weight", None, persistent=False)
        self.register_buffer("_latent_down_gate_shared_gu_weight", None, persistent=False)
        self.register_buffer("_latent_shared_gates_weight", None, persistent=False)
        self._fp8_latent_down_gate: Optional[SM100FP8Linear] = None
        self._fp8_latent_down: Optional[SM100FP8Linear] = None
        self._fp8_latent_gate: Optional[SM100FP8Linear] = None
        self._fp8_latent_up: Optional[SM100FP8Linear] = None
        self.enable_shared_overlap = bool(enable_shared_overlap)
        self.backend = str(backend)
        self.use_triton_mxfp4 = bool(use_triton_mxfp4)
        self.compute_mode = str(compute_mode)
        self.weights_transposed = False
        self._mxfp4_weights_cache: Optional[MXFP4MoEWeights] = None
        self._shared_stream: Optional[torch.cuda.Stream] = None
        self._shared_input_ready: Optional[torch.cuda.Event] = None
        self._shared_output_ready: Optional[torch.cuda.Event] = None

        # Latent projections & pre-MoE Gemma RMSNorm (BF16)
        self.latent_down_proj = nn.Linear(
            self.hidden_size, self.latent_dim, bias=False, dtype=dtype, device=device
        )
        self.pre_moe_latent_layernorm = GemmaRMSNorm(
            self.latent_dim,
            eps=self.rms_norm_eps,
            use_triton=self.use_triton_mxfp4,
            dtype=dtype,
            device=device,
        )
        self.latent_gate_proj = nn.Linear(
            self.hidden_size, self.latent_dim, bias=False, dtype=dtype, device=device
        )
        self.latent_up_proj = nn.Linear(
            self.latent_dim, self.hidden_size, bias=False, dtype=dtype, device=device
        )
        self.latent_scale_gate = nn.Linear(
            self.hidden_size, 1, bias=False, dtype=dtype, device=device
        )

        # Router (BF16 gate weight + FP32 expert bias)
        self.gate = nn.Linear(
            self.latent_dim, self.num_experts, bias=False, dtype=dtype, device=device
        )
        if moe_router_enable_expert_bias:
            self.expert_bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("expert_bias", None)

        # Shared expert (BF16/FP8 Dense MLP + sigmoid scalar gate)
        self.shared_expert = Qwen4_5DenseMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_expert_intermediate_size,
            gate_beta=self.gate_beta,
            up_beta=self.up_beta,
            use_triton=self.use_triton_mxfp4,
            dense_linear_mode=self.dense_linear_mode,
            dtype=dtype,
            device=device,
        )
        if shared_expert_gate:
            self.shared_expert_gate: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, 1, bias=False, dtype=dtype, device=device
            )
        else:
            self.shared_expert_gate = None

        # Routed experts MXFP4 packed buffers (uint8; local shard [E_local, N, K//2])
        if init_mxfp4_weights:
            self.register_buffer(
                "w13_u8",
                torch.zeros(
                    (
                        self.local_num_experts,
                        2 * self.moe_intermediate_size,
                        self.latent_dim // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w13_scale",
                torch.full(
                    (
                        self.local_num_experts,
                        2 * self.moe_intermediate_size,
                        self.latent_dim // MXFP4_BLOCK_SIZE,
                    ),
                    127,
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w2_u8",
                torch.zeros(
                    (
                        self.local_num_experts,
                        self.latent_dim,
                        self.moe_intermediate_size // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w2_scale",
                torch.full(
                    (
                        self.local_num_experts,
                        self.latent_dim,
                        self.moe_intermediate_size // MXFP4_BLOCK_SIZE,
                    ),
                    127,
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.register_buffer("w13_u8", None)
            self.register_buffer("w13_scale", None)
            self.register_buffer("w2_u8", None)
            self.register_buffer("w2_scale", None)

        if self.enable_shared_overlap and device is not None:
            dev = torch.device(device)
            if dev.type == "cuda":
                self.prepare_shared_expert_overlap(device=dev)

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        all_to_all_comm: Any = None,
        dp_equal_tokens: Optional[bool] = None,
        use_symm_p2p_combine: Optional[bool] = None,
        dense_linear_mode: Optional[str] = None,
        enable_shared_overlap: bool = True,
        backend: str = "auto",
    ) -> "Qwen4_5LatentMoE":
        return cls(
            config=config,
            use_triton_mxfp4=use_triton_mxfp4,
            compute_mode=compute_mode,
            dtype=dtype,
            device=device,
            init_mxfp4_weights=init_mxfp4_weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
            all_to_all_comm=all_to_all_comm,
            dp_equal_tokens=dp_equal_tokens,
            use_symm_p2p_combine=use_symm_p2p_combine,
            dense_linear_mode=dense_linear_mode,
            enable_shared_overlap=enable_shared_overlap,
            backend=backend,
        )

    @torch.no_grad()
    def load_mxfp4_weights(
        self,
        *,
        w13_u8: Optional[torch.Tensor] = None,
        w13_scale: Optional[torch.Tensor] = None,
        w_gate_u8: Optional[torch.Tensor] = None,
        w_gate_scale: Optional[torch.Tensor] = None,
        w_up_u8: Optional[torch.Tensor] = None,
        w_up_scale: Optional[torch.Tensor] = None,
        w2_u8: torch.Tensor,
        w2_scale: torch.Tensor,
        transpose_weights: bool = False,
    ) -> None:
        """Load MXFP4 routed expert weights from fused or separate checkpoint tensors."""
        prepared = prepare_mxfp4_moe_weights(
            w13_u8=w13_u8,
            w13_scale=w13_scale,
            w_gate_u8=w_gate_u8,
            w_gate_scale=w_gate_scale,
            w_up_u8=w_up_u8,
            w_up_scale=w_up_scale,
            w2_u8=w2_u8,
            w2_scale=w2_scale,
            transpose_weights=transpose_weights,
        )
        self.w13_u8 = prepared.w13_u8
        self.w13_scale = prepared.w13_scale
        self.w2_u8 = prepared.w2_u8
        self.w2_scale = prepared.w2_scale
        self.weights_transposed = prepared.weights_transposed
        self._mxfp4_weights_cache = None

    def prepare_shared_expert_overlap(
        self, device: Optional[torch.device] = None
    ) -> None:
        """Initialize auxiliary CUDA stream and event pair for shared-expert overlap."""
        if not self.enable_shared_overlap or not torch.cuda.is_available() or torch.version.hip is not None:
            return
        dev = device if device is not None else self.shared_expert.gate_proj.weight.device
        if dev.type != "cuda":
            return
        dev_idx = 0 if dev.index is None else int(dev.index)
        self._shared_stream = _shared_stream_for_device(dev_idx)
        self._shared_input_ready = torch.cuda.Event()
        self._shared_output_ready = torch.cuda.Event()

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        transpose_mxfp4: bool = False,
        fuse_shared_gate_up: bool = True,
        fuse_latent_down_gate: bool = True,
        prepare_deepgemm: bool = False,
        prepare_flashinfer: bool = False,
        dense_linear_mode: Optional[str] = None,
    ) -> None:
        """Prepare MXFP4 routed weights, SM100 scale caches, shared expert, and overlap stream."""
        if self.w13_u8 is not None and self.w2_u8 is not None:
            if transpose_mxfp4 and not self.weights_transposed:
                self.w13_u8 = self.w13_u8.transpose(1, 2).contiguous()
                self.w2_u8 = self.w2_u8.transpose(1, 2).contiguous()
                self.weights_transposed = True
            elif not transpose_mxfp4 and self.weights_transposed:
                self.w13_u8 = self.w13_u8.transpose(1, 2).contiguous()
                self.w2_u8 = self.w2_u8.transpose(1, 2).contiguous()
                self.weights_transposed = False
            else:
                self.w13_u8 = self.w13_u8.contiguous()
                self.w13_scale = self.w13_scale.contiguous()
                self.w2_u8 = self.w2_u8.contiguous()
                self.w2_scale = self.w2_scale.contiguous()
            self._mxfp4_weights_cache = None
            weights = self._mxfp4_weights()
            if prepare_deepgemm and not self.weights_transposed:
                weights.get_deepgemm_pair()
            if prepare_flashinfer and not self.weights_transposed:
                weights.get_flashinfer_pair()
        if dense_linear_mode is not None:
            self.dense_linear_mode = str(dense_linear_mode).lower()
        eff_mode = (
            self.dense_linear_mode
            or os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "bf16").strip().lower()
        )
        self.shared_expert.prepare_weights(
            fuse_gate_up=fuse_shared_gate_up,
            dense_linear_mode=eff_mode,
            allow_mxfp8=False,
        )
        if (
            fuse_latent_down_gate
            and fuse_shared_gate_up
            and self.shared_expert._gate_up_weight is not None
        ):
            w_3x = torch.cat(
                [
                    self.latent_down_proj.weight.data,
                    self.latent_gate_proj.weight.data,
                    self.shared_expert._gate_up_weight,
                ],
                dim=0,
            ).contiguous()
            self._latent_down_gate_shared_gu_weight = w_3x
            self._latent_down_gate_weight = w_3x[: 2 * self.latent_dim]
            self.shared_expert._gate_up_weight = w_3x[2 * self.latent_dim :]
            if (
                self.shared_expert._fp8_gate_up is not None
                and self.shared_expert._fp8_gate_up.weight is not None
            ):
                self.shared_expert._fp8_gate_up.weight.data = self.shared_expert._gate_up_weight
        elif fuse_latent_down_gate:
            self._latent_down_gate_shared_gu_weight = None
            self._latent_down_gate_weight = torch.cat(
                [self.latent_down_proj.weight.data, self.latent_gate_proj.weight.data], dim=0
            ).contiguous()
        else:
            self._latent_down_gate_shared_gu_weight = None
            self._latent_down_gate_weight = None

        if self.shared_expert_gate is not None:
            self._latent_shared_gates_weight = torch.cat(
                [self.latent_scale_gate.weight.data, self.shared_expert_gate.weight.data], dim=0
            ).contiguous()
        else:
            self._latent_shared_gates_weight = None

        if eff_mode != "bf16" and self.latent_down_proj.weight.is_cuda:
            # Keep pre-router/pre-SiTU-GLU latent inputs in BF16 under auto so zero router flips
            # or SiTU-GLU error amplification occur; quantize pure output projection latent_up to MXFP8.
            latent_in_kw = dict(
                mode=eff_mode,
                keep_bf16=True,
                prepare_trtllm_ll=False,
                prepare_mxfp8=(eff_mode == "mxfp8"),
                prepare_w4a8=(eff_mode == "w4a8"),
                allow_decode_scaled_mm=False,
            )
            latent_out_kw = dict(latent_in_kw, prepare_mxfp8=True)
            if self._latent_down_gate_weight is not None:
                self._fp8_latent_down_gate = SM100FP8Linear(
                    self.hidden_size,
                    2 * self.latent_dim,
                    **latent_in_kw,
                    dtype=self.latent_down_proj.weight.dtype,
                    device=self.latent_down_proj.weight.device,
                )
                self._fp8_latent_down_gate.prepare_weights(self._latent_down_gate_weight)
                self._fp8_latent_down = None
                self._fp8_latent_gate = None
            else:
                self._fp8_latent_down_gate = None
                self._fp8_latent_down = SM100FP8Linear.from_linear(
                    self.latent_down_proj, **latent_in_kw
                )
                self._fp8_latent_gate = SM100FP8Linear.from_linear(
                    self.latent_gate_proj, **latent_in_kw
                )
            self._fp8_latent_up = SM100FP8Linear.from_linear(
                self.latent_up_proj, **latent_out_kw
            )
        else:
            self._fp8_latent_down_gate = None
            self._fp8_latent_down = None
            self._fp8_latent_gate = None
            self._fp8_latent_up = None

        self.prepare_shared_expert_overlap()

    def _mxfp4_weights(self) -> MXFP4MoEWeights:
        if (
            self.w13_u8 is None
            or self.w13_scale is None
            or self.w2_u8 is None
            or self.w2_scale is None
        ):
            raise RuntimeError(
                "MXFP4 routed expert weights are not initialized; call load_mxfp4_weights() first."
            )
        cached = self._mxfp4_weights_cache
        if (
            cached is not None
            and cached.w13_u8 is self.w13_u8
            and cached.w13_scale is self.w13_scale
            and cached.w2_u8 is self.w2_u8
            and cached.w2_scale is self.w2_scale
            and cached.weights_transposed == self.weights_transposed
        ):
            return cached
        self._mxfp4_weights_cache = MXFP4MoEWeights(
            w13_u8=self.w13_u8,
            w13_scale=self.w13_scale,
            w2_u8=self.w2_u8,
            w2_scale=self.w2_scale,
            weights_transposed=self.weights_transposed,
        )
        return self._mxfp4_weights_cache

    def route(
        self, routed: torch.Tensor, *, use_triton: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k expert IDs and normalized weights from latent activations.

        Exact specification (``router_precision="fp32_logits"``):
          1. ``logits = F.linear(routed, gate_weight).float()``
          2. ``scores = torch.sigmoid(logits)``
          3. ``selection = scores + expert_bias.float()``
          4. ``topk_ids = selection.topk(top_k, dim=-1).indices``
          5. ``topk_weights = scores.gather(-1, topk_ids)``
          6. ``topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-8)``
          7. ``topk_weights = topk_weights.to(routed.dtype).float()``
        """
        if self.router_precision == "fp32":
            logits = F.linear(routed.float(), self.gate.weight.float())
        else:
            # Default "fp32_logits": linear in input dtype, cast logits to FP32.
            logits = F.linear(routed, self.gate.weight)

        run_triton = (
            self.use_triton_mxfp4 if use_triton is None else bool(use_triton)
        )
        if (
            run_triton
            and logits.is_cuda
            and self.num_experts <= 2048
            and self.top_k <= 16
            and routed.dtype in (torch.bfloat16, torch.float32)
        ):
            if logits.stride(-1) != 1:
                logits = logits.contiguous()
            M = logits.shape[0]
            topk_ids = torch.empty((M, self.top_k), dtype=torch.int32, device=logits.device)
            topk_weights = torch.empty((M, self.top_k), dtype=torch.float32, device=logits.device)
            if M == 0:
                return topk_ids, topk_weights
            bias_t = self.expert_bias if self.expert_bias is not None else logits
            E_PAD = triton.next_power_of_2(self.num_experts)
            TOP_K_PAD = triton.next_power_of_2(max(self.top_k, 2))
            with _cuda_device_ctx(logits.device):
                _qwen45_sigmoid_topk_router_kernel[(M,)](
                    logits,
                    bias_t,
                    topk_ids,
                    topk_weights,
                    self.num_experts,
                    logits.stride(0),
                    E_PAD=E_PAD,
                    TOP_K=self.top_k,
                    TOP_K_PAD=TOP_K_PAD,
                    HAS_BIAS=(self.expert_bias is not None),
                    NORMALIZE=self.normalize_topk_scores,
                    ROUND_BF16=(routed.dtype == torch.bfloat16),
                    num_warps=4,
                )
            return topk_ids, topk_weights

        logits_f = logits.float()
        scores = torch.sigmoid(logits_f)
        if self.expert_bias is not None:
            selection = scores + self.expert_bias.float()
        else:
            selection = scores
        topk_ids = torch.topk(selection, k=self.top_k, dim=-1).indices
        topk_weights = scores.gather(-1, topk_ids)
        if self.normalize_topk_scores:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-8
            )
        topk_weights = topk_weights.to(routed.dtype).float()
        return topk_ids, topk_weights

    def _run_latent_norm_mul_sigmoid(
        self, dg: torch.Tensor, *, run_triton: bool
    ) -> torch.Tensor:
        if run_triton and dg.is_cuda:
            dg_2d = dg if dg.stride(-1) == 1 else dg.contiguous()
            M_dg = int(dg_2d.shape[0])
            routed = torch.empty((M_dg, self.latent_dim), dtype=dg_2d.dtype, device=dg_2d.device)
            if M_dg > 0:
                block_k = min(512, triton.next_power_of_2(max(self.latent_dim, 64)))
                with _cuda_device_ctx(dg_2d.device):
                    _latent_norm_mul_sigmoid_kernel[(M_dg,)](
                        dg_2d,
                        self.pre_moe_latent_layernorm.weight.contiguous(),
                        routed,
                        M_dg,
                        self.latent_dim,
                        dg_2d.stride(0),
                        routed.stride(0),
                        eps=float(self.rms_norm_eps),
                        BLOCK_K=block_k,
                        num_warps=4,
                    )
            return routed
        d_part, g_part = dg.split(self.latent_dim, dim=-1)
        return self.pre_moe_latent_layernorm(d_part, use_triton=run_triton) * torch.sigmoid(g_part)

    def _project_and_norm_latent(
        self, x: torch.Tensor, *, run_triton: bool
    ) -> torch.Tensor:
        if self._latent_down_gate_weight is not None:
            dg = (
                self._fp8_latent_down_gate(x)
                if self._fp8_latent_down_gate is not None
                else F.linear(x, self._latent_down_gate_weight)
            )
            return self._run_latent_norm_mul_sigmoid(dg, run_triton=run_triton)
        routed = (
            self._fp8_latent_down(x)
            if self._fp8_latent_down is not None
            else F.linear(x, self.latent_down_proj.weight)
        )
        routed = self.pre_moe_latent_layernorm(routed, use_triton=run_triton)
        l_gate = (
            self._fp8_latent_gate(x)
            if self._fp8_latent_gate is not None
            else F.linear(x, self.latent_gate_proj.weight)
        )
        return routed * torch.sigmoid(l_gate)

    def _forward_routed_from_norm(
        self,
        x: torch.Tensor,
        routed: torch.Tensor,
        *,
        run_triton: bool,
        mode: str,
        backend: str,
        apply_scale_gate: bool = True,
    ) -> torch.Tensor:
        topk_ids, topk_weights = self.route(routed, use_triton=run_triton)
        weights = self._mxfp4_weights()
        use_a2a = (
            self.all_to_all_comm is not None
            and self.ep_size > 1
            and getattr(self.all_to_all_comm, "world_size", 1) > 1
        )
        if use_a2a:
            eff_m_est = int(routed.shape[0]) * self.ep_size
            resolved_b = resolve_moe_backend_for_shape(
                eff_m_est,
                self.latent_dim,
                routed.device,
                weights,
                backend,
                run_triton,
                compute_mode=mode,
            )
            want_fp8_a2a = run_triton and resolved_b in (
                "deepgemm_masked",
                "deepgemm_contiguous",
                "flashinfer",
                "flashinfer_noswap",
            )
            a2a_state = self.all_to_all_comm.dispatch(
                routed,
                topk_ids,
                topk_weights,
                num_experts=self.num_experts,
                equal_tokens=self.dp_equal_tokens,
                quantize_fp8=want_fp8_a2a,
            )
            if run_triton:
                routed_ep = mxfp4_moe_forward(
                    a2a_state.hidden,
                    a2a_state.topk_ids,
                    a2a_state.topk_weights,
                    weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    compute_mode=mode,
                    use_triton=True,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                    backend=backend,
                    prequant_fp8=a2a_state.hidden_fp8,
                    out_dtype=routed.dtype,
                )
            else:
                assert a2a_state.hidden is not None
                routed_ep = mxfp4_moe_forward_ref(
                    a2a_state.hidden,
                    a2a_state.topk_ids,
                    a2a_state.topk_weights,
                    weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                )
            routed = self.all_to_all_comm.combine(
                routed_ep, a2a_state, use_symm_p2p=self.use_symm_p2p_combine
            )
        else:
            if run_triton:
                routed = mxfp4_moe_forward(
                    routed,
                    topk_ids,
                    topk_weights,
                    weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    compute_mode=mode,
                    use_triton=True,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                    backend=backend,
                )
            else:
                routed = mxfp4_moe_forward_ref(
                    routed,
                    topk_ids,
                    topk_weights,
                    weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                )
            if self.ep_size > 1 and self.ep_group is not None:
                if (
                    self.fast_all_reduce is not None
                    and self.fast_all_reduce.group is self.ep_group
                ):
                    routed = self.fast_all_reduce(routed)
                else:
                    if not routed.is_contiguous():
                        routed = routed.contiguous()
                    torch.distributed.all_reduce(
                        routed, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
                    )

        routed = (
            self._fp8_latent_up(routed)
            if self._fp8_latent_up is not None
            else F.linear(routed, self.latent_up_proj.weight)
        )
        if apply_scale_gate:
            scale = torch.sigmoid(F.linear(x, self.latent_scale_gate.weight))
            return routed * scale
        return routed

    def _forward_routed(
        self,
        x: torch.Tensor,
        *,
        run_triton: bool,
        mode: str,
        backend: str,
        apply_scale_gate: bool = True,
    ) -> torch.Tensor:
        routed = self._project_and_norm_latent(x, run_triton=run_triton)
        return self._forward_routed_from_norm(
            x,
            routed,
            run_triton=run_triton,
            mode=mode,
            backend=backend,
            apply_scale_gate=apply_scale_gate,
        )

    def _forward_shared(
        self, x: torch.Tensor, *, run_triton: bool, apply_gate: bool = True
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty_like(x)
        shared = self.shared_expert(x, use_triton=run_triton, skip_allreduce=True)
        if apply_gate and self.shared_expert_gate is not None:
            shared = shared * torch.sigmoid(F.linear(x, self.shared_expert_gate.weight))
        return shared

    def _combine_latent_and_shared(
        self,
        x: torch.Tensor,
        routed_up: torch.Tensor,
        shared: torch.Tensor,
        *,
        run_triton: bool,
    ) -> torch.Tensor:
        M = int(x.shape[0])
        if M == 0:
            return routed_up + shared
        has_sg = self.shared_expert_gate is not None
        if has_sg:
            if self._latent_shared_gates_weight is not None:
                gates = F.linear(x, self._latent_shared_gates_weight)
            else:
                gates = torch.cat(
                    [
                        F.linear(x, self.latent_scale_gate.weight),
                        F.linear(x, self.shared_expert_gate.weight),
                    ],
                    dim=-1,
                )
        else:
            gates = F.linear(x, self.latent_scale_gate.weight)
        if run_triton and x.is_cuda:
            r_2d = routed_up if routed_up.stride(-1) == 1 else routed_up.contiguous()
            s_2d = shared if shared.stride(-1) == 1 else shared.contiguous()
            g_2d = gates if gates.stride(-1) == 1 else gates.contiguous()
            out = torch.empty((M, self.hidden_size), dtype=r_2d.dtype, device=r_2d.device)
            block_h = min(1024, triton.next_power_of_2(max(self.hidden_size, 64)))
            with _cuda_device_ctx(r_2d.device):
                _combine_latent_shared_out_kernel[(M, triton.cdiv(self.hidden_size, block_h))](
                    r_2d,
                    s_2d,
                    g_2d,
                    out,
                    r_2d.stride(0),
                    s_2d.stride(0),
                    g_2d.stride(0),
                    out.stride(0),
                    H=self.hidden_size,
                    HAS_SHARED_GATE=has_sg,
                    IS_BF16=(
                        r_2d.dtype == torch.bfloat16
                        and s_2d.dtype == torch.bfloat16
                        and g_2d.dtype == torch.bfloat16
                    ),
                    BLOCK_H=block_h,
                    num_warps=4,
                )
            return out
        if has_sg:
            if gates.shape[-1] == 2:
                g_r, g_s = gates[:, :1], gates[:, 1:2]
            else:
                g_r = gates
                g_s = F.linear(x, self.shared_expert_gate.weight)
            return routed_up * torch.sigmoid(g_r) + shared * torch.sigmoid(g_s)
        return routed_up * torch.sigmoid(gates) + shared

    def _run_with_shared_stream(
        self,
        device: torch.device,
        shared_fn: Any,
        main_fn: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            self._shared_stream is None
            or self._shared_input_ready is None
            or self._shared_output_ready is None
            or self._shared_stream.device != device
        ) and not capturing:
            self.prepare_shared_expert_overlap(device=device)

        stream = self._shared_stream
        if (
            stream is not None
            and self._shared_input_ready is not None
            and self._shared_output_ready is not None
            and stream.device == device
        ):
            main = torch.cuda.current_stream(device)
            self._shared_input_ready.record(main)
            shared: Optional[torch.Tensor] = None
            inner_exc = False
            try:
                with torch.cuda.stream(stream):
                    self._shared_input_ready.wait(stream)
                    shared_exc = False
                    try:
                        shared = shared_fn()
                    except BaseException:
                        shared_exc = True
                        raise
                    finally:
                        if not (capturing and shared_exc):
                            self._shared_output_ready.record(stream)
                routed = main_fn()
            except BaseException:
                inner_exc = True
                raise
            finally:
                if capturing and inner_exc:
                    try:
                        self._shared_output_ready.wait(main)
                    except Exception:
                        pass
                else:
                    self._shared_output_ready.wait(main)
            assert shared is not None
            if not capturing:
                shared.record_stream(main)
            return routed, shared
        routed = main_fn()
        shared = shared_fn()
        return routed, shared

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton_mxfp4: Optional[bool] = None,
        compute_mode: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> torch.Tensor:
        """Run Latent MoE + Shared Expert forward pass with optional aux_stream overlap.

        Accepts 2D ``[M, hidden_size]`` or 3D ``[B, T, hidden_size]`` tensors.
        """
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        has_a2a = (
            self.all_to_all_comm is not None
            and self.ep_size > 1
            and getattr(self.all_to_all_comm, "world_size", 1) > 1
        )
        # Never skip DP-EP collective dispatch on a 0-token rank: other DP ranks may
        # route tokens to this rank's local experts inside all_to_all_comm.
        if x.shape[0] == 0 and not has_a2a:
            return torch.empty_like(hidden_states)

        use_triton = (
            self.use_triton_mxfp4 if use_triton_mxfp4 is None else bool(use_triton_mxfp4)
        )
        mode = self.compute_mode if compute_mode is None else compute_mode
        eff_backend = self.backend if backend is None else backend
        run_triton = use_triton and x.is_cuda and supports_triton_mxfp4(x.device)
        use_overlap = self.enable_shared_overlap and x.is_cuda and torch.version.hip is None
        M = int(x.shape[0])

        if (
            not has_a2a
            and self._latent_down_gate_shared_gu_weight is not None
            and M > 0
            and x.is_cuda
        ):
            env_mode = os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "")
            if (
                not env_mode
                and self.dense_linear_mode in ("bf16", "auto")
                and (
                    self.dense_linear_mode == "bf16"
                    or self._fp8_latent_down_gate is None
                    or self._fp8_latent_down_gate.w_mxfp8 is None
                )
            ):
                ldg_is_bf16 = True
            else:
                ldg_is_bf16 = (
                    self._fp8_latent_down_gate is None
                    or self._fp8_latent_down_gate._resolve_backend(M, x.device) == "bf16"
                )
            if ldg_is_bf16:
                if not x.is_contiguous():
                    x = x.contiguous()
                all_up = F.linear(x, self._latent_down_gate_shared_gu_weight)
                dg = all_up[:, : 2 * self.latent_dim]
                sgu = all_up[:, 2 * self.latent_dim :]
                routed_norm = self._run_latent_norm_mul_sigmoid(dg, run_triton=run_triton)
                capturing = torch.cuda.is_current_stream_capturing() if use_overlap else False
                if use_overlap and (capturing or M >= 1024):
                    routed_up, shared = self._run_with_shared_stream(
                        x.device,
                        lambda: self.shared_expert.forward_from_gate_up(
                            sgu, use_triton=run_triton, skip_allreduce=True
                        ),
                        lambda: self._forward_routed_from_norm(
                            x,
                            routed_norm,
                            run_triton=run_triton,
                            mode=mode,
                            backend=eff_backend,
                            apply_scale_gate=False,
                        ),
                    )
                else:
                    routed_up = self._forward_routed_from_norm(
                        x,
                        routed_norm,
                        run_triton=run_triton,
                        mode=mode,
                        backend=eff_backend,
                        apply_scale_gate=False,
                    )
                    shared = self.shared_expert.forward_from_gate_up(
                        sgu, use_triton=run_triton, skip_allreduce=True
                    )
                return self._combine_latent_and_shared(
                    x, routed_up, shared, run_triton=run_triton
                ).reshape(orig_shape)

        if use_overlap:
            if not x.is_contiguous():
                x = x.contiguous()
            routed_up, shared = self._run_with_shared_stream(
                x.device,
                lambda: self._forward_shared(x, run_triton=run_triton, apply_gate=False),
                lambda: self._forward_routed(
                    x,
                    run_triton=run_triton,
                    mode=mode,
                    backend=eff_backend,
                    apply_scale_gate=False,
                ),
            )
            return self._combine_latent_and_shared(
                x, routed_up, shared, run_triton=run_triton
            ).reshape(orig_shape)

        routed_up = self._forward_routed(
            x,
            run_triton=run_triton,
            mode=mode,
            backend=eff_backend,
            apply_scale_gate=False,
        )
        shared = self._forward_shared(x, run_triton=run_triton, apply_gate=False)
        return self._combine_latent_and_shared(
            x, routed_up, shared, run_triton=run_triton
        ).reshape(orig_shape)

    def _compute_latent_and_route(
        self, x: torch.Tensor, *, run_triton: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routed = self._project_and_norm_latent(x, run_triton=run_triton)
        topk_ids, topk_weights = self.route(routed, use_triton=run_triton)
        return routed, topk_ids, topk_weights

    def forward_route_and_dispatch(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton_mxfp4: Optional[bool] = None,
        compute_mode: Optional[str] = None,
        backend: Optional[str] = None,
        equal_tokens: Optional[bool] = None,
        slot: int = 0,
    ) -> LatentMoETBOContext:
        """TBO Stage 1: project latent + route + launch async AllToAll dispatch."""
        orig_shape = tuple(hidden_states.shape)
        x = hidden_states.reshape(-1, self.hidden_size)
        has_a2a = (
            self.all_to_all_comm is not None
            and self.ep_size > 1
            and getattr(self.all_to_all_comm, "world_size", 1) > 1
        )
        use_triton = (
            self.use_triton_mxfp4 if use_triton_mxfp4 is None else bool(use_triton_mxfp4)
        )
        mode = self.compute_mode if compute_mode is None else compute_mode
        eff_backend = self.backend if backend is None else backend
        run_triton = use_triton and x.is_cuda and supports_triton_mxfp4(x.device)
        weights = self._mxfp4_weights()
        if x.shape[0] == 0 and not has_a2a:
            return LatentMoETBOContext(
                orig_shape=orig_shape,
                x=x,
                run_triton=run_triton,
                mode=mode,
                backend=eff_backend,
                weights=weights,
                has_a2a=False,
                early_return=torch.empty_like(hidden_states),
            )
        routed, topk_ids, topk_weights = self._compute_latent_and_route(x, run_triton=run_triton)
        dispatch_handle = None
        if has_a2a:
            assert self.all_to_all_comm is not None
            eff_m_est = int(routed.shape[0]) * self.ep_size
            resolved_b = resolve_moe_backend_for_shape(
                eff_m_est,
                self.latent_dim,
                routed.device,
                weights,
                eff_backend,
                run_triton,
                compute_mode=mode,
            )
            want_fp8_a2a = run_triton and resolved_b in (
                "deepgemm_masked",
                "deepgemm_contiguous",
                "flashinfer",
                "flashinfer_noswap",
            )
            eq_tok = self.dp_equal_tokens if equal_tokens is None else equal_tokens
            dispatch_handle = self.all_to_all_comm.dispatch_async(
                routed,
                topk_ids,
                topk_weights,
                num_experts=self.num_experts,
                equal_tokens=eq_tok,
                quantize_fp8=want_fp8_a2a,
                slot=slot,
            )
        return LatentMoETBOContext(
            orig_shape=orig_shape,
            x=x,
            run_triton=run_triton,
            mode=mode,
            backend=eff_backend,
            weights=weights,
            has_a2a=has_a2a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            routed=routed,
            dispatch_handle=dispatch_handle,
        )

    def forward_routed_compute_and_combine(
        self, ctx: LatentMoETBOContext, *, slot: int = 0
    ) -> LatentMoETBOContext:
        """TBO Stage 2: wait dispatch -> routed MXFP4 -> launch async combine + shared."""
        if ctx.early_return is not None:
            return ctx
        assert ctx.routed is not None and ctx.topk_ids is not None and ctx.topk_weights is not None
        if ctx.has_a2a:
            assert self.all_to_all_comm is not None and ctx.dispatch_handle is not None
            a2a_state = self.all_to_all_comm.wait_dispatch(ctx.dispatch_handle)
            if ctx.run_triton:
                routed_ep = mxfp4_moe_forward(
                    a2a_state.hidden,
                    a2a_state.topk_ids,
                    a2a_state.topk_weights,
                    ctx.weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    compute_mode=ctx.mode,
                    use_triton=True,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                    backend=ctx.backend,
                    prequant_fp8=a2a_state.hidden_fp8,
                    out_dtype=ctx.routed.dtype,
                )
            else:
                assert a2a_state.hidden is not None
                routed_ep = mxfp4_moe_forward_ref(
                    a2a_state.hidden,
                    a2a_state.topk_ids,
                    a2a_state.topk_weights,
                    ctx.weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                )
            ctx.combine_handle = self.all_to_all_comm.combine_async(
                routed_ep,
                a2a_state,
                use_symm_p2p=self.use_symm_p2p_combine,
                slot=slot,
            )
            # Compute collective-free shared branch on main stream while combine runs on comm_stream.
            ctx.shared = self._forward_shared(ctx.x, run_triton=ctx.run_triton, apply_gate=False)
        else:
            if ctx.run_triton:
                routed = mxfp4_moe_forward(
                    ctx.routed,
                    ctx.topk_ids,
                    ctx.topk_weights,
                    ctx.weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    compute_mode=ctx.mode,
                    use_triton=True,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                    backend=ctx.backend,
                )
            else:
                routed = mxfp4_moe_forward_ref(
                    ctx.routed,
                    ctx.topk_ids,
                    ctx.topk_weights,
                    ctx.weights,
                    gate_beta=self.gate_beta,
                    up_beta=self.up_beta,
                    ep_start=self.ep_start,
                    ep_size=self.ep_size,
                )
            if self.ep_size > 1 and self.ep_group is not None:
                if self.fast_all_reduce is not None and self.fast_all_reduce.group is self.ep_group:
                    routed = self.fast_all_reduce(routed)
                else:
                    if not routed.is_contiguous():
                        routed = routed.contiguous()
                    torch.distributed.all_reduce(
                        routed, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
                    )
            ctx.routed = routed
            ctx.shared = self._forward_shared(ctx.x, run_triton=ctx.run_triton, apply_gate=False)
        return ctx

    def forward_finish_combine(self, ctx: LatentMoETBOContext) -> torch.Tensor:
        """TBO Stage 3: wait combine -> latent up + fused scale gates + add shared expert."""
        if ctx.early_return is not None:
            return ctx.early_return
        if ctx.has_a2a:
            assert self.all_to_all_comm is not None and ctx.combine_handle is not None
            routed = self.all_to_all_comm.wait_combine(ctx.combine_handle)
        else:
            assert ctx.routed is not None
            routed = ctx.routed
        routed = (
            self._fp8_latent_up(routed)
            if self._fp8_latent_up is not None
            else F.linear(routed, self.latent_up_proj.weight)
        )
        shared = (
            ctx.shared
            if ctx.shared is not None
            else self._forward_shared(ctx.x, run_triton=ctx.run_triton, apply_gate=False)
        )
        return self._combine_latent_and_shared(
            ctx.x, routed, shared, run_triton=ctx.run_triton
        ).reshape(ctx.orig_shape)

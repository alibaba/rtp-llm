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

from rtp_llm.models_py.modules.qwen4_5.activation import (
    SituAndMul,
    bf16_linear_tgv_or_torch,
    can_sm10x_bf16_tgv,
    get_sm10x_bf16_tgv_backend,
    run_sm10x_bf16_tgv_gemm,
    select_sm10x_bf16_tgv_tactic,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    FastAllReduce,
    sp_slice_bounds,
)
from rtp_llm.models_py.modules.qwen4_5.fp8_linear import (
    SM100FP8Linear,
    _cuda_device_ctx,
    has_deepgemm_fp8,
)


def _resolve_sp_comm(
    sp_group: Any,
    sp_fast_comm: Any,
    fast_all_reduce: Any,
) -> Any:
    if sp_fast_comm is not None:
        return sp_fast_comm
    if fast_all_reduce is not None and (
        sp_group is None or getattr(fast_all_reduce, "group", None) is sp_group
    ):
        return fast_all_reduce
    return None


def _sp_all_gather(
    x: torch.Tensor,
    *,
    sp_size: int,
    sp_rank: int,
    sp_group: Any,
    sp_comm: Any,
    total_tokens: Optional[int],
) -> torch.Tensor:
    if sp_size <= 1 or sp_group is None:
        return x[:total_tokens] if total_tokens is not None and x.shape[0] != total_tokens else x
    if sp_comm is not None and hasattr(sp_comm, "all_gather_sp"):
        return sp_comm.all_gather_sp(x, total_tokens=total_tokens)
    dist = torch.distributed
    T_in = int(x.shape[0])
    rest = tuple(x.shape[1:])
    T = T_in * sp_size if total_tokens is None else int(total_tokens)
    if T == 0:
        return torch.empty((0, *rest), dtype=x.dtype, device=x.device)
    chunk = (T + sp_size - 1) // sp_size
    padded_T = sp_size * chunk
    x_c = x if x.is_contiguous() else x.contiguous()
    if T_in == chunk:
        src_pad = x_c
    else:
        src_pad = torch.empty((chunk, *rest), dtype=x_c.dtype, device=x_c.device)
        copy_rows = min(T_in, chunk)
        if copy_rows > 0:
            src_pad[:copy_rows].copy_(x_c[:copy_rows])
    if dist.get_backend(sp_group) == "gloo" or not x_c.is_cuda:
        cpu_in = src_pad.detach().cpu() if src_pad.is_cuda else src_pad
        chunks = [torch.empty_like(cpu_in) for _ in range(sp_size)]
        dist.all_gather(chunks, cpu_in, group=sp_group)
        full = torch.cat(chunks, dim=0)[:T]
        return full.to(device=x_c.device, dtype=x_c.dtype) if x_c.is_cuda else full
    dst_full = torch.empty((padded_T, *rest), dtype=x_c.dtype, device=x_c.device)
    dist.all_gather_into_tensor(dst_full.view(sp_size, chunk, *rest), src_pad, group=sp_group)
    return dst_full[:T]


def _sp_reduce_scatter(
    x: torch.Tensor,
    *,
    sp_size: int,
    sp_rank: int,
    sp_group: Any,
    sp_comm: Any,
    total_tokens: Optional[int],
) -> torch.Tensor:
    if sp_size <= 1 or sp_group is None:
        return x[:total_tokens] if total_tokens is not None and x.shape[0] != total_tokens else x
    if sp_comm is not None and hasattr(sp_comm, "reduce_scatter_sp"):
        return sp_comm.reduce_scatter_sp(x, total_tokens=total_tokens)
    dist = torch.distributed
    T_in = int(x.shape[0])
    rest = tuple(x.shape[1:])
    T = T_in if total_tokens is None else int(total_tokens)
    if T == 0:
        return torch.empty((0, *rest), dtype=x.dtype, device=x.device)
    s0, s1 = sp_slice_bounds(T, sp_size, sp_rank)
    local_T = s1 - s0
    chunk = (T + sp_size - 1) // sp_size
    padded_T = sp_size * chunk
    x_c = x if x.is_contiguous() else x.contiguous()
    if T_in == padded_T:
        src_full = x_c
    else:
        src_full = torch.empty((padded_T, *rest), dtype=x_c.dtype, device=x_c.device)
        src_full[:T].copy_(x_c[:T])
        if padded_T > T:
            src_full[T:padded_T].zero_()
    if dist.get_backend(sp_group) == "gloo" or not x_c.is_cuda:
        cpu_in = (src_full.detach().cpu() if src_full.is_cuda else src_full).clone()
        dist.all_reduce(cpu_in, op=dist.ReduceOp.SUM, group=sp_group)
        loc = cpu_in[s0:s1]
        return loc.to(device=x_c.device, dtype=x_c.dtype) if x_c.is_cuda else loc.clone()
    dst_chunk = torch.empty((chunk, *rest), dtype=x_c.dtype, device=x_c.device)
    dist.reduce_scatter_tensor(dst_chunk, src_full, op=dist.ReduceOp.SUM, group=sp_group)
    return dst_chunk if local_T == chunk else dst_chunk[:local_T].clone()


def _bind_fp8_bf16_alias(fp8_mod: SM100FP8Linear, w_tensor: torch.Tensor) -> None:
    """Bind BF16 weight view as a non-persistent buffer alias (zero duplicate BF16/state_dict)."""
    fp8_mod._parameters.pop("weight", None)
    fp8_mod.register_buffer("weight", w_tensor.data, persistent=False)
    if hasattr(fp8_mod, "_refresh_fast_dispatch"):
        fp8_mod._refresh_fast_dispatch()


def _wrap_moe_fp8_linear(
    in_features: int,
    out_features: int,
    w_tensor: torch.Tensor,
    **lin_kw: Any,
) -> SM100FP8Linear:
    """Build zero-copy SM100FP8Linear aliasing existing BF16 tensor before quantizing."""
    kw = dict(lin_kw, keep_bf16=False)
    inst = SM100FP8Linear(
        in_features,
        out_features,
        **kw,
        dtype=w_tensor.dtype,
        device=w_tensor.device,
    )
    _bind_fp8_bf16_alias(inst, w_tensor)
    inst.prepare_weights()
    return inst
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    MXFP4_BLOCK_SIZE,
    MXFP4MoEWeights,
    mxfp4_moe_forward,
    mxfp4_moe_forward_ref,
    prepare_mxfp4_moe_weights,
    resolve_moe_backend_for_shape,
    supports_triton_mxfp4,
    warmup_deepgemm_mxfp4_moe_kernels,
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



@triton.jit(do_not_specialize=["M", "stride_xm", "stride_wm", "stride_om"])
def _gates_2col_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    H: tl.constexpr,
    C: tl.constexpr,
    stride_xm: tl.int64,
    stride_wm: tl.int64,
    stride_om: tl.int64,
    BLOCK_K: tl.constexpr,
):
    """Anti-cutlass_80 Triton projection [M, H] @ [H, C] for small M (C in {1, 2})."""
    row = tl.program_id(0).to(tl.int64)
    if row >= M:
        return
    xb = x_ptr + row * stride_xm
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for off in range(0, H, BLOCK_K):
        ko = off + tl.arange(0, BLOCK_K)
        mk = ko < H
        xv = tl.load(xb + ko, mask=mk, other=0.0).to(tl.float32)
        w0 = tl.load(w_ptr + ko, mask=mk, other=0.0).to(tl.float32)
        acc0 += xv * w0
        if C == 2:
            w1 = tl.load(w_ptr + stride_wm + ko, mask=mk, other=0.0).to(tl.float32)
            acc1 += xv * w1
    ob = out_ptr + row * stride_om
    tl.store(ob, tl.sum(acc0, axis=0).to(out_ptr.dtype.element_ty))
    if C == 2:
        tl.store(ob + 1, tl.sum(acc1, axis=0).to(out_ptr.dtype.element_ty))


@triton.jit
def _qwen45_router_splitk_gemm_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    K: tl.constexpr,
    E: tl.constexpr,
    stride_xm: tl.int64,
    stride_wn: tl.int64,
    stride_ps: tl.int64,
    stride_pm: tl.int64,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute Split-K partial BF16->FP32 router logits ``[SPLIT_K, M, E]`` without splitKreduce."""
    pid_n = tl.program_id(0)
    pid_sk = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < E
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * k_per
    k_end = tl.minimum(K, k_start + k_per)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < k_end
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mk[None, :],
            other=0.0,
        )
        wv = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None],
            mask=mask_n[None, :] & mk[:, None],
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc=acc)

    tl.store(
        part_ptr + pid_sk * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _qwen45_router_splitk_reduce_topk_kernel(
    part_ptr,
    bias_ptr,
    ids_ptr,
    weights_ptr,
    E,
    stride_ps: tl.int64,
    stride_pm: tl.int64,
    SPLIT_K: tl.constexpr,
    E_PAD: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    """Sum Split-K router partials in registers and fuse sigmoid + bias + topk + renormalize."""
    row = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, E_PAD)
    mask_e = offs_e < E
    acc = tl.zeros((E_PAD,), dtype=tl.float32)
    for sk in tl.static_range(SPLIT_K):
        acc += tl.load(
            part_ptr + sk * stride_ps + row * stride_pm + offs_e,
            mask=mask_e,
            other=0.0,
        )

    raw = acc.to(tl.bfloat16).to(tl.float32) if ROUND_BF16 else acc
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


# Re-export circular-dependency-free SM100+ TGV BF16 helpers from activation.py.
_get_flashinfer_tgv_backend = get_sm10x_bf16_tgv_backend
_select_bf16_tgv_tactic = select_sm10x_bf16_tgv_tactic
_can_bf16_tgv = can_sm10x_bf16_tgv
_run_bf16_tgv = run_sm10x_bf16_tgv_gemm
_bf16_linear_tgv_or_torch = bf16_linear_tgv_or_torch


def _dense_linear_or_tgv(
    x: torch.Tensor, w: torch.Tensor, fp8_mod: Optional[SM100FP8Linear]
) -> torch.Tensor:
    """Dispatch to BF16 TGV when active or ``fp8_mod`` resolves to BF16, else ``fp8_mod``/``F.linear``."""
    spec = _can_bf16_tgv(x, w)
    if spec is not None and (fp8_mod is None or fp8_mod._resolve_backend(spec[3], x.device) == "bf16"):
        return _run_bf16_tgv(x, w, spec)
    if fp8_mod is not None:
        return fp8_mod(x)
    return F.linear(x, w)


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
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        sp_fast_comm: Optional[FastAllReduce] = None,
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
        self.sp_size = int(sp_size)
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group if sp_group is not None else tp_group
        self.sp_fast_comm = _resolve_sp_comm(self.sp_group, sp_fast_comm, fast_all_reduce)
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
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        sp_fast_comm: Optional[FastAllReduce] = None,
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
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_fast_comm=sp_fast_comm,
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
            # SiTU-GLU amplifies activation kurtosis; keep gate/up and down_proj BF16 under
            # auto/fp8 (zero w_dg_fp8/w_mxfp8 memory & zero GDN/Dense FP8 drift), while
            # preserving w4a8 and explicit deepgemm_fp8 / mxfp8 / allow_mxfp8 opt-ins.
            use_mxfp8 = bool(allow_mxfp8 or eff_mode == "mxfp8")
            down_use_dg = bool(allow_mxfp8 or eff_mode == "deepgemm_fp8")
            lin_kw = dict(
                mode=eff_mode,
                prepare_trtllm_ll=False,
                prepare_mxfp8=use_mxfp8,
                prepare_deepgemm_fp8=down_use_dg,
                prepare_w4a8=(eff_mode == "w4a8"),
                allow_decode_scaled_mm=False,
            )
            if self._gate_up_weight is not None:
                self._fp8_gate_up = _wrap_moe_fp8_linear(
                    self.hidden_size,
                    2 * self.local_intermediate_size,
                    self._gate_up_weight,
                    **lin_kw,
                )
                self._fp8_gate = None
                self._fp8_up = None
            else:
                self._fp8_gate_up = None
                self._fp8_gate = _wrap_moe_fp8_linear(
                    self.hidden_size, self.local_intermediate_size, self.gate_proj.weight, **lin_kw
                )
                self._fp8_up = _wrap_moe_fp8_linear(
                    self.hidden_size, self.local_intermediate_size, self.up_proj.weight, **lin_kw
                )
            self._fp8_down = _wrap_moe_fp8_linear(
                self.local_intermediate_size, self.hidden_size, self.down_proj.weight, **lin_kw
            )
        else:
            self._fp8_gate_up = None
            self._fp8_gate = None
            self._fp8_up = None
            self._fp8_down = None

    def _finish_down_and_reduce(
        self, act: torch.Tensor, *, skip_allreduce: bool
    ) -> torch.Tensor:
        out = _dense_linear_or_tgv(act, self.down_proj.weight, self._fp8_down)
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
        scale_gate: Optional[torch.Tensor] = None,
        scale_gate_col: int = 0,
    ) -> torch.Tensor:
        """Run SiTU-GLU + down projection from precomputed ``gate_up`` [..., 2*D]."""
        act = self.act_fn(
            gate_up,
            use_triton=use_triton,
            scale_gate=scale_gate,
            scale_gate_col=scale_gate_col,
        )
        return self._finish_down_and_reduce(act, skip_allreduce=skip_allreduce)

    def _forward_compute(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton: Optional[bool],
        skip_allreduce: bool,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return torch.empty_like(hidden_states)
        if self._gate_up_weight is not None and (use_triton is None or use_triton):
            gate_up = _dense_linear_or_tgv(hidden_states, self._gate_up_weight, self._fp8_gate_up)
            return self.forward_from_gate_up(
                gate_up, use_triton=use_triton, skip_allreduce=skip_allreduce
            )
        gate = _dense_linear_or_tgv(hidden_states, self.gate_proj.weight, self._fp8_gate)
        up = _dense_linear_or_tgv(hidden_states, self.up_proj.weight, self._fp8_up)
        act = self.act_fn(gate, up, use_triton=use_triton)
        return self._finish_down_and_reduce(act, skip_allreduce=skip_allreduce)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton: Optional[bool] = None,
        skip_allreduce: bool = False,
        sp_input: bool = False,
        sp_total_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        eff_sp_size = (
            self.sp_size if self.sp_size > 1 else (self.tp_size if self.tp_size > 1 else 1)
        )
        eff_sp_rank = self.sp_rank if self.sp_size > 1 else self.tp_rank
        eff_sp_group = self.sp_group if self.sp_group is not None else self.tp_group
        if sp_input and eff_sp_size > 1 and self.tp_size > 1 and eff_sp_group is not None:
            orig_ndim = hidden_states.ndim
            orig_b = int(hidden_states.shape[0]) if orig_ndim >= 3 else 0
            x_2d = hidden_states.reshape(-1, self.hidden_size)
            if sp_total_tokens == 0 or (sp_total_tokens is None and x_2d.shape[0] == 0):
                return torch.empty_like(hidden_states)
            sp_comm = _resolve_sp_comm(eff_sp_group, self.sp_fast_comm, self.fast_all_reduce)
            x_full = _sp_all_gather(
                x_2d,
                sp_size=eff_sp_size,
                sp_rank=eff_sp_rank,
                sp_group=eff_sp_group,
                sp_comm=sp_comm,
                total_tokens=sp_total_tokens,
            )
            partial_full = self._forward_compute(
                x_full, use_triton=use_triton, skip_allreduce=True
            )
            out_2d = _sp_reduce_scatter(
                partial_full,
                sp_size=eff_sp_size,
                sp_rank=eff_sp_rank,
                sp_group=eff_sp_group,
                sp_comm=sp_comm,
                total_tokens=sp_total_tokens,
            )
            if orig_ndim == 2:
                return out_2d
            if orig_ndim == 3:
                return out_2d.reshape(orig_b, -1, self.hidden_size)
            return out_2d.reshape(-1, self.hidden_size)

        if hidden_states.numel() == 0:
            return torch.empty_like(hidden_states)
        eff_skip_ar = skip_allreduce or (sp_input and self.tp_size == 1)
        return self._forward_compute(
            hidden_states, use_triton=use_triton, skip_allreduce=eff_skip_ar
        )


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
    shared_gate_up: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None
    sp_input: bool = False
    sp_total_tokens: Optional[int] = None
    sp_local_tokens: Optional[int] = None
    fused_epilogue: bool = False
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
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        sp_fast_comm: Optional[FastAllReduce] = None,
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
        self.sp_size = int(sp_size)
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group if sp_group is not None else (tp_group if tp_size > 1 else ep_group)
        self.sp_fast_comm = _resolve_sp_comm(self.sp_group, sp_fast_comm, fast_all_reduce)
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
        self.register_buffer("_w3x_and_gates_weight", None, persistent=False)
        self.register_buffer("_latent_shared_gates_weight", None, persistent=False)
        self.register_buffer("_latent_shared_gates_pad8", None, persistent=False)
        self.register_buffer("_latent_scale_gate_pad8", None, persistent=False)
        self._w3x_and_gates_fast: Optional[torch.Tensor] = None
        self._w3x_gates_cols: int = 0
        self._w3x_base_g: int = 2 * self.latent_dim + 2 * self.shared_expert_intermediate_size
        self._pre_moe_ln_w_c: Optional[torch.Tensor] = None
        self._pre_moe_ln_w_ptr: int = -1
        self._router_e_pad: int = triton.next_power_of_2(self.num_experts)
        self._router_topk_pad: int = triton.next_power_of_2(max(self.top_k, 2))
        self._latent_ln_block_k: int = min(512, triton.next_power_of_2(max(self.latent_dim, 64)))
        self._combine_block_h: int = min(1024, triton.next_power_of_2(max(self.hidden_size, 64)))
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

        if self.all_to_all_comm is not None and hasattr(self.all_to_all_comm, "reset_step_equal_cache"):
            self.all_to_all_comm.reset_step_equal_cache()
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
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        sp_fast_comm: Optional[FastAllReduce] = None,
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
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_fast_comm=sp_fast_comm,
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
            if self.use_triton_mxfp4 and not self.weights_transposed:
                warmup_deepgemm_mxfp4_moe_kernels(
                    weights,
                    ep_size=self.ep_size,
                    ep_rank=self.ep_rank,
                    backend=self.backend,
                    compute_mode=self.compute_mode,
                    device=self.w13_u8.device,
                )
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
        ln_w = self.pre_moe_latent_layernorm.weight.data
        self._pre_moe_ln_w_c = ln_w if ln_w.is_contiguous() else ln_w.contiguous()
        self._pre_moe_ln_w_ptr = int(ln_w.data_ptr())
        if (
            fuse_latent_down_gate
            and fuse_shared_gate_up
            and self.shared_expert._gate_up_weight is not None
        ):
            # Fuse W_3x + [latent_scale_gate, (shared_expert_gate)] + zero-pad(8) into one
            # contiguous [2*L + 2*S + 8, H] GEMM weight: zero extra memory vs separate pad8
            # and eliminates 1 kernel launch + 1 HBM read of x per MoE layer.
            gates_parts = [self.latent_scale_gate.weight.data]
            if self.shared_expert_gate is not None:
                gates_parts.append(self.shared_expert_gate.weight.data)
            gates_c = len(gates_parts)
            pad_tail = torch.zeros(
                (8 - gates_c, self.hidden_size),
                dtype=self.latent_scale_gate.weight.dtype,
                device=self.latent_scale_gate.weight.device,
            )
            w_full = torch.cat(
                [
                    self.latent_down_proj.weight.data,
                    self.latent_gate_proj.weight.data,
                    self.shared_expert._gate_up_weight,
                    *gates_parts,
                    pad_tail,
                ],
                dim=0,
            ).contiguous()
            base_g = 2 * self.latent_dim + 2 * self.shared_expert_intermediate_size
            self._w3x_base_g = base_g
            self._w3x_gates_cols = gates_c
            self._w3x_and_gates_weight = w_full
            self._w3x_and_gates_fast = w_full
            self._latent_down_gate_shared_gu_weight = w_full[:base_g]
            self._latent_down_gate_weight = w_full[: 2 * self.latent_dim]
            self.shared_expert._gate_up_weight = w_full[2 * self.latent_dim : base_g]
            if (
                self.shared_expert._fp8_gate_up is not None
                and self.shared_expert._fp8_gate_up.weight is not None
            ):
                _bind_fp8_bf16_alias(
                    self.shared_expert._fp8_gate_up, self.shared_expert._gate_up_weight
                )
            if gates_c == 2:
                self._latent_shared_gates_weight = w_full[base_g : base_g + 2]
                self._latent_shared_gates_pad8 = w_full[base_g : base_g + 8]
                self._latent_scale_gate_pad8 = None
            else:
                self._latent_shared_gates_weight = None
                self._latent_shared_gates_pad8 = None
                self._latent_scale_gate_pad8 = w_full[base_g : base_g + 8]
        else:
            self._w3x_and_gates_weight = None
            self._w3x_and_gates_fast = None
            self._w3x_gates_cols = 0
            self._latent_down_gate_shared_gu_weight = None
            if fuse_latent_down_gate:
                self._latent_down_gate_weight = torch.cat(
                    [self.latent_down_proj.weight.data, self.latent_gate_proj.weight.data], dim=0
                ).contiguous()
            else:
                self._latent_down_gate_weight = None
            if self.shared_expert_gate is not None:
                gates_w = torch.cat(
                    [self.latent_scale_gate.weight.data, self.shared_expert_gate.weight.data], dim=0
                ).contiguous()
                self._latent_shared_gates_weight = gates_w
                pad8 = torch.zeros(
                    (8, self.hidden_size), dtype=gates_w.dtype, device=gates_w.device
                )
                pad8[:2].copy_(gates_w)
                self._latent_shared_gates_pad8 = pad8
                self._latent_scale_gate_pad8 = None
            else:
                self._latent_shared_gates_weight = None
                self._latent_shared_gates_pad8 = None
                sw = self.latent_scale_gate.weight.data
                pad8_s = torch.zeros((8, self.hidden_size), dtype=sw.dtype, device=sw.device)
                pad8_s[:1].copy_(sw)
                self._latent_scale_gate_pad8 = pad8_s

        if eff_mode != "bf16" and self.latent_down_proj.weight.is_cuda:
            # Keep pre-router/pre-SiTU-GLU latent inputs in BF16 under auto so zero router flips
            # or SiTU-GLU error amplification occur; quantize pure output projection latent_up.
            dev_idx = int(self.latent_down_proj.weight.device.index or 0)
            latent_in_kw = dict(
                mode=eff_mode,
                prepare_trtllm_ll=False,
                prepare_mxfp8=(eff_mode == "mxfp8"),
                prepare_deepgemm_fp8=(eff_mode == "deepgemm_fp8"),
                prepare_w4a8=(eff_mode == "w4a8"),
                allow_decode_scaled_mm=False,
            )
            latent_out_kw = dict(
                latent_in_kw,
                prepare_mxfp8=bool(eff_mode == "mxfp8" or (eff_mode in ("auto", "fp8") and not has_deepgemm_fp8(dev_idx))),
                prepare_deepgemm_fp8=(eff_mode in ("auto", "fp8", "deepgemm_fp8")),
            )
            if self._latent_down_gate_weight is not None:
                self._fp8_latent_down_gate = _wrap_moe_fp8_linear(
                    self.hidden_size,
                    2 * self.latent_dim,
                    self._latent_down_gate_weight,
                    **latent_in_kw,
                )
                self._fp8_latent_down = None
                self._fp8_latent_gate = None
            else:
                self._fp8_latent_down_gate = None
                self._fp8_latent_down = _wrap_moe_fp8_linear(
                    self.hidden_size, self.latent_dim, self.latent_down_proj.weight, **latent_in_kw
                )
                self._fp8_latent_gate = _wrap_moe_fp8_linear(
                    self.hidden_size, self.latent_dim, self.latent_gate_proj.weight, **latent_in_kw
                )
            self._fp8_latent_up = _wrap_moe_fp8_linear(
                self.latent_dim, self.hidden_size, self.latent_up_proj.weight, **latent_out_kw
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
        needs_grad = torch.is_grad_enabled() and (
            routed.requires_grad or self.gate.weight.requires_grad
        )
        run_triton = (
            (self.use_triton_mxfp4 if use_triton is None else bool(use_triton))
            and not needs_grad
        )
        M_r = int(routed.shape[0])
        if (
            run_triton
            and routed.is_cuda
            and routed.dtype == torch.bfloat16
            and self.gate.weight.is_cuda
            and self.gate.weight.dtype == torch.bfloat16
            and self.gate.weight.stride(-1) == 1
            and self.router_precision != "fp32"
            and 1 <= M_r <= 64
            and self.latent_dim >= 1024
            and (self.latent_dim % 64) == 0
            and self.num_experts <= 2048
            and self.top_k <= 16
            and os.environ.get("QWEN4_5_TRITON_SPLITK_ROUTER", "1") != "0"
        ):
            routed_2d = routed if routed.stride(-1) == 1 else routed.contiguous()
            topk_ids = torch.empty((M_r, self.top_k), dtype=torch.int32, device=routed_2d.device)
            topk_weights = torch.empty((M_r, self.top_k), dtype=torch.float32, device=routed_2d.device)
            sk_n = 10 if (self.latent_dim % 640) == 0 else 8
            bn = 32 if M_r >= 8 else 64
            bm = 16 if M_r <= 16 else (32 if M_r <= 32 else 64)
            nw = 2 if (M_r >= 8 and bn == 32) else 4
            part = torch.empty(
                (sk_n, M_r, self.num_experts), dtype=torch.float32, device=routed_2d.device
            )
            bias_t = self.expert_bias if self.expert_bias is not None else routed_2d
            with _cuda_device_ctx(routed_2d.device):
                _qwen45_router_splitk_gemm_kernel[(triton.cdiv(self.num_experts, bn), sk_n)](
                    routed_2d,
                    self.gate.weight,
                    part,
                    M_r,
                    self.latent_dim,
                    self.num_experts,
                    routed_2d.stride(0),
                    self.gate.weight.stride(0),
                    part.stride(0),
                    part.stride(1),
                    SPLIT_K=sk_n,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=64,
                    num_warps=nw,
                    num_stages=3,
                )
                _qwen45_router_splitk_reduce_topk_kernel[(M_r,)](
                    part,
                    bias_t,
                    topk_ids,
                    topk_weights,
                    self.num_experts,
                    part.stride(0),
                    part.stride(1),
                    SPLIT_K=sk_n,
                    E_PAD=self._router_e_pad,
                    TOP_K=self.top_k,
                    TOP_K_PAD=self._router_topk_pad,
                    HAS_BIAS=(self.expert_bias is not None),
                    NORMALIZE=self.normalize_topk_scores,
                    ROUND_BF16=True,
                    num_warps=4,
                )
            return topk_ids, topk_weights

        if self.router_precision == "fp32":
            logits = F.linear(routed.float(), self.gate.weight.float())
        else:
            # Default "fp32_logits": linear in input dtype, cast logits to FP32.
            logits = F.linear(routed, self.gate.weight)

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
            with _cuda_device_ctx(logits.device):
                _qwen45_sigmoid_topk_router_kernel[(M,)](
                    logits,
                    bias_t,
                    topk_ids,
                    topk_weights,
                    self.num_experts,
                    logits.stride(0),
                    E_PAD=self._router_e_pad,
                    TOP_K=self.top_k,
                    TOP_K_PAD=self._router_topk_pad,
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
                ln_w_raw = self.pre_moe_latent_layernorm.weight.data
                ptr = int(ln_w_raw.data_ptr())
                ln_w = self._pre_moe_ln_w_c
                if ln_w is None or self._pre_moe_ln_w_ptr != ptr:
                    ln_w = ln_w_raw if ln_w_raw.is_contiguous() else ln_w_raw.contiguous()
                    self._pre_moe_ln_w_c = ln_w
                    self._pre_moe_ln_w_ptr = ptr
                with _cuda_device_ctx(dg_2d.device):
                    _latent_norm_mul_sigmoid_kernel[(M_dg,)](
                        dg_2d,
                        ln_w,
                        routed,
                        M_dg,
                        self.latent_dim,
                        dg_2d.stride(0),
                        self.latent_dim,
                        eps=self.rms_norm_eps,
                        BLOCK_K=self._latent_ln_block_k,
                        num_warps=4,
                    )
            return routed
        d_part, g_part = dg.split(self.latent_dim, dim=-1)
        return self.pre_moe_latent_layernorm(d_part, use_triton=run_triton) * torch.sigmoid(g_part)

    def _project_and_norm_latent(
        self, x: torch.Tensor, *, run_triton: bool
    ) -> torch.Tensor:
        if self._latent_down_gate_weight is not None:
            dg = _dense_linear_or_tgv(x, self._latent_down_gate_weight, self._fp8_latent_down_gate)
            return self._run_latent_norm_mul_sigmoid(dg, run_triton=run_triton)
        routed = _dense_linear_or_tgv(x, self.latent_down_proj.weight, self._fp8_latent_down)
        routed = self.pre_moe_latent_layernorm(routed, use_triton=run_triton)
        l_gate = _dense_linear_or_tgv(x, self.latent_gate_proj.weight, self._fp8_latent_gate)
        return routed * torch.sigmoid(l_gate)

    def _effective_sp_size(self, sp_input: bool) -> int:
        if not sp_input:
            return 1
        if self.sp_size > 1:
            return self.sp_size
        if self.tp_size > 1:
            return self.tp_size
        return self.ep_size if self.ep_size > 1 else 1

    def _resolve_a2a_equal_tokens(
        self,
        r_disp: torch.Tensor,
        eq_tok: Optional[bool],
        *,
        sp_input: bool,
        slot: int = -1,
    ) -> Optional[bool]:
        if (
            eq_tok is None
            and sp_input
            and self.sp_size > 1
            and self.all_to_all_comm is not None
            and hasattr(self.all_to_all_comm, "resolve_step_equal_tokens")
        ):
            sig = (
                self.sp_size,
                self.tp_size,
                self.ep_size,
                self.hidden_size,
                self.num_experts,
            )
            return self.all_to_all_comm.resolve_step_equal_tokens(
                int(r_disp.shape[0]),
                r_disp.device,
                caller_id=id(self),
                step_sig=sig,
                slot=slot,
            )
        return eq_tok

    def _prepare_sp_dispatch(
        self,
        routed: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        sp_input: bool,
        sp_total_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[bool], Optional[int]]:
        local_T = int(routed.shape[0])
        if not sp_input:
            return routed, topk_ids, topk_weights, self.dp_equal_tokens, None

        eff_sp = self._effective_sp_size(True)
        if sp_total_tokens is None:
            if eff_sp == self.ep_size and self.dp_equal_tokens is True:
                return routed, topk_ids, topk_weights, True, None
            return routed, topk_ids, topk_weights, self.dp_equal_tokens, None

        T_sp = int(sp_total_tokens)
        can_pad_equal = (
            (eff_sp == self.ep_size and self.dp_equal_tokens is not False)
            or self.dp_equal_tokens is True
        )
        can_pad_sp_chunk = can_pad_equal or (
            eff_sp > 1 and self.dp_equal_tokens is not False
        )
        if can_pad_sp_chunk and T_sp > 0:
            chunk = (T_sp + eff_sp - 1) // eff_sp
            eq_ret = True if can_pad_equal else self.dp_equal_tokens
            if local_T == chunk:
                return routed, topk_ids, topk_weights, eq_ret, None
            if local_T < chunk:
                routed_p = torch.zeros(
                    (chunk, self.latent_dim), dtype=routed.dtype, device=routed.device
                )
                ids_p = torch.full(
                    (chunk, self.top_k), -1, dtype=torch.int32, device=routed.device
                )
                w_p = torch.zeros(
                    (chunk, self.top_k), dtype=torch.float32, device=routed.device
                )
                if local_T > 0:
                    routed_p[:local_T].copy_(routed)
                    ids_p[:local_T].copy_(topk_ids)
                    w_p[:local_T].copy_(topk_weights)
                return routed_p, ids_p, w_p, eq_ret, local_T
        return routed, topk_ids, topk_weights, self.dp_equal_tokens, None

    def _fallback_sp_ep_non_a2a(
        self,
        routed: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: MXFP4MoEWeights,
        *,
        run_triton: bool,
        mode: str,
        backend: str,
        sp_total_tokens: Optional[int],
    ) -> torch.Tensor:
        """Exact SP-domain fallback when all_to_all_comm is disabled and ep_size > 1."""
        eff_sp = self._effective_sp_size(True)
        eff_sp_rk = (
            self.sp_rank
            if self.sp_size > 1
            else (self.tp_rank if self.tp_size > 1 else self.ep_rank)
        )
        eff_sp_grp = self.sp_group if self.sp_group is not None else self.ep_group
        sp_comm = _resolve_sp_comm(eff_sp_grp, self.sp_fast_comm, self.fast_all_reduce)
        routed_full = _sp_all_gather(
            routed,
            sp_size=eff_sp,
            sp_rank=eff_sp_rk,
            sp_group=eff_sp_grp,
            sp_comm=sp_comm,
            total_tokens=sp_total_tokens,
        )
        ids_full = _sp_all_gather(
            topk_ids.to(torch.int32),
            sp_size=eff_sp,
            sp_rank=eff_sp_rk,
            sp_group=eff_sp_grp,
            sp_comm=sp_comm,
            total_tokens=sp_total_tokens,
        )
        w_full = _sp_all_gather(
            topk_weights.to(torch.float32),
            sp_size=eff_sp,
            sp_rank=eff_sp_rk,
            sp_group=eff_sp_grp,
            sp_comm=sp_comm,
            total_tokens=sp_total_tokens,
        )
        if run_triton:
            routed_part = mxfp4_moe_forward(
                routed_full,
                ids_full,
                w_full,
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
            routed_part = mxfp4_moe_forward_ref(
                routed_full,
                ids_full,
                w_full,
                weights,
                gate_beta=self.gate_beta,
                up_beta=self.up_beta,
                ep_start=self.ep_start,
                ep_size=self.ep_size,
            )
        if self.ep_group is eff_sp_grp and eff_sp == self.ep_size:
            return _sp_reduce_scatter(
                routed_part,
                sp_size=eff_sp,
                sp_rank=eff_sp_rk,
                sp_group=eff_sp_grp,
                sp_comm=sp_comm,
                total_tokens=sp_total_tokens,
            )
        if eff_sp < self.ep_size and self.ep_group is not None:
            # DP x SP hybrid without A2A: isolate each DP group across ep_group cleanly.
            payload = (
                routed_full.detach().cpu(),
                ids_full.detach().cpu(),
                w_full.detach().cpu(),
            )
            gathered: Any = [None] * self.ep_size
            torch.distributed.all_gather_object(gathered, payload, group=self.ep_group)
            dp_idx = self.ep_rank // eff_sp
            dp_heads = list(range(0, self.ep_size, eff_sp))
            local_dp_partials = {}
            for h_r in dp_heads:
                rb_h, ib_h, wb_h = gathered[h_r]
                rd_h = rb_h.to(routed.device, dtype=routed.dtype)
                id_h = ib_h.to(routed.device, dtype=torch.int32)
                wd_h = wb_h.to(routed.device, dtype=torch.float32)
                lp_h = (
                    mxfp4_moe_forward(
                        rd_h,
                        id_h,
                        wd_h,
                        weights,
                        gate_beta=self.gate_beta,
                        up_beta=self.up_beta,
                        compute_mode=mode,
                        use_triton=True,
                        ep_start=self.ep_start,
                        ep_size=self.ep_size,
                        backend=backend,
                    )
                    if run_triton
                    else mxfp4_moe_forward_ref(
                        rd_h,
                        id_h,
                        wd_h,
                        weights,
                        gate_beta=self.gate_beta,
                        up_beta=self.up_beta,
                        ep_start=self.ep_start,
                        ep_size=self.ep_size,
                    )
                )
                local_dp_partials[h_r] = lp_h.detach().cpu()
            all_ep_dp = [None] * self.ep_size
            torch.distributed.all_gather_object(all_ep_dp, local_dp_partials, group=self.ep_group)
            my_head = dp_idx * eff_sp
            acc = torch.zeros_like(local_dp_partials[my_head].to(routed.device), dtype=torch.float32)
            for ep_dict in all_ep_dp:
                acc.add_(ep_dict[my_head].to(device=routed.device, dtype=torch.float32))
            routed_res = acc.to(routed.dtype)
            s0, s1 = sp_slice_bounds(int(routed_res.shape[0]), eff_sp, eff_sp_rk)
            return routed_res[s0:s1]
        if self.ep_group is not None:
            if not routed_part.is_contiguous():
                routed_part = routed_part.contiguous()
            torch.distributed.all_reduce(
                routed_part, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
            )
        T_tot = int(routed_part.shape[0])
        s0, s1 = sp_slice_bounds(T_tot, eff_sp, eff_sp_rk)
        return routed_part[s0:s1]

    def _can_fuse_latent_shared_epilogue(
        self,
        M: int,
        device: torch.device,
        *,
        run_triton: bool,
        has_a2a: bool,
        sp_input: bool,
        sp_total_tokens: Optional[int],
    ) -> bool:
        if os.environ.get("QWEN4_5_FUSE_MOE_EPILOGUE", "0") != "1":
            return False
        if not (run_triton and device.type == "cuda" and M > 0):
            return False
        if self.latent_up_proj.weight.dtype != self.gate.weight.dtype:
            return False
        if self.shared_expert.down_proj.weight.dtype != self.gate.weight.dtype:
            return False
        if self._fp8_latent_up is not None and self._fp8_latent_up._resolve_backend(M, device) != "bf16":
            return False
        if self.shared_expert._fp8_down is not None and self.shared_expert._fp8_down._resolve_backend(M, device) != "bf16":
            return False
        if has_a2a and not hasattr(self.all_to_all_comm, "can_symm_p2p_combine"):
            return False
        if not has_a2a and self.ep_size > 1:
            return False
        if sp_input and sp_total_tokens is not None:
            eff_sp = max(1, self._effective_sp_size(True))
            if int(sp_total_tokens) % eff_sp != 0:
                return False
        return True

    def _forward_routed_from_norm(
        self,
        x: torch.Tensor,
        routed: torch.Tensor,
        *,
        run_triton: bool,
        mode: str,
        backend: str,
        apply_scale_gate: bool = True,
        sp_input: bool = False,
        sp_total_tokens: Optional[int] = None,
        return_pre_up: bool = False,
        routed_scale_gate: Optional[torch.Tensor] = None,
        routed_scale_gate_col: int = 0,
    ) -> torch.Tensor:
        topk_ids, topk_weights = self.route(routed, use_triton=run_triton)
        weights = self._mxfp4_weights()
        use_a2a = (
            self.all_to_all_comm is not None
            and self.ep_size > 1
            and getattr(self.all_to_all_comm, "world_size", 1) > 1
        )
        if use_a2a:
            r_disp, ids_disp, w_disp, eq_tok, sp_loc_t = self._prepare_sp_dispatch(
                routed,
                topk_ids,
                topk_weights,
                sp_input=sp_input,
                sp_total_tokens=sp_total_tokens,
            )
            eff_m_est = int(r_disp.shape[0]) * self.ep_size
            resolved_b = resolve_moe_backend_for_shape(
                eff_m_est,
                self.latent_dim,
                r_disp.device,
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
            want_packed = want_fp8_a2a and resolved_b == "deepgemm_masked"
            eq_tok = self._resolve_a2a_equal_tokens(
                r_disp, eq_tok, sp_input=sp_input, slot=-1
            )
            a2a_state = self.all_to_all_comm.dispatch(
                r_disp,
                ids_disp,
                w_disp,
                num_experts=self.num_experts,
                equal_tokens=eq_tok,
                quantize_fp8=want_fp8_a2a,
                return_packed=want_packed,
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
                    prequant_packed=getattr(a2a_state, "packed_fp8", None),
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
            comb_sg = routed_scale_gate if (routed_scale_gate is not None and sp_loc_t is None) else None
            routed = self.all_to_all_comm.combine(
                routed_ep,
                a2a_state,
                use_symm_p2p=self.use_symm_p2p_combine,
                scale_gate=comb_sg,
                scale_gate_col=routed_scale_gate_col,
            )
            if sp_loc_t is not None:
                routed = routed[:sp_loc_t]
                if routed_scale_gate is not None and routed.shape[0] > 0:
                    sg = routed_scale_gate.reshape(int(x.shape[0]), -1)[:, routed_scale_gate_col : routed_scale_gate_col + 1]
                    routed = (routed.float() * torch.sigmoid(sg.float()).to(routed.dtype).float()).to(routed.dtype)
        elif sp_input and self.ep_size > 1 and self._effective_sp_size(True) > 1:
            routed = self._fallback_sp_ep_non_a2a(
                routed,
                topk_ids,
                topk_weights,
                weights,
                run_triton=run_triton,
                mode=mode,
                backend=backend,
                sp_total_tokens=sp_total_tokens,
            )
            if routed_scale_gate is not None and routed.shape[0] > 0:
                sg = routed_scale_gate.reshape(int(x.shape[0]), -1)[:, routed_scale_gate_col : routed_scale_gate_col + 1]
                routed = (routed.float() * torch.sigmoid(sg.float()).to(routed.dtype).float()).to(routed.dtype)
        else:
            local_sg = routed_scale_gate if (self.ep_size <= 1 and routed_scale_gate is not None) else None
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
                    scale_gate=local_sg,
                    scale_gate_col=routed_scale_gate_col,
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
                if local_sg is not None and routed.shape[0] > 0:
                    sg = local_sg.reshape(int(x.shape[0]), -1)[:, routed_scale_gate_col : routed_scale_gate_col + 1]
                    routed = (routed.float() * torch.sigmoid(sg.float()).to(routed.dtype).float()).to(routed.dtype)
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
                if routed_scale_gate is not None and routed.shape[0] > 0:
                    sg = routed_scale_gate.reshape(int(x.shape[0]), -1)[:, routed_scale_gate_col : routed_scale_gate_col + 1]
                    routed = (routed.float() * torch.sigmoid(sg.float()).to(routed.dtype).float()).to(routed.dtype)

        if x.shape[0] == 0:
            dim_out = self.latent_dim if return_pre_up else self.hidden_size
            return torch.empty((0, dim_out), dtype=x.dtype, device=x.device)

        if return_pre_up:
            return routed

        routed = _dense_linear_or_tgv(routed, self.latent_up_proj.weight, self._fp8_latent_up)
        if apply_scale_gate:
            sg_logits = self._project_gate_no_cutlass80(
                x,
                self.latent_scale_gate.weight,
                self._latent_scale_gate_pad8,
                run_triton=run_triton,
            )
            return routed * torch.sigmoid(sg_logits)
        return routed

    def _forward_routed(
        self,
        x: torch.Tensor,
        *,
        run_triton: bool,
        mode: str,
        backend: str,
        apply_scale_gate: bool = True,
        sp_input: bool = False,
        sp_total_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        routed = self._project_and_norm_latent(x, run_triton=run_triton)
        return self._forward_routed_from_norm(
            x,
            routed,
            run_triton=run_triton,
            mode=mode,
            backend=backend,
            apply_scale_gate=apply_scale_gate,
            sp_input=sp_input,
            sp_total_tokens=sp_total_tokens,
        )

    def _project_gate_no_cutlass80(
        self,
        x: torch.Tensor,
        w_raw: torch.Tensor,
        w_pad8: Optional[torch.Tensor] = None,
        *,
        run_triton: bool,
    ) -> torch.Tensor:
        """Compute [M, H] @ [H, C] (C in {1, 2}) avoiding cuBLAS cutlass_80 kernels."""
        M = int(x.shape[0])
        C = int(w_raw.shape[0])
        if M == 0:
            return torch.empty((0, C), dtype=x.dtype, device=x.device)
        if M <= 512 and run_triton and x.is_cuda and C in (1, 2):
            x_2d = x if x.stride(-1) == 1 else x.contiguous()
            w_c = w_raw if w_raw.stride(-1) == 1 else w_raw.contiguous()
            out = torch.empty((M, C), dtype=x_2d.dtype, device=x_2d.device)
            block_k = min(512, triton.next_power_of_2(max(self.hidden_size, 64)))
            with _cuda_device_ctx(x_2d.device):
                _gates_2col_gemv_kernel[(M,)](
                    x_2d,
                    w_c,
                    out,
                    M,
                    H=self.hidden_size,
                    C=C,
                    stride_xm=x_2d.stride(0),
                    stride_wm=w_c.stride(0),
                    stride_om=out.stride(0),
                    BLOCK_K=block_k,
                    num_warps=4,
                )
            return out
        if x.is_cuda and w_pad8 is not None:
            return F.linear(x, w_pad8)[:, :C]
        return F.linear(x, w_raw)

    def _forward_w3x_latent_and_sgu(
        self, x: torch.Tensor, *, run_triton: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fused W_3x projection [M, H] -> [routed_norm, shared_gate_up] and contiguous x."""
        if not x.is_contiguous():
            x = x.contiguous()
        assert self._latent_down_gate_shared_gu_weight is not None
        all_up = _bf16_linear_tgv_or_torch(x, self._latent_down_gate_shared_gu_weight)
        dg = all_up[:, : 2 * self.latent_dim]
        sgu = all_up[:, 2 * self.latent_dim :]
        routed_norm = self._run_latent_norm_mul_sigmoid(dg, run_triton=run_triton)
        return x, routed_norm, sgu

    def _forward_w3x_latent_sgu_and_gates(
        self, x: torch.Tensor, *, run_triton: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run fused W_3x+gates projection [M, H] -> (x, routed_norm, shared_gate_up, gates)."""
        if not x.is_contiguous():
            x = x.contiguous()
        w_fused = self._w3x_and_gates_fast
        if w_fused is None:
            w_fused = self._w3x_and_gates_weight
        if w_fused is not None and self._w3x_gates_cols > 0 and w_fused.device == x.device:
            all_up = _bf16_linear_tgv_or_torch(x, w_fused)
            base_g = self._w3x_base_g
            dg = all_up[:, : 2 * self.latent_dim]
            sgu = all_up[:, 2 * self.latent_dim : base_g]
            gates = all_up[:, base_g : base_g + self._w3x_gates_cols]
            routed_norm = self._run_latent_norm_mul_sigmoid(dg, run_triton=run_triton)
            return x, routed_norm, sgu, gates
        x_c, routed_norm, sgu = self._forward_w3x_latent_and_sgu(x, run_triton=run_triton)
        return x_c, routed_norm, sgu, None

    def _should_use_w3x_bf16(self, M: int, device: torch.device) -> bool:
        if (
            self._latent_down_gate_shared_gu_weight is None
            or M <= 0
            or device.type != "cuda"
        ):
            return False
        env_raw = os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "")
        env_mode = env_raw.strip().lower() if env_raw else ""
        if (
            not env_mode
            and self.dense_linear_mode in ("bf16", "auto")
            and (
                self.dense_linear_mode == "bf16"
                or self._fp8_latent_down_gate is None
                or (
                    self._fp8_latent_down_gate.w_mxfp8 is None
                    and self._fp8_latent_down_gate.w_dg_fp8 is None
                )
            )
        ):
            return True
        return (
            self._fp8_latent_down_gate is None
            or self._fp8_latent_down_gate._resolve_backend(M, device) == "bf16"
        )

    def _forward_shared(
        self, x: torch.Tensor, *, run_triton: bool, apply_gate: bool = True
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty_like(x)
        shared = self.shared_expert(x, use_triton=run_triton, skip_allreduce=True)
        if apply_gate and self.shared_expert_gate is not None:
            sg = self._project_gate_no_cutlass80(
                x, self.shared_expert_gate.weight, run_triton=run_triton
            )
            shared = shared * torch.sigmoid(sg)
        return shared

    def _combine_latent_and_shared(
        self,
        x: torch.Tensor,
        routed_up: torch.Tensor,
        shared: torch.Tensor,
        *,
        run_triton: bool,
        precomputed_gates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = int(x.shape[0])
        if M == 0:
            return routed_up + shared
        has_sg = self.shared_expert_gate is not None
        if precomputed_gates is not None:
            gates = precomputed_gates
        elif has_sg:
            if self._latent_shared_gates_weight is not None:
                gates = self._project_gate_no_cutlass80(
                    x,
                    self._latent_shared_gates_weight,
                    self._latent_shared_gates_pad8,
                    run_triton=run_triton,
                )
            else:
                gates = torch.cat(
                    [
                        self._project_gate_no_cutlass80(
                            x, self.latent_scale_gate.weight, run_triton=run_triton
                        ),
                        self._project_gate_no_cutlass80(
                            x, self.shared_expert_gate.weight, run_triton=run_triton
                        ),
                    ],
                    dim=-1,
                )
        else:
            gates = self._project_gate_no_cutlass80(
                x,
                self.latent_scale_gate.weight,
                self._latent_scale_gate_pad8,
                run_triton=run_triton,
            )
        if run_triton and x.is_cuda:
            r_2d = routed_up if routed_up.stride(-1) == 1 else routed_up.contiguous()
            s_2d = shared if shared.stride(-1) == 1 else shared.contiguous()
            g_2d = gates if gates.stride(-1) == 1 else gates.contiguous()
            out = torch.empty((M, self.hidden_size), dtype=r_2d.dtype, device=r_2d.device)
            block_h = self._combine_block_h
            with _cuda_device_ctx(r_2d.device):
                _combine_latent_shared_out_kernel[(M, triton.cdiv(self.hidden_size, block_h))](
                    r_2d,
                    s_2d,
                    g_2d,
                    out,
                    r_2d.stride(0),
                    s_2d.stride(0),
                    g_2d.stride(0),
                    self.hidden_size,
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
        sp_input: bool = False,
        sp_total_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Run Latent MoE + Shared Expert forward pass in replicated or SP domain.

        When ``sp_input=True``, ``hidden_states`` is the SP-sharded local slice
        ``[T_loc, H]``; projections, router, shared expert, and EP AllToAll stay
        strictly in the SP domain with zero AllReduce on MoE.
        """
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        has_a2a = (
            self.all_to_all_comm is not None
            and self.ep_size > 1
            and getattr(self.all_to_all_comm, "world_size", 1) > 1
        )
        eff_sp = self._effective_sp_size(sp_input)
        sp_empty = (
            sp_input
            and sp_total_tokens == 0
            and eff_sp == self.ep_size
            and self.dp_equal_tokens is not False
        )
        if sp_empty or (x.shape[0] == 0 and not has_a2a and not (sp_input and self.ep_size > 1)):
            return torch.empty_like(hidden_states)

        use_triton = (
            self.use_triton_mxfp4 if use_triton_mxfp4 is None else bool(use_triton_mxfp4)
        )
        mode = self.compute_mode if compute_mode is None else compute_mode
        eff_backend = self.backend if backend is None else backend
        run_triton = use_triton and x.is_cuda and supports_triton_mxfp4(x.device)
        use_overlap = self.enable_shared_overlap and x.is_cuda and torch.version.hip is None
        M = int(x.shape[0])

        if self._should_use_w3x_bf16(M, x.device):
            x, routed_norm, sgu, pre_gates = self._forward_w3x_latent_sgu_and_gates(
                x, run_triton=run_triton
            )
            capturing = torch.cuda.is_current_stream_capturing() if use_overlap else False
            fuse_epi = (
                pre_gates is not None
                and self._can_fuse_latent_shared_epilogue(
                    M,
                    x.device,
                    run_triton=run_triton,
                    has_a2a=has_a2a,
                    sp_input=sp_input,
                    sp_total_tokens=sp_total_tokens,
                )
            )
            if fuse_epi:
                assert pre_gates is not None
                has_sg = self.shared_expert_gate is not None
                sg_shared = pre_gates if has_sg else None
                if use_overlap and (capturing or M >= 1024):
                    routed_sc, shared_sc = self._run_with_shared_stream(
                        x.device,
                        lambda: self.shared_expert.forward_from_gate_up(
                            sgu,
                            use_triton=run_triton,
                            skip_allreduce=True,
                            scale_gate=sg_shared,
                            scale_gate_col=1,
                        ),
                        lambda: self._forward_routed_from_norm(
                            x,
                            routed_norm,
                            run_triton=run_triton,
                            mode=mode,
                            backend=eff_backend,
                            apply_scale_gate=False,
                            sp_input=sp_input,
                            sp_total_tokens=sp_total_tokens,
                            return_pre_up=True,
                            routed_scale_gate=pre_gates,
                            routed_scale_gate_col=0,
                        ),
                    )
                else:
                    routed_sc = self._forward_routed_from_norm(
                        x,
                        routed_norm,
                        run_triton=run_triton,
                        mode=mode,
                        backend=eff_backend,
                        apply_scale_gate=False,
                        sp_input=sp_input,
                        sp_total_tokens=sp_total_tokens,
                        return_pre_up=True,
                        routed_scale_gate=pre_gates,
                        routed_scale_gate_col=0,
                    )
                    shared_sc = self.shared_expert.forward_from_gate_up(
                        sgu,
                        use_triton=run_triton,
                        skip_allreduce=True,
                        scale_gate=sg_shared,
                        scale_gate_col=1,
                    )
                r2d = routed_sc if routed_sc.is_contiguous() else routed_sc.contiguous()
                s2d = shared_sc if shared_sc.is_contiguous() else shared_sc.contiguous()
                w_up_t = self.latent_up_proj.weight.t()
                if not s2d.requires_grad and not r2d.requires_grad:
                    torch.addmm(s2d, r2d, w_up_t, beta=1.0, alpha=1.0, out=s2d)
                    return s2d.reshape(orig_shape)
                return torch.addmm(s2d, r2d, w_up_t, beta=1.0, alpha=1.0).reshape(orig_shape)

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
                        sp_input=sp_input,
                        sp_total_tokens=sp_total_tokens,
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
                    sp_input=sp_input,
                    sp_total_tokens=sp_total_tokens,
                )
                shared = self.shared_expert.forward_from_gate_up(
                    sgu, use_triton=run_triton, skip_allreduce=True
                )
            return self._combine_latent_and_shared(
                x,
                routed_up,
                shared,
                run_triton=run_triton,
                precomputed_gates=pre_gates,
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
                    sp_input=sp_input,
                    sp_total_tokens=sp_total_tokens,
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
            sp_input=sp_input,
            sp_total_tokens=sp_total_tokens,
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
        sp_input: bool = False,
        sp_total_tokens: Optional[int] = None,
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
        eff_sp = self._effective_sp_size(sp_input)
        sp_empty = (
            sp_input
            and sp_total_tokens == 0
            and eff_sp == self.ep_size
            and self.dp_equal_tokens is not False
        )
        if sp_empty or (x.shape[0] == 0 and not has_a2a and not (sp_input and self.ep_size > 1)):
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
        shared_gate_up = None
        pre_gates = None
        if self._should_use_w3x_bf16(int(x.shape[0]), x.device):
            x, routed, shared_gate_up, pre_gates = self._forward_w3x_latent_sgu_and_gates(
                x, run_triton=run_triton
            )
            topk_ids, topk_weights = self.route(routed, use_triton=run_triton)
        else:
            routed, topk_ids, topk_weights = self._compute_latent_and_route(x, run_triton=run_triton)
        dispatch_handle = None
        sp_loc_t: Optional[int] = None
        if has_a2a:
            assert self.all_to_all_comm is not None
            r_disp, ids_disp, w_disp, sp_eq, sp_loc_t = self._prepare_sp_dispatch(
                routed,
                topk_ids,
                topk_weights,
                sp_input=sp_input,
                sp_total_tokens=sp_total_tokens,
            )
            eff_m_est = int(r_disp.shape[0]) * self.ep_size
            resolved_b = resolve_moe_backend_for_shape(
                eff_m_est,
                self.latent_dim,
                r_disp.device,
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
            want_packed = want_fp8_a2a and resolved_b == "deepgemm_masked"
            eq_tok = sp_eq if equal_tokens is None else equal_tokens
            eq_tok = self._resolve_a2a_equal_tokens(
                r_disp, eq_tok, sp_input=sp_input, slot=slot
            )
            dispatch_handle = self.all_to_all_comm.dispatch_async(
                r_disp,
                ids_disp,
                w_disp,
                num_experts=self.num_experts,
                equal_tokens=eq_tok,
                quantize_fp8=want_fp8_a2a,
                return_packed=want_packed,
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
            shared_gate_up=shared_gate_up,
            gates=pre_gates,
            sp_input=sp_input,
            sp_total_tokens=sp_total_tokens,
            sp_local_tokens=sp_loc_t,
        )

    def forward_routed_compute_and_combine(
        self, ctx: LatentMoETBOContext, *, slot: int = 0
    ) -> LatentMoETBOContext:
        """TBO Stage 2: wait dispatch -> routed MXFP4 -> launch async combine + shared."""
        if ctx.early_return is not None:
            return ctx
        assert ctx.routed is not None and ctx.topk_ids is not None and ctx.topk_weights is not None
        fuse_epi = (
            ctx.gates is not None
            and ctx.shared_gate_up is not None
            and ctx.sp_local_tokens is None
            and self._can_fuse_latent_shared_epilogue(
                int(ctx.x.shape[0]),
                ctx.x.device,
                run_triton=ctx.run_triton,
                has_a2a=ctx.has_a2a,
                sp_input=ctx.sp_input,
                sp_total_tokens=ctx.sp_total_tokens,
            )
        )
        ctx.fused_epilogue = fuse_epi
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
                    prequant_packed=getattr(a2a_state, "packed_fp8", None),
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
                scale_gate=(ctx.gates if fuse_epi else None),
                scale_gate_col=0,
                slot=slot,
            )
        else:
            if ctx.sp_input and self.ep_size > 1 and self._effective_sp_size(True) > 1:
                routed = self._fallback_sp_ep_non_a2a(
                    ctx.routed,
                    ctx.topk_ids,
                    ctx.topk_weights,
                    ctx.weights,
                    run_triton=ctx.run_triton,
                    mode=ctx.mode,
                    backend=ctx.backend,
                    sp_total_tokens=ctx.sp_total_tokens,
                )
            else:
                local_sg = ctx.gates if (fuse_epi and self.ep_size <= 1) else None
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
                        scale_gate=local_sg,
                        scale_gate_col=0,
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
        # Compute collective-free shared branch on main stream (overlapping combine_async on comm_stream when has_a2a).
        sg_shared = (ctx.gates if (fuse_epi and self.shared_expert_gate is not None) else None)
        ctx.shared = (
            self.shared_expert.forward_from_gate_up(
                ctx.shared_gate_up,
                use_triton=ctx.run_triton,
                skip_allreduce=True,
                scale_gate=sg_shared,
                scale_gate_col=1,
            )
            if ctx.shared_gate_up is not None
            else self._forward_shared(ctx.x, run_triton=ctx.run_triton, apply_gate=False)
        )
        ctx.shared_gate_up = None
        return ctx

    def forward_finish_combine(self, ctx: LatentMoETBOContext) -> torch.Tensor:
        """TBO Stage 3: wait combine -> latent up + fused scale gates + add shared expert."""
        if ctx.early_return is not None:
            return ctx.early_return
        if ctx.has_a2a:
            assert self.all_to_all_comm is not None and ctx.combine_handle is not None
            routed = self.all_to_all_comm.wait_combine(ctx.combine_handle)
            if ctx.sp_local_tokens is not None:
                routed = routed[:ctx.sp_local_tokens]
        else:
            assert ctx.routed is not None
            routed = ctx.routed
        if ctx.x.shape[0] == 0:
            return torch.empty(ctx.orig_shape, dtype=ctx.x.dtype, device=ctx.x.device)
        shared = (
            ctx.shared
            if ctx.shared is not None
            else self._forward_shared(ctx.x, run_triton=ctx.run_triton, apply_gate=False)
        )
        if ctx.fused_epilogue:
            r2d = routed if routed.is_contiguous() else routed.contiguous()
            s2d = shared if shared.is_contiguous() else shared.contiguous()
            w_up_t = self.latent_up_proj.weight.t()
            if not s2d.requires_grad and not r2d.requires_grad:
                torch.addmm(s2d, r2d, w_up_t, beta=1.0, alpha=1.0, out=s2d)
                return s2d.reshape(ctx.orig_shape)
            return torch.addmm(s2d, r2d, w_up_t, beta=1.0, alpha=1.0).reshape(ctx.orig_shape)
        routed = _dense_linear_or_tgv(routed, self.latent_up_proj.weight, self._fp8_latent_up)
        return self._combine_latent_and_shared(
            ctx.x, routed, shared, run_triton=ctx.run_triton, precomputed_gates=ctx.gates
        ).reshape(ctx.orig_shape)

"""DeepSeek-V4 Transformer Block: mHC + Attention + MoE.

Mirrors `inference/model.py:Block`. Each call applies hc_pre/F/hc_post
twice — once for Attention and once for MoE FFN.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.mhc import hc_split_sinkhorn
from rtp_llm.models_py.modules.dsv4.moe import MoE

# P1 (prefill_opt/final_plan.md): TileKernels fused mHC pre/post.  Drop-in
# replaces the per-block split-K GEMM + RMSNorm + sigmoid×3 + softmax +
# 20-iter Sinkhorn (mhc.py:hc_split_sinkhorn) with a single fused TileLang
# kernel, plus a fused post kernel.  Falls back to the REF path when:
#   - DSV4_USE_TILEKERNELS_MHC=0 explicitly disables it, or
#   - tile_kernels is not importable, or
#   - residual dtype != bf16 / mhc_mult != 4 / autograd enabled / B*S == 0
#     (TileKernels' big_fuse requires bf16 residual + mhc_mult=4 + inference
#     mode; empty batch trips internal asserts in pre_big_fuse_kernel).
try:
    from tile_kernels.modeling.mhc.functional import (
        mhc_post as _tk_mhc_post,
        mhc_pre as _tk_mhc_pre,
    )
    _TK_MHC_OK = True
except Exception:  # pragma: no cover — keep V4 importable without tile_kernels
    _tk_mhc_pre = None
    _tk_mhc_post = None
    _TK_MHC_OK = False


def _use_tk_mhc(residual: torch.Tensor, hc_mult: int) -> bool:
    if not _TK_MHC_OK:
        return False
    if os.environ.get("DSV4_USE_TILEKERNELS_MHC", "1") == "0":
        return False
    if torch.is_grad_enabled():
        return False
    if hc_mult != 4:
        return False
    if residual.dtype != torch.bfloat16:
        return False
    if residual.numel() == 0:
        return False
    return True


class _RMSNorm(nn.Module):
    """Standalone RMSNorm with FP32 weight, matches `inference/model.py:RMSNorm`."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x32).to(dtype)


class Block(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str,
        route_scale: float,
        swiglu_limit: float,
        n_hash_layers: int,
        vocab_size: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float = 1e-6,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self._factory_mode = weights is not None

        self.attn = Attention(
            layer_id=layer_id,
            dim=dim, n_heads=n_heads, q_lora_rank=q_lora_rank,
            head_dim=head_dim, rope_head_dim=rope_head_dim,
            o_lora_rank=o_lora_rank, o_groups=o_groups,
            window_size=window_size, compress_ratio=compress_ratio,
            compress_rope_theta=compress_rope_theta, rope_theta=rope_theta,
            rope_factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
            original_seq_len=original_seq_len,
            max_batch_size=max_batch_size, max_seq_len=max_seq_len,
            index_n_heads=index_n_heads, index_head_dim=index_head_dim,
            index_topk=index_topk, norm_eps=norm_eps,
            weights=weights,
            prefix=f"{prefix}.attn" if self._factory_mode else "",
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.ffn = MoE(
            layer_id=layer_id, dim=dim, moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts, n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts, score_func=score_func,
            route_scale=route_scale, swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers, vocab_size=vocab_size,
            weights=weights,
            prefix=f"{prefix}.ffn" if self._factory_mode else "",
            ep_size=ep_size, ep_rank=ep_rank,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        self.attn_norm = _RMSNorm(dim, norm_eps)
        self.ffn_norm = _RMSNorm(dim, norm_eps)
        if self._factory_mode:
            self.attn_norm.weight = nn.Parameter(
                weights[f"{prefix}.attn_norm.weight"].float(), requires_grad=False,
            )
            self.ffn_norm.weight = nn.Parameter(
                weights[f"{prefix}.ffn_norm.weight"].float(), requires_grad=False,
            )

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        if self._factory_mode:
            self.hc_attn_fn = nn.Parameter(weights[f"{prefix}.hc_attn_fn"].float(), requires_grad=False)
            self.hc_ffn_fn = nn.Parameter(weights[f"{prefix}.hc_ffn_fn"].float(), requires_grad=False)
            self.hc_attn_base = nn.Parameter(weights[f"{prefix}.hc_attn_base"].float(), requires_grad=False)
            self.hc_ffn_base = nn.Parameter(weights[f"{prefix}.hc_ffn_base"].float(), requires_grad=False)
            self.hc_attn_scale = nn.Parameter(weights[f"{prefix}.hc_attn_scale"].float(), requires_grad=False)
            self.hc_ffn_scale = nn.Parameter(weights[f"{prefix}.hc_ffn_scale"].float(), requires_grad=False)
        else:
            # Match official param naming for ckpt loading.
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [B,S,hc,d] -> y: [B,S,d], post: [B,S,hc[,1]], comb: [B,S,hc,hc].

        TileKernels path returns ``post`` shaped ``[B,S,hc,1]``; the REF path
        returns ``[B,S,hc]``.  ``_hc_post`` accepts either.
        """
        if _use_tk_mhc(x, self.hc_mult):
            y, (post, comb) = _tk_mhc_pre(
                x, hc_fn, hc_scale, hc_base,
                norm_eps=self.norm_eps,
                mhc_mult=self.hc_mult,
                post_mult_value=2.0,
                pre_eps=self.hc_eps,
                sinkhorn_eps=self.hc_eps,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post, comb

        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base,
            hc_mult=self.hc_mult, sinkhorn_iters=self.hc_sinkhorn_iters, eps=self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # TileKernels' fused post kernel — requires bf16 x/residual + fp32
        # post (shape [..., hc, 1]) + fp32 comb.  Falls back to the broadcast
        # REF when the gate is off OR post is in the legacy [..., hc] shape.
        if (
            _use_tk_mhc(residual, self.hc_mult)
            and x.dtype == torch.bfloat16
            and post.dim() == residual.dim()  # post=[...,hc,1], residual=[...,hc,d]
        ):
            return _tk_mhc_post(x, residual, post, comb)

        if post.dim() == residual.dim():
            # Came from the TK path but TK gate now off — drop the trailing 1
            post_b = post.squeeze(-1)
        else:
            post_b = post
        y = (post_b.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # Attention path
        residual = x
        x_pre, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x_pre = self.attn_norm(x_pre)
        attn_out = self.attn(x_pre, start_pos)
        x = self._hc_post(attn_out, residual, post, comb)

        # FFN path
        residual = x
        x_pre, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x_pre = self.ffn_norm(x_pre)
        ffn_out = self.ffn(x_pre, input_ids if input_ids is not None
                           else torch.zeros(x.size(0), x.size(1), dtype=torch.long, device=x.device))
        x = self._hc_post(ffn_out, residual, post, comb)
        return x


class MTPBlock(Block):
    """Multi-Token Prediction draft block — V4 speculative decode head.

    Given the last-layer hidden ``x [B, S, hc, dim]`` (from the main model)
    and the NEXT-step ``input_ids`` (shifted one position), this block
    fuses the shifted embed with the hidden state via ``e_proj + h_proj +
    enorm/hnorm``, runs the fused tensor through the standard V4 Block
    (attention + MoE-FFN + mHC), then produces per-position logits via
    its own ``hc_head_*`` reduce and the shared LM head.

    Ckpt keys live under ``mtp.{i}.*`` and mirror the regular block keys
    plus:
      - ``e_proj.{weight,scale}`` — FP8 e4m3fn + UE8M0 block-128 scale
      - ``h_proj.{weight,scale}`` — same format
      - ``enorm.weight``, ``hnorm.weight``, ``norm.weight`` — BF16 (cast
        to FP32 on load, matching regular block norm params)
      - ``hc_head_{fn,base,scale}`` — FP32

    Mirrors ``inference/model.py:MTPBlock``; the speculative-decoding
    driver (prefill + draft-step sampler) is a framework-side concern and
    lives outside this class.
    """

    def __init__(
        self, layer_id: int, dim: int, n_heads: int, q_lora_rank: int,
        head_dim: int, rope_head_dim: int, o_lora_rank: int, o_groups: int,
        window_size: int, compress_ratio: int, compress_rope_theta: float,
        rope_theta: float, rope_factor: float, beta_fast: int, beta_slow: int,
        original_seq_len: int, max_batch_size: int, max_seq_len: int,
        index_n_heads: int, index_head_dim: int, index_topk: int,
        moe_inter_dim: int, n_routed_experts: int, n_activated_experts: int,
        n_shared_experts: int, score_func: str, route_scale: float,
        swiglu_limit: float, n_hash_layers: int, vocab_size: int,
        hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float,
        norm_eps: float = 1e-6,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        tp_size: int = 1, tp_rank: int = 0,
        ep_size: int = 1, ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
    ):
        super().__init__(
            layer_id=layer_id, dim=dim, n_heads=n_heads, q_lora_rank=q_lora_rank,
            head_dim=head_dim, rope_head_dim=rope_head_dim,
            o_lora_rank=o_lora_rank, o_groups=o_groups,
            window_size=window_size, compress_ratio=compress_ratio,
            compress_rope_theta=compress_rope_theta, rope_theta=rope_theta,
            rope_factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
            original_seq_len=original_seq_len,
            max_batch_size=max_batch_size, max_seq_len=max_seq_len,
            index_n_heads=index_n_heads, index_head_dim=index_head_dim,
            index_topk=index_topk,
            moe_inter_dim=moe_inter_dim, n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts, n_shared_experts=n_shared_experts,
            score_func=score_func, route_scale=route_scale,
            swiglu_limit=swiglu_limit, n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters, hc_eps=hc_eps,
            norm_eps=norm_eps, weights=weights, prefix=prefix,
            tp_size=tp_size, tp_rank=tp_rank,
            ep_size=ep_size, ep_rank=ep_rank,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear

        self.dim = dim
        if self._factory_mode:
            from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear_from_dict
            self.e_proj = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.e_proj.weight", f"{prefix}.e_proj.scale",
            )
            self.h_proj = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.h_proj.weight", f"{prefix}.h_proj.scale",
            )
            self._h_proj_is_factory = True
        else:
            self.e_proj = QuantizedLinear(dim, dim, storage="fp8")
            self.h_proj = QuantizedLinear(dim, dim, storage="fp8")
            self._h_proj_is_factory = False

        self.enorm = _RMSNorm(dim, norm_eps)
        self.hnorm = _RMSNorm(dim, norm_eps)
        self.norm = _RMSNorm(dim, norm_eps)

        hc_dim = hc_mult * dim
        if self._factory_mode:
            self.enorm.weight = nn.Parameter(weights[f"{prefix}.enorm.weight"].float(), requires_grad=False)
            self.hnorm.weight = nn.Parameter(weights[f"{prefix}.hnorm.weight"].float(), requires_grad=False)
            self.norm.weight = nn.Parameter(weights[f"{prefix}.norm.weight"].float(), requires_grad=False)
            self.hc_head_fn = nn.Parameter(weights[f"{prefix}.hc_head_fn"].float(), requires_grad=False)
            self.hc_head_base = nn.Parameter(weights[f"{prefix}.hc_head_base"].float(), requires_grad=False)
            self.hc_head_scale = nn.Parameter(weights[f"{prefix}.hc_head_scale"].float(), requires_grad=False)
        else:
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _apply_proj(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Factory FP8 linears want 2D input; QuantizedLinear accepts N-D."""
        if self._h_proj_is_factory and x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward_draft(
        self,
        x: torch.Tensor,                            # [B, S, hc, dim]
        start_pos: int,
        input_ids: torch.Tensor,                    # [B, S] shifted one step
        embed: nn.Module,                           # shared V4Transformer.embed
        lm_head_weight: torch.Tensor,               # shared [vocab, dim] FP32
    ) -> torch.Tensor:
        """Draft-token logits. See class docstring."""
        e = embed(input_ids)                              # [B, S, dim]
        e = self.enorm(e)
        x_norm = self.hnorm(x)                            # [B, S, hc, dim] (norm over last dim)
        e_proj_out = self._apply_proj(self.e_proj, e).unsqueeze(2)     # [B, S, 1, dim]
        h_proj_out = self._apply_proj(self.h_proj, x_norm)             # [B, S, hc, dim]
        x_fused = e_proj_out + h_proj_out                              # [B, S, hc, dim]

        # Parent Block runs attn + FFN + mHC on [B, S, hc, dim]
        x_after = super().forward(x_fused, start_pos, input_ids)       # [B, S, hc, dim]

        # hc_head_reduce (same math as V4Transformer._hc_head_reduce)
        shape, dtype = x_after.size(), x_after.dtype
        x_flat = x_after.flatten(2).float()                             # [B, S, hc*dim]
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        h = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)    # [B, S, dim]
        h = self.norm(h.to(dtype))                                      # [B, S, dim]
        return F.linear(h.float(), lm_head_weight)                      # [B, S, vocab]

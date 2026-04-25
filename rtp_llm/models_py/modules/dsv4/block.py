"""DeepSeek-V4 Transformer Block: mHC + Attention + MoE.

Mirrors `inference/model.py:Block`. Each call applies hc_pre/F/hc_post
twice — once for Attention and once for MoE FFN.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.mhc import hc_split_sinkhorn
from rtp_llm.models_py.modules.dsv4.moe import MoE


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
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

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
        )
        self.ffn = MoE(
            layer_id=layer_id, dim=dim, moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts, n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts, score_func=score_func,
            route_scale=route_scale, swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers, vocab_size=vocab_size,
        )
        self.attn_norm = _RMSNorm(dim, norm_eps)
        self.ffn_norm = _RMSNorm(dim, norm_eps)

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # Match official param naming for ckpt loading.
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [B,S,hc,d] -> y: [B,S,d], post: [B,S,hc], comb: [B,S,hc,hc]"""
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
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
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

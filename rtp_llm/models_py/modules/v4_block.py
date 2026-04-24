"""DeepSeek-V4 transformer block: mHC residual + V4 attention + V4 MoE.

Wraps `MHCMixing` (twice — for attention sub-block and FFN sub-block) around
`V4Attention` and `V4MoE`. The MTP variant adds an `e_proj`/`h_proj`/`enorm`/
`hnorm` pre-projection before the Block forward.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.mhc import MHCMixing
from rtp_llm.models_py.modules.v4_attention import V4Attention
from rtp_llm.models_py.modules.v4_moe import V4MoE


class V4Block(nn.Module):
    """One transformer layer in V4.

    Forward:
      pre/post/comb = mhc_attn.pre(x)              # x: [b,s,hc,d] → ([b,s,d], [b,s,hc], [b,s,hc,hc])
      a             = attn(attn_norm(pre))         # [b,s,d]
      x             = mhc_attn.post(a, x, post, comb)  # [b,s,hc,d]

      pre/post/comb = mhc_ffn.pre(x)
      f             = moe(ffn_norm(pre), input_ids)  # [b,s,d]
      x             = mhc_ffn.post(f, x, post, comb)
      return x
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        o_lora_rank: int,
        n_groups: int,
        window_size: int,
        compress_ratio: int,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_theta: float,
        compress_rope_theta: float,
        rope_factor: float,
        beta_fast: float,
        beta_slow: float,
        original_seq_len: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: int,
        route_scale: float,
        swiglu_limit: float,
        moe_hash_routing_layers: int,
        vocab_size: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = norm_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.attn = V4Attention(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            window_size=window_size,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_theta=rope_theta,
            compress_rope_theta=compress_rope_theta,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_seq_len=original_seq_len,
            world_size=world_size,
        )
        self.ffn = V4MoE(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            moe_hash_routing_layers=moe_hash_routing_layers,
            vocab_size=vocab_size,
            world_size=world_size,
            rank=rank,
        )
        self.attn_norm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ffn_norm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.mhc_attn = MHCMixing(dim, hc_mult, norm_eps)
        self.mhc_ffn = MHCMixing(dim, hc_mult, norm_eps)

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.norm_eps)
        return (weight * xf).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        pre, post, comb = self.mhc_attn.pre(x, self.hc_sinkhorn_iters, self.hc_eps)
        a = self.attn(self._rmsnorm(pre, self.attn_norm_weight), start_pos)
        x = self.mhc_attn.post(a, residual, post, comb)

        residual = x
        pre, post, comb = self.mhc_ffn.pre(x, self.hc_sinkhorn_iters, self.hc_eps)
        f = self.ffn(self._rmsnorm(pre, self.ffn_norm_weight), input_ids)
        x = self.mhc_ffn.post(f, residual, post, comb)
        return x


class V4MTPBlock(V4Block):
    """MTP nextN block: V4Block + e_proj/h_proj/enorm/hnorm pre-projection.

    e = enorm(embed(input_ids))
    x = hnorm(x_from_main)
    x = e_proj(e).unsqueeze(hc_axis) + h_proj(x)
    return Block.forward(x, ...)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.attn.dim
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.enorm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.hnorm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward_mtp(
        self,
        x: torch.Tensor,  # [b, s, hc, d] from main model
        start_pos: int,
        input_ids: torch.Tensor,
        embed_module: nn.Module,
    ) -> torch.Tensor:
        e = embed_module(input_ids)  # [b, s, d]
        e = self._rmsnorm(e, self.enorm_weight)
        # hnorm is applied to the hc state — needs to handle [b, s, hc, d]
        x_normed = self._rmsnorm(x, self.hnorm_weight)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x_normed)
        return super().forward(x, start_pos, input_ids)

"""DeepSeek-V4 standalone Transformer.

Top-level model: embed -> hc-expand -> N Blocks -> hc_head -> lm_head.

Mirrors `inference/model.py:Transformer` for TP=1 (full vocab embed/lm_head,
all experts on one device). Used to validate end-to-end correctness with
mock per-layer KV cache before wiring into RTP-LLM's GptModelBase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.block import Block, _RMSNorm


class _LMHead(nn.Module):
    """LM head — single weight matrix [vocab_size, dim] in FP32."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))


@dataclass
class V4Args:
    # geometry
    vocab_size: int = 129280
    dim: int = 4096
    n_heads: int = 64
    n_layers: int = 43
    n_mtp_layers: int = 1
    # attention
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: List[int] = field(default_factory=lambda: [0, 0] + [4, 128] * 20 + [4, 0])
    # rope
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    # indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # moe
    moe_inter_dim: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    n_hash_layers: int = 3
    # mhc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # general
    norm_eps: float = 1e-6
    # runtime
    max_batch_size: int = 4
    max_seq_len: int = 4096


def _build_block(layer_id: int, args: V4Args,
                 weights: Optional[Dict[str, torch.Tensor]] = None,
                 prefix: str = "") -> Block:
    return Block(
        layer_id=layer_id,
        dim=args.dim, n_heads=args.n_heads, q_lora_rank=args.q_lora_rank,
        head_dim=args.head_dim, rope_head_dim=args.rope_head_dim,
        o_lora_rank=args.o_lora_rank, o_groups=args.o_groups,
        window_size=args.window_size,
        compress_ratio=args.compress_ratios[layer_id],
        compress_rope_theta=args.compress_rope_theta, rope_theta=args.rope_theta,
        rope_factor=args.rope_factor, beta_fast=args.beta_fast, beta_slow=args.beta_slow,
        original_seq_len=args.original_seq_len,
        max_batch_size=args.max_batch_size, max_seq_len=args.max_seq_len,
        index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        moe_inter_dim=args.moe_inter_dim, n_routed_experts=args.n_routed_experts,
        n_activated_experts=args.n_activated_experts, n_shared_experts=args.n_shared_experts,
        score_func=args.score_func, route_scale=args.route_scale,
        swiglu_limit=args.swiglu_limit, n_hash_layers=args.n_hash_layers,
        vocab_size=args.vocab_size,
        hc_mult=args.hc_mult, hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps,
        norm_eps=args.norm_eps,
        weights=weights, prefix=prefix,
    )


class V4Transformer(nn.Module):
    """Standalone V4 forward. No TP/EP/PP sharding (world_size=1)."""

    def __init__(self, args: V4Args, weights: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult
        self._factory_mode = weights is not None

        self.embed = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList([
            _build_block(
                i, args,
                weights=weights,
                prefix=f"layers.{i}" if self._factory_mode else "",
            )
            for i in range(args.n_layers)
        ])
        self.norm = _RMSNorm(args.dim, args.norm_eps)

        # MTP layers
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            mtp_args = args
            # NOTE: MTPBlock ckpt keys live under `mtp.{i}.*`. Loading not
            # yet implemented — the block is a placeholder today. Pass
            # weights=None to keep legacy empty-param construction.
            blk = _build_block(args.n_layers + i, mtp_args, weights=None, prefix="")
            self.mtp.append(blk)

        # Final LM head + hc_head reduce
        # head.weight FP32 per official ParallelHead.
        self.head = _LMHead(args.vocab_size, args.dim)
        hc_dim = args.hc_mult * args.dim
        if self._factory_mode:
            # Embedding + LM head are BF16 in ckpt; cast to FP32 head (match
            # official ParallelHead.weight.dtype) but keep embed in BF16.
            self.embed.weight = nn.Parameter(weights["embed.weight"], requires_grad=False)
            self.head.weight = nn.Parameter(weights["head.weight"].float(), requires_grad=False)
            self.norm.weight = nn.Parameter(weights["norm.weight"].float(), requires_grad=False)
            self.hc_head_fn = nn.Parameter(weights["hc_head_fn"].float(), requires_grad=False)
            self.hc_head_base = nn.Parameter(weights["hc_head_base"].float(), requires_grad=False)
            self.hc_head_scale = nn.Parameter(weights["hc_head_scale"].float(), requires_grad=False)
        else:
            self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, hc, d] -> [B, S, d]"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0,
                apply_lm_head: bool = True) -> torch.Tensor:
        """Standalone forward.

        Returns:
          if apply_lm_head: logits of LAST token [B, vocab_size] (official behavior)
          else:            pre-lm-head hidden state of ALL tokens [B, S, d] — for
                           framework wrapper which applies lm_head externally
        """
        h = self.embed(input_ids)                                  # [B, S, d]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)           # [B, S, hc, d]
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        h = self._hc_head_reduce(h)                                # [B, S, d]
        h = self.norm(h)
        if apply_lm_head:
            return F.linear(h[:, -1].float(), self.head.weight)
        return h

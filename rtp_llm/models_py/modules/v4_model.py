"""DeepSeek-V4 top-level Transformer.

A freestanding port of `inference/model.py:Transformer` adapted to RTP-LLM's
parallelism conventions but otherwise faithful to the official reference.

This is the **mock-KV-cache milestone** target: end-to-end forward that runs in
pure PyTorch with inline `register_buffer` per-layer KV cache. Once the
heterogeneous framework KV cache (M4) lands, the `v4_attention.V4Attention`
inline cache can be swapped for the engine's paged blocks without touching the
top-level model.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from rtp_llm.models_py.modules.mhc import MHCHead, MHCMixing
from rtp_llm.models_py.modules.v4_attention import (
    V4Attention,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from rtp_llm.models_py.modules.v4_block import V4Block, V4MTPBlock
from rtp_llm.models_py.modules.v4_moe import V4MoE


# =============================================================================
# Parallelism primitives — TP-aware embedding / linear / lm_head.
# These are intentionally lightweight ports of the official ParallelEmbedding /
# ColumnParallelLinear / RowParallelLinear / ParallelHead. Production should use
# RTP-LLM's `models_py/distributed/collective_torch` and `LinearFactory` for
# fused all-reduce / overlap. Kept inline here for the M0/M1/M2 mock-KV path.
# =============================================================================


class V4ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, world_size: int, rank: int):
        super().__init__()
        assert vocab_size % world_size == 0, (
            f"vocab_size {vocab_size} must be divisible by world_size {world_size}"
        )
        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        self.rank = rank
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x = x.masked_fill(mask, 0)
        y = F.embedding(x, self.weight)
        if self.world_size > 1:
            y = y.masked_fill(mask.unsqueeze(-1), 0.0)
            dist.all_reduce(y)
        return y


class V4ParallelHead(nn.Module):
    """Final lm_head: applies hc_head fold then column-sharded vocab projection.

    For the *last* token only (V4 generation pattern: predict next token from
    last position). The full per-position logits version takes negligible extra
    work — change `x[:, -1]` to `x` if needed for training.
    """

    def __init__(self, vocab_size: int, dim: int, world_size: int):
        super().__init__()
        assert vocab_size % world_size == 0
        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        self.part_vocab_size = vocab_size // world_size
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, d] (post hc_head fold + final norm)
        returns logits: [b, vocab_size]   (last position)
        """
        logits = F.linear(x[:, -1].float(), self.weight)
        if self.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


# =============================================================================
# Top-level V4 Transformer
# =============================================================================


class V4Transformer(nn.Module):
    """DeepSeek-V4 model.

    Forward (prefill, start_pos == 0):
        h = embed(input_ids)                    # [b, s, d]
        h = h.unsqueeze(2).repeat(1,1,hc,1)     # [b, s, hc, d]  ← expand to HC
        for layer in layers:
            h = layer(h, start_pos, input_ids)
        logits = head(hc_head_fold(final_norm(h)))   # [b, vocab]
    Decode (start_pos > 0, single token): same pipeline, just with seqlen=1.

    MTP: Transformer.mtp[i] takes the post-main `h` and predicts (i+1)-token-ahead
    logits. See `mtp_forward` for the call pattern.
    """

    def __init__(
        self,
        # Top-level
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_mtp_layers: int,
        n_hash_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        # Attention
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        o_lora_rank: int,
        n_groups: int,
        window_size: int,
        compress_ratios: list,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_theta: float,
        compress_rope_theta: float,
        rope_factor: float,
        beta_fast: float,
        beta_slow: float,
        original_seq_len: int,
        # MoE
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: int,
        route_scale: float,
        swiglu_limit: float,
        # mHC
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
        # Parallelism
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        assert (
            len(compress_ratios) == n_layers + n_mtp_layers
        ), f"compress_ratios len {len(compress_ratios)} != n_layers+n_mtp {n_layers}+{n_mtp_layers}"
        self.world_size = world_size
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.dim = dim

        # Embedding (TP-sharded along vocab)
        self.embed = V4ParallelEmbedding(vocab_size, dim, world_size, rank)

        # Main transformer layers
        block_kwargs = dict(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            window_size=window_size,
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
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            moe_hash_routing_layers=n_hash_layers,
            vocab_size=vocab_size,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            hc_eps=hc_eps,
            world_size=world_size,
            rank=rank,
        )
        self.layers = nn.ModuleList(
            [
                V4Block(layer_id=i, compress_ratio=compress_ratios[i], **block_kwargs)
                for i in range(n_layers)
            ]
        )
        # MTP layers (one per nextN prediction depth)
        self.mtp = nn.ModuleList(
            [
                V4MTPBlock(
                    layer_id=n_layers + i,
                    compress_ratio=compress_ratios[n_layers + i],
                    **block_kwargs,
                )
                for i in range(n_mtp_layers)
            ]
        )

        # Final norm + head
        self.final_norm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.head = V4ParallelHead(vocab_size, dim, world_size)
        # MTP layers share embed + head with main model.
        for mtp_block in self.mtp:
            mtp_block.embed = self.embed
            mtp_block.head = self.head

        # Top-level mHC head fold (4 → 1).
        self.hc_head = MHCHead(dim, hc_mult, norm_eps, hc_eps)
        # MTP shares the same hc_head pattern but each MTP block has its own
        # `hc_head_fn / hc_head_base / hc_head_scale` parameters in the official
        # checkpoint; expose them as attributes on each MTP block (already done
        # via `V4MTPBlock` if needed — the MTP weight loading wires them up).

    def _final_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.norm_eps)
        return (self.final_norm_weight * xf).to(dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """input_ids: [b, s] (s == seqlen for prefill, s == 1 for decode steps)
        Returns: logits [b, vocab_size]   for the LAST position only.
        """
        h = self.embed(input_ids)  # [b, s, d]
        # Expand to hc_mult copies of the residual stream.
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [b, s, hc, d]
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        # Top-level fold + final norm + lm_head
        h = self.hc_head(h)  # [b, s, d]
        h = self._final_rmsnorm(h)
        return self.head(h)

    @torch.inference_mode()
    def mtp_forward(
        self,
        h: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
        mtp_idx: int = 0,
    ) -> torch.Tensor:
        """Run one MTP block on top of the main model's output `h: [b, s, hc, d]`.
        Returns logits for the next-N-token prediction.
        """
        return self.mtp[mtp_idx].forward_mtp(h, start_pos, input_ids, self.embed)


def from_v4_config(config_dict: dict, max_batch_size: int = 4, world_size: int = 1, rank: int = 0) -> V4Transformer:
    """Build a `V4Transformer` directly from a HF V4 config dict (the JSON shape
    that lives at `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json`).

    Useful for standalone mock-KV experiments — bypasses the C++ engine entirely.
    """
    rope_scaling = config_dict.get("rope_scaling", {}) or {}
    return V4Transformer(
        vocab_size=config_dict["vocab_size"],
        dim=config_dict["hidden_size"],
        n_layers=config_dict["num_hidden_layers"],
        n_mtp_layers=config_dict.get("num_nextn_predict_layers", 0),
        n_hash_layers=config_dict.get("num_hash_layers", 0),
        max_batch_size=max_batch_size,
        max_seq_len=config_dict.get("max_position_embeddings", 4096),
        n_heads=config_dict["num_attention_heads"],
        head_dim=config_dict["head_dim"],
        rope_head_dim=config_dict["qk_rope_head_dim"],
        q_lora_rank=config_dict["q_lora_rank"],
        o_lora_rank=config_dict["o_lora_rank"],
        n_groups=config_dict["o_groups"],
        window_size=config_dict["sliding_window"],
        compress_ratios=config_dict["compress_ratios"],
        index_n_heads=config_dict["index_n_heads"],
        index_head_dim=config_dict["index_head_dim"],
        index_topk=config_dict["index_topk"],
        rope_theta=float(config_dict["rope_theta"]),
        compress_rope_theta=float(config_dict["compress_rope_theta"]),
        rope_factor=float(rope_scaling.get("factor", 1.0)),
        beta_fast=float(rope_scaling.get("beta_fast", 32)),
        beta_slow=float(rope_scaling.get("beta_slow", 1)),
        original_seq_len=int(rope_scaling.get("original_max_position_embeddings", 0)),
        moe_inter_dim=config_dict["moe_intermediate_size"],
        n_routed_experts=config_dict["n_routed_experts"],
        n_activated_experts=config_dict["num_experts_per_tok"],
        n_shared_experts=config_dict.get("n_shared_experts", 1),
        score_func={"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}[
            config_dict.get("scoring_func", "sqrtsoftplus")
        ],
        route_scale=float(config_dict.get("routed_scaling_factor", 1.0)),
        swiglu_limit=float(config_dict.get("swiglu_limit", 0.0)),
        hc_mult=int(config_dict["hc_mult"]),
        hc_sinkhorn_iters=int(config_dict.get("hc_sinkhorn_iters", 20)),
        hc_eps=float(config_dict.get("hc_eps", 1e-6)),
        norm_eps=float(config_dict.get("rms_norm_eps", 1e-6)),
        world_size=world_size,
        rank=rank,
    )

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

from rtp_llm.models_py.modules.dsv4.block import Block, MTPBlock, _RMSNorm
from rtp_llm.models_py.modules.dsv4.cp import CPContext, build_cp_context


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
    compress_ratios: List[int] = field(
        default_factory=lambda: [0, 0] + [4, 128] * 20 + [4, 0]
    )
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
    # Peak per-rank token count that the routed-MoE path must budget
    # for (Mega MoE's symmetric-memory dispatch buffer is sized from
    # this at init).  Defaults to ``max_seq_len`` downstream when not
    # explicitly set; kept separate so non-MoE engines can leave the
    # buffer small.
    max_tokens_per_rank: int = 8192
    # parallelism (S7 scaffold — currently unused; populated from
    # ParallelismConfig in deepseek_v4_model.py and threaded down so
    # downstream patches can shard without re-wiring the constructor.)
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    world_size: int = 1
    world_rank: int = 0


def _block_kwargs(
    layer_id: int, args: V4Args, weights: Optional[Dict[str, torch.Tensor]], prefix: str
) -> Dict:
    """Kwargs common to Block and MTPBlock construction.

    ``compress_ratios`` is sized ``n_layers + n_mtp_layers`` (44 for
    V4-Flash default) so ``compress_ratios[layer_id]`` works directly
    for both main layers and MTP layers.
    """
    return dict(
        layer_id=layer_id,
        dim=args.dim,
        n_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        o_lora_rank=args.o_lora_rank,
        o_groups=args.o_groups,
        window_size=args.window_size,
        compress_ratio=args.compress_ratios[layer_id],
        compress_rope_theta=args.compress_rope_theta,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        original_seq_len=args.original_seq_len,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        moe_inter_dim=args.moe_inter_dim,
        n_routed_experts=args.n_routed_experts,
        n_activated_experts=args.n_activated_experts,
        n_shared_experts=args.n_shared_experts,
        score_func=args.score_func,
        route_scale=args.route_scale,
        swiglu_limit=args.swiglu_limit,
        n_hash_layers=args.n_hash_layers,
        vocab_size=args.vocab_size,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
        norm_eps=args.norm_eps,
        weights=weights,
        prefix=prefix,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        ep_size=args.ep_size,
        ep_rank=args.ep_rank,
        max_tokens_per_rank=args.max_tokens_per_rank,
    )


def _build_block(
    layer_id: int,
    args: V4Args,
    weights: Optional[Dict[str, torch.Tensor]] = None,
    prefix: str = "",
) -> Block:
    return Block(**_block_kwargs(layer_id, args, weights, prefix))


def _build_mtp_block(
    layer_id: int,
    args: V4Args,
    weights: Optional[Dict[str, torch.Tensor]] = None,
    prefix: str = "",
) -> MTPBlock:
    return MTPBlock(**_block_kwargs(layer_id, args, weights, prefix))


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

        self.layers = nn.ModuleList(
            [
                _build_block(
                    i,
                    args,
                    weights=weights,
                    prefix=f"layers.{i}" if self._factory_mode else "",
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = _RMSNorm(args.dim, args.norm_eps)

        # MTP layers — draft heads for speculative decoding.  Full impl
        # in block.py::MTPBlock; each layer owns its own e_proj/h_proj +
        # enorm/hnorm/norm + per-layer hc_head_*, plus all the regular
        # Block machinery (attn, ffn, mHC). The shared embedding and LM
        # head flow through ``forward_draft`` explicitly at inference
        # time rather than being stashed on the MTPBlock instance.
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            prefix = f"mtp.{i}" if self._factory_mode else ""
            blk = _build_mtp_block(
                args.n_layers + i, args, weights=weights, prefix=prefix
            )
            self.mtp.append(blk)

        # Final LM head + hc_head reduce
        # head.weight FP32 per official ParallelHead.
        self.head = _LMHead(args.vocab_size, args.dim)
        hc_dim = args.hc_mult * args.dim
        if self._factory_mode:
            # Embedding + LM head are BF16 in ckpt; cast to FP32 head (match
            # official ParallelHead.weight.dtype) but keep embed in BF16.
            self.embed.weight = nn.Parameter(
                weights["embed.weight"], requires_grad=False
            )
            self.head.weight = nn.Parameter(
                weights["head.weight"].float(), requires_grad=False
            )
            self.norm.weight = nn.Parameter(
                weights["norm.weight"].float(), requires_grad=False
            )
            self.hc_head_fn = nn.Parameter(
                weights["hc_head_fn"].float(), requires_grad=False
            )
            self.hc_head_base = nn.Parameter(
                weights["hc_head_base"].float(), requires_grad=False
            )
            self.hc_head_scale = nn.Parameter(
                weights["hc_head_scale"].float(), requires_grad=False
            )
        else:
            self.hc_head_fn = nn.Parameter(
                torch.empty(args.hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(args.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def set_cp_info(self, cp_info, cp_size: int, cp_rank: int) -> None:
        """Bind / clear the framework's Context-Parallel metadata for the
        NEXT forward.  Only stashes the raw metadata; the derived
        ``CPContext`` (global positions, stripped restore indices, etc.)
        is built inside ``forward`` once chunk_length is known and then
        propagated to every layer's attn + compressor + indexer via
        ``set_cp_ctx``."""
        self._cp_info = cp_info
        self._cp_size = int(cp_size)
        self._cp_rank = int(cp_rank)

    def _propagate_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            attn.set_cp_ctx(cp_ctx)
            c = getattr(attn, "compressor", None)
            if c is not None:
                c.set_cp_ctx(cp_ctx)
            idx = getattr(attn, "indexer", None)
            if idx is not None:
                idx.set_cp_ctx(cp_ctx)
                ic = getattr(idx, "compressor", None)
                if ic is not None:
                    ic.set_cp_ctx(cp_ctx)

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, hc, d] -> [B, S, d]"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    @torch.inference_mode()
    def forward_decode(
        self,
        input_ids: torch.Tensor,  # [T_total] int (== [B] for q_len=1)
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Decode-only forward.

        Returns ``hidden [T_total, dim]`` (caller applies lm_head).
        Prefill ``forward`` is untouched — PD-disagg later splits cleanly.
        """
        B = attn_metadata.batch_size
        q_len = attn_metadata.q_len_per_req
        if input_ids.dim() == 1:
            input_ids_2d = input_ids.view(B, q_len)
        else:
            input_ids_2d = input_ids
        h = self.embed(input_ids_2d)  # [B, q_len, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, q_len, hc, dim]
        for layer in self.layers:
            h = layer.forward_decode(h, attn_metadata, input_ids_2d)
        h = self._hc_head_reduce(h)  # [B, q_len, dim]
        h = self.norm(h)  # [B, q_len, dim]
        # Return packed [T_total, dim] for the framework.
        return h.reshape(B * q_len, self.args.dim)

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, start_pos=0, apply_lm_head: bool = True
    ) -> torch.Tensor:
        """Standalone forward.

        Returns:
          if apply_lm_head: logits of LAST token [B, vocab_size] (official behavior)
          else:            pre-lm-head hidden state of ALL tokens [B, S, d] — for
                           framework wrapper which applies lm_head externally

        Empty-batch contract: when ``input_ids`` has zero tokens (S=0),
        every downstream module — ``apply_rotary_emb`` in particular,
        which does ``view_as_complex`` on an S=0 unflatten — would
        crash.  Under DP/EP the framework routinely issues forwards to
        ranks that have no local work to do; short-circuit with a
        correctly-shaped empty output so those ranks still participate
        in any collective that other ranks trigger (DeepEP dispatch,
        all_reduce, etc.).
        """
        B = input_ids.size(0)
        S = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
        # S=0 short-circuit only applies for TP=EP=1.  With ep_size>1,
        # DeepEP dispatch/combine are collectives that ALL ranks must
        # enter together, even the ones with zero local tokens — else
        # the missing rank's absence deadlocks the buffer with "CPU
        # recv timeout".  The caller is responsible for clamping /
        # padding ``start_pos`` so downstream indexing stays in range
        # (see ``DeepSeekV4Model.forward``); individual layers are
        # empty-batch-safe (``apply_rotary_emb`` and the sparse-attn
        # path both short-circuit on zero-element inputs).
        if S == 0 and self.args.ep_size <= 1:
            device = next(self.parameters()).device
            if apply_lm_head:
                return torch.zeros(
                    (B, self.head.weight.size(0)), dtype=torch.float32, device=device
                )
            return torch.zeros(
                (B, 0, self.args.dim), dtype=torch.bfloat16, device=device
            )

        # Build + propagate CP context once per forward.  When CP is
        # active (set_cp_info previously called with a real cp_info +
        # cp_size>1 + this is a prefill step), every layer's attn /
        # compressor / indexer gets a fresh CPContext with the derived
        # global_positions + unpad_restore tensors.  Otherwise we clear
        # any stale context with None so the modules fall through to
        # the single-rank path unchanged.
        _cp_info = getattr(self, "_cp_info", None)
        _cp_size = getattr(self, "_cp_size", 1)
        _cp_rank = getattr(self, "_cp_rank", 0)
        cp_ctx: Optional[CPContext] = None
        if _cp_info is not None and _cp_size > 1 and start_pos == 0 and S > 0:
            device = input_ids.device
            cp_ctx = build_cp_context(_cp_info, _cp_size, _cp_rank, S, device)
        self._propagate_cp_ctx(cp_ctx)

        h = self.embed(input_ids)  # [B, S, d]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, S, hc, d]
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        h = self._hc_head_reduce(h)  # [B, S, d]
        h = self.norm(h)
        if apply_lm_head:
            return F.linear(h[:, -1].float(), self.head.weight)
        return h

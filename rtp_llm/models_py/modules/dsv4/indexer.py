"""DeepSeek-V4 lightning Indexer for CSA.

Faithful BF16 port of `inference/model.py:Indexer`. Skips Hadamard rotation
+ FP4 quant (BF16-only path for M2/M3 correctness validation).

Has its own dedicated Compressor (rotate=True in official code; we keep
the parameter for ckpt-loader symmetry but don't apply Hadamard).
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_freqs_cis_local
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

# P0 (prefill_opt/final_plan.md): fused indexer score Triton kernel.
# Replaces einsum + relu + weighted-sum chunked path.  Set
# DSV4_INDEXER_FAST=0 to force the chunked REF (debug only).
try:
    from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
    _INDEXER_FAST_OK = True
except Exception:  # pragma: no cover — keep V4 importable without Triton
    v4_indexer_score = None
    _INDEXER_FAST_OK = False


def _use_indexer_fast() -> bool:
    if not _INDEXER_FAST_OK:
        return False
    return os.environ.get("DSV4_INDEXER_FAST", "1") != "0"


class Indexer(nn.Module):
    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        compress_ratio: int,
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self._factory_mode = weights is not None

        if self._factory_mode:
            from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear_from_dict
            self.wq_b = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.wq_b.weight", f"{prefix}.wq_b.scale",
            )
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)
            # weights_proj stays in the ckpt dtype (BF16); the legacy path
            # used `nn.Linear(...)` under `torch.set_default_dtype(bf16)`
            # context in DeepSeekV4Model, so BF16 matches behavior.
            self.weights_proj.weight = nn.Parameter(
                weights[f"{prefix}.weights_proj.weight"], requires_grad=False,
            )
        else:
            self.wq_b = QuantizedLinear(q_lora_rank, index_n_heads * index_head_dim, storage="fp8")
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)

        self.compressor = Compressor(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            norm_eps=norm_eps,
            rotate=True,
            weights=weights,
            prefix=f"{prefix}.compressor" if self._factory_mode else "",
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, max_seq_len // compress_ratio, index_head_dim),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None
        # CP context bound per-forward by V4Transformer; None = no CP.
        self._cp_ctx: Optional[CPContext] = None

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind CP context.  When active, rank-local Q applies RoPE at
        GLOBAL positions (not rank-local row indices); the causal mask
        over the compressed-KV axis uses global positions; and the score
        einsum reads the nested compressor's ``kv_cache[:, :seq_len_full
        // ratio]`` which was just populated with the full compressed KV
        by ``Compressor.forward``'s all-gather path.

        The outer ``offset`` passed by ``Attention`` is the number of
        sliding-window KV slots that precede the compressed-KV block in
        the concatenated ``[sliding | compressed]`` layout — under CP
        the sliding slots equal ``seq_len_full``, not ``chunk_length``.
        ``Attention`` passes the already-computed offset; this method
        only needs to expose the context so Indexer can position the
        causal mask / topk-mask relative to global Q positions."""
        self._cp_ctx = cp_ctx

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        cp_ctx = self._cp_ctx
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and start_pos == 0

        if cp_on:
            # Rank-local Q at its GLOBAL positions.
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            # Compressed KV spans [0, seq_len_full // ratio) after the
            # nested compressor.forward populates kv_cache via gather.
            end_pos = cp_ctx.seq_len_full
        else:
            freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
            end_pos = start_pos + seqlen

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        if self._factory_mode and qr.dim() > 2:
            shape = qr.shape
            q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                *shape[:-1], self.n_heads * self.head_dim,
            )
        else:
            q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        # Skip rotate_activation + fp4_act_quant in this BF16 path

        # Nested compressor: reads its own _cp_ctx set by V4Transformer,
        # all-gathers rank-local kv/score → writes full compressed KV
        # into our self.kv_cache (bound above).
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        # Per-token index score: for each Q token we score every key in
        # ``kv_cache[:, :end_pos//ratio]``, ReLU, weight-sum over heads.
        # The naive einsum materializes ``[B, S, n_heads, T]`` fp32 which
        # is ``O(S*T*H)`` — 64 GB at S=64K, T=16K, H=64.  Two paths:
        #
        # * fast path (DSV4_INDEXER_FAST=1, default): single Triton
        #   kernel `v4_indexer_score` streams the H dim and never
        #   materializes the [B,S,H,T] intermediate; causal mask folded
        #   in at write-time.  See _indexer_score_triton.py.
        # * REF path (DSV4_INDEXER_FAST=0): chunked einsum + relu + sum.
        kv = self.kv_cache[:bsz, :end_pos // ratio]
        T = kv.size(1)

        # Build q_pos for the causal mask (only on the prefill chunk).
        if start_pos == 0:
            if cp_on:
                q_pos = cp_ctx.global_positions.unsqueeze(0).expand(bsz, -1).contiguous()
            else:
                q_pos = (
                    torch.arange(seqlen, dtype=torch.int32, device=x.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                    .contiguous()
                )
        else:
            q_pos = None

        if _use_indexer_fast() and T > 0 and seqlen > 0:
            # contiguous required by the Triton kernel
            kv_c = kv.contiguous() if not kv.is_contiguous() else kv
            q_c = q if q.is_contiguous() else q.contiguous()
            index_score = v4_indexer_score(
                q_c, kv_c, weights,
                q_pos=q_pos,
                compress_ratio=ratio,
            )
        else:
            q_f = q.float()
            w_f = weights.float()
            S = q_f.size(1)
            # REF chunked path — peak chunk_size * n_heads * T * 4 bytes ≤ 2 GB.
            max_chunk_bytes = 2 * (1 << 30)
            denom = max(self.n_heads * max(T, 1) * 4, 1)
            chunk_size = max(1, min(S, max_chunk_bytes // denom))
            if chunk_size >= S:
                index_score = torch.einsum("bshd,btd->bsht", q_f, kv.float())
                index_score = (index_score.relu_() * w_f.unsqueeze(-1)).sum(dim=2)
            else:
                parts = []
                kv_f = kv.float()
                for i in range(0, S, chunk_size):
                    end = min(i + chunk_size, S)
                    score = torch.einsum(
                        "bshd,btd->bsht", q_f[:, i:end], kv_f,
                    )
                    score = (score.relu_() * w_f[:, i:end].unsqueeze(-1)).sum(dim=2)
                    parts.append(score)
                index_score = torch.cat(parts, dim=1)
                del parts, kv_f

            # REF path applies the causal mask separately; the fast path
            # folds it into the kernel write.
            if q_pos is not None and T > 0:
                kv_cols = torch.arange(T, device=x.device)
                thr = (q_pos.long() + 1) // ratio  # [B, S]
                mask = kv_cols.unsqueeze(0).unsqueeze(0) >= thr.unsqueeze(-1)
                index_score = index_score + torch.where(mask, float("-inf"), 0.0)

        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            # topk_idxs are in compressed-KV space [0, T_comp).  Entries
            # past this Q token's allowed window become -1 (masked).  The
            # remaining ones shift by ``offset`` — the base index in the
            # attention-side concatenated tensor (offset = S_sliding).
            # ``q_pos`` was built above for the score causal mask; reuse it.
            q_pos_1b = (q_pos.long() + 1).unsqueeze(-1)  # [B, S_local, 1]
            mask = topk_idxs >= (q_pos_1b // ratio)
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

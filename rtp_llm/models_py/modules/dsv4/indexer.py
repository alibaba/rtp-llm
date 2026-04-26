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
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)

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
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self._factory_mode = weights is not None

        if self._factory_mode:
            from rtp_llm.models_py.modules.dsv4.attention import (
                _v4_fp8_linear_from_dict,
            )

            self.wq_b = _v4_fp8_linear_from_dict(
                weights,
                f"{prefix}.wq_b.weight",
                f"{prefix}.wq_b.scale",
            )
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)
            # weights_proj stays in the ckpt dtype (BF16); the legacy path
            # used `nn.Linear(...)` under `torch.set_default_dtype(bf16)`
            # context in DeepSeekV4Model, so BF16 matches behavior.
            self.weights_proj.weight = nn.Parameter(
                weights[f"{prefix}.weights_proj.weight"],
                requires_grad=False,
            )
        else:
            self.wq_b = QuantizedLinear(
                q_lora_rank, index_n_heads * index_head_dim, storage="fp8"
            )
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

    def forward_decode(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16 — single decode token per req
        qr: torch.Tensor,  # [B, 1, q_lora_rank] bf16 — q_a output
        start_pos: torch.Tensor,  # [B] int32 — abs pos per request
        out_topk_buffer: torch.Tensor,  # [B, 1, K=index_topk] int32 — pre-allocated by metadata builder
    ) -> torch.Tensor:
        """Batched decode-time indexer.

        Per request r: runs the small Compressor step (already
        ``self.compressor.forward_decode``-friendly), computes per-token
        index score against ``self.kv_cache[r, :compressed_len[r]]``,
        and writes top-K indices into ``out_topk_buffer[r]``.

        For requests with ``compressed_len < K``, the unused slots are
        filled with -1 (downstream sparse_attn masks them).

        Decode-only — does NOT touch the prefill ``forward`` arm.
        """
        assert x.shape[1] == 1, "decode-only: q_len must be 1"
        bsz = x.size(0)
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        K = self.index_topk

        # Compressor decode step writes new compressed K to its kv_cache
        # (== self.kv_cache via bind in attention forward_decode).
        # We don't use its return value here; we just need the cache to be
        # current for the score computation below.
        self.compressor.forward_decode(x, start_pos)

        # qr -> wq_b -> [B, 1, n_heads * head_dim] -> unflatten
        if self._factory_mode and qr.dim() > 2:
            shape = qr.shape
            q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                *shape[:-1],
                self.n_heads * self.head_dim,
            )
        else:
            q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))  # [B, 1, H_idx, D_idx]

        # Per-request RoPE on q_pe (each request has its own start_pos).
        # Rather than loop, we gather per-request freqs_cis once.
        freqs_cis_per_req = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
        # apply_rotary_emb expects a contiguous [seq, freqs_dim] view to
        # broadcast over [B, S, H, last_dim]. Loop over B for simplicity.
        for r in range(bsz):
            apply_rotary_emb(q[r : r + 1, :, :, -rd:], freqs_cis_per_req[r : r + 1])

        # weights = weights_proj(x) * scale
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads**-0.5
        )  # [B, 1, H_idx]

        # score against per-request compressed-K cache slice + topk.
        # compressed_len[r] = (start_pos[r] + 1) // ratio (post-step length).
        compressed_len = ((start_pos + 1) // ratio).to(torch.int64)  # [B]
        out_topk_buffer.fill_(-1)
        for r in range(bsz):
            T_r = int(compressed_len[r].item())
            if T_r <= 0:
                continue
            kv_r = self.kv_cache[r : r + 1, :T_r]  # [1, T_r, D_idx]
            q_r = q[r : r + 1].float()  # [1, 1, H_idx, D_idx]
            w_r = weights[r : r + 1].float()  # [1, 1, H_idx]
            score = torch.einsum("bshd,btd->bsht", q_r, kv_r.float())
            score = (score.relu_() * w_r.unsqueeze(-1)).sum(dim=2)  # [1, 1, T_r]
            k_r = min(K, T_r)
            topk_r = score.topk(k_r, dim=-1)[1].to(torch.int32)
            out_topk_buffer[r : r + 1, :, :k_r] = topk_r
        return out_topk_buffer

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16
        qr: torch.Tensor,  # [B, 1, q_lora_rank] bf16
        start_pos: torch.Tensor,  # [B] int32
        out_topk_buffer: torch.Tensor,  # [B, 1, K]
    ) -> torch.Tensor:
        """Stage 3B vectorized variant of :meth:`forward_decode`.

        No Python loops over B, no ``.item()`` calls. The compressor step
        uses the vectorized variant; per-request RoPE uses
        ``apply_rotary_emb_batched``; the score / topk is computed
        batched across B with a length mask (positions beyond
        ``compressed_len[r]`` are set to ``-inf`` so ``topk`` returns
        valid indices for the leading prefix and arbitrary indices for
        the masked tail). The caller masks via ``compressed_lens`` from
        the metadata (already pre-computed).

        Result shape: ``out_topk_buffer`` modified in place; returns
        the same tensor for caller convenience.
        """
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        K = self.index_topk

        # Compressor decode (vectorized) writes new compressed K.
        self.compressor.forward_decode_vectorized(x, start_pos)

        # qr -> wq_b -> [B, 1, n_heads * head_dim] -> [B, 1, H_idx, D_idx]
        if self._factory_mode and qr.dim() > 2:
            shape = qr.shape
            q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                *shape[:-1],
                self.n_heads * self.head_dim,
            )
        else:
            q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))  # [B, 1, H_idx, D_idx]

        # Per-request batched RoPE on q_pe.
        freqs_per_b = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
        apply_rotary_emb_batched(q[..., -rd:], freqs_per_b)

        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads**-0.5
        )  # [B, 1, H_idx]

        # Batched score against the FULL kv_cache prefix (length
        # max_seq_len // ratio). Mask invalid positions to -inf so topk
        # only returns valid leading indices.
        compressed_len = (
            ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)
        )  # [B, 1, 1]
        T_max = self.kv_cache.shape[1]
        kv_full = self.kv_cache[:bsz].float()  # [B, T_max, D_idx]
        q32 = q.float()  # [B, 1, H_idx, D_idx]
        w32 = weights.float()  # [B, 1, H_idx]
        # einsum: [B, S, H, D] x [B, T, D] -> [B, S, H, T]
        score = torch.einsum("bshd,btd->bsht", q32, kv_full)
        score = (score.relu_() * w32.unsqueeze(-1)).sum(dim=2)  # [B, S=1, T_max]

        # Mask T positions >= compressed_len[r] to -inf.
        t_arange = torch.arange(T_max, device=score.device).view(1, 1, T_max)
        score = torch.where(
            t_arange < compressed_len,
            score,
            torch.full_like(score, float("-inf")),
        )

        # topk over T_max — returns valid indices for the leading
        # prefix; for requests with compressed_len < K the tail of the
        # topk result is meaningless (all scores were -inf so any tie
        # is broken arbitrarily). Match the loop variant's contract by
        # masking positions ``k >= compressed_len[r]`` back to -1.
        K_eff = min(K, T_max)
        out_topk_buffer.fill_(-1)
        if K_eff > 0:
            topk_idxs = score.topk(K_eff, dim=-1)[1].to(torch.int32)
            out_topk_buffer[:, :, :K_eff].copy_(topk_idxs)
            # Per-request: zero out positions beyond compressed_len[r].
            # Loop variant writes only k_r = min(K, T_r) entries and leaves
            # the rest at the initial -1; replicate that by masking.
            k_arange = torch.arange(K, device=out_topk_buffer.device).view(1, 1, K)
            out_topk_buffer.masked_fill_(
                k_arange >= compressed_len,
                -1,
            )

        return out_topk_buffer

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
    ) -> torch.Tensor:
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
                *shape[:-1],
                self.n_heads * self.head_dim,
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
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

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

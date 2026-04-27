"""DeepSeek-V4 lightning Indexer for CSA.

Faithful BF16 port of `inference/model.py:Indexer`. Skips Hadamard rotation
+ FP4 quant (BF16-only path for M2/M3 correctness validation).

Has its own dedicated Compressor (rotate=True in official code; we keep
the parameter for ckpt-loader symmetry but don't apply Hadamard).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_freqs_cis_local
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb


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

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos, offset: int
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        cp_ctx = self._cp_ctx
        is_batched = isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        cp_on = (
            cp_ctx is not None
            and cp_ctx.cp_size > 1
            and not is_batched
            and start_pos == 0
        )

        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            end_pos = cp_ctx.seq_len_full
        elif is_batched:
            positions = start_pos.long()
            freqs_cis = self.freqs_cis[positions].unsqueeze(1)  # [B, 1, rope_dim//2]
            end_pos = (start_pos.max() + seqlen).item()
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]
            end_pos = sp + seqlen

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
        # is ``O(S*T*H)`` — 64 GB at S=64K, T=16K, H=64.  Chunk in S to
        # keep peak memory bounded (under a few GB) at any seqlen.
        # Without this, warmup prefill at max_seq_len ≥ 32K OOMs.
        # TODO(K9 in kernel_audit): port V4-official ``index_score_kernel``
        # TileLang kernel — eliminates the materialized intermediate
        # entirely (online ReLU+reduce in shared memory).
        kv = self.kv_cache[:bsz, : end_pos // ratio]
        q_f = q.float()
        w_f = weights.float()
        S = q_f.size(1)
        # Chunk size picked so peak = chunk_size * n_heads * T * 4 bytes
        # stays under ~2 GB for typical T up to 32K.
        max_chunk_bytes = 2 * (1 << 30)
        T = kv.size(1)
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
                    "bshd,btd->bsht",
                    q_f[:, i:end],
                    kv_f,
                )
                score = (score.relu_() * w_f[:, i:end].unsqueeze(-1)).sum(dim=2)
                parts.append(score)
            index_score = torch.cat(parts, dim=1)
            del parts, kv_f

        # Causal mask: each Q token at GLOBAL position g can only read
        # compressed KV blocks [0, (g+1)//ratio).  Only needed for prefill
        # (start_pos == 0).  Decode (scalar or batched) always has start_pos > 0.
        is_prefill = not is_batched and (
            isinstance(start_pos, int)
            and start_pos == 0
            or isinstance(start_pos, torch.Tensor)
            and start_pos.item() == 0
        )
        if is_prefill:
            T_comp = index_score.size(-1)
            if cp_on:
                q_pos_1b = (cp_ctx.global_positions + 1).unsqueeze(
                    1
                )  # [chunk_length, 1]
            else:
                q_pos_1b = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(
                    1
                )  # [S, 1]
            kv_cols = torch.arange(T_comp, device=x.device)  # [T_comp]
            mask = kv_cols >= (q_pos_1b // ratio)  # [S_local, T_comp]
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)

        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if is_prefill:
            # topk_idxs are in compressed-KV space [0, T_comp).  Entries
            # past this Q token's allowed window become -1 (masked).  The
            # remaining ones shift by ``offset`` — the base index in the
            # attention-side concatenated tensor (offset = S_sliding).
            if cp_on:
                q_pos_1b = (cp_ctx.global_positions + 1).unsqueeze(1)
            else:
                q_pos_1b = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
            mask = topk_idxs >= (q_pos_1b // ratio)
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

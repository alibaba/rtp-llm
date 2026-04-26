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

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
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

        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        # Per-token index score: for each Q token we score every key in
        # ``kv_cache[:, :end_pos//ratio]``, ReLU, weight-sum over heads.
        # The naive einsum materializes ``[B, S, n_heads, T]`` fp32 which
        # is ``O(S*T*H)`` — 64 GB at S=64K, T=16K, H=64.  Chunk in S to
        # keep peak memory bounded (under a few GB) at any seqlen.
        # Without this, warmup prefill at max_seq_len ≥ 32K OOMs.
        # TODO(K9 in kernel_audit): port V4-official ``index_score_kernel``
        # TileLang kernel — eliminates the materialized intermediate
        # entirely (online ReLU+reduce in shared memory).
        kv = self.kv_cache[:bsz, :end_pos // ratio]
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
                    "bshd,btd->bsht", q_f[:, i:end], kv_f,
                )
                score = (score.relu_() * w_f[:, i:end].unsqueeze(-1)).sum(dim=2)
                parts.append(score)
            index_score = torch.cat(parts, dim=1)
            del parts, kv_f

        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= \
                   torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)

        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

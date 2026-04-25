"""DeepSeek-V4 Attention with HCA / CSA / SWA-only path selection.

Direct port of `inference/model.py:Attention` (BF16-only, mock per-layer
KV cache via register_buffer). Skips Hadamard rotate / FP4 / FP8 quant.

Layer schedule via `compress_ratio`:
  0   -> SWA-only (no Compressor, no Indexer)
  4   -> CSA (Compressor with overlap=True + Indexer for sparse top-k)
  128 -> HCA (Compressor with overlap=False, dense compressed MQA)

Sparse attention reference uses `gather`-based PyTorch implementation —
slow but correct. M6 will swap in FlashMLA sparse impl.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.indexer import Indexer
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb, precompute_freqs_cis


class _NormHolder(nn.Module):
    """Wraps an FP32 norm-weight parameter so that ckpt key `.weight` matches."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))


def _get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device) -> torch.Tensor:
    """Returns int64 [bsz, seqlen, window_size] giving the (cyclic) absolute slot indices
    in the sliding-window KV ring buffer that each query position should read."""
    if start_pos >= window_size - 1:
        sp = start_pos % window_size
        matrix = torch.cat(
            [torch.arange(sp + 1, window_size, device=device),
             torch.arange(0, sp + 1, device=device)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1, device=device), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        matrix = torch.where(matrix > base, -1, matrix)
        if matrix.size(1) < window_size:
            matrix = F.pad(matrix, (0, window_size - matrix.size(1)), value=-1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int, device) -> torch.Tensor:
    if start_pos > 0:
        n = (start_pos + 1) // ratio
        matrix = (torch.arange(0, n, device=device) + offset).unsqueeze(0).expand(seqlen, -1)
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _sparse_attn(
    q: torch.Tensor,           # [B, S, H, D]
    kv: torch.Tensor,          # [B, T_kv, D]   (single KV head, shared across H)
    sink: torch.Tensor,        # [H]   FP32 logit added to softmax denom (per-head sink)
    topk_idxs: torch.Tensor,   # [B, S, K] long; -1 entries are masked out
    softmax_scale: float,
) -> torch.Tensor:
    """Reference PyTorch sparse attention with attention sink.

    Output: [B, S, H, D]
    """
    bsz, seqlen, n_heads, head_dim = q.size()
    K = topk_idxs.size(-1)
    valid = (topk_idxs >= 0)                                            # [B, S, K]
    safe_idxs = topk_idxs.clamp_min(0)                                  # [B, S, K]

    # gather selected KV: [B, S, K, D]
    idx_expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_exp = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)                 # [B, S, T_kv, D]
    selected = torch.gather(kv_exp, 2, idx_expanded)                    # [B, S, K, D]

    # logits: [B, S, H, K] = einsum(qhd, kd)
    q_f = q.float()
    selected_f = selected.float()
    logits = torch.einsum("bshd,bskd->bshk", q_f, selected_f) * softmax_scale
    # mask invalid slots
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # Softmax with attn_sink — matches official `sparse_attn_kernel`:
    #   scores_max = max over logits only (NOT including sink)
    #   exp_logits = exp(logits - scores_max)
    #   acc_o = Σ exp_logits · v
    #   sum_exp = Σ exp_logits + exp(sink - scores_max)
    #   out = acc_o / sum_exp
    # Note: we do NOT include sink in `scores_max`, and the numerator has no sink term.
    scores_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e30)     # [B, S, H, 1]
    exp_logits = torch.exp(logits - scores_max)                          # [B, S, H, K]
    sink_logit = sink.view(1, 1, n_heads, 1).expand_as(scores_max)
    exp_sink = torch.exp(sink_logit - scores_max)                        # [B, S, H, 1]
    sum_exp = exp_logits.sum(dim=-1, keepdim=True) + exp_sink            # [B, S, H, 1]

    # acc_o = Σ_k exp_logits[k] · selected[k]
    acc_o = torch.einsum("bshk,bskd->bshd", exp_logits, selected_f)      # [B, S, H, D]
    out = acc_o / sum_exp                                                 # divide each head by its denom
    return out.to(q.dtype)


class Attention(nn.Module):
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
        # Indexer
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.n_groups = o_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.eps = norm_eps
        self.softmax_scale = head_dim ** -0.5

        # Q / KV / O — all attention linear weights stored as FP8 in V4-Flash ckpt.
        self.wq_a = QuantizedLinear(dim, q_lora_rank, storage="fp8")
        self.q_norm = _NormHolder(q_lora_rank)
        self.wq_b = QuantizedLinear(q_lora_rank, n_heads * head_dim, storage="fp8")

        self.wkv = QuantizedLinear(dim, head_dim, storage="fp8")
        self.kv_norm = _NormHolder(head_dim)

        # Grouped output projection
        assert (n_heads * head_dim) % o_groups == 0
        self.wo_a = QuantizedLinear(n_heads * head_dim // o_groups, o_groups * o_lora_rank, storage="fp8")
        self.wo_b = QuantizedLinear(o_groups * o_lora_rank, dim, storage="fp8")

        # per-head learnable attention sink (added to softmax denominator)
        self.attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))

        # Compressor + Indexer (only for compressed layers)
        if compress_ratio:
            self.compressor = Compressor(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=max_batch_size,
                norm_eps=norm_eps,
            )
            if compress_ratio == 4:
                self.indexer = Indexer(
                    dim=dim,
                    q_lora_rank=q_lora_rank,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    rope_head_dim=rope_head_dim,
                    index_topk=index_topk,
                    compress_ratio=compress_ratio,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # KV cache: [B, window_size + max_seq_len // ratio, head_dim]
        kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, kv_cache_size, head_dim),
            persistent=False,
        )

        # Per-layer freqs_cis: SWA-only uses base rope_theta with no yarn,
        # CSA/HCA uses compress_rope_theta with yarn (when original_seq_len > 0).
        # Store scalars so we can re-compute after `to_empty`(meta) — otherwise
        # the buffer ends up all zeros.
        if compress_ratio:
            self._rope_base = compress_rope_theta
            self._rope_o_seq_len = original_seq_len
        else:
            self._rope_base = rope_theta
            self._rope_o_seq_len = 0
        self._rope_factor = rope_factor
        self._rope_beta_fast = beta_fast
        self._rope_beta_slow = beta_slow
        self._rope_dim = rope_head_dim
        self._rope_max_seq_len = max_seq_len
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, max_seq_len, self._rope_o_seq_len, self._rope_base,
            rope_factor, beta_fast, beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def reset_rope_cache(self, device=None):
        """Recompute `freqs_cis` on the actual device — MUST be called after
        `model.to_empty(device=...)` since meta-tensor construction leaves the
        cached freqs as zeros."""
        freqs_cis = precompute_freqs_cis(
            self._rope_dim, self._rope_max_seq_len, self._rope_o_seq_len,
            self._rope_base, self._rope_factor, self._rope_beta_fast, self._rope_beta_slow,
        )
        if device is not None:
            freqs_cis = freqs_cis.to(device)
        self.freqs_cis = freqs_cis
        # Clear compressor / indexer bound references so they rebind on next forward
        if self.compressor is not None:
            self.compressor.freqs_cis = None
        if self.indexer is not None:
            self.indexer.freqs_cis = None
            if self.indexer.compressor is not None:
                self.indexer.compressor.freqs_cis = None

    def _rmsnorm_weighted(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (weight * x32).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x.device

        # bind compressor cache + freqs on first call
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q path
        qr = self._rmsnorm_weighted(self.wq_a(x), self.q_norm.weight)  # [B, S, q_lora_rank]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        # QK RMSNorm (no learnable scale here, per official code)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV path (single KV head)
        kv = self._rmsnorm_weighted(self.wkv(x), self.kv_norm.weight)  # [B, S, head_dim]
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # build topk_idxs
        topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, start_pos, device)
        if self.compress_ratio:
            # During prefill, the concatenated KV is [sliding kv (seqlen tokens), compressed tail].
            # The compressed entries start at index `seqlen` in the cat'd sequence.
            # During decode, the persistent layout is [0:win] = sliding ring buffer,
            # [win:] = compressed entries, so offset = win.
            offset = seqlen if start_pos == 0 else win
            if self.indexer is not None:
                compress_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_idxs = _get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, device)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
        topk_idxs = topk_idxs.long()

        # Write KV cache + sparse attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                # Place the last `win` tokens into the ring buffer in correct order.
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = _sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = _sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

        # Inverse RoPE on output (cancels K's absolute position)
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # Grouped output projection: split heads into n_groups groups
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        # wo_a storage is native FP8; dequant on-the-fly and view into group-wise form.
        wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
        wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

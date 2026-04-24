"""DeepSeek-V4 hybrid attention (CSA + HCA + SWA-bypass).

Single-file port of `inference/model.py:Compressor`, `Indexer`, `Attention` from
the official DeepSeek-V4 release. The three layer types are dispatched by
`compress_ratio` (the per-layer entry of `compress_ratios`):

| compress_ratio | Layer kind | Has compressor? | Has indexer? |
|---------------:|-----------:|----------------:|-------------:|
|              0 | SWA-only   | no              | no           |
|              4 | CSA (m=4)  | yes (overlap)   | yes          |
|            128 | HCA (m=128)| yes (no overlap)| no           |

This file provides a *PyTorch reference* implementation that uses an inline
`register_buffer` KV cache per layer (matching the official reference exactly).
Production paged KV cache is the M4 task — see `DEEPSEEK_V4_DESIGN.md` §4.

The sparse_attn op is the V4 attention with attention sink + topk gather. The
PyTorch reference here does the full gather + softmax inline; M2.5 will swap
in a FlashMLA-V4 backend.
"""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn


# =============================================================================
# Helpers
# =============================================================================


def _yarn_corr_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    import math
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_corr_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int):
    import math
    low = math.floor(_yarn_corr_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_yarn_corr_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _yarn_ramp(min_v: int, max_v: int, dim: int) -> torch.Tensor:
    if min_v == max_v:
        max_v += 0.001
    f = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(f, 0, 1)


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Yarn-scaled complex RoPE frequencies. lru_cache keys on all args so each
    (dim, seqlen, base) combination is computed once.
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    if original_seq_len > 0:
        low, high = _yarn_corr_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - _yarn_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place partial RoPE on the last `freqs_cis.size(-1) * 2` dims of x.

    `inverse=True` applies the conjugate, which is V4's "inverse RoPE on output"
    trick to undo the position encoding carried by KV cache entries.
    """
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(out)
    return y


@lru_cache(maxsize=4)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """SWA bypass: per-(batch, query) the indices of up to window_size most-recent
    KV slots in the *cyclic* SWA region of the cache. -1 = no slot (mask out).
    """
    if start_pos >= window_size - 1:
        s = start_pos % window_size
        m = torch.cat([torch.arange(s + 1, window_size), torch.arange(0, s + 1)], dim=0)
    elif start_pos > 0:
        m = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        m = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        m = torch.where(m > base, -1, m)
    return m.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(maxsize=4)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int) -> torch.Tensor:
    """HCA-only (no indexer): dense top-k = "all compressed entries seen so far",
    offset by the SWA region size. CSA replaces this with the indexer top-k.
    """
    if start_pos > 0:
        m = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        m = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = m >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        m = torch.where(mask, -1, m + offset)
    return m.unsqueeze(0).expand(bsz, -1, -1)


def sparse_attn_pytorch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """PyTorch reference for the official `kernel.py:sparse_attn` op.

    q:         [b, s_q, n_heads, d]
    kv:        [b, s_kv_total, d]   (single MQA head)
    sink:      [n_heads]            per-head learnable sink logit
    topk_idxs: [b, s_q, k]          int gather indices into kv along s_kv axis;
                                    -1 = mask out
    Returns o: [b, s_q, n_heads, d]
    """
    b, s_q, n_heads, d = q.shape
    k = topk_idxs.shape[-1]
    # Build masked gather: replace -1 with 0, then mask the corresponding logits.
    mask_invalid = topk_idxs < 0  # [b, s_q, k]
    safe_idx = topk_idxs.clamp(min=0)
    # Gather: [b, s_q, k, d] from kv [b, s_kv, d]
    safe_idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, -1, d).long()
    kv_gathered = torch.gather(kv.unsqueeze(1).expand(-1, s_q, -1, -1), 2, safe_idx_exp)
    # Logits: [b, s_q, n_heads, k] = einsum bshd,bskd → bshk
    logits = torch.einsum("bshd,bskd->bshk", q.float(), kv_gathered.float()) * softmax_scale
    logits = logits.masked_fill(mask_invalid.unsqueeze(2), float("-inf"))
    # Sink trick: append per-head learnable sink to softmax denominator.
    # Equivalent to softmax([..., sink_h]) and dropping that sink slot in the weighted sum.
    # Expand sink to [b, s_q, n_heads, 1].
    sink_logits = sink.view(1, 1, n_heads, 1).expand(b, s_q, -1, 1).float()
    logits_with_sink = torch.cat([logits, sink_logits], dim=-1)  # [b, s_q, n_heads, k+1]
    weights = logits_with_sink.softmax(dim=-1)
    weights = weights[..., :-1]  # drop sink slot
    o = torch.einsum("bshk,bskd->bshd", weights, kv_gathered.float())
    return o.to(q.dtype)


# =============================================================================
# Compressor (CSA m=4 with overlap windows; HCA m=128 without overlap)
# =============================================================================


class V4Compressor(nn.Module):
    """Token-level KV compressor: maps `[bsz, seqlen, dim]` hidden states into
    one compressed entry per `compress_ratio` consecutive tokens.

    Two modes:
    - **m=4 (CSA), `overlap=True`**: each full window has size `2*ratio`, with
      half overlap between adjacent windows. Trailing tail held in `kv_state` /
      `score_state` buffers until next ratio of tokens arrives.
    - **m=128 (HCA), `overlap=False`**: disjoint windows of size `ratio`.

    The Indexer wraps another Compressor instance with `rotate=True` (Hadamard
    rotation + FP4 quant) for its own scoring KV stream.

    The `kv_cache` buffer is **not owned** by this module — it's set lazily by
    the parent Attention module to point into a slice of the layer's KV cache.

    NOTE: this is a faithful port of `inference/model.py:Compressor`. The
    `register_buffer` state buffers (kv_state / score_state) are decode-phase
    accumulators; for prefill, the operation is done in one shot over the
    full sequence.
    """

    def __init__(
        self,
        dim: int,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        norm_eps: float,
        max_batch_size: int,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + (1 if self.overlap else 0)

        # Absolute position embedding for the m positions inside one window.
        # When overlap=True, the second half of `coff*head_dim` is the "current"
        # window, the first half is the "overlap" window (one prior step).
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # wkv / wgate projections — fp32 in inference.
        self.wkv = nn.Linear(dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(
            dim, coff * self.head_dim, bias=False, dtype=torch.float32
        )
        # RMSNorm on the final compressed entry (head_dim wide).
        self.norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float32))
        self.norm_eps = norm_eps

        # Decode-phase accumulators: hold the trailing partial window until full.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (max_batch_size, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        # Filled by parent Attention.
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.norm_eps)
        return (self.norm_weight * xf).to(dtype)

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        """Reshape `[b, s, ratio, 2*d]` → `[b, s, 2*ratio, d]` interleaving
        prior and current overlap windows. Filled with `value` where invalid.
        """
        b, s, _, _ = tensor.shape
        ratio, d = self.compress_ratio, self.head_dim
        new = tensor.new_full((b, s, 2 * ratio, d), value)
        new[:, :, ratio:] = tensor[:, :, :, d:]
        new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        """Compress `x: [bsz, seqlen, dim]` into compressed KV entries written
        into `self.kv_cache` at positions `[start_pos // ratio, ...]`.

        Prefill (start_pos == 0): processes whole seqlen, fills `cutoff // ratio`
            entries; trailing `seqlen % ratio` tokens land in kv_state.
        Decode (start_pos > 0, seqlen == 1): accumulates one token; only when
            `(start_pos + 1) % ratio == 0` is a new compressed entry produced.

        Returns the new compressed entries (only for prefill); decode flushes
        directly into `self.kv_cache`.
        """
        assert self.kv_cache is not None and self.freqs_cis is not None, (
            "V4Compressor.kv_cache and freqs_cis must be set by parent Attention "
            "before forward()"
        )
        bsz, seqlen, _ = x.shape
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))  # [b, n_windows, ratio, ...]
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self._overlap_transform(kv, 0)
                score = self._overlap_transform(score, float("-inf"))
            # Weighted sum over the ratio-sized window with softmax-of-score weights.
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # Decode: single-token accumulation.
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz]
                        * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self._rmsnorm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[: cutoff : ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # NOTE: official also applies fp4_act_quant for indexer compressor with rotate,
        # and act_quant (FP8 simulate) on the non-RoPE dims for main compressor.
        # The PyTorch reference here skips the simulate-quant since we run BF16 / FP32.
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


# =============================================================================
# Indexer (CSA only) — lightning indexer with FP4 simulation
# =============================================================================


class V4Indexer(nn.Module):
    """Lightning indexer for CSA layers: builds its own compressed KV stream
    (via a `V4Compressor(rotate=True)`), scores against per-head queries, and
    returns the top-k=`index_topk` most relevant compressed positions.

    The score is `Σ_h w_h · ReLU(q_h · K_s)` per (batch, query, head). FP4 is
    applied on Q and K via `rotate_activation + fp4_act_quant` in the official
    reference; the PyTorch reference here keeps BF16 throughout.
    """

    def __init__(
        self,
        dim: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
        compress_ratio: int = 4,
        world_size: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = index_n_heads
        self.n_local_heads = index_n_heads // world_size
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = index_head_dim ** -0.5

        # Q low-rank projection (consumes the q_norm output from main attention).
        self.wq_b = nn.Linear(q_lora_rank, self.n_local_heads * index_head_dim, bias=False)
        # Per-head weights for combining ReLU'd dot products.
        self.weights_proj = nn.Linear(dim, self.n_local_heads, bias=False, dtype=torch.bfloat16)
        # Internal compressor with Hadamard rotation (FP4 simulation in official).
        self.compressor = V4Compressor(
            dim=dim,
            compress_ratio=compress_ratio,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
            max_batch_size=max_batch_size,
            rotate=True,
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                max_batch_size, max_seq_len // compress_ratio, index_head_dim
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        """x:        [b, s, dim]    hidden state for compressor + weights_proj
        qr:       [b, s, q_lora_rank]  output of attention's q_norm(wq_a(x))
        offset:   start position offset into the parent KV cache (window_size
                  for decode, current cache size for prefill)

        Returns topk_idxs: [b, s, index_topk] — gather indices into the parent's
        compressed-KV region, with -1 for invalid (masked) positions.
        """
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        # NOTE: official applies rotate_activation + fp4_act_quant here. Skipped
        # in PyTorch reference for clarity.

        # Build the indexer's own compressed KV via its compressor.
        self.compressor(x, start_pos)

        weights = self.weights_proj(x.to(self.weights_proj.weight.dtype)) * (
            self.softmax_scale * self.n_heads ** -0.5
        )
        # index_score: [b, s_q, n_heads, n_compressed_so_far]
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        # TP all-reduce skipped — caller/parallelism layer handles it.

        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= (
                torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# =============================================================================
# Main attention block (HCA + CSA + SWA-only dispatcher)
# =============================================================================


class V4Attention(nn.Module):
    """V4 attention with per-layer CSA / HCA / SWA-only dispatch.

    Q path:     wq_a → q_norm → wq_b → per-head Q-RMSNorm → partial RoPE
    KV path:    wkv → kv_norm → partial RoPE → SWA write
    Index sel:  window topk_idxs   (always)
              + CSA: indexer.forward(x, qr, ...)
              + HCA: get_compress_topk_idxs (dense)
              + SWA-only: nothing extra
    Compress:   compressor(x, start_pos) → entry into kv_cache[:, window:]
    Attention:  sparse_attn(q, kv_cache, attn_sink, topk_idxs, scale)
    Output:     inverse RoPE → grouped wo_a (n_groups groups) → wo_b → hidden

    KV cache layout (per layer, single MQA head, head_dim wide):
        [bsz, window_size + (max_seq_len // compress_ratio if compress_ratio else 0), head_dim]
        ╰────── SWA region ──────╯╰──────────── compressed entries ───────────╯
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
        world_size: int = 1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads // world_size
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.n_groups = n_groups
        self.n_local_groups = n_groups // world_size
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.eps = norm_eps
        self.softmax_scale = head_dim ** -0.5

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))

        # Q path
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(q_lora_rank, dtype=torch.float32))
        self.wq_b = nn.Linear(q_lora_rank, self.n_local_heads * head_dim, bias=False)

        # KV path (single MQA head)
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.kv_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=torch.float32))

        # Grouped output projection: heads → groups → o_lora_rank → dim
        # wo_a is column-parallel: shape [n_local_groups * o_lora_rank, n_heads * head_dim / n_groups]
        self.wo_a = nn.Linear(
            n_heads * head_dim // n_groups,
            self.n_local_groups * o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        # wo_b is row-parallel: shape [dim, n_groups * o_lora_rank]
        self.wo_b = nn.Linear(n_groups * o_lora_rank, dim, bias=False)

        # Compressor + Indexer (only for compress_ratio != 0)
        self.compressor = None
        self.indexer = None
        if compress_ratio:
            self.compressor = V4Compressor(
                dim, compress_ratio, head_dim, rope_head_dim, norm_eps, max_batch_size
            )
            if compress_ratio == 4:
                self.indexer = V4Indexer(
                    dim,
                    index_n_heads,
                    index_head_dim,
                    rope_head_dim,
                    index_topk,
                    q_lora_rank,
                    norm_eps,
                    max_batch_size,
                    max_seq_len,
                    compress_ratio=compress_ratio,
                    world_size=world_size,
                )

        # KV cache: SWA region + (optional) compressed entries.
        kv_cache_size = window_size + (
            max_seq_len // compress_ratio if compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, kv_cache_size, head_dim),
            persistent=False,
        )
        # Different RoPE base for compressed branch vs SWA-only.
        if compress_ratio:
            base, orig = compress_rope_theta, original_seq_len
        else:
            base, orig = rope_theta, 0
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, max_seq_len, orig, base, rope_factor, beta_fast, beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (weight * xf).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """x: [b, s, dim] (already past `attn_norm`)
        Returns: [b, s, dim]
        """
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Lazy-bind compressor / indexer to share kv_cache + freqs_cis with attn.
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q
        qr = self._rmsnorm(self.wq_a(x), self.q_norm_weight)
        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # Per-head Q-RMSNorm (no learnable scale — pure normalize).
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV (single head, head_dim=512)
        kv = self.wkv(x)
        kv = self._rmsnorm(kv, self.kv_norm_weight)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # SWA topk
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                csa_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                csa_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                ).to(x.device)
            topk_idxs = torch.cat([topk_idxs, csa_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Write KV into cache + sparse attention
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn_pytorch(
                q, kv, self.attn_sink, topk_idxs, self.softmax_scale
            )
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor(x, start_pos)
            o = sparse_attn_pytorch(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )

        # Inverse RoPE on output to remove the position carried in KV entries.
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # Grouped output projection
        # o: [b, s, n_local_heads, head_dim] → view as [b, s, n_local_groups, group_dim]
        o = o.reshape(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # einsum: per-group small linear, then concat across groups.
        o = torch.einsum("bsgd,grd->bsgr", o.to(wo_a.dtype), wo_a)
        return self.wo_b(o.flatten(2).to(self.wo_b.weight.dtype))

"""DeepSeek-V4 Compressor — token-level KV pooling.

Faithful BF16 port of `inference/model.py:Compressor`. Skips FP4/FP8
quantization paths (will re-enable in M6 perf pass). Supports both
overlap=False (HCA, ratio=128) and overlap=True (CSA, ratio=4).

The compressor pools `compress_ratio` consecutive tokens via learned
softmax-gated weighting, applies RMSNorm + RoPE on the compressed
result, and writes into a target `kv_cache` buffer.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext, cp_all_gather_full, cp_should_gather,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

# P2 (prefill_opt/final_plan.md): fused softmax+weighted-sum Triton kernel.
# Replaces the prefill `(kv * score.softmax(dim=2)).sum(dim=2)` chain
# (~10 launches) with one kernel.  Set DSV4_COMPRESSOR_FAST=0 to force
# the REF path (debug only).
try:
    from rtp_llm.models_py.modules.dsv4._compressor_triton import v4_compressor_pool
    _COMPRESSOR_FAST_OK = True
except Exception:  # pragma: no cover — keep V4 importable without Triton
    v4_compressor_pool = None
    _COMPRESSOR_FAST_OK = False


def _use_compressor_fast() -> bool:
    if not _COMPRESSOR_FAST_OK:
        return False
    return os.environ.get("DSV4_COMPRESSOR_FAST", "1") != "0"


class _CompressorNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))


class Compressor(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self._factory_mode = weights is not None

        if self._factory_mode:
            self.ape = nn.Parameter(
                weights[f"{prefix}.ape"].float(), requires_grad=False,
            )
            self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
            self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
            with torch.no_grad():
                self.wkv.weight = nn.Parameter(
                    weights[f"{prefix}.wkv.weight"].float(), requires_grad=False,
                )
                self.wgate.weight = nn.Parameter(
                    weights[f"{prefix}.wgate.weight"].float(), requires_grad=False,
                )
            self.norm = _CompressorNorm(head_dim)
            self.norm.weight = nn.Parameter(
                weights[f"{prefix}.norm.weight"].float(), requires_grad=False,
            )
        else:
            self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
            self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
            self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
            with torch.no_grad():
                self.wkv.weight = nn.Parameter(self.wkv.weight.float())
                self.wgate.weight = nn.Parameter(self.wgate.weight.float())
            self.norm = _CompressorNorm(head_dim)
        self.norm_eps = norm_eps

        # State buffers for incremental decode-phase compression.
        self.register_buffer(
            "kv_state",
            torch.zeros(max_batch_size, coff * compress_ratio, coff * head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (max_batch_size, coff * compress_ratio, coff * head_dim), float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.kv_cache: Optional[torch.Tensor] = None  # bind from caller (Attention)
        self.freqs_cis: Optional[torch.Tensor] = None
        # CP context bound per-forward by V4Transformer.  None = no CP,
        # falls through to single-rank path unchanged.
        self._cp_ctx: Optional[CPContext] = None

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.norm_eps)
        return (self.norm.weight * x32).to(dtype)

    def _overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d] -> [b,s,2r,d]; first ratio rows pull from previous window's tail
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind CP context for the next prefill forward.  When set and
        ``cp_ctx.cp_size > 1`` and ``start_pos == 0``, the rank-local
        wkv / wgate projections are all-gathered to the full sequence
        before the S-dim pool step — so every rank's local ``kv_cache``
        ends up holding the SAME full compressed KV, which lets
        attention run with rank-local Q × full-KV downstream."""
        self._cp_ctx = cp_ctx

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        assert self.kv_cache is not None, "Compressor.kv_cache must be bound by caller"
        assert self.freqs_cis is not None, "Compressor.freqs_cis must be bound by caller"
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x32 = x.float()
        kv = torch.nn.functional.linear(x32, self.wkv.weight)
        score = torch.nn.functional.linear(x32, self.wgate.weight)

        # CP prefill: all-gather rank-local kv / score to full sequence
        # before the S-pool step so the pool sees all tokens in logical
        # order.  Decode (start_pos > 0) runs rank-local as usual — the
        # kv_cache was already populated with the full KV during prefill.
        cp_ctx = self._cp_ctx
        if cp_should_gather(cp_ctx, start_pos):
            kv = cp_all_gather_full(kv, cp_ctx)
            score = cp_all_gather_full(score, cp_ctx)
            # After gather the effective seqlen is the FULL un-padded
            # prefill len; the caller's ``seqlen`` / ``bsz`` above
            # reflected the rank-local padded slice.  Rebind so
            # downstream pool/pooling-mask/RoPE math operates on full.
            bsz, seqlen = kv.size(0), kv.size(1)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape

            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]

            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape

            if overlap:
                kv = self._overlap_transform(kv, 0)
                score = self._overlap_transform(score, float("-inf"))

            # Fast path: single Triton kernel fuses per-D softmax over G
            # + weighted sum.  REF chain materializes 3 fp32 intermediates
            # of size [B, NB, G, D] each (softmax exp/sum, masked mul,
            # then sum).  See _compressor_triton.py.
            if _use_compressor_fast() and kv.is_cuda and kv.numel() > 0:
                kv_c = kv if kv.is_contiguous() else kv.contiguous()
                sc_c = score if score.is_contiguous() else score.contiguous()
                kv = v4_compressor_pool(kv_c, sc_c)
            else:
                kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]

            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1
                    )
                    score_state = torch.cat(
                        [self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )

        if not should_compress:
            return None

        kv = self._rmsnorm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # NOTE: skip rotate_activation/fp4/fp8 quant for M2 (BF16-only path).

        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

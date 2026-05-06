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

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    freqs_cis_to_cos_sin,
    v4_compressor_fused,
)
from rtp_llm.models_py.modules.dsv4._compressor_kv_fused_triton import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
    v4_compressor_kv_fused,
)
from rtp_llm.models_py.modules.dsv4._compressor_triton import v4_compressor_pool
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full,
    cp_should_gather,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    PoolBackedModule,
    is_fp8_indexer_pool,
    is_fp8_swa_slot_pool,
)
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _fused_compressor_enabled() -> bool:
    """Fused compressor fast-path — primary implementation. The 4-launch
    pool/rmsnorm/rope/scatter chain remains as reference fallback; set
    ``DSV4_COMPRESSOR_FUSED=0`` to force it (e.g. for bisection)."""
    return os.environ.get("DSV4_COMPRESSOR_FUSED", "1") != "0"


class _CompressorNorm(nn.Module):
    """Weight holder for Compressor RMSNorm.  BF16 — ``rtp_llm_ops.rmsnorm``
    requires bf16 weight (silent NaN on fp32), matching vLLM default."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class Compressor(PoolBackedModule):
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
                weights[f"{prefix}.ape"].float(),
                requires_grad=False,
            )
            self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
            self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
            with torch.no_grad():
                self.wkv.weight = nn.Parameter(
                    weights[f"{prefix}.wkv.weight"].float(),
                    requires_grad=False,
                )
                self.wgate.weight = nn.Parameter(
                    weights[f"{prefix}.wgate.weight"].float(),
                    requires_grad=False,
                )
            self.norm = _CompressorNorm(head_dim)
            self.norm.weight = nn.Parameter(
                weights[f"{prefix}.norm.weight"].to(torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.ape = nn.Parameter(
                torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32)
            )
            self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
            self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
            with torch.no_grad():
                self.wkv.weight = nn.Parameter(self.wkv.weight.float())
                self.wgate.weight = nn.Parameter(self.wgate.weight.float())
            self.norm = _CompressorNorm(head_dim)
        self.norm_eps = norm_eps

        self._state_rows = coff * compress_ratio
        self._state_dim = coff * head_dim
        self._kv_cache_d = head_dim
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cp_ctx: Optional[CPContext] = None
        self._dbg_prefix: Optional[str] = None

        # ------------------------------------------------------------------

    # #50 — pool-only lifecycle: no persistent Python-owned buffers.
    # ------------------------------------------------------------------

    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        self._kv_cache_t = kv_cache_t

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        # Framework C++ ``rtp_llm_ops.rmsnorm`` (single launch, bf16 weight).
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out = torch.empty_like(x_2d)
        rtp_llm_ops.rmsnorm(
            out,
            x_2d,
            self.norm.weight,
            self.norm_eps,
            torch.cuda.current_stream().cuda_stream,
        )
        return out.view(orig_shape)

    def _fp8_pool_active(self) -> bool:
        """True iff bound KV pool is the indexer's FP8 packed cache (uint8/132B)."""
        return (
            is_fp8_indexer_pool(self._kv_pool_view)
            and self._kv_block_table is not None
            and self.head_dim == INDEXER_HEAD_DIM
        )

    def _fp8_kv_pool_active(self) -> bool:
        """True iff bound KV pool is the canonical 584B FP8 CSA/HCA cache.

        Triggers the fused {pool→RMSNorm→RoPE→FP8 quant→cache scatter}
        path that writes the fp8_model1_mla layout consumed by
        ``flash_mla_sparse_fwd``. See ``DSV4CacheConfig.h:78-91``."""
        return (
            is_fp8_swa_slot_pool(self._kv_pool_view)
            and self._kv_block_table is not None
            and self.head_dim == KV_HEAD_DIM
        )

    def _kv_cache_block_stride_bytes(self) -> int:
        """Per-block byte stride of the bound 584B FP8 KV pool.

        TMA padding (576B alignment, see ``DSV4PoolSpec::padded_block_size_bytes``)
        means this is **not** ``block_size * 584`` in general. Read it from
        the pool tensor's ``stride(0)`` after viewing as
        ``[num_blocks, block_size, 584]``.
        """
        eb = self._kv_eb
        view = self._kv_pool_view.view(-1, eb, KV_ENTRY_BYTES)
        # uint8 element stride == byte stride.
        return int(view.stride(0))

    def _logical_to_pool_slots(
        self,
        logical: torch.Tensor,  # [N] int64, logical compressed positions per row
        b_idx: torch.Tensor,  # [N] int64, batch index per row
        valid_in: torch.Tensor,  # [N] bool, caller-side validity (e.g. boundary mask)
    ) -> torch.Tensor:
        """Translate (b, logical) → flat pool slot via self._kv_block_table; -1 when invalid."""
        bt = self._kv_block_table.to(torch.long)
        eb = self._kv_eb
        max_blocks = bt.shape[1]
        pool_capacity = max_blocks * eb
        in_capacity = (logical >= 0) & (logical < pool_capacity)
        safe_logical = torch.where(in_capacity, logical, torch.zeros_like(logical))
        block_in_seq = safe_logical // eb
        in_block = safe_logical % eb
        block_id = bt[b_idx, block_in_seq]
        safe_slot = block_id * eb + in_block
        pool_rows = int(self._kv_pool_view.numel() // self._kv_pool_view.shape[-1])
        valid = valid_in & in_capacity & (block_id > 0) & (safe_slot < pool_rows)
        return torch.where(valid, safe_slot, torch.full_like(safe_slot, -1))

    def _run_fused_cache_write(
        self,
        kv_state_3d: torch.Tensor,  # [B', G, D_in] fp32 contiguous
        score_state_3d: torch.Tensor,  # [B', G, D_in] fp32 contiguous
        slots: torch.Tensor,  # [B'] int64, -1 for skip
        freq_idx: torch.Tensor,  # [B'] int64
        overlap_flag: Optional[bool] = None,
    ) -> None:
        """Run v4_compressor_fused for an already-staged [B', G, D_in] state."""
        freqs = self.freqs_cis[freq_idx]  # [B', rope_head_dim/2] complex
        cos, sin = freqs_cis_to_cos_sin(freqs)
        # Pool view is [num_total_slots, 132] uint8; fused kernel expects
        # [num_blocks, block_size, 132]. Use block_size=1 — slot indices are
        # already absolute, so block_idx == slot, block_off == 0.
        pool_blocks = self._kv_pool_view.view(-1, 1, INDEXER_ENTRY_BYTES)
        v4_compressor_fused(
            kv_state_3d,
            score_state_3d,
            slots,
            self.norm.weight,
            cos,
            sin,
            pool_blocks,
            overlap=self.overlap if overlap_flag is None else overlap_flag,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            norm_eps=self.norm_eps,
        )

    def _run_fused_kv_cache_write(
        self,
        kv_state_3d: torch.Tensor,  # [B', G, D_in] fp32 contiguous
        score_state_3d: torch.Tensor,  # [B', G, D_in] fp32 contiguous
        slots: torch.Tensor,  # [B'] int64, -1 for skip
        freq_idx: torch.Tensor,  # [B'] int64
        overlap_flag: Optional[bool] = None,
    ) -> None:
        """Run v4_compressor_kv_fused for an already-staged [B', G, D_in] state.

        Writes into the canonical 584B CSA/HCA pool. Block stride is read
        from the pool view (TMA-padded) — must not be derived from
        ``block_size * 584``."""
        freqs = self.freqs_cis[freq_idx]  # [B', rope_head_dim/2] complex
        cos, sin = freqs_cis_to_cos_sin(freqs)
        eb = self._kv_eb
        pool_blocks = self._kv_pool_view.view(-1, eb, KV_ENTRY_BYTES)
        v4_compressor_kv_fused(
            kv_state_3d,
            score_state_3d,
            slots,
            self.norm.weight,
            cos,
            sin,
            pool_blocks,
            cache_block_stride_bytes=int(pool_blocks.stride(0)),
            overlap=self.overlap if overlap_flag is None else overlap_flag,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            norm_eps=self.norm_eps,
        )

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

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,  # [B, 1, dim]
        start_pos: torch.Tensor,  # [B] int32
    ) -> "Optional[torch.Tensor]":
        """Batched decode: all B requests processed in one pass, no
        per-request Python loop.  Graph-capturable.

        Non-boundary requests perform the same compute but side effects
        are masked via ``torch.where`` (boundary writes KV cache,
        non-boundary is a no-op overwrite of the existing slot).
        """
        assert (
            self.freqs_cis is not None
        ), "Compressor.freqs_cis must be bound by caller"
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        device = x.device
        # PP4: when bound pool is an FP8 packed cache (indexer 132B/slot or
        # CSA/HCA 584B/slot uint8) AND the env gate is on, take the fused
        # {pool→RMSNorm→RoPE→FP8 quant→scatter} fast path which writes the
        # cache directly. State buffers still need their normal pool bind;
        # the kv_cache materialization is skipped (cache=None makes the
        # scatter step a no-op).
        use_fused_indexer = self._fp8_pool_active() and _fused_compressor_enabled()
        use_fused_kv = self._fp8_kv_pool_active() and _fused_compressor_enabled()
        use_fused = use_fused_indexer or use_fused_kv
        # #50: pool-only lifecycle.
        self._bind_state_from_pool(bsz, is_fresh_prefill=False, device=device)
        if use_fused:
            self.kv_cache = None
        else:
            self._bind_kv_cache_from_pool(
                bsz, is_fresh_prefill=False, device=device, dtype=dtype
            )
        try:
            x32 = x.float()
            kv_all = torch.nn.functional.linear(x32, self.wkv.weight)
            score_all = torch.nn.functional.linear(x32, self.wgate.weight)

            sp = start_pos.to(torch.long)
            sp_mod = sp % ratio  # [B]
            boundary = ((sp + 1) % ratio) == 0  # [B] bool
            b_idx = torch.arange(bsz, device=device, dtype=torch.long)

            # Per-request ape gather: ape is [ratio, coff*d].
            ape_rows = self.ape[sp_mod].to(score_all.dtype)  # [B, coff*d]
            score_all = score_all + ape_rows.unsqueeze(1)  # [B, 1, coff*d]

            # Slot in kv_state / score_state.
            slot = (sp_mod + ratio) if overlap else sp_mod  # [B]
            # Vectorized per-request write.
            self.kv_state[b_idx, slot] = kv_all[:, 0]
            self.score_state[b_idx, slot] = score_all[:, 0]

            # P2: pool kernel reads the raw [B, 2r, 2d] CSA state directly
            # and synthesizes the post-cat view ``cat([:, :r, :d], [:, r:, d:])``
            # in its load step — saves the two torch.cat allocs per layer.
            # HCA (overlap=False) still feeds the kernel its [B, ratio, d]
            # state as-is.
            kv_4d = self.kv_state[:bsz].unsqueeze(1).contiguous()
            sc_4d = self.score_state[:bsz].unsqueeze(1).contiguous()
            if overlap:
                kv_compressed = v4_compressor_pool(
                    kv_4d, sc_4d, overlap=True, out_d=d
                )  # [B, 1, d]
            else:
                kv_compressed = v4_compressor_pool(kv_4d, sc_4d)  # [B, 1, D]

            if use_fused:
                # Fused fast path: feed the raw [B, G, D_in] state straight
                # into the kernel which does pool+RMSNorm+RoPE+FP8 quant+
                # scatter into the packed cache. Skips the bf16 kv_compressed
                # materialization & per-batch index_put scatter. ``boundary``
                # mask folded into slots so non-boundary requests no-op
                # (kernel early-exits on slot < 0).
                cache_logical = torch.clamp(sp // ratio, min=0)
                slots = self._logical_to_pool_slots(cache_logical, b_idx, boundary)
                rope_idx = torch.clamp(sp + 1 - ratio, min=0)
                kv_state_3d = self.kv_state[:bsz].contiguous()
                score_state_3d = self.score_state[:bsz].contiguous()
                writer = (
                    self._run_fused_kv_cache_write
                    if use_fused_kv
                    else self._run_fused_cache_write
                )
                writer(kv_state_3d, score_state_3d, slots, rope_idx)
                # Replicate the bf16 path's post-pool/rmsnorm/rope value for
                # callers that may consume the return; keep it cheap by
                # running just the bf16 chain (no cache write).
                kv_compressed = self._rmsnorm(kv_compressed.to(dtype))
                freqs_per_b = self.freqs_cis[rope_idx]
                apply_rotary_emb_batched(kv_compressed[..., -rd:], freqs_per_b)
            else:
                kv_compressed = self._rmsnorm(kv_compressed.to(dtype))  # [B, 1, d]

                # Per-request RoPE — index into freqs_cis at sp + 1 - ratio. For
                # non-boundary requests the index might be negative; clamp.
                rope_idx = torch.clamp(sp + 1 - ratio, min=0)  # [B]
                freqs_per_b = self.freqs_cis[rope_idx]  # [B, freqs_dim] complex
                # Apply on the trailing rope_head_dim slice in place.
                apply_rotary_emb_batched(kv_compressed[..., -rd:], freqs_per_b)

                # Cache write: only boundary requests may overwrite their slot.
                # Use a "safe slot" (clamp to >=0) for the addressing, then mask
                # the value via torch.where so non-boundary requests no-op
                # (overwrite the existing slot with itself).
                cache_slot = torch.clamp(sp // ratio, min=0)  # [B]
                cache_slot = torch.where(
                    cache_slot < self.kv_cache.shape[1],
                    cache_slot,
                    torch.zeros_like(cache_slot),
                )
                existing = self.kv_cache[b_idx, cache_slot]  # [B, d]
                new_val = torch.where(
                    boundary.unsqueeze(-1),
                    kv_compressed.squeeze(1).to(self.kv_cache.dtype),
                    existing,
                )
                self.kv_cache[b_idx, cache_slot] = new_val

            # Roll kv_state/score_state for overlap=True (boundary requests only).
            if overlap:
                new_first_kv = torch.where(
                    boundary.view(bsz, 1, 1),
                    self.kv_state[:bsz, ratio:],  # rolled
                    self.kv_state[:bsz, :ratio],  # unchanged
                )
                self.kv_state[:bsz, :ratio] = new_first_kv
                new_first_score = torch.where(
                    boundary.view(bsz, 1, 1),
                    self.score_state[:bsz, ratio:],
                    self.score_state[:bsz, :ratio],
                )
                self.score_state[:bsz, :ratio] = new_first_score

            out = torch.where(
                boundary.view(bsz, 1, 1),
                kv_compressed,
                torch.zeros_like(kv_compressed),
            )
            return out
        finally:
            # #50: scatter state + kv_cache back to framework pools and clear.
            self._scatter_state_to_pool(bsz)
            self._scatter_kv_cache_to_pool(bsz)
            self.kv_state = None
            self.score_state = None
            self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos,
    ) -> Optional[torch.Tensor]:
        """Prefill-only entry point.  Decode goes through
        :meth:`forward_decode_vectorized` which handles batched decode
        natively.

        Attention enforces ``bsz==1`` for prefill (FIFO scheduler
        ``max_context_batch_size=1``), so no per-row batched dispatch."""
        bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        is_fresh_prefill = sp == 0
        use_fused = (
            self._fp8_pool_active() or self._fp8_kv_pool_active()
        ) and _fused_compressor_enabled()
        self._bind_state_from_pool(bsz, is_fresh_prefill, device=x.device)
        if use_fused:
            self.kv_cache = None
        else:
            self._bind_kv_cache_from_pool(
                bsz, is_fresh_prefill, device=x.device, dtype=x.dtype
            )
        try:
            return self._forward_prefill_body(x, sp)
        finally:
            self._scatter_state_to_pool(bsz)
            self._scatter_kv_cache_to_pool(bsz)
            self.kv_state = None
            self.score_state = None
            self.kv_cache = None

    def _forward_prefill_body(
        self,
        x: torch.Tensor,
        start_pos: int,
    ) -> Optional[torch.Tensor]:
        """Prefill body (single-row, seqlen > 1).  Called from :meth:`forward`
        (directly or via the batched-prefill per-row loop), which has already
        bound self.kv_state / self.score_state / self.kv_cache from the
        framework pool.  Decode goes through :meth:`forward_decode_vectorized`.
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        assert (
            self.freqs_cis is not None
        ), "Compressor.freqs_cis must be bound by caller"
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        _dbg = self._dbg_prefix if _rt.ENABLED else None
        sp_int = start_pos

        x32 = x.float()
        kv = torch.nn.functional.linear(x32, self.wkv.weight)
        score = torch.nn.functional.linear(x32, self.wgate.weight)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_x_in", x)
            _rt.record_if_level(2, f"{_dbg}_kv_local", kv)
            _rt.record_if_level(2, f"{_dbg}_score_local", score)

        cp_ctx = self._cp_ctx
        if cp_should_gather(cp_ctx, start_pos):
            kv = cp_all_gather_full(kv, cp_ctx)
            score = cp_all_gather_full(score, cp_ctx)
            bsz, seqlen = kv.size(0), kv.size(1)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_kv_full", kv)
            _rt.record_if_level(2, f"{_dbg}_score_full", score)

        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0

        # Continuation prefill needs the PRIOR call's "save for next call"
        # kv_state / score_state to fill window 0's overlap slots.
        # Snapshot BEFORE the save-for-next-call writes clobber them.
        prior_kv_state_ratio = (
            self.kv_state[:bsz, :ratio, :d].clone() if overlap and sp_int > 0 else None
        )
        prior_score_state_ratio = (
            self.score_state[:bsz, :ratio, :d].clone()
            if overlap and sp_int > 0
            else None
        )

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

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
            if sp_int > 0:
                kv[:bsz, 0, :ratio] = prior_kv_state_ratio
                score[:bsz, 0, :ratio] = prior_score_state_ratio

        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_pool_in_kv", kv)
            _rt.record_if_level(2, f"{_dbg}_pool_in_score", score)

        # PP4: FP8 fused fast path. Stage [B, NB, G, D_in] → [B*NB, G, D_in],
        # build slot_mapping/freq idx, fuse pool+RMSNorm+RoPE+FP8 quant+
        # cache scatter into a single Triton launch. State buffers (above)
        # still needed for the next call's overlap window, so this only
        # replaces the post-pool tail.
        #
        # Two pool flavors share the same staging:
        #   * indexer 132B (head_dim=128) → ``_run_fused_cache_write``
        #   * CSA/HCA  584B (head_dim=512) → ``_run_fused_kv_cache_write``
        #     (fp8_model1_mla layout for ``flash_mla_sparse_fwd``)
        use_fused_indexer = (
            self._fp8_pool_active() and _fused_compressor_enabled() and should_compress
        )
        use_fused_kv = (
            self._fp8_kv_pool_active()
            and _fused_compressor_enabled()
            and should_compress
        )
        if use_fused_indexer or use_fused_kv:
            # kv/score are [B, NB, G, D_in] — fold NB into batch.
            B = kv.size(0)
            NB = kv.size(1)
            G = kv.size(2)
            D_in = kv.size(3)
            kv_flat = kv.reshape(B * NB, G, D_in).contiguous()
            score_flat = score.reshape(B * NB, G, D_in).contiguous()
            write_start = sp_int // ratio
            # Per-position logical slot in the compressor's cache axis;
            # batch index repeated NB times per batch row.
            device = kv.device
            pos_local = torch.arange(NB, device=device, dtype=torch.long)
            cache_logical = write_start + pos_local  # [NB]
            cache_logical = cache_logical.unsqueeze(0).expand(B, NB).reshape(-1)
            b_idx = (
                torch.arange(B, device=device, dtype=torch.long)
                .unsqueeze(1)
                .expand(B, NB)
                .reshape(-1)
            )
            valid_in = torch.ones_like(cache_logical, dtype=torch.bool)
            slots = self._logical_to_pool_slots(cache_logical, b_idx, valid_in)
            # freq idx per compressed token: sp + p*ratio + (1 - ratio)
            freq_idx = sp_int + pos_local * ratio + (1 - ratio)
            freq_idx = torch.clamp(freq_idx, min=0)
            freq_idx = freq_idx.unsqueeze(0).expand(B, NB).reshape(-1)
            # Data is already in post-_overlap_transform layout [B', G, d]
            # (G = 2r for CSA, r for HCA), so call with overlap_flag=False
            # — the kernel just consumes G rows of d-elements each.
            writer = (
                self._run_fused_kv_cache_write
                if use_fused_kv
                else self._run_fused_cache_write
            )
            writer(kv_flat, score_flat, slots, freq_idx, overlap_flag=False)
            return None

        kv = v4_compressor_pool(kv.contiguous(), score.contiguous())
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_pool_out", kv)

        if not should_compress:
            return None

        kv = self._rmsnorm(kv.to(dtype))
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_norm_out", kv)
        freqs_cis = self.freqs_cis[sp_int : sp_int + cutoff : ratio]
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_freqs_cis", freqs_cis)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_rope_out", kv)

        write_start = sp_int // ratio
        self.kv_cache[:bsz, write_start : write_start + cutoff // ratio] = kv
        return kv

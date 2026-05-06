"""DeepSeek-V4 Compressor — FP8 KV pool path.

Companion to ``compressor.py`` (BF16 path). Single class for both pool
flavors; ``head_dim`` selects the writer kernel + layout:

  * ``head_dim == 512`` (CSA / HCA): writes 584B striped layout via
    ``v4_compressor_kv_fused``. Per-token entry = 448 fp8 NoPE +
    64 bf16 RoPE + 8 UE8M0 scales. Reader: ``flash_mla_sparse_fwd``
    after dequant via ``_kv_fp8_pool_io.dequantize_and_gather_k_cache``.

  * ``head_dim == 128`` (indexer compressor): writes 132B grouped
    layout via ``v4_compressor_fused``. Per-token entry = 128 fp8 K +
    4-byte fp32 UE8M0 scale. Reader: DeepGEMM
    ``fp8_paged_mqa_logits`` (consumes the pool directly).

What this class does NOT do — by design:

  * NEVER allocates ``self.kv_cache`` (no bf16 materialization). Reads
    of the FP8 pool happen at the attention / indexer layer via the
    dedicated readers. ``PoolBackedModule._bind/scatter_kv_cache_*``
    machinery is unused.
  * NO runtime fp8/bf16 dispatch. The class is FP8-only; pick the right
    class at attention construction time (see ``attention.py`` factory).
  * NO env gates / fallbacks. ``_compressor_triton.v4_compressor_pool``
    and the 4-launch reference path live in ``compressor.py`` (BF16).

State buffers (kv_state / score_state) for overlap continuation are
still bound from the framework's fp32 STATE pool — same pattern as the
BF16 class (PoolBackedModule's state half).
"""

from __future__ import annotations

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
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full,
    cp_should_gather,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import PoolBackedModule


class _CompressorNorm(nn.Module):
    """RMSNorm weight holder — bf16 (rtp_llm_ops.rmsnorm requires bf16
    weight; same convention as the BF16 class)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class CompressorFP8(PoolBackedModule):
    """FP8 KV cache compressor — single class for both pool flavors.

    head_dim selects the writer kernel + per-slot pool layout:
      * ``head_dim == 512`` → 584B striped (CSA / HCA), via
        ``v4_compressor_kv_fused``.
      * ``head_dim == 128`` → 132B grouped (indexer), via
        ``v4_compressor_fused``.

    Compress ratio: CSA uses ratio=4 (overlap=True), HCA uses ratio=128
    (overlap=False). The indexer compressor follows the host attention
    layer's ratio (typically 4, overlap=True).
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        compressor_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``compressor_weights`` is a 4-key dict ``{"ape", "wkv", "wgate",
        "norm"}`` extracted by the caller from ``layer_weights[W.v4_*compressor_*]``
        — same shape as the BF16 ``Compressor`` ctor."""
        super().__init__()
        assert head_dim in (KV_HEAD_DIM, INDEXER_HEAD_DIM), (
            f"CompressorFP8 supports head_dim in {{{KV_HEAD_DIM}, "
            f"{INDEXER_HEAD_DIM}}} (CSA/HCA 584B and indexer 132B); got {head_dim}"
        )
        assert compressor_weights is not None, (
            "CompressorFP8 requires compressor_weights — meta-tensor / "
            "stand-alone construction is not supported (use the BF16 path "
            "for that)."
        )
        self.dim = dim
        self.head_dim = head_dim
        # Per-slot pool entry size — selects layout & writer kernel.
        self._pool_entry_bytes = (
            KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
        )
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        # ape / wkv / wgate stay FP32 — CompressorFP8._forward_prefill_body
        # accumulates in FP32 for the (kv * softmax(score)).sum reduction
        # before the FP8 quant write. (BF16 ``Compressor`` casts wkv/wgate
        # to BF16 because its body keeps the partials in BF16; FP8 path
        # diverges here.)
        self.ape = compressor_weights["ape"].float()
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        with torch.no_grad():
            self.wkv.weight = nn.Parameter(
                compressor_weights["wkv"].float(), requires_grad=False
            )
            self.wgate.weight = nn.Parameter(
                compressor_weights["wgate"].float(), requires_grad=False
            )
        self.norm = _CompressorNorm(head_dim)
        self.norm.weight = nn.Parameter(
            compressor_weights["norm"].to(torch.bfloat16),
            requires_grad=False,
        )
        self.norm_eps = norm_eps

        # State buffer geometry — PoolBackedModule reads these in
        # ``_bind_state_from_pool``.
        self._state_rows = coff * compress_ratio
        self._state_dim = coff * head_dim
        # Unused (no kv_cache binding); kept zero so any accidental call
        # short-circuits to None.
        self._kv_cache_t = 0
        self._kv_cache_d = 0

        self.freqs_cis: Optional[torch.Tensor] = None
        self._cp_ctx: Optional[CPContext] = None

    # -- exposure for legacy callers reading ``_kv_cache_t`` (e.g.
    #    attention.py:1452); FP8 path stores no per-step bf16 cache,
    #    so just track the maximum compressed length the layer expects.
    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        # Stored only as informational metadata — NOT used for any
        # FP8-side allocation. Kept for compatibility with the BF16
        # class's API surface (some callers read ``_kv_cache_t``).
        self._kv_cache_t = kv_cache_t

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx

    # --------------------------------------------------------------
    # Helpers (private)
    # --------------------------------------------------------------
    def _kv_cache_block_stride_bytes(self) -> int:
        return int(self._pool_view_3d().stride(0))

    def _pool_view_3d(self) -> torch.Tensor:
        """Return ``_kv_pool_view`` as 3D ``[num_blocks, eb, entry_bytes]``
        uint8. Production path (``Attention._set_compressor_pool_context``)
        already passes 3D (padded-aware via ``as_strided`` for 584B
        CSA/HCA pools); standalone unit tests still pass flat 2D
        ``[num_blocks*eb, entry_bytes]`` and we reshape on the fly. dim-0
        stride in elements (uint8 → 1 byte/elem) == per-block byte stride.
        """
        v = self._kv_pool_view
        if v.dim() == 3:
            return v
        return v.view(-1, self._kv_eb, self._pool_entry_bytes)

    def _logical_to_pool_slots(
        self,
        logical: torch.Tensor,  # [N] int64 — logical compressed positions
        b_idx: torch.Tensor,  # [N] int64 — batch index per row
        valid_in: torch.Tensor,  # [N] bool — caller-side validity mask
    ) -> torch.Tensor:
        bt = self._kv_block_table.to(torch.long)
        eb = self._kv_eb
        pool_capacity = bt.shape[1] * eb
        in_capacity = (logical >= 0) & (logical < pool_capacity)
        safe_logical = torch.where(in_capacity, logical, torch.zeros_like(logical))
        block_in_seq = safe_logical // eb
        in_block = safe_logical % eb
        block_id = bt[b_idx, block_in_seq]
        valid = valid_in & in_capacity & (block_id > 0)
        safe_slot = block_id * eb + in_block
        return torch.where(valid, safe_slot, torch.full_like(safe_slot, -1))

    def _run_fused_pool_write(
        self,
        kv_state_3d: torch.Tensor,  # [B', G, D_in] fp32 contig
        score_state_3d: torch.Tensor,
        slots: torch.Tensor,  # [B'] int64; -1 = skip
        freq_idx: torch.Tensor,  # [B'] int64
        *,
        kernel_overlap: bool,
    ) -> None:
        """Dispatch to the right fused writer based on head_dim.

        ``kernel_overlap`` selects how the kernel reads the input:
          * ``True``  — raw CSA state ``[B, 2r, 2d]``: kernel synthesizes
            the post-cat view internally (used by the decode path).
          * ``False`` — already post-_overlap_transform ``[B, G, d]``:
            kernel reads G rows of d-elements directly (used by the
            prefill path).

        Both kernels (584B and 132B) accept the same shape and overlap
        convention; they differ only in the per-token byte layout and
        quant scheme.
        """
        freqs = self.freqs_cis[freq_idx]
        cos, sin = freqs_cis_to_cos_sin(freqs)
        pool_blocks = self._pool_view_3d()
        if self.head_dim == KV_HEAD_DIM:
            v4_compressor_kv_fused(
                kv_state_3d,
                score_state_3d,
                slots,
                self.norm.weight,
                cos,
                sin,
                pool_blocks,
                cache_block_stride_bytes=int(pool_blocks.stride(0)),
                overlap=kernel_overlap,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                norm_eps=self.norm_eps,
            )
        else:
            # head_dim == INDEXER_HEAD_DIM (128): 132B grouped layout.
            # Kernel internally hardcodes block stride = block_size *
            # ENTRY_BYTES (no TMA padding for 132B), so no stride arg.
            v4_compressor_fused(
                kv_state_3d,
                score_state_3d,
                slots,
                self.norm.weight,
                cos,
                sin,
                pool_blocks,
                overlap=kernel_overlap,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                norm_eps=self.norm_eps,
            )

    def _overlap_transform(self, tensor: torch.Tensor, value=0):
        """[b, s, r, 2d] → [b, s, 2r, d]; first ratio rows pull from
        previous window's tail (matches BF16 class)."""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    # --------------------------------------------------------------
    # Forward (prefill)
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, start_pos) -> Optional[torch.Tensor]:
        """Prefill entry. ``bsz==1`` (FIFO scheduler).

        Writes FP8 pool directly via ``v4_compressor_kv_fused``. State
        buffer (kv_state / score_state) is bound/scattered from the fp32
        STATE pool for overlap continuation across calls. Returns None
        — the FP8 path has no bf16 ``kv_compressed`` to emit; downstream
        readers gather from the pool.
        """
        bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        is_fresh_prefill = sp == 0
        # Warmup forward (no pool bound by framework): no-op. Same gate as
        # BF16 Compressor (`_kv_block_table is None` → skip write).
        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        self._bind_state_from_pool(bsz, is_fresh_prefill, device=x.device)
        # FP8 path never allocates self.kv_cache — leave None to make any
        # accidental read explode immediately.
        self.kv_cache = None
        try:
            return self._forward_prefill_body(x, sp)
        finally:
            self._scatter_state_to_pool(bsz)
            self.kv_state = None
            self.score_state = None

    def _forward_prefill_body(
        self, x: torch.Tensor, start_pos: int
    ) -> Optional[torch.Tensor]:
        assert (
            self.freqs_cis is not None
        ), "CompressorFP8.freqs_cis must be bound by caller"
        bsz, seqlen, _ = x.size()
        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        sp_int = start_pos

        x32 = x.float()
        kv = torch.nn.functional.linear(x32, self.wkv.weight)
        score = torch.nn.functional.linear(x32, self.wgate.weight)

        cp_ctx = self._cp_ctx
        if cp_should_gather(cp_ctx, start_pos):
            kv = cp_all_gather_full(kv, cp_ctx)
            score = cp_all_gather_full(score, cp_ctx)
            bsz, seqlen = kv.size(0), kv.size(1)

        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0

        # Continuation prefill needs the PRIOR call's stash to fill the
        # overlap slots of window 0. Snapshot BEFORE the save-for-next-call
        # writes clobber them.
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

        if not should_compress:
            return None

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
            if sp_int > 0:
                kv[:bsz, 0, :ratio] = prior_kv_state_ratio
                score[:bsz, 0, :ratio] = prior_score_state_ratio

        # FP8 fused write. Stage [B, NB, G, D_in] → [B*NB, G, D_in], build
        # slot_mapping / freq idx via the fused prelude kernel (one launch
        # vs ~10 small ops in the original pure-torch prelude).
        from rtp_llm.models_py.modules.dsv4._compressor_prelude_triton import (
            compressor_prelude_fused,
        )

        B, NB, G, D_in = kv.shape
        kv_flat = kv.reshape(B * NB, G, D_in).contiguous()
        score_flat = score.reshape(B * NB, G, D_in).contiguous()
        write_start = sp_int // ratio
        slots, freq_idx = compressor_prelude_fused(
            B=B,
            NB=NB,
            write_start=write_start,
            sp=sp_int,
            ratio=ratio,
            eb=self._kv_eb,
            block_table=self._kv_block_table,
        )
        # Prefill: data is already post-_overlap_transform layout
        # ([B', G, d] for both CSA G=2r and HCA G=r); kernel reads
        # G rows of d-elements directly.
        self._run_fused_pool_write(
            kv_flat, score_flat, slots, freq_idx, kernel_overlap=False
        )
        return None

    # --------------------------------------------------------------
    # Forward (decode, vectorized over B)
    # --------------------------------------------------------------
    def forward_decode_vectorized(
        self, x: torch.Tensor, start_pos: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Batched decode entry. Boundary requests trigger an FP8 write
        for their compressed token; non-boundary requests are masked out
        of the slot mapping (kernel early-exits on slot < 0)."""
        assert (
            self.freqs_cis is not None
        ), "CompressorFP8.freqs_cis must be bound by caller"
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        device = x.device
        # Warmup forward (no pool bound by framework): no-op.
        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        self._bind_state_from_pool(bsz, is_fresh_prefill=False, device=device)
        self.kv_cache = None
        try:
            x32 = x.float()
            kv_all = torch.nn.functional.linear(x32, self.wkv.weight)
            score_all = torch.nn.functional.linear(x32, self.wgate.weight)

            sp = start_pos.to(torch.long)
            sp_mod = sp % ratio
            boundary = ((sp + 1) % ratio) == 0
            b_idx = torch.arange(bsz, device=device, dtype=torch.long)

            ape_rows = self.ape[sp_mod].to(score_all.dtype)
            score_all = score_all + ape_rows.unsqueeze(1)

            slot = (sp_mod + ratio) if overlap else sp_mod
            self.kv_state[b_idx, slot] = kv_all[:, 0]
            self.score_state[b_idx, slot] = score_all[:, 0]

            cache_logical = torch.clamp(sp // ratio, min=0)
            slots = self._logical_to_pool_slots(cache_logical, b_idx, boundary)
            rope_idx = torch.clamp(sp + 1 - ratio, min=0)
            kv_state_3d = self.kv_state[:bsz].contiguous()
            score_state_3d = self.score_state[:bsz].contiguous()
            # Decode: feeds the RAW state buffer ([B, 2r, 2d] for CSA,
            # [B, r, d] for HCA) — kernel synthesizes the post-cat view
            # internally. ``kernel_overlap`` mirrors the layer's overlap.
            self._run_fused_pool_write(
                kv_state_3d,
                score_state_3d,
                slots,
                rope_idx,
                kernel_overlap=overlap,
            )

            # Roll kv_state/score_state for overlap=True (boundary only).
            if overlap:
                new_first_kv = torch.where(
                    boundary.view(bsz, 1, 1),
                    self.kv_state[:bsz, ratio:],
                    self.kv_state[:bsz, :ratio],
                )
                self.kv_state[:bsz, :ratio] = new_first_kv
                new_first_score = torch.where(
                    boundary.view(bsz, 1, 1),
                    self.score_state[:bsz, ratio:],
                    self.score_state[:bsz, :ratio],
                )
                self.score_state[:bsz, :ratio] = new_first_score
            return None
        finally:
            self._scatter_state_to_pool(bsz)
            self.kv_state = None
            self.score_state = None

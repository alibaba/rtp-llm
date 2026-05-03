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
    CPContext,
    cp_all_gather_full,
    cp_should_gather,
)
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

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
    """Weight holder for Compressor RMSNorm.  BF16 — ``rtp_llm_ops.rmsnorm``
    requires bf16 weight (silent NaN on fp32), matching vLLM default."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


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

        # Pool-only (#50): kv_state / score_state / kv_cache are NEVER
        # persistent Python-owned tensors.  Each ``forward`` / ``forward_decode``
        # call:
        #   (a) gathers the state / kv_cache rows from the framework pools
        #       (CSA_STATE / HCA_STATE / INDEXER_STATE for state;
        #        CSA_KV / HCA_KV / INDEXER_KV for kv_cache) at entry,
        #       binding them onto these ephemeral attrs so the body math
        #       (which reads/writes self.kv_state / self.score_state /
        #       self.kv_cache) stays unchanged
        #   (b) scatters back to the pools at exit via write_kv_to_pool
        #   (c) clears these to None in try/finally so no stale tensors
        #       survive across forwards.
        #
        # Layout (unchanged from the retired persistent buffers):
        #   kv_state     : [B, coff * compress_ratio, coff * head_dim] fp32
        #   score_state  : [B, coff * compress_ratio, coff * head_dim] fp32
        #                  (fresh-prefill fill value -inf so softmax masks
        #                  unfilled slots)
        #   kv_cache     : [B, max_seq_len // compress_ratio, head_dim] bf16
        #                  (fresh-prefill fill value 0)
        self._state_rows = coff * compress_ratio
        self._state_dim = coff * head_dim
        self._kv_cache_d = head_dim
        self.kv_state: Optional[torch.Tensor] = None
        self.score_state: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None
        # CP context bound per-forward by V4Transformer.  None = no CP,
        # falls through to single-rank path unchanged.
        self._cp_ctx: Optional[CPContext] = None
        # Optional per-forward debug prefix (e.g. "L01_attn_cmp"); when set,
        # internal forward checkpoints are recorded via _record_tensor.
        self._dbg_prefix: Optional[str] = None

        # Pool context — set by Attention before each ``forward`` /
        # ``forward_decode*`` call via ``set_pool_context`` and cleared
        # by ``clear_pool_context``.  When unbound (e.g. standalone unit
        # tests), the forward helpers fall back to ephemeral zero/-inf
        # tensors (no scatter, one-call lifecycle).
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0
        self._state_pool_view: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0
        # Caller-provided capacity hint (``max_seq_len // compress_ratio``)
        # for the ephemeral kv_cache tensor's T dim when pool context is
        # not bound (standalone tests).  Pool-bound path derives T from
        # the pool view directly.
        self._kv_cache_t: int = 0

    # ------------------------------------------------------------------
    # #50 — pool-only lifecycle: no persistent Python-owned buffers.
    # ------------------------------------------------------------------

    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        """Caller-provided ``max_seq_len // compress_ratio`` — used only
        for the fallback ephemeral kv_cache tensor shape when no pool
        context is bound (standalone unit tests).  Pool-bound path reads
        T from the pool view directly."""
        self._kv_cache_t = kv_cache_t

    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
    ) -> None:
        """Bind framework-pool views + block tables for this forward call.
        Attention sets this before calling ``forward`` / ``forward_decode*``;
        ``clear_pool_context`` clears after."""
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_view = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_view = None
        self._state_block_table = None
        self._state_eb = 0

    def _compute_pool_slots(
        self,
        bsz: int,
        T: int,
        block_table: torch.Tensor,
        eb: int,
        device: torch.device,
    ) -> tuple:
        """Common slot math for gather/scatter: returns ``(valid, safe_slot)``
        both ``[B, T]`` long.  ``valid`` masks positions past pool capacity
        and sentinel (block_id <= 0) slots; ``safe_slot`` is addressed with
        zeros for invalid positions so gather/scatter are in-bounds."""
        max_blocks = block_table.shape[1]
        pool_capacity = max_blocks * eb
        pos = torch.arange(T, device=device, dtype=torch.long)
        in_capacity_row = pos < pool_capacity
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb
        in_block = safe_pos % eb
        bt_long = block_table.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)
        valid = (block_id > 0) & in_capacity
        safe_slot = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
        )
        return valid, safe_slot

    def _bind_state_from_pool(
        self, bsz: int, is_fresh_prefill: bool, device: torch.device
    ) -> None:
        """Populate ``self.kv_state`` / ``self.score_state`` at the start of
        a forward call.  Fresh prefill → zero / -inf.  Continuation /
        decode → gather from STATE pool (zero for sentinel slots).  When
        no pool context is bound (standalone tests) → zero / -inf."""
        T = self._state_rows
        half_dim = self._state_dim
        # Fresh prefill always starts zeros/-inf (matches retired
        # reset_state_for_new_prefill).  Decode / continuation need the
        # pool-gathered prefix.
        if (
            is_fresh_prefill
            or self._state_pool_view is None
            or self._state_block_table is None
            or self._state_eb <= 0
        ):
            self.kv_state = torch.zeros(
                bsz, T, half_dim, dtype=torch.float32, device=device
            )
            self.score_state = torch.full(
                (bsz, T, half_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
            return
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._state_block_table, self._state_eb, device
        )
        gathered = self._state_pool_view.index_select(
            0, safe_slot.reshape(-1)
        )  # [B*T, 2*half]
        valid_bcast = valid.reshape(-1).unsqueeze(-1)
        zero_row_kv = torch.zeros((), dtype=torch.float32, device=device)
        # Use -inf for unfilled score slots so softmax excludes them.
        neg_inf_row = torch.full((), float("-inf"), dtype=torch.float32, device=device)
        kv_rows = torch.where(valid_bcast, gathered[:, :half_dim], zero_row_kv)
        sc_rows = torch.where(valid_bcast, gathered[:, half_dim:], neg_inf_row)
        self.kv_state = kv_rows.view(bsz, T, half_dim).contiguous()
        self.score_state = sc_rows.view(bsz, T, half_dim).contiguous()

    def _scatter_state_to_pool(self, bsz: int) -> None:
        """Write ``self.kv_state`` / ``self.score_state`` to the STATE pool
        (CSA_STATE / HCA_STATE / INDEXER_STATE).  Pool slot layout per entry
        ``[2 * half_dim]`` fp32 = ``kv_row || score_row``.  No-op when pool
        context unbound (standalone tests)."""
        if (
            self._state_pool_view is None
            or self._state_block_table is None
            or self._state_eb <= 0
            or self.kv_state is None
            or self.score_state is None
        ):
            return
        device = self.kv_state.device
        T = self._state_rows
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._state_block_table, self._state_eb, device
        )
        merged = torch.cat(
            [self.kv_state[:bsz], self.score_state[:bsz]], dim=-1
        )  # [B, T, 2*half]
        merged_flat = merged.reshape(bsz * T, -1)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        write_kv_to_pool(
            merged_flat, slot_mapping, self._state_pool_view, mask_negative=True
        )

    def _bind_kv_cache_from_pool(
        self,
        bsz: int,
        is_fresh_prefill: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Populate ``self.kv_cache`` at the start of a forward call.  Fresh
        prefill → zeros; decode / continuation → gather from KV pool (zero
        for sentinel slots)."""
        if self._kv_pool_view is not None and self._kv_eb > 0:
            # Derive T from pool capacity (num_blocks of this request's
            # block_table × entries_per_block); equivalently use configure
            # hint when set.
            T = (
                self._kv_cache_t
                if self._kv_cache_t > 0
                else (
                    self._kv_block_table.shape[1] * self._kv_eb
                    if self._kv_block_table is not None
                    else 0
                )
            )
        else:
            T = self._kv_cache_t
        if T <= 0:
            # No capacity configured and no pool context — callers that
            # never compress (pure SWA forward path) will never touch
            # self.kv_cache, so leave it None.
            self.kv_cache = None
            return
        D = self._kv_cache_d
        if (
            is_fresh_prefill
            or self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
        ):
            self.kv_cache = torch.zeros(bsz, T, D, dtype=dtype, device=device)
            return
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._kv_block_table, self._kv_eb, device
        )
        gathered = self._kv_pool_view.index_select(0, safe_slot.reshape(-1))  # [B*T, D]
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out_flat = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        self.kv_cache = out_flat.view(bsz, T, D).contiguous()

    def _scatter_kv_cache_to_pool(
        self, bsz: int, block_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Write ``self.kv_cache`` to the KV pool.  ``block_mask`` (optional
        ``[B, T]`` bool) marks which blocks were written this forward; other
        positions are skipped.  When unset, all T blocks of every row are
        written (correct but bandwidth-wasteful)."""
        if (
            self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
            or self.kv_cache is None
        ):
            return
        device = self.kv_cache.device
        T = int(self.kv_cache.shape[1])
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._kv_block_table, self._kv_eb, device
        )
        if block_mask is not None:
            valid = valid & block_mask[:bsz].to(device)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        D = int(self.kv_cache.shape[2])
        flat = self.kv_cache[:bsz].reshape(bsz * T, D)
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        write_kv_to_pool(flat, slot_mapping, self._kv_pool_view, mask_negative=True)

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

    def forward_decode(
        self,
        x: torch.Tensor,  # [B, 1, dim]  — single decode token per request (q_len=1)
        start_pos: torch.Tensor,  # [B] int32 — absolute pos of this token per request
    ) -> "Optional[torch.Tensor]":
        """Batched decode-time per-token compression.

        Returns ``compressed_kv [B, 1, head_dim] bf16`` for **every** request,
        with -inf-padded entries for requests not on a compression boundary
        (caller masks via ``(start_pos + 1) % ratio == 0``). Updates
        ``self.kv_state`` / ``self.score_state`` / ``self.kv_cache`` per
        request from the new token. Decode-only — does NOT touch the
        prefill ``forward`` arm (which stays bit-identical for PD-disagg).

        For each request r:
          1. score[r] += ape[start_pos[r] % ratio]
          2. write kv_state[r, slot] = kv[r] (overlap-aware slot)
          3. write score_state[r, slot] = score[r]
          4. if (start_pos[r] + 1) % ratio == 0:
               compressed_kv[r] = sum(kv_state[r] * softmax(score_state[r]), dim=0)
               apply_rotary_emb on compressed
               write kv_cache[r, start_pos[r] // ratio] = compressed_kv[r]
               (overlap variant also rolls kv_state/score_state)

        Implementation: Python loop over B for correctness on the first
        cut. B is small (≤~16 typical decode), layer count dominates;
        vectorizing is a Phase 4+ perf improvement.
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
        # #50: pool-only lifecycle.  Bind state + kv_cache from the
        # framework pools at entry; scatter back + clear at exit.
        self._bind_state_from_pool(bsz, is_fresh_prefill=False, device=x.device)
        self._bind_kv_cache_from_pool(
            bsz, is_fresh_prefill=False, device=x.device, dtype=dtype
        )
        try:
            x32 = x.float()  # [B, 1, dim]
            kv_all = torch.nn.functional.linear(
                x32, self.wkv.weight
            )  # [B, 1, coff*head_dim]
            score_all = torch.nn.functional.linear(
                x32, self.wgate.weight
            )  # [B, 1, coff*head_dim]

            # Output buffer — populated only for boundary requests; the rest
            # carry zeros (caller skips them via the boundary mask).
            out_compressed = torch.zeros(bsz, 1, d, device=x.device, dtype=dtype)
            any_compressed = False

            for r in range(bsz):
                sp = int(start_pos[r].item())
                sp_mod = sp % ratio
                ape_row = self.ape[sp_mod].to(score_all.dtype)
                kv_r = kv_all[r : r + 1]  # [1, 1, coff*head_dim]
                score_r = score_all[r : r + 1] + ape_row  # [1, 1, coff*head_dim]
                should_compress = ((sp + 1) % ratio) == 0

                if overlap:
                    # CSA path — overlap=True, write at slot ratio + sp_mod
                    self.kv_state[r : r + 1, ratio + sp_mod] = kv_r.squeeze(1)
                    self.score_state[r : r + 1, ratio + sp_mod] = score_r.squeeze(1)
                    if should_compress:
                        kv_state = torch.cat(
                            [
                                self.kv_state[r : r + 1, :ratio, :d],
                                self.kv_state[r : r + 1, ratio:, d:],
                            ],
                            dim=1,
                        )
                        score_state = torch.cat(
                            [
                                self.score_state[r : r + 1, :ratio, :d],
                                self.score_state[r : r + 1, ratio:, d:],
                            ],
                            dim=1,
                        )
                        kv_compressed = (kv_state * score_state.softmax(dim=1)).sum(
                            dim=1, keepdim=True
                        )
                        self.kv_state[r : r + 1, :ratio] = self.kv_state[
                            r : r + 1, ratio:
                        ]
                        self.score_state[r : r + 1, :ratio] = self.score_state[
                            r : r + 1, ratio:
                        ]
                    else:
                        continue
                else:
                    # HCA path — overlap=False, single-pool
                    self.kv_state[r : r + 1, sp_mod] = kv_r.squeeze(1)
                    self.score_state[r : r + 1, sp_mod] = score_r.squeeze(1)
                    if should_compress:
                        kv_compressed = (
                            self.kv_state[r : r + 1]
                            * self.score_state[r : r + 1].softmax(dim=1)
                        ).sum(dim=1, keepdim=True)
                    else:
                        continue

                # RMSNorm + RoPE + cache-write for boundary requests
                kv_compressed = self._rmsnorm(
                    kv_compressed.to(dtype)
                )  # [1, 1, head_dim]
                freqs_cis = self.freqs_cis[sp + 1 - ratio].unsqueeze(0)
                apply_rotary_emb(kv_compressed[..., -rd:], freqs_cis)
                self.kv_cache[r : r + 1, sp // ratio] = kv_compressed.squeeze(1)
                out_compressed[r : r + 1] = kv_compressed
                any_compressed = True

            if not any_compressed:
                return None
            return out_compressed  # [B, 1, head_dim] bf16, zeros for non-boundary reqs
        finally:
            # #50: write state + kv_cache back to framework pools and clear
            # the ephemeral Python tensors.
            self._scatter_state_to_pool(bsz)
            self._scatter_kv_cache_to_pool(bsz)
            self.kv_state = None
            self.score_state = None
            self.kv_cache = None

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,  # [B, 1, dim]
        start_pos: torch.Tensor,  # [B] int32
    ) -> "Optional[torch.Tensor]":
        """Stage 3B vectorized variant of :meth:`forward_decode`.

        Removes the per-request Python loop + ``.item()`` boundary check
        so the entire body is graph-capturable. Math is equivalent to
        the loop variant for boundary requests; for non-boundary
        requests the compute is still performed but its side effects
        are masked away via ``torch.where``.

        Trade-off: O(B) extra work for non-boundary requests vs the
        loop variant. Acceptable because (a) decode B is small (≤16
        typical), (b) the kernel launch overhead saved by avoiding
        per-request scalar copies / Python branches dominates, and
        (c) only the CUDA-graph path uses this — Phase 2 eager keeps
        the loop variant for byte-equal regression safety.
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
        # #50: pool-only lifecycle.
        self._bind_state_from_pool(bsz, is_fresh_prefill=False, device=device)
        self._bind_kv_cache_from_pool(
            bsz, is_fresh_prefill=False, device=device, dtype=dtype
        )
        try:
            x32 = x.float()
            kv_all = torch.nn.functional.linear(x32, self.wkv.weight)  # [B, 1, coff*d]
            score_all = torch.nn.functional.linear(
                x32, self.wgate.weight
            )  # [B, 1, coff*d]

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

            # Build the compute view (post-write).
            if overlap:
                kv_state_view = torch.cat(
                    [self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]],
                    dim=1,
                )  # [B, 2*ratio, d]
                score_state_view = torch.cat(
                    [
                        self.score_state[:bsz, :ratio, :d],
                        self.score_state[:bsz, ratio:, d:],
                    ],
                    dim=1,
                )
            else:
                kv_state_view = self.kv_state[:bsz]  # [B, ratio, coff*d]
                score_state_view = self.score_state[:bsz]

            kv_compressed = (kv_state_view * score_state_view.softmax(dim=1)).sum(
                dim=1, keepdim=True
            )  # [B, 1, d]

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

            # Return the compressed-K with non-boundary rows zeroed (matches
            # the loop variant's contract).
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

    def _forward_batched_prefill(
        self,
        x: torch.Tensor,  # [B, max_S, dim]
        start_pos: torch.Tensor,  # [B] int
        sequence_lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Batched prefill per-row dispatch.

        Loops over ``b`` and reuses the scalar forward body via state view
        swapping (``self.kv_state = saved[b:]`` etc.), so each row's
        ``self.kv_state[:1]`` write in the scalar body hits its own row of
        the base storage.  Maintains bit-equal behavior with the scalar
        bsz==1 path because the same code runs per row.

        Returns ``[B, K_max, head_dim] bf16`` stacked compressed output
        (zero-padded for rows with fewer compressed blocks), or ``None``
        if no row produced any compressed output.
        """
        assert (
            self.freqs_cis is not None
        ), "Compressor.freqs_cis must be bound by caller"
        bsz = int(x.size(0))
        max_S = int(x.size(1))
        ratio = self.compress_ratio
        d = self.head_dim
        device = x.device
        dtype = x.dtype

        if sequence_lengths is None:
            seq_t = torch.full((bsz,), max_S, device=device, dtype=torch.long)
        else:
            seq_t = sequence_lengths.to(device=device, dtype=torch.long)

        sp_t = start_pos.to(device=device, dtype=torch.long)

        # Per-row compressed-block count.  For fresh prefill (sp==0),
        # the scalar path emits ``cutoff // ratio = (seq // ratio)`` blocks.
        # For continuation prefill (sp>0), same formula applies at forward
        # time (cutoff is the trimmed seqlen regardless of sp).
        cutoff_t = (seq_t // ratio) * ratio  # [B]
        n_blocks_per_row = cutoff_t // ratio  # [B]
        K_max = int(n_blocks_per_row.max().item()) if bsz > 0 else 0

        if K_max == 0:
            return None

        out = torch.zeros(bsz, K_max, d, device=device, dtype=dtype)
        any_compressed = False

        # #50: per-row pool-context narrowing.  Each row's scalar
        # ``self.forward(x_b, sp_b)`` call does its own bind/scatter
        # against the narrowed [1, max_blocks] block_tables; pool views
        # are kept global (addressed by absolute slot id).
        saved_kv_bt = self._kv_block_table
        saved_state_bt = self._state_block_table
        saved_dbg = self._dbg_prefix
        try:
            self._dbg_prefix = None
            for b in range(bsz):
                seq_b = int(seq_t[b].item())
                if seq_b == 0:
                    continue
                sp_b = int(sp_t[b].item())
                x_b = x[b : b + 1, :seq_b]
                self._kv_block_table = (
                    saved_kv_bt[b : b + 1] if saved_kv_bt is not None else None
                )
                self._state_block_table = (
                    saved_state_bt[b : b + 1] if saved_state_bt is not None else None
                )
                out_b = self.forward(x_b, sp_b)
                if out_b is not None:
                    t_b = min(int(out_b.size(1)), K_max)
                    if t_b > 0:
                        out[b : b + 1, :t_b] = out_b[:, :t_b]
                        any_compressed = True
        finally:
            self._kv_block_table = saved_kv_bt
            self._state_block_table = saved_state_bt
            self._dbg_prefix = saved_dbg

        if not any_compressed:
            return None
        return out

    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Pool-only wrapper (#50).  Dispatches batched prefill, otherwise
        binds state + kv_cache from framework pools, runs the scalar body
        via :meth:`_forward_scalar_impl`, scatters back on exit."""
        bsz, seqlen, _ = x.size()
        # Batched prefill dispatch (seqlen > 1 with [B] start_pos, B > 1).
        # The inner per-row forward call handles its own bind/scatter.
        if seqlen > 1 and isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1:
            return self._forward_batched_prefill(x, start_pos, sequence_lengths)
        # Determine fresh-prefill state so bind initializes zero / -inf
        # instead of reading pool slots that haven't been written for
        # this request yet.
        if isinstance(start_pos, torch.Tensor):
            sp_scalar = int(start_pos.item()) if start_pos.numel() == 1 else 0
        else:
            sp_scalar = int(start_pos)
        is_fresh_prefill = seqlen > 1 and sp_scalar == 0
        self._bind_state_from_pool(bsz, is_fresh_prefill, device=x.device)
        self._bind_kv_cache_from_pool(
            bsz, is_fresh_prefill, device=x.device, dtype=x.dtype
        )
        try:
            return self._forward_scalar_impl(x, start_pos, sequence_lengths)
        finally:
            self._scatter_state_to_pool(bsz)
            self._scatter_kv_cache_to_pool(bsz)
            self.kv_state = None
            self.score_state = None
            self.kv_cache = None

    def _forward_scalar_impl(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Scalar-prefill / scalar-decode / batched-decode body.  Called
        only from :meth:`forward`, which has already bound self.kv_state /
        self.score_state / self.kv_cache from the framework pool.
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
        # Master switch: gated by both MOEDBG (process-wide) and per-instance
        # _dbg_prefix (set by parent indexer only when its own _dbg is on).
        _dbg = self._dbg_prefix if _rt.ENABLED else None

        x32 = x.float()
        kv = torch.nn.functional.linear(x32, self.wkv.weight)
        score = torch.nn.functional.linear(x32, self.wgate.weight)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_x_in", x)
            _rt.record_if_level(2, f"{_dbg}_kv_local", kv)
            _rt.record_if_level(2, f"{_dbg}_score_local", score)

        is_batched_decode = (
            isinstance(start_pos, torch.Tensor)
            and start_pos.numel() > 1
            and seqlen == 1
        )

        # CP prefill: all-gather rank-local kv / score to full sequence
        # before the S-pool step so the pool sees all tokens in logical
        # order.  Decode (start_pos > 0) runs rank-local as usual — the
        # kv_cache was already populated with the full KV during prefill.
        cp_ctx = self._cp_ctx
        if not is_batched_decode and cp_should_gather(cp_ctx, start_pos):
            kv = cp_all_gather_full(kv, cp_ctx)
            score = cp_all_gather_full(score, cp_ctx)
            # After gather the effective seqlen is the FULL un-padded
            # prefill len; the caller's ``seqlen`` / ``bsz`` above
            # reflected the rank-local padded slice.  Rebind so
            # downstream pool/pooling-mask/RoPE math operates on full.
            bsz, seqlen = kv.size(0), kv.size(1)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_kv_full", kv)
            _rt.record_if_level(2, f"{_dbg}_score_full", score)

        if is_batched_decode:
            # ---- batched decode: each batch at different position ----
            slot = start_pos % ratio  # [B]
            batch_idx = torch.arange(bsz, device=x.device)

            # APE: per-batch slot
            ape_per_batch = self.ape[slot]  # [B, coff*head_dim]
            score = score.squeeze(1) + ape_per_batch  # [B, coff*head_dim]
            kv = kv.squeeze(1)  # [B, coff*head_dim]

            if overlap:
                write_slot = ratio + slot  # [B]
                self.kv_state[batch_idx, write_slot] = kv
                self.score_state[batch_idx, write_slot] = score
            else:
                self.kv_state[batch_idx, slot] = kv
                self.score_state[batch_idx, slot] = score

            emit_mask = (start_pos + 1) % ratio == 0  # [B] bool
            should_compress = emit_mask.any().item()

            if should_compress:
                emit_idx = batch_idx[emit_mask]
                if overlap:
                    kv_s = torch.cat(
                        [
                            self.kv_state[emit_idx, :ratio, :d],
                            self.kv_state[emit_idx, ratio:, d:],
                        ],
                        dim=1,
                    )
                    sc_s = torch.cat(
                        [
                            self.score_state[emit_idx, :ratio, :d],
                            self.score_state[emit_idx, ratio:, d:],
                        ],
                        dim=1,
                    )
                    compressed = (kv_s * sc_s.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[emit_idx, :ratio] = self.kv_state[emit_idx, ratio:]
                    self.score_state[emit_idx, :ratio] = self.score_state[
                        emit_idx, ratio:
                    ]
                else:
                    compressed = (
                        self.kv_state[emit_idx]
                        * self.score_state[emit_idx].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

                compressed = self._rmsnorm(compressed.to(dtype))
                emit_pos = start_pos[emit_mask]
                freqs = self.freqs_cis[emit_pos + 1 - ratio].unsqueeze(
                    1
                )  # [E, 1, rope_dim//2]
                apply_rotary_emb(compressed[..., -rd:], freqs)

                write_pos = emit_pos // ratio
                self.kv_cache[emit_idx, write_pos] = compressed.squeeze(1)

            # Decode attention reads kv_cache directly, return None
            return None

        elif seqlen > 1:
            # Prefill (fresh or continuation)
            sp_int = (
                int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            )
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            # Continuation prefill needs the PRIOR call's "save for next call"
            # kv_state / score_state to fill window 0's overlap slots (see
            # restore at line ~540 below). Snapshot them BEFORE the save-for-
            # next-call writes below clobber them.  For sp_int == 0 the snapshot
            # is unused.
            prior_kv_state_ratio = (
                self.kv_state[:bsz, :ratio, :d].clone()
                if overlap and sp_int > 0
                else None
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
                # Continuation prefill: inject saved overlap from prefix tail.
                # Use the PRIOR-call snapshot taken above — the save-for-next-
                # call writes at lines ~520/527 have already overwritten
                # kv_state[:, :ratio] / score_state[:, :ratio] with this
                # call's own tail, which would feed wrong data into window 0's
                # overlap slots.
                if sp_int > 0:
                    kv[:bsz, 0, :ratio] = prior_kv_state_ratio
                    score[:bsz, 0, :ratio] = prior_score_state_ratio

            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_pool_in_kv", kv)
                _rt.record_if_level(2, f"{_dbg}_pool_in_score", score)
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
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_pool_out", kv)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
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
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self._rmsnorm(kv.to(dtype))
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_norm_out", kv)
        if seqlen > 1:
            # Prefill (fresh or continuation)
            freqs_cis = self.freqs_cis[sp_int : sp_int + cutoff : ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_freqs_cis", freqs_cis)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_rope_out", kv)
        # NOTE: skip rotate_activation/fp4/fp8 quant for M2 (BF16-only path).

        if seqlen > 1:
            write_start = sp_int // ratio
            self.kv_cache[:bsz, write_start : write_start + cutoff // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

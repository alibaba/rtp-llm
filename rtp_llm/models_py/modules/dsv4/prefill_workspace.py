"""Per-forward prefill scratch (``PrefillWorkspace``).

A single prefill ``forward`` allocates ONE ``PrefillWorkspace`` up front and
threads it through ``PrefillMeta`` (see ``fp8/prefill_meta.py``). It folds the
two big prefill-only scratch buffers — the Q-projection output buffer and the
CP gather/restore scratch — into one per-forward object, so both are released
back to the caching allocator the moment the forward returns. That release at
the main→MTP-draft boundary is what lets the MTP forward borrow the ~16+16 GiB
those buffers would otherwise pin process-wide on a near-full 1M+CP8 card.

Each compressor role owns a fixed sub-region of ONE union buffer carved by byte
offsets (see :class:`PrefillWorkspace` for the layout). Sizes are the per-forward
MAXIMUM (derived from ``max_seq_len + cp_size + max_context_batch_size``),
identical on every forward, so a freed block is exactly reusable by the next
forward — no fragmentation. Getters take the live token count and assert it fits.

The CP region is split per gather ROLE, because two concurrent compressor gather
lifetimes can be in flight within a single CSA layer:
  * ``main`` — the CSA/HCA compressor gather
  * ``indexer`` — the nested indexer compressor gather; the overlap
    orchestrator queues it back-to-back with ``main`` before either is consumed
Each role owns a DEDICATED ``cp_gather``/``cp_restore`` sub-region; sharing
storage would let an in-flight NCCL overwrite another role's result. Overlap is
strictly within-layer (layer N drains all in-flight gathers before layer N+1
starts), so one pair per role suffices — no double-buffering. The gather impl
skips ``record_stream`` for these workspace buffers (they are reused across
layers, never recycled by the allocator) and relies on the
``gather_stream.wait_stream`` edge for cross-layer ordering. See
``cp._CP_ROLE_*``.
"""

import torch

# Default union-buffer alignment. Rounding every per-forward union block up to a
# clean 1 GiB multiple makes the caching allocator hand back an identically-sized
# block each forward, so the next forward (incl. the MTP draft right after)
# reuses it with ZERO fragmentation — the whole point of this per-forward buffer.
# The cost is up to <1 GiB rounded-up slack on small/warmup forwards, freed at
# forward exit (acceptable: the card is not full then). CPU unit tests pass a
# small ``align_bytes`` (e.g. 1) so the rounding does not force a 1 GiB host
# allocation.
_PREFILL_WS_ALIGN_BYTES = 1 << 30


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class PrefillWorkspace:
    """Per-forward prefill scratch: ONE union ``uint8`` buffer time-multiplexed
    between the Q projection output and the compressor CP gather/restore pairs.

    The big Q (``[cap_local, n_heads*head_dim]`` bf16, ~16 GiB at 1M/CP8) and the
    compressor CP buffers have DISJOINT lifetimes within a layer with respect to
    Q: those gathers are fully consumed *before* Q is materialized (Q's
    ``q_lora_b`` + RoPE are deferred to just before ``flash_mla_sparse_fwd``; see
    ``attention._materialize_prefill_q``). Current-layer SWA ``kv_full`` is not
    workspace-backed because that tensor remains live across Q materialization.
    The compressor CP roles get DISTINCT sub-regions and Q time-multiplexes with
    the WHOLE CP region:

        prefill_q       : [0,                                      q_bytes)
        cp_gather_main  : [0,                                      main_bytes)
        cp_restore_main : [main_bytes,                             2*main_bytes)
        cp_gather_idx   : [2*main_bytes,                           2*main_bytes + idx_bytes)
        cp_restore_idx  : [2*main_bytes + idx_bytes,               2*main_bytes + 2*idx_bytes)

    Union size = ``round_up(max(q_bytes, 2*main_bytes + 2*idx_bytes),
    align_bytes)`` (``align_bytes`` defaults to 1 GiB so the per-forward block is
    a clean multiple → the caching allocator reuses the same-sized block across
    forwards with zero fragmentation, and hands it back at the main→MTP draft
    boundary). Folding Q into the compressor CP region saves the standalone
    16 GiB Q buffer.

    Read/write ordering between the aliasing roles is guaranteed only for CP
    buffers whose contents are no longer live when Q is materialized. The Q vs
    compressor-CP overlap is DELIBERATE — do NOT assert physical disjointness for
    those roles.

    Each CP sub-region is sized to its role's MAXIMUM byte footprint at the
    role's natural dtype (main/idx: fp32 to admit the compressor's fp32 fused
    gather). Sub-offsets are fp32-aligned (each is a multiple of ``main_bytes`` /
    ``idx_bytes``, themselves multiples of 4). Getters take the live row count
    and assert it fits — no silent growth, no fallback allocation.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        q_rows: int,
        q_dim: int,
        reserve_cp: bool,
        cp_rows: int = 0,
        main_w: int = 0,
        idx_w: int = 0,
        # 1 GiB default is DELIBERATE — see ``_PREFILL_WS_ALIGN_BYTES`` for why.
        align_bytes: int = _PREFILL_WS_ALIGN_BYTES,
    ) -> None:
        self._device = device
        self._q_rows = int(q_rows)
        self._q_dim = int(q_dim)
        self._q_bytes = self._q_rows * self._q_dim * _dtype_size(torch.bfloat16)

        cp_rows = int(cp_rows)
        fp32 = _dtype_size(torch.float32)
        self._has_main = bool(reserve_cp) and int(main_w) > 0
        self._has_idx = bool(reserve_cp) and int(idx_w) > 0
        self._main_bytes = cp_rows * int(main_w) * fp32 if self._has_main else 0
        self._idx_bytes = cp_rows * int(idx_w) * fp32 if self._has_idx else 0

        # Fixed byte offsets into the union buffer. Q ([0, q_bytes)) overlaps the
        # front of the CP region by design (disjoint lifetimes vs Q).
        self._off_gather_main = 0
        self._off_restore_main = self._main_bytes
        self._off_gather_idx = 2 * self._main_bytes
        self._off_restore_idx = 2 * self._main_bytes + self._idx_bytes
        cp_region_bytes = 2 * self._main_bytes + 2 * self._idx_bytes

        align = int(align_bytes)
        assert align > 0, f"align_bytes must be positive, got {align}"
        union_bytes = max(self._q_bytes, cp_region_bytes)
        union_bytes = ((union_bytes + align - 1) // align) * align
        self._union = torch.empty(union_bytes, dtype=torch.uint8, device=device)
        self._attn_workspace: torch.Tensor | None = None
        self._attn_workspace_numel = 0

    def prefill_q(self, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, q_dim]`` bf16 view at the front of the union buffer."""
        num_tokens = int(num_tokens)
        assert (
            0 <= num_tokens <= self._q_rows
        ), f"prefill_q overflow: num_tokens={num_tokens} > capacity {self._q_rows}"
        return (
            self._union[: self._q_bytes]
            .view(torch.bfloat16)
            .view(self._q_rows, self._q_dim)[:num_tokens]
        )

    def cp_gather_main(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the main compressor's CP gather buffer."""
        return self._cp_view(
            "cp_gather_main",
            self._has_main,
            self._off_gather_main,
            self._main_bytes,
            rows,
            dim,
            dtype,
        )

    def cp_restore_main(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the main compressor's CP restore buffer."""
        return self._cp_view(
            "cp_restore_main",
            self._has_main,
            self._off_restore_main,
            self._main_bytes,
            rows,
            dim,
            dtype,
        )

    def cp_gather_idx(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the indexer compressor's CP gather buffer."""
        return self._cp_view(
            "cp_gather_idx",
            self._has_idx,
            self._off_gather_idx,
            self._idx_bytes,
            rows,
            dim,
            dtype,
        )

    def cp_restore_idx(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the indexer compressor's CP restore buffer."""
        return self._cp_view(
            "cp_restore_idx",
            self._has_idx,
            self._off_restore_idx,
            self._idx_bytes,
            rows,
            dim,
            dtype,
        )

    def attention_workspace(
        self,
        batch: int,
        rows: int,
        dim: int,
        dtype: torch.dtype,
        *,
        zero_on_allocate: bool = True,
    ) -> torch.Tensor:
        """Reusable ``[batch, rows, dim]`` attention workspace for one forward.

        This is intentionally separate from the Q/CP union buffer: sparse
        attention reads this workspace while Q and CP scratch may still be live,
        so aliasing it into the union would be unsafe.  The buffer grows within
        a forward if a later layer needs a larger ``M``; otherwise layers reuse
        the same allocation.
        """

        batch = int(batch)
        rows = int(rows)
        dim = int(dim)
        assert batch >= 0 and rows >= 0 and dim >= 0
        required = batch * rows * dim
        buf = self._attn_workspace
        needs_alloc = (
            buf is None
            or buf.dtype != dtype
            or buf.device != self._device
            or int(buf.numel()) < required
        )
        if needs_alloc:
            new_numel = max(int(self._attn_workspace_numel), required)
            buf = torch.empty((new_numel,), dtype=dtype, device=self._device)
            if zero_on_allocate:
                buf.zero_()
            self._attn_workspace = buf
            self._attn_workspace_numel = int(buf.numel())
        return buf[:required].view(batch, rows, dim)

    def _cp_view(
        self,
        name: str,
        has: bool,
        off: int,
        region_bytes: int,
        rows: int,
        dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert has, f"{name} region not reserved (reserve_cp=False)"
        rows = int(rows)
        dim = int(dim)
        nbytes = rows * dim * _dtype_size(dtype)
        assert nbytes <= region_bytes, (
            f"{name} overflow: need {nbytes} B ({rows}x{dim}x{_dtype_size(dtype)}) "
            f"> reserved {region_bytes} B"
        )
        return self._union[off : off + nbytes].view(dtype).view(rows, dim)

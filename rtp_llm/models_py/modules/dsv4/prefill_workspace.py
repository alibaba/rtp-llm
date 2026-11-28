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

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch

try:
    from rtp_llm.models_py.triton_kernels.fast_launch import is_cuda_stream_capturing
except ImportError:  # pragma: no cover - CPU/decode minimal targets

    def is_cuda_stream_capturing() -> bool:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# Default union-buffer alignment. Rounding every per-forward union block up to a
# clean 1 GiB multiple makes the caching allocator hand back an identically-sized
# block each forward, so the next forward (incl. the MTP draft right after)
# reuses it with ZERO fragmentation — the whole point of this per-forward buffer.
# The cost is up to <1 GiB rounded-up slack on small/warmup forwards, freed at
# forward exit (acceptable: the card is not full then). CPU unit tests pass a
# small ``align_bytes`` (e.g. 1) so the rounding does not force a 1 GiB host
# allocation.
_PREFILL_WS_ALIGN_BYTES = 1 << 30

_DTYPE_SIZES: Dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
    torch.float8_e4m3fn: 1,
}


def _dtype_size(dtype: torch.dtype) -> int:
    size = _DTYPE_SIZES.get(dtype)
    if size is not None:
        return size
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
        self._hca_disjoint_gather: Optional[torch.Tensor] = None
        self._hca_disjoint_restore: Optional[torch.Tensor] = None
        self._disjoint_main_gather_buf: Optional[torch.Tensor] = None
        self._disjoint_live_q_end_bytes: int = 0
        self._scratch: Dict[str, torch.Tensor] = {}
        self.reuse_layer_out_scratch: bool = False

    def scratch(
        self,
        key: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a per-forward grow-only reusable scratch view of ``(shape, dtype)``."""
        elem_bytes = _dtype_size(dtype)
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * elem_bytes
        if nbytes == 0:
            return self._union[:0].view(dtype).view(shape)
        buf = self._scratch.get(key)
        if buf is None or int(buf.numel()) < nbytes:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            self._scratch[key] = buf
        out = buf[:nbytes].view(dtype).view(shape)
        if zero:
            out.zero_()
        return out

    def scratch_ue8m0_scale(
        self,
        key: str,
        rows: int,
        width: int,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Return ``[rows, aligned_k // 4]`` transposed int32 TMA-aligned UE8M0 scale view."""
        scale_cols = int(width) // int(group_size)
        aligned_m = ((int(rows) + 3) // 4) * 4
        aligned_k = ((scale_cols + 3) // 4) * 4
        raw = self.scratch(key, (aligned_k // 4, aligned_m), torch.int32)
        return raw.transpose(-1, -2)[: int(rows), :]

    def scratch_inv_rope_scale(
        self,
        key: str,
        m_rows: int,
        n_groups: int,
        packed_sf_k: int,
    ) -> torch.Tensor:
        """Return column-major TMA-aligned int32 scale view for inverse RoPE FP8 quant."""
        tma_m = ((int(m_rows) + 3) // 4) * 4
        raw = self.scratch(
            key, (int(n_groups) * int(packed_sf_k) * tma_m,), torch.int32
        )
        return raw.as_strided(
            (int(n_groups), tma_m, int(packed_sf_k)),
            (int(packed_sf_k) * tma_m, 1, tma_m),
        )

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """Return True if ``tensor`` aliases storage owned by this workspace."""
        ptr = int(tensor.data_ptr())
        if ptr == 0:
            return False
        union_ptr = int(self._union.data_ptr())
        if union_ptr <= ptr < union_ptr + int(self._union.numel()):
            return True
        for buf in self._scratch.values():
            bptr = int(buf.data_ptr())
            if bptr <= ptr < bptr + int(buf.numel()):
                return True
        if self._hca_disjoint_gather is not None:
            gptr = int(self._hca_disjoint_gather.data_ptr())
            if gptr <= ptr < gptr + int(self._hca_disjoint_gather.numel()):
                return True
        if self._hca_disjoint_restore is not None:
            rptr = int(self._hca_disjoint_restore.data_ptr())
            if rptr <= ptr < rptr + int(self._hca_disjoint_restore.numel()):
                return True
        return False

    def try_disjoint_hca_gather(
        self,
        q_tokens: int,
        cp_rows: int,
        dim: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return a gather buffer disjoint from ``prefill_q[:q_tokens]`` if safe.

        Order of preference:
          1. Unused tail of ``self._union`` after ``[0, q_end)`` (zero extra HBM).
          2. One lazily cached per-forward ``_hca_disjoint_gather`` buffer when
             non-capturing CUDA free HBM comfortably exceeds ``2 * nbytes + 4 GiB``.
          3. ``None`` so the caller falls back to the serialized ``cp_gather_main``
             overlap-free schedule (zero OOM risk at near-full 1M context).
        """
        q_tokens = int(q_tokens)
        cp_rows = int(cp_rows)
        dim = int(dim)
        if (
            not self._has_main
            or q_tokens < 0
            or q_tokens > self._q_rows
            or cp_rows < 0
            or dim < 0
        ):
            return None

        elem_bytes = _dtype_size(dtype)
        nbytes = cp_rows * dim * elem_bytes
        if nbytes == 0:
            return torch.empty((cp_rows, dim), dtype=dtype, device=self._device)
        if nbytes > self._main_bytes:
            return None

        q_end = q_tokens * self._q_dim * _dtype_size(torch.bfloat16)
        union_bytes = int(self._union.numel())
        align16 = max(16, elem_bytes)
        tail_start_16 = ((q_end + align16 - 1) // align16) * align16
        tail_start_elem = ((q_end + elem_bytes - 1) // elem_bytes) * elem_bytes
        if union_bytes - tail_start_16 >= nbytes:
            tail_start = tail_start_16
        elif union_bytes - tail_start_elem >= nbytes:
            tail_start = tail_start_elem
        else:
            tail_start = -1

        if tail_start >= 0:
            return (
                self._union[tail_start : tail_start + nbytes]
                .view(dtype)
                .view(cp_rows, dim)
            )

        if self._device.type != "cuda" or is_cuda_stream_capturing():
            return None

        if (
            self._hca_disjoint_gather is not None
            and int(self._hca_disjoint_gather.numel()) >= nbytes
        ):
            return (
                self._hca_disjoint_gather[:nbytes]
                .view(dtype)
                .view(cp_rows, dim)
            )

        free_bytes, _ = torch.cuda.mem_get_info(self._device)
        if free_bytes > 2 * nbytes + (4 << 30):
            self._hca_disjoint_gather = torch.empty(
                nbytes, dtype=torch.uint8, device=self._device
            )
            return self._hca_disjoint_gather.view(dtype).view(cp_rows, dim)
        return None

    @contextmanager
    def use_disjoint_main_gather(
        self,
        buf: Optional[torch.Tensor],
        *,
        live_q_tokens: int = 0,
    ) -> Iterator[None]:
        """Temporarily redirect ``cp_gather_main`` to ``buf`` and guard ``cp_restore_main``."""
        prev_buf = self._disjoint_main_gather_buf
        prev_q_end = self._disjoint_live_q_end_bytes
        self._disjoint_main_gather_buf = buf
        self._disjoint_live_q_end_bytes = (
            max(0, int(live_q_tokens)) * self._q_dim * _dtype_size(torch.bfloat16)
            if buf is not None
            else 0
        )
        try:
            yield
        finally:
            self._disjoint_main_gather_buf = prev_buf
            self._disjoint_live_q_end_bytes = prev_q_end

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
        if self._disjoint_main_gather_buf is not None:
            assert self._has_main, "cp_gather_main region not reserved (reserve_cp=False)"
            rows = int(rows)
            dim = int(dim)
            nbytes = rows * dim * _dtype_size(dtype)
            assert nbytes <= self._main_bytes, (
                f"cp_gather_main overflow: need {nbytes} B ({rows}x{dim}x{_dtype_size(dtype)}) "
                f"> reserved {self._main_bytes} B"
            )
            buf = self._disjoint_main_gather_buf
            assert (
                buf.device == self._device
                and buf.dtype == dtype
                and buf.is_contiguous()
            ), (
                "disjoint cp_gather_main buffer mismatch: "
                f"got device={buf.device}, dtype={buf.dtype}, contiguous={buf.is_contiguous()}"
            )
            assert (
                buf.numel() >= rows * dim
            ), f"disjoint cp_gather_main underflow: {buf.numel()} < {rows * dim}"
            if rows == 0 or dim == 0:
                return buf.view(-1)[:0].view(rows, dim)
            return buf.view(-1)[: rows * dim].view(rows, dim)
        return self._cp_view(
            "cp_gather_main",
            self._has_main,
            self._off_gather_main,
            self._main_bytes,
            rows,
            dim,
            dtype,
        )

    def _union_byte_span(self, tensor: torch.Tensor) -> Optional[tuple[int, int]]:
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if nbytes == 0:
            return None
        base_ptr = int(self._union.data_ptr())
        base_end = base_ptr + int(self._union.numel())
        ptr = int(tensor.data_ptr())
        if base_ptr <= ptr and ptr + nbytes <= base_end:
            return (ptr - base_ptr, ptr - base_ptr + nbytes)
        return None

    def cp_restore_main(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the main compressor's CP restore buffer."""
        if self._disjoint_main_gather_buf is not None:
            assert self._has_main, "cp_restore_main region not reserved (reserve_cp=False)"
            rows = int(rows)
            dim = int(dim)
            elem_bytes = _dtype_size(dtype)
            nbytes = rows * dim * elem_bytes
            assert nbytes <= self._main_bytes, (
                f"cp_restore_main overflow: need {nbytes} B ({rows}x{dim}x{elem_bytes}) "
                f"> reserved {self._main_bytes} B"
            )
            if nbytes == 0:
                return self._cp_view(
                    "cp_restore_main",
                    self._has_main,
                    self._off_restore_main,
                    self._main_bytes,
                    rows,
                    dim,
                    dtype,
                )
            r_start = self._off_restore_main
            r_end = r_start + nbytes
            q_end = self._disjoint_live_q_end_bytes
            gather_span = self._union_byte_span(self._disjoint_main_gather_buf)
            overlaps_q = r_start < q_end
            overlaps_gather = (
                gather_span is not None
                and r_start < gather_span[1]
                and gather_span[0] < r_end
            )
            if overlaps_q or overlaps_gather:
                min_free = max(q_end, gather_span[1] if gather_span is not None else 0)
                tail_start = ((min_free + elem_bytes - 1) // elem_bytes) * elem_bytes
                if int(self._union.numel()) - tail_start >= nbytes:
                    return (
                        self._union[tail_start : tail_start + nbytes]
                        .view(dtype)
                        .view(rows, dim)
                    )
                if (
                    self._hca_disjoint_restore is None
                    or int(self._hca_disjoint_restore.numel()) < nbytes
                ):
                    self._hca_disjoint_restore = torch.empty(
                        nbytes, dtype=torch.uint8, device=self._device
                    )
                return self._hca_disjoint_restore[:nbytes].view(dtype).view(rows, dim)
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

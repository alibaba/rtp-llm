"""Per-forward prefill scratch (``PrefillWorkspace``).

A single prefill ``forward`` allocates ONE ``PrefillWorkspace`` up front and
threads it through ``PrefillMeta`` (see ``fp8/prefill_meta.py``). It folds the
two big prefill-only scratch buffers — the Q-projection output buffer and the
CP gather/restore scratch — into one per-forward object, so both are released
back to the caching allocator the moment the forward returns. That release at
the main→MTP-draft boundary is what lets the MTP forward borrow the ~16+16 GiB
those buffers would otherwise pin process-wide on a near-full 1M+CP8 card.

Each compressor role owns a fixed sub-region of ONE union buffer carved by byte
offsets (see :class:`PrefillWorkspace` for the layout). Ordinary V4 retains
fixed maximum Q and CP row capacities, identical on every forward for allocator
reuse. V4.1 bounds Q to one attention chunk of rank-local padded tokens and
passes ``reserve_cp=False``: it owns its CP gather buffers separately, without V4's
compressor or nested indexer modules. Widths remain model-level maxima, and the
union retains its 1 GiB allocation buckets for both models.

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

import os

import torch


def prefill_q_workspace_rows(rows: int) -> int:
    """V4.1 Q capacity: one attention chunk, or all rows for the fallback."""
    if os.environ.get("DSV41_PREFILL_Q_CHUNKED", "1") != "0":
        from rtp_llm.models_py.modules.dsv4.chunk_env import FLASH_MLA_SPARSE_Q_CHUNK

        return min(int(rows), FLASH_MLA_SPARSE_Q_CHUNK)
    return int(rows)


# Default union-buffer alignment. Rounding every per-forward union block up to a
# clean 1 GiB multiple allows adjacent input sizes to reuse one allocation
# bucket, including target/draft forwards with the same live row footprint.
# The cost is <1 GiB rounded-up slack, freed at forward exit. CPU unit tests pass a
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
    a clean multiple for reuse across similarly sized forwards, and is released
    at the main→MTP draft boundary). Q shares the larger of the two regions.

    Read/write ordering between the aliasing roles is guaranteed only for CP
    buffers whose contents are no longer live when Q is materialized. The Q vs
    compressor-CP overlap is DELIBERATE — do NOT assert physical disjointness for
    those roles.

    Each CP sub-region is sized to its role's MAXIMUM byte footprint at the
    role's natural dtype (main/idx: fp32 to admit the compressor's fp32 fused
    gather). Sub-offsets are fp32-aligned (each is a multiple of ``main_bytes`` /
    ``idx_bytes``, themselves multiples of 4). Getters use the metadata-sized
    live row count without silent growth or fallback allocation.
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
        union_bytes = max(self._q_bytes, cp_region_bytes)
        union_bytes = ((union_bytes + align - 1) // align) * align
        self._union = torch.empty(union_bytes, dtype=torch.uint8, device=device)
        # Cache fixed role sub-regions once so hot getters cannot cross into a
        # neighboring gather/restore buffer even when caller metadata drifts.
        self._gather_main_region = self._union[
            self._off_gather_main : self._off_gather_main + self._main_bytes
        ]
        self._restore_main_region = self._union[
            self._off_restore_main : self._off_restore_main + self._main_bytes
        ]
        self._gather_idx_region = self._union[
            self._off_gather_idx : self._off_gather_idx + self._idx_bytes
        ]
        self._restore_idx_region = self._union[
            self._off_restore_idx : self._off_restore_idx + self._idx_bytes
        ]

    def prefill_q(self, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, q_dim]`` bf16 view at the front of the union buffer."""
        num_tokens = int(num_tokens)
        return (
            self._union[: self._q_bytes]
            .view(torch.bfloat16)
            .view(self._q_rows, self._q_dim)[:num_tokens]
        )

    def cp_gather_main(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the main compressor's CP gather buffer."""
        return self._cp_view(
            self._gather_main_region,
            rows,
            dim,
            dtype,
        )

    def cp_restore_main(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the main compressor's CP restore buffer."""
        return self._cp_view(
            self._restore_main_region,
            rows,
            dim,
            dtype,
        )

    def cp_gather_idx(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the indexer compressor's CP gather buffer."""
        return self._cp_view(
            self._gather_idx_region,
            rows,
            dim,
            dtype,
        )

    def cp_restore_idx(self, rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """``[rows, dim]`` view of the indexer compressor's CP restore buffer."""
        return self._cp_view(
            self._restore_idx_region,
            rows,
            dim,
            dtype,
        )

    def _cp_view(
        self,
        region: torch.Tensor,
        rows: int,
        dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = int(rows)
        dim = int(dim)
        nbytes = rows * dim * _dtype_size(dtype)
        return region[:nbytes].view(dtype).view(rows, dim)

"""Shared-prefix decode for contiguous and ROCm shuffled paged KV caches.

Query tiles reuse K/V through matrix instructions for the common complete
pages. A second kernel evaluates each query's private suffix and merges the
FP32 softmax states. Planning happens outside HIP Graph capture; every launch
and device address is fixed for a given sequence-row capacity.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _shared_prefix(
    Q,
    KV,
    TABLE,
    CUTS,
    Q_PTR,
    PART_OUT,
    PART_LSE,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    KV_PLANE: tl.constexpr,
    KV_HEAD: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    LINEAR_V: tl.constexpr,
    SPLITS: tl.constexpr,
    BEAMS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    CUT_BEAMS: tl.constexpr,
    INDEX64: tl.constexpr,
    HND: tl.constexpr = False,
):
    tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    G: tl.constexpr = HEADS // KV_HEADS
    X: tl.constexpr = 8  # FP16/BF16 shuffled K/V vector width (16 bytes).
    m = tl.arange(0, M)
    d = tl.arange(0, DIM)
    subtiles: tl.constexpr = CUT_BEAMS // BEAMS
    group_index = tile // subtiles
    group_begin = tl.load(Q_PTR + group_index)
    group_end = tl.load(Q_PTR + group_index + 1)
    rows = group_begin + (tile % subtiles) * BEAMS + m // G
    valid_rows = (rows < group_end) & (rows < ROWS)
    qheads = head * G + m % G
    q = tl.load(
        Q + rows[:, None] * Q_ROW + qheads[:, None] * Q_HEAD + d[None, :],
        valid_rows[:, None],
        other=0,
    )
    shared = tl.load(CUTS + group_index)
    per_split = tl.cdiv(shared, SPLITS)
    begin = split * per_split
    end = tl.minimum(shared, begin + per_split)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, DIM), 0, tl.float32)
    for start in range(begin, end, N):
        n = start + tl.arange(0, N)
        block = tl.load(TABLE + group_begin * TABLE_ROW + n // PAGE, n < end, other=0)
        if INDEX64:
            block = block.to(tl.int64)
        t = n % PAGE
        if HND:
            koff = t[None, :] * DIM + d[:, None]
        else:
            koff = (d[:, None] // X) * PAGE * X + t[None, :] * X + d[:, None] % X
        k = tl.load(
            KV + block[None, :] * KV_BLOCK + head * KV_HEAD + koff,
            n[None, :] < end,
            other=0,
        )
        logits = tl.dot(q, k).to(tl.float32) * SCALE
        logits = tl.where(n[None, :] < end, logits, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        alpha = tl.exp2(maximum - new_maximum)
        p = tl.exp2(logits - new_maximum[:, None])
        if HND:
            voff = t[:, None] * DIM + d[None, :]
        elif LINEAR_V:
            voff = d[None, :] * PAGE + t[:, None]
        else:
            voff = (t[:, None] // X) * DIM * X + d[None, :] * X + t[:, None] % X
        v = tl.load(
            KV + block[:, None] * KV_BLOCK + KV_PLANE + head * KV_HEAD + voff,
            n[:, None] < end,
            other=0,
        )
        if HND:
            # BF16 probabilities and normalized partial outputs each introduce
            # another rounding before the private tail is merged. Quantized
            # models amplify that loss. Keep FP32 states and retain the low
            # probability bits with a second matrix multiply.
            p_hi = p.to(v.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_maximum
    safe_denom = tl.where(denom > 0, denom, 1.0)
    pos = (split * ROWS + rows) * HEADS + qheads
    tl.store(
        PART_OUT + pos[:, None] * DIM + d[None, :],
        acc / safe_denom[:, None],
        valid_rows[:, None],
    )
    tl.store(
        PART_LSE + pos,
        tl.where(denom > 0, maximum + tl.log2(safe_denom), -float("inf")),
        valid_rows,
    )


@triton.jit
def _private_tail_merge(
    Q,
    KV,
    TABLE,
    LENGTHS,
    CUTS,
    PART_OUT,
    PART_LSE,
    OUT,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    KV_PLANE: tl.constexpr,
    KV_HEAD: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    LINEAR_V: tl.constexpr,
    SPLITS: tl.constexpr,
    BEAMS: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    INDEX64: tl.constexpr,
    NATIVE_PREFIX: tl.constexpr = False,
    ROW_TILE: tl.constexpr = 1,
    HND: tl.constexpr = False,
    GROUP_TILE: tl.constexpr = 0,
):
    tile, head_tile = tl.program_id(0), tl.program_id(1)
    X: tl.constexpr = 8
    GROUP: tl.constexpr = G if GROUP_TILE == 0 else GROUP_TILE
    GROUPS_PER_HEAD: tl.constexpr = G // GROUP
    head = head_tile // GROUPS_PER_HEAD
    M: tl.constexpr = ROW_TILE * GROUP
    m = tl.arange(0, M)
    row = tile * ROW_TILE + m // GROUP
    d = tl.arange(0, DIM)
    n = tl.arange(0, N)
    qheads = head * G + (head_tile % GROUPS_PER_HEAD) * GROUP + m % GROUP
    q = tl.load(
        Q + row[:, None] * Q_ROW + qheads[:, None] * Q_HEAD + d[None, :],
        row[:, None] < ROWS,
        other=0,
    ).to(tl.float32)
    length = tl.load(LENGTHS + row, row < ROWS, other=0)
    begin = tl.load(CUTS + tile * ROW_TILE)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, DIM), 0, tl.float32)
    for start in range(begin, tl.max(length, 0), N):
        tokens = start + n
        valid = (row[:, None] < ROWS) & (tokens[None, :] < length[:, None])
        block = tl.load(
            TABLE + row[:, None] * TABLE_ROW + tokens[None, :] // PAGE, valid, other=0
        )
        if INDEX64:
            block = block.to(tl.int64)
        t = tokens % PAGE
        if HND:
            koff = t[:, None] * DIM + d[None, :]
        else:
            koff = (d[None, :] // X) * PAGE * X + t[:, None] * X + d[None, :] % X
        k = tl.load(
            KV + block[:, :, None] * KV_BLOCK + head * KV_HEAD + koff[None, :, :],
            valid[:, :, None],
            other=0,
        ).to(tl.float32)
        logits = tl.sum(q[:, None, :] * k, 2) * SCALE
        logits = tl.where(valid, logits, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        p = tl.exp2(logits - safe_maximum[:, None])
        if HND:
            voff = t[:, None] * DIM + d[None, :]
        elif LINEAR_V:
            voff = d[None, :] * PAGE + t[:, None]
        else:
            voff = (t[:, None] // X) * DIM * X + d[None, :] * X + t[:, None] % X
        v = tl.load(
            KV
            + block[:, :, None] * KV_BLOCK
            + KV_PLANE
            + head * KV_HEAD
            + voff[None, :, :],
            valid[:, :, None],
            other=0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v, 1)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_maximum
    safe_denom = tl.where(denom > 0, denom, 1.0)
    tail_lse = tl.where(denom > 0, maximum + tl.log2(safe_denom), -float("inf"))
    splits = tl.arange(0, SPLITS)
    positions = (splits[:, None] * ROWS + row[None, :]) * HEADS + qheads[None, :]
    if NATIVE_PREFIX:
        prefix_lse = (
            tl.load(
                PART_LSE + qheads[None, :] * ROWS + row[None, :],
                row[None, :] < ROWS,
                other=-float("inf"),
            )
            * 1.4426950408889634
        )
        prefix_lse = tl.where(begin > 0, prefix_lse, -float("inf"))
    else:
        prefix_lse = tl.load(
            PART_LSE + positions, row[None, :] < ROWS, other=-float("inf")
        )
    merge_max = tl.maximum(tail_lse, tl.max(prefix_lse, 0))
    merge_max = tl.where(merge_max == -float("inf"), 0.0, merge_max)
    prefix_weight = tl.exp2(prefix_lse - merge_max[None, :])
    tail_weight = tl.exp2(tail_lse - merge_max)
    prefix_out = tl.load(
        PART_OUT + positions[:, :, None] * DIM + d[None, None, :],
        row[None, :, None] < ROWS,
        other=0,
    )
    prefix_out = tl.where(begin > 0, prefix_out, 0.0)
    merged = tl.sum(prefix_out * prefix_weight[:, :, None], 0)
    merged += acc / safe_denom[:, None] * tail_weight[:, None]
    merged /= tl.maximum(tl.sum(prefix_weight, 0) + tail_weight, 1e-30)[:, None]
    tl.store(
        OUT + (row[:, None] * HEADS + qheads[:, None]) * DIM + d[None, :],
        merged,
        row[:, None] < ROWS,
    )


def _query_partition_boundaries(table, lengths, page, tile_rows, capacity, min_beams):
    """Propose query tiles from physical histories; cuts still verify every page.

    ``lengths`` includes the current token. Only complete history pages take
    part in the signature: a copied partial page is normally private per beam.
    Long equal-history runs also reveal requests whose first page was reused
    across requests. Short runs are not added, so divergent beam histories do
    not consume a tile for every row. First-page boundaries retain the previous
    behavior when a request has divergent complete private-history pages.
    """
    rows, width = table.shape
    first_page_changes = np.flatnonzero(table[1:, 0] != table[:-1, 0]) + 1
    history_pages = (lengths - 1) // page
    common_pages = np.minimum(history_pages[1:], history_pages[:-1])
    history_changed = np.any(
        (table[1:] != table[:-1]) & (np.arange(width)[None, :] < common_pages[:, None]),
        axis=1,
    )
    changes = np.flatnonzero((lengths[1:] != lengths[:-1]) | history_changed) + 1
    runs = np.r_[0, changes, rows]
    long_runs = np.diff(runs) >= min_beams
    edges = np.unique(
        np.r_[0, first_page_changes, runs[:-1][long_runs], runs[1:][long_runs], rows]
    )
    boundaries = np.concatenate(
        [
            np.arange(first, last, tile_rows)
            for first, last in zip(edges[:-1], edges[1:])
        ]
    )
    if len(boundaries) > capacity:
        # Fixed launch capacity is part of the graph contract. An arbitrary
        # page table may propose too many runs; the normal cut checks make
        # fixed tiling correct even when a tile crosses a request boundary.
        boundaries = np.arange(0, rows, tile_rows)
    return boundaries


class TritonBeamSearchCascadeAttention:
    MIN_BEAMS = 32
    MIN_SHARED_TOKENS = 128
    BEAMS_PER_TILE = 16
    # Validated byte-span bound for this native route; independent of Triton
    # address width and the underlying allocation's storage offset.
    NATIVE_PREFIX_BYTE_LIMIT = 1 << 31

    def __init__(
        self,
        rows,
        heads,
        kv_heads,
        head_dim,
        page_size,
        max_pages,
        dtype,
        device,
        *,
        linear_v=False,
        max_requests=1,
        kv_layout="shuffled",
    ):
        if (
            heads <= 0
            or kv_heads <= 0
            or heads % kv_heads
            or heads // kv_heads
            not in ((1, 2, 4, 8, 16, 32) if kv_layout == "HND" else (1, 2, 4, 8))
            or head_dim not in (64, 128, 256)
            or page_size <= 0
            or (kv_layout == "shuffled" and page_size % 8)
            or dtype not in (torch.bfloat16, torch.float16)
            or rows <= 0
            or max_pages <= 0
            or max_requests <= 0
            or kv_layout not in ("shuffled", "HND")
            or (kv_layout == "HND" and linear_v)
        ):
            raise ValueError("Unsupported Triton cascade geometry, layout or dtype")
        self.rows, self.heads, self.kv_heads = rows, heads, kv_heads
        self.dim, self.page, self.max_pages = head_dim, page_size, max_pages
        self.dtype, self.device, self.linear_v = dtype, torch.device(device), linear_v
        self.hnd = kv_layout == "HND"
        # CK's vectorized paged prefill kernels are specialized for selected
        # page sizes. Use the verified 16-token variant; other geometries use
        # the generic Triton implementation below.
        self.native_prefix = (
            not self.hnd
            and not linear_v
            and dtype == torch.bfloat16
            and page_size == 16
        )
        self.BEAMS_PER_TILE = (
            min(16, 64 // (heads // kv_heads))
            if self.hnd
            else (128 if self.native_prefix else 16)
        )
        if self.hnd:
            # Concurrency bounds beam rows. Useful independent shared groups
            # each contain MIN_BEAMS rows; excessive fragmentation uses the
            # existing verified fixed-tiling fallback in the host planner.
            max_requests = min(max_requests, max(1, rows // self.MIN_BEAMS))
        # Reserve enough fixed-address partitions to split at request boundaries.
        # Excessively fragmented inputs safely fall back to the original tiling.
        self.tiles = min(
            rows, triton.cdiv(rows, self.BEAMS_PER_TILE) + max_requests - 1
        )
        self.splits = 1 if self.native_prefix else (4 if rows <= 512 else 1)
        self.table_host = torch.zeros(
            (rows, max_pages), dtype=torch.int32, pin_memory=True
        )
        self.lengths_host = torch.zeros(rows, dtype=torch.int32, pin_memory=True)
        self.cuts_host = torch.zeros(self.tiles, dtype=torch.int32, pin_memory=True)
        self.qptr_host = torch.full(
            (self.tiles + 1,), rows, dtype=torch.int32, pin_memory=True
        )
        self.row_cuts_host = torch.zeros(rows, dtype=torch.int32, pin_memory=True)
        self.prefix_qptr = torch.empty_like(self.qptr_host, device=device)
        self.row_cuts = torch.empty_like(self.row_cuts_host, device=device)
        self.table = torch.empty_like(self.table_host, device=device)
        self.lengths = torch.empty_like(self.lengths_host, device=device)
        self.cuts = torch.empty_like(self.cuts_host, device=device)
        if not self.hnd:
            self.partial_out = torch.empty(
                (self.splits, rows, heads, head_dim), dtype=torch.float32, device=device
            )
            self.partial_lse = torch.empty(
                (self.splits, rows, heads), dtype=torch.float32, device=device
            )
            self.output = torch.empty(
                (rows, heads, head_dim), dtype=dtype, device=device
            )
        self._prefix_launch_options = {"num_warps": 4}
        if self.hnd or (heads // kv_heads == 8 and head_dim == 256):
            self._prefix_launch_options["num_stages"] = 1
        if not self.hnd:
            self._prefix_launch_options["waves_per_eu"] = 2
        self.shared_tokens = 0
        self._runtime_path_logged = False
        if self.native_prefix:
            # CK may prefetch up to 128 tokens beyond a tile. Use a stable
            # padded table and sanitize inactive columns before every plan.
            self.prefix_table_width = max_pages + (128 + page_size - 1) // page_size
            self.prefix_table = torch.empty(
                (self.tiles, self.prefix_table_width), dtype=torch.int32, device=device
            )
            self.prefix_table_host = torch.zeros(
                (self.tiles, self.prefix_table_width),
                dtype=torch.int32,
                pin_memory=True,
            )
            self.prefix_indptr = (
                torch.arange(self.tiles + 1, dtype=torch.int32, device=device)
                * self.prefix_table_width
            )
            self.prefix_output = torch.empty_like(self.output)
        self._staging_done = torch.cuda.Event()
        self._staging_in_flight = False

    def plan(self, block_table, sequence_lengths):
        """sequence_lengths counts history; the current token adds one KV entry."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cascade planning must run outside device Graph capture")
        if self._staging_in_flight and not self._staging_done.query():
            self._staging_done.synchronize()
        if (
            block_table.device.type != "cpu"
            or sequence_lengths.device.type != "cpu"
            or block_table.dtype != torch.int32
            or sequence_lengths.dtype != torch.int32
            or sequence_lengths.shape != (self.rows,)
            or block_table.dim() != 2
            or block_table.shape[0] != self.rows
        ):
            raise ValueError(
                "Triton cascade needs CPU int32 page tables and fixed-capacity history lengths"
            )
        lengths = sequence_lengths.numpy().astype(np.int64) + 1
        if np.any(lengths <= 0) or np.any(lengths > self.max_pages * self.page):
            raise ValueError("ROCm cascade sequence length exceeds page capacity")
        width = int(((lengths + self.page - 1) // self.page).max())
        if width > block_table.shape[1]:
            raise ValueError("ROCm cascade page table is too narrow")
        table = block_table.numpy()[:, :width]
        live_pages = (
            np.arange(width)[None, :]
            < ((lengths + self.page - 1) // self.page)[:, None]
        )
        if np.any(table[live_pages] < 0):
            raise ValueError("ROCm cascade page IDs must be nonnegative")
        boundaries = _query_partition_boundaries(
            table,
            lengths,
            self.page,
            self.BEAMS_PER_TILE,
            self.tiles,
            self.MIN_BEAMS,
        )
        qptr = self.qptr_host.numpy()
        qptr.fill(self.rows)
        qptr[: len(boundaries)] = boundaries
        cuts = self.cuts_host.numpy()
        cuts.fill(0)
        row_cuts = self.row_cuts_host.numpy()
        row_cuts.fill(0)
        for tile in range(len(boundaries)):
            first, last = int(qptr[tile]), int(qptr[tile + 1])
            limit = int((lengths[first:last] - 1).min() // self.page)
            same = np.all(table[first:last, :limit] == table[first, :limit], axis=0)
            mismatch = np.flatnonzero(~same)
            shared = int(mismatch[0]) if len(mismatch) else limit
            if shared * self.page >= self.MIN_SHARED_TOKENS:
                cuts[tile] = shared * self.page
                row_cuts[first:last] = shared * self.page
        self.shared_tokens = int(row_cuts.sum())
        # A CPU Tensor.copy_ may wake the full OpenMP pool for this small
        # metadata copy. Keep planning single-threaded and write pinned memory.
        np.copyto(self.table_host.numpy()[:, :width], table)
        self.lengths_host.numpy()[:] = lengths
        if self.hnd:
            # Kernels mask table reads to each row's live sequence. Preserve
            # the fixed graph stride without staging unused capacity columns.
            self.table[:, :width].copy_(self.table_host[:, :width], non_blocking=True)
        else:
            self.table.copy_(self.table_host, non_blocking=True)
        self.lengths.copy_(self.lengths_host, non_blocking=True)
        self.cuts.copy_(self.cuts_host, non_blocking=True)
        self.prefix_qptr.copy_(self.qptr_host, non_blocking=True)
        self.row_cuts.copy_(self.row_cuts_host, non_blocking=True)
        if self.native_prefix:
            prefix_table = self.prefix_table_host.numpy()
            source_rows = self.table_host.numpy()[np.minimum(qptr[:-1], self.rows - 1)]
            for tile in range(self.tiles):
                live_pages = int(cuts[tile] // self.page)
                # Empty prefixes use reserved page zero for speculative reads.
                prefix_table[tile].fill(
                    source_rows[tile, live_pages - 1] if live_pages else 0
                )
                prefix_table[tile, :live_pages] = source_rows[tile, :live_pages]
            self.prefix_table.copy_(self.prefix_table_host, non_blocking=True)
        self._staging_done.record()
        self._staging_in_flight = True

    def run(self, q, kv_cache):
        if (
            q.shape != (self.rows, self.heads, self.dim)
            or q.dtype != self.dtype
            or q.device != self.table.device
            or q.stride(-1) != 1
            or kv_cache.device != q.device
            or kv_cache.dtype != self.dtype
            or kv_cache.dim() != 5
            or kv_cache.shape[1:] != (2, self.kv_heads, self.page, self.dim)
            or not kv_cache.is_contiguous()
        ):
            raise ValueError(
                "Triton cascade Q/KV tensors do not match the captured layout"
            )
        if self.hnd:
            # Allocate temporaries inside capture so buckets use the runner's
            # shared Graph pool. Keeping them on each attention object would
            # retain a separate large allocation for every decode bucket.
            partial_out = torch.empty(
                (self.splits, self.rows, self.heads, self.dim),
                dtype=torch.float32,
                device=q.device,
            )
            partial_lse = torch.empty(
                (self.splits, self.rows, self.heads),
                dtype=torch.float32,
                device=q.device,
            )
            output = torch.empty_like(q, memory_format=torch.contiguous_format)
        else:
            partial_out, partial_lse, output = (
                self.partial_out,
                self.partial_lse,
                self.output,
            )
        group = self.heads // self.kv_heads
        args = (
            q.stride(0),
            q.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            self.table.stride(0),
            self.rows,
            self.heads,
            self.kv_heads,
            self.dim,
            self.page,
            self.linear_v,
            self.splits,
            self.BEAMS_PER_TILE,
        )
        scale = self.dim**-0.5 * 1.4426950408889634
        kv_span_bytes = kv_cache.shape[0] * kv_cache.stride(0) * kv_cache.element_size()
        # Triton address width is independent of the native backend's limit.
        # Keep 64-bit indexing even when a native large-pool path is enabled.
        large_pool = kv_span_bytes >= (1 << 31)
        native_prefix = (
            self.native_prefix and kv_span_bytes < self.NATIVE_PREFIX_BYTE_LIMIT
        )
        if not self._runtime_path_logged:
            logging.info(
                "%s beam cascade execution: prefix=%s, kv_span_bytes=%d, "
                "rows=%d, partition_capacity=%d",
                "HND" if self.hnd else "ROCm",
                "aiter_bf16" if native_prefix else "triton_fp32",
                kv_cache.shape[0] * kv_cache.stride(0) * kv_cache.element_size(),
                self.rows,
                self.tiles,
            )
            self._runtime_path_logged = True
        if native_prefix:
            import aiter

            k = kv_cache[:, 0].view(
                kv_cache.shape[0], self.kv_heads, self.dim // 8, self.page, 8
            )
            v = kv_cache[:, 1].view(
                kv_cache.shape[0], self.kv_heads, self.page // 8, self.dim, 8
            )
            prefix_output, prefix_lse = aiter.mha_batch_prefill_func(
                q,
                k,
                v,
                self.prefix_qptr,
                self.prefix_indptr,
                self.prefix_table.flatten(),
                self.BEAMS_PER_TILE,
                self.max_pages * self.page,
                causal=False,
                return_lse=True,
                out=self.prefix_output,
                block_table=self.prefix_table,
                seqlen_k=self.cuts,
            )
        else:
            # Preserve planned cuts while bounding the FP32 accumulator tile
            # on both layouts, including the large-pool native-prefix fallback.
            prefix_beams = min(self.BEAMS_PER_TILE, 16, 64 // group)
            prefix_args = (*args[:-1], prefix_beams)
            _shared_prefix[
                (
                    self.tiles * (self.BEAMS_PER_TILE // prefix_beams),
                    self.kv_heads,
                    self.splits,
                )
            ](
                q,
                kv_cache,
                self.table,
                self.cuts,
                self.prefix_qptr,
                partial_out,
                partial_lse,
                *prefix_args,
                prefix_beams * group,
                64,
                scale,
                self.BEAMS_PER_TILE,
                large_pool,
                HND=self.hnd,
                **self._prefix_launch_options,
            )
            prefix_output, prefix_lse = partial_out, partial_lse
        # Bound the FP32 reduction tile for wide GQA. Keeping all 32 heads in
        # one tile with head_dim=256 exceeds the shared-memory limit on SM120.
        tail_group = min(group, 8) if self.hnd else group
        _private_tail_merge[(self.rows, self.kv_heads * (group // tail_group))](
            q,
            kv_cache,
            self.table,
            self.lengths,
            self.row_cuts,
            prefix_output,
            prefix_lse,
            output,
            *args,
            group,
            8,
            scale,
            large_pool,
            NATIVE_PREFIX=native_prefix,
            ROW_TILE=1,
            HND=self.hnd,
            GROUP_TILE=tail_group if self.hnd else 0,
            num_warps=2,
        )
        return output

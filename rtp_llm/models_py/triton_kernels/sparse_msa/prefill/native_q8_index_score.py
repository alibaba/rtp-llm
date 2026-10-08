"""Forward-owned Q8 OnlyScore workspace for packed MiniMax-M3.1 index-K.

This path normally stages all physical index pages, including a zero sentinel.
If that exceeds the budget, it stages only a chunk's distinct referenced pages,
using the existing FP4 -> FP16 multiply-RN -> E4M3 satfinite reader. It never
expands main KV and leaves the production TopK untouched. Ineligible geometry
must use the existing native Q8K4 reader, not a BF16 working-page fallback.

The dependency contract is fmha_sm100.api's private dense paged planner and
OnlyScore entrypoint: 128-aligned max_k_tiles, original-head score layout
[H,K,Q], explicit max_score ownership, output_o=False and unit Q/K scales.
Different dependency versions require validation; API errors propagate.
Scratch belongs to one forward/producer epoch and runs serially on the current
stream. Rebuild on epoch/geometry changes; stage after every layer's K writer.
The 512 MiB admission budget covers this module's stage, score and safe tables;
it excludes caller Q/TopK buffers and the dependency's own planner workspace.
"""

import inspect
from numbers import Integral

import torch
import triton
import triton.language as tl

from ..decode.nvfp4_q8_index_score import _load_index_page_fp8

_WORKSPACE_LIMIT = 512 * 1024 * 1024


def _chunk_geometry(chunk):
    """Read Python host geometry only; never copy device metadata to CPU."""
    meta = chunk.host_metadata
    if meta is None:
        raise ValueError("native score requires producer-owned host metadata")
    qlens, klens, prefixes = meta.query_lens, meta.seq_lens, meta.prefix_lens
    if not qlens or not (len(qlens) == len(klens) == len(prefixes)):
        raise ValueError("native score host segment lengths differ")
    if any(
        not isinstance(v, Integral)
        for values in (qlens, klens, prefixes)
        for v in values
    ):
        raise ValueError("native score host geometry must contain integers")
    if any(v > 2**31 - 1 for values in (qlens, klens, prefixes) for v in values):
        raise ValueError("native score host lengths exceed int32 metadata ABI")
    # CP's final query segment may include padded rows beyond the real KV tail.
    # Native causal/KV-length masks handle these; the caller strips padded Q.
    if any(q <= 0 or k <= 0 or p < 0 for q, k, p in zip(qlens, klens, prefixes)):
        raise ValueError("native score host causal geometry is invalid")
    rows = chunk.q_end - chunk.q_start
    if (
        sum(qlens) != rows
        or chunk.max_seqlen_q != max(qlens)
        or chunk.max_seqlen_k != max(klens)
    ):
        raise ValueError("native score chunk bounds disagree with host metadata")
    count = len(qlens)
    for tensor, size, name in (
        (chunk.cu_seqlens, count + 1, "cu_seqlens"),
        (chunk.seq_lens, count, "seq_lens"),
        (chunk.prefix_lens, count, "prefix_lens"),
    ):
        _check_int_vector(tensor, size, name)
    entries = sum((k + 127) // 128 for k in klens)
    _check_int_vector(chunk.kv_indices, entries, "kv_indices")
    blocks = (max(klens) + 127) // 128
    tiles = ((blocks + 127) // 128) * 128
    return rows, blocks, tiles, entries


def _check_int_vector(tensor, size, name):
    if (
        tensor is None
        or tensor.dtype != torch.int32
        or tensor.ndim != 1
        or tensor.numel() != size
        or tensor.stride(0) != 1
    ):
        raise ValueError(f"{name} must be contiguous int32 [{size}]")


def native_index_workspace_bytes(chunks, pages, heads, compact=False):
    """Exact local scratch bytes for the supported private planner geometry."""
    if not chunks or pages <= 0 or heads != 4:
        raise ValueError("native score requires positive physical pages and H4 chunks")
    capacity = 0
    table_entries = 0
    covered = 0
    for chunk in chunks:
        if chunk.q_start != covered:
            raise ValueError("native score chunks must partition query rows")
        rows, _, tiles, entries = _chunk_geometry(chunk)
        covered = chunk.q_end
        capacity = max(capacity, heads * tiles * rows)
        table_entries += entries
    stage_pages = (
        min(pages, max(_chunk_geometry(c)[3] for c in chunks)) if compact else pages
    )
    extra = (pages + 1 + stage_pages + 1) * 4 if compact else 0
    return (stage_pages + 1) * 16384 + capacity * 4 + table_entries * 4 + extra


def supported_native_index_workspace(
    chunks, pages, heads, total_q, max_chunk_q, max_pages, compact=False
):
    """Host-only admission; rejection selects the existing Q8K4 score reader.

    Missing host metadata is a normal rejection. Invalid device tensor contents
    remain the producer's responsibility, as in the existing score reader.
    """
    if (
        heads != 4
        or total_q < 4096
        or not 0 < max_chunk_q <= 16384
        or not 0 < max_pages <= 1024
        or pages <= 0
        or not chunks
    ):
        return False
    try:
        nbytes = native_index_workspace_bytes(chunks, pages, heads, compact)
        geometry = [_chunk_geometry(c) for c in chunks]
    except ValueError:
        return False
    return (
        chunks[-1].q_end == total_q
        and max(g[0] for g in geometry) == max_chunk_q
        and max(g[1] for g in geometry) <= max_pages
        and nbytes <= _WORKSPACE_LIMIT
    )


# Physical strides are cache-ABI constants: retaining their compile-time
# alignment preserves the packed FP4 reader's vector layout. The changing
# active page count must not create a new kernel for each request shape.
@triton.jit(do_not_specialize=["NP"])
def _stage_index_pages(K, S, O, NP, KS: tl.constexpr, SS: tl.constexpr):
    page = tl.program_id(0).to(tl.int64)
    key = _load_index_page_fp8(K, S, page, page < NP, KS, SS, True)
    token = tl.arange(0, 128)
    dim = tl.arange(0, 128)
    tl.store(O + page * 16384 + token[:, None] * 128 + dim[None, :], key)


@triton.jit(do_not_specialize=["N", "NP"])
def _claim_index_pages(TABLE, MAP, LIST, COUNT, N, NP, B: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    page = tl.load(TABLE + offset, mask=offset < N, other=-1)
    valid = (offset < N) & (page >= 0) & (page < NP)
    target = tl.where(valid, page, NP)
    old = tl.atomic_cas(
        MAP + target,
        tl.full((B,), -1, tl.int32),
        tl.full((B,), -2, tl.int32),
        sem="relaxed",
    )
    winner = valid & (old == -1)
    dense = tl.atomic_add(
        COUNT + tl.full((B,), 0, tl.int32), 1, mask=winner, sem="relaxed"
    )
    tl.store(MAP + page, dense, mask=winner)
    tl.store(LIST + dense, page, mask=winner)


@triton.jit(do_not_specialize=["NP"])
def _stage_compact_index_pages(
    K, S, O, LIST, COUNT, KS: tl.constexpr, SS: tl.constexpr, NP
):
    dense = tl.program_id(0).to(tl.int64)
    if dense < tl.load(COUNT):
        page = tl.load(LIST + dense).to(tl.int64)
        key = _load_index_page_fp8(K, S, page, page < NP, KS, SS, True)
        token = tl.arange(0, 128)
        dim = tl.arange(0, 128)
        tl.store(O + dense * 16384 + token[:, None] * 128 + dim[None, :], key)


@triton.jit(do_not_specialize=["N", "NP", "CAP"])
def _remap_index_pages(TABLE, MAP, OUT, N, NP, CAP, B: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    page = tl.load(TABLE + offset, mask=offset < N, other=-1)
    valid = (page >= 0) & (page < NP)
    dense = tl.load(MAP + tl.where(valid, page, NP), mask=offset < N, other=CAP)
    tl.store(OUT + offset, tl.where(valid, dense, CAP), mask=offset < N)


@triton.jit(do_not_specialize=["N", "NP"])
def _safe_index_pages(TABLE, OUT, N, NP, B: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    page = tl.load(TABLE + offset, mask=offset < N, other=-1)
    tl.store(
        OUT + offset, tl.where((page >= 0) & (page < NP), page, NP), mask=offset < N
    )


@triton.jit(do_not_specialize=["NQ", "NB", "NP", "IH", "IK", "IQ", "OH", "OQ"])
def _copy_index_scores(
    IN,
    OUT,
    CU,
    LENS,
    PRE,
    POFF,
    TABLE,
    NQ,
    NB,
    NP,
    IH,
    IK,
    IQ,
    OH,
    OQ,
    B: tl.constexpr = 32,
    P: tl.constexpr = 32,
):
    row = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    seg = tl.program_id(1)
    page_tiles = tl.cdiv(NB, P)
    head = tl.program_id(2).to(tl.int64) // page_tiles
    block = (tl.program_id(2) % page_tiles) * P + tl.arange(0, P)
    start = tl.load(CU + seg).to(tl.int64)
    stop = tl.load(CU + seg + 1).to(tl.int64)
    row += start
    length = tl.load(LENS + seg)
    prefix = tl.load(PRE + seg).to(tl.int64)
    ps = tl.load(POFF + seg).to(tl.int64)
    pe = tl.load(POFF + seg + 1).to(tl.int64)
    physical = tl.load(
        TABLE + ps + block, mask=(ps + block < pe) & (block < NB), other=-1
    )
    visible = (
        (block[None, :] * 128 < length)
        & (block[None, :] * 128 <= prefix + row[:, None] - start)
        & (physical[None, :] >= 0)
        & (physical[None, :] < NP)
    )
    valid = (row[:, None] < stop) & (row[:, None] < NQ) & (block[None, :] < NB)
    source = (
        head * IH.to(tl.int64) + block[None, :].to(tl.int64) * IK + row[:, None] * IQ
    )
    destination = head * OH.to(tl.int64) + row[:, None] * OQ + block[None, :]
    value = tl.load(IN + source, mask=valid, other=0)
    tl.store(OUT + destination, tl.where(visible, value, float("-inf")), mask=valid)


def _native_api():
    from fmha_sm100.api import _fmha_sm100, _fmha_sm100_plan

    requirements = (
        (
            _fmha_sm100_plan,
            (
                "num_kv_heads",
                "qo_offset",
                "page_size",
                "output_maxscore",
                "causal",
                "num_kv_splits",
                "device",
            ),
        ),
        (
            _fmha_sm100,
            (
                "kv_indices",
                "max_score",
                "sm_scale",
                "q_scale",
                "k_scale",
                "output_maxscore",
                "output_o",
            ),
        ),
    )
    for function, names in requirements:
        if not set(names).issubset(inspect.signature(function).parameters):
            raise RuntimeError("fmha_sm100 private OnlyScore API is incompatible")
    return _fmha_sm100_plan, _fmha_sm100


def _build_native_plan(chunk, heads, device, planner):
    _, _, tiles, _ = _chunk_geometry(chunk)
    meta = chunk.host_metadata
    plan = planner(
        torch.tensor(meta.query_lens, dtype=torch.int32, device="cpu"),
        torch.tensor(meta.seq_lens, dtype=torch.int32, device="cpu"),
        heads,
        num_kv_heads=1,
        qo_offset=torch.tensor(meta.prefix_lens, dtype=torch.int32, device="cpu"),
        page_size=128,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
        device=device,
    )
    if (
        plan["max_k_tiles"] != tiles
        or plan["orig_num_qo_heads"] != heads
        or plan["MM-SA-Nv"]
        or plan["num_kv_splits"] != 1
    ):
        raise RuntimeError("fmha_sm100 plan violates dense OnlyScore geometry")
    return plan


class NativeIndexWorkspace:
    """One forward's immutable plans and sequential layer/chunk scratch."""

    def __init__(self, chunks, pages, heads, device, compact=False):
        self.chunks = tuple(chunks)
        geometry = [_chunk_geometry(c) for c in self.chunks]
        total_q = self.chunks[-1].q_end if self.chunks else 0
        if not supported_native_index_workspace(
            self.chunks,
            pages,
            heads,
            total_q,
            max((g[0] for g in geometry), default=0),
            max((g[1] for g in geometry), default=0),
            compact,
        ):
            raise ValueError("native score geometry exceeds admission limits")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("native score requires an explicit CUDA device ordinal")
        self.pages, self.heads = pages, heads
        self.compact = compact
        self.stage_pages = min(pages, max(g[3] for g in geometry)) if compact else pages
        self._geometry = tuple(geometry)
        planner, self._score = _native_api()
        self.plans = tuple(
            _build_native_plan(c, heads, self.device, planner) for c in self.chunks
        )
        capacity = max(
            heads * p["max_k_tiles"] * g[0] for p, g in zip(self.plans, geometry)
        )
        self.workspace_bytes = native_index_workspace_bytes(
            self.chunks, pages, heads, compact
        )
        if self.workspace_bytes > _WORKSPACE_LIMIT:
            raise RuntimeError("native plan scratch exceeds 512 MiB")
        for chunk in self.chunks:
            self._check_device(
                chunk.cu_seqlens, chunk.seq_lens, chunk.prefix_lens, chunk.kv_indices
            )
        self.staged = torch.empty(
            (self.stage_pages + 1, 1, 128, 128),
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        self.staged[self.stage_pages].zero_()
        if compact:
            self.page_map = torch.empty(
                pages + 1, device=self.device, dtype=torch.int32
            )
            self.page_list = torch.empty(
                self.stage_pages, device=self.device, dtype=torch.int32
            )
            self.page_count = torch.empty(1, device=self.device, dtype=torch.int32)
        self.native_score = torch.empty(
            capacity, device=self.device, dtype=torch.float32
        )
        self.safe_tables = tuple(torch.empty_like(c.kv_indices) for c in self.chunks)
        self._staged = False

    def _check_device(self, *tensors):
        if any(t.device != self.device for t in tensors):
            raise ValueError("native score tensors must share the forward device")

    @torch.no_grad()
    def stage(self, packed, scales):
        """Stage the current layer, preserving padded physical page strides."""
        self._staged = False
        validate_native_index_cache(packed, scales, self.pages)
        self._check_device(packed, scales)
        if self.compact:
            # Only hold this layer's inputs. Each chunk rebuilds its map on the
            # current stream; no IDs or decoded K survive a layer/forward.
            self._packed, self._scales = packed, scales
            self._staged = True
            return
        _stage_index_pages[(self.pages,)](
            packed,
            scales.view(torch.uint8),
            self.staged,
            self.pages,
            packed.stride(0),
            scales.stride(0),
            num_warps=4,
        )
        self._staged = True

    @torch.no_grad()
    def score(self, index, q8, page_offsets, output):
        """Overwrite caller-owned [H,Q,logical-page] scores; refresh IDs each call."""
        if not self._staged:
            raise RuntimeError("stage current layer's index K before score")
        chunk = self.chunks[index]
        if _chunk_geometry(chunk) != self._geometry[index]:
            raise ValueError(
                "native score producer geometry changed; rebuild workspace"
            )
        rows, blocks, _, _ = self._geometry[index]
        _check_int_vector(
            page_offsets, len(chunk.host_metadata.seq_lens) + 1, "page_offsets"
        )
        if (
            q8.dtype != torch.float8_e4m3fn
            or tuple(q8.shape) != (rows, 4, 128)
            or not q8.is_contiguous()
        ):
            raise ValueError("native score Q must be contiguous E4M3 [chunk_q,4,128]")
        if (
            output.dtype != torch.float32
            or tuple(output.shape) != (4, rows, blocks)
            or output.stride(2) != 1
            or output.stride(1) < blocks
            or output.stride(0) < rows * output.stride(1)
        ):
            raise ValueError(
                "native output must be nonoverlapping float32 [4,chunk_q,logical_pages]"
            )
        table, safe = chunk.kv_indices, self.safe_tables[index]
        self._check_device(q8, page_offsets, output, table)
        if (
            output.untyped_storage().data_ptr()
            == self.native_score.untyped_storage().data_ptr()
        ):
            raise ValueError("caller output must not alias native scratch")
        if self.compact:
            self.page_map.fill_(-1)
            self.page_count.zero_()
            grid = (triton.cdiv(table.numel(), 256),)
            _claim_index_pages[grid](
                table,
                self.page_map,
                self.page_list,
                self.page_count,
                table.numel(),
                self.pages,
                256,
                num_warps=4,
            )
            _stage_compact_index_pages[(self.stage_pages,)](
                self._packed,
                self._scales.view(torch.uint8),
                self.staged,
                self.page_list,
                self.page_count,
                self._packed.stride(0),
                self._scales.stride(0),
                self.pages,
                num_warps=4,
            )
            _remap_index_pages[grid](
                table,
                self.page_map,
                safe,
                table.numel(),
                self.pages,
                self.stage_pages,
                256,
                num_warps=4,
            )
        else:
            _safe_index_pages[(triton.cdiv(table.numel(), 256),)](
                table, safe, table.numel(), self.pages, 256, num_warps=4
            )
        plan = self.plans[index]
        shape = (self.heads, plan["max_k_tiles"], rows)
        result = self.native_score[: shape[0] * shape[1] * shape[2]].view(shape)
        result.fill_(float("-inf"))
        attention, returned = self._score(
            q8,
            self.staged,
            self.staged,
            plan,
            kv_indices=safe,
            max_score=result,
            sm_scale=1.0,
            q_scale=1.0,
            k_scale=1.0,
            output_maxscore=True,
            output_o=False,
        )
        if (
            attention is not None
            or returned is None
            or returned.data_ptr() != result.data_ptr()
            or tuple(returned.shape) != shape
            or returned.dtype != result.dtype
            or returned.device != result.device
            or returned.stride() != result.stride()
        ):
            raise RuntimeError("native score ownership/layout violates OnlyScore ABI")
        grid = (
            triton.cdiv(chunk.max_seqlen_q, 32),
            chunk.cu_seqlens.numel() - 1,
            self.heads * triton.cdiv(blocks, 32),
        )
        _copy_index_scores[grid](
            result,
            output,
            chunk.cu_seqlens,
            chunk.seq_lens,
            chunk.prefix_lens,
            page_offsets,
            table,
            rows,
            blocks,
            self.pages,
            *result.stride(),
            output.stride(0),
            output.stride(1),
            num_warps=4,
        )
        return output


def validate_native_index_cache(packed, scales, pages):
    """Validate page-local MMA ABI without requiring contiguous physical pages."""
    if (
        packed.dtype != torch.uint8
        or tuple(packed.shape) != (pages, 1, 128, 64)
        or packed.stride(3) != 1
        or packed.stride(2) != 64
        or packed.stride(0) < 8192
        or packed.data_ptr() % 4
        or packed.stride(0) % 4
    ):
        raise ValueError(
            "packed index K requires aligned uint8 [page,1,128,64] page-local ABI"
        )
    if (
        scales.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or tuple(scales.shape) != (pages, 1, 2, 32, 4, 4)
        or scales.stride()[2:] != (512, 16, 4, 1)
        or scales.stride(0) < 1024
    ):
        raise ValueError("index scales require E4M3/uint8 [page,1,2,32,4,4] MMA bytes")
    if scales.view(torch.uint8).stride() != scales.stride():
        raise ValueError("E4M3 scale byte view changed cache strides")

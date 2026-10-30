"""M3.1 CP Prefill IndexScore using staged FP8 pages and native OnlyScore."""

import inspect
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_index_score import (
    _load_index_page_fp8,
)
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_topk import m31_topk16

_MAX_SCRATCH_BYTES = 896 * 1024 * 1024


@triton.jit
def _stage_pages(K, S, O, PAGES, KS: tl.constexpr, SS: tl.constexpr):
    page = tl.program_id(0).to(tl.int64)
    key = _load_index_page_fp8(K, S, page, page < PAGES, KS, SS, True)
    token = tl.arange(0, 128)
    dim = tl.arange(0, 128)
    tl.store(O + page * 16384 + token[:, None] * 128 + dim[None, :], key)


@triton.jit
def _copy_scores(
    NATIVE,
    OUT,
    CU,
    LENS,
    PREFIX,
    ROWS,
    BLOCKS,
    IH: tl.constexpr,
    IK: tl.constexpr,
    IQ: tl.constexpr,
    OH: tl.constexpr,
    OQ: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
):
    segment = tl.program_id(1)
    start = tl.load(CU + segment)
    stop = tl.load(CU + segment + 1)
    row = start + tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(2) // tl.cdiv(BLOCKS, 32)
    block = (tl.program_id(2) % tl.cdiv(BLOCKS, 32)) * 32 + tl.arange(0, 32)
    length = tl.load(LENS + segment)
    prefix = tl.load(PREFIX + segment)
    position = prefix + row - start
    source = head * IH + block[None, :] * IK + row[:, None] * IQ
    value = tl.load(
        NATIVE + source,
        mask=(row[:, None] < stop) & (block[None, :] < BLOCKS),
        other=float("-inf"),
    ) * (128**-0.5)
    visible = (
        (row[:, None] < stop)
        & (block[None, :] < BLOCKS)
        & (block[None, :] * 128 < length)
        & (block[None, :] * 128 <= position[:, None])
    )
    value = tl.where(visible, value, float("-inf"))
    local_start = tl.maximum(0, (position + 128) // 128 - LOCAL)
    value = tl.where(visible & (block[None, :] >= local_start[:, None]), 1.0e29, value)
    value = tl.where(
        visible & (block[None, :] < INIT) & (block[None, :] < local_start[:, None]),
        1.0e30,
        value,
    )
    target = head * OH + row[:, None] * OQ + block[None, :]
    tl.store(
        OUT + target, value, mask=(row[:, None] < stop) & (block[None, :] < BLOCKS)
    )


@lru_cache(maxsize=1)
def _native_api():
    try:
        from fmha_sm100.api import _fmha_sm100, _fmha_sm100_plan
    except ModuleNotFoundError as error:
        if error.name not in ("fmha_sm100", "fmha_sm100.api"):
            raise
        return None
    required_plan = {
        "num_kv_heads",
        "qo_offset",
        "page_size",
        "output_maxscore",
        "causal",
        "num_kv_splits",
        "device",
    }
    required_run = {
        "kv_indices",
        "max_score",
        "sm_scale",
        "q_scale",
        "k_scale",
        "output_maxscore",
        "output_o",
    }
    if not required_plan <= set(inspect.signature(_fmha_sm100_plan).parameters):
        return None
    if not required_run <= set(inspect.signature(_fmha_sm100).parameters):
        return None
    return _fmha_sm100_plan, _fmha_sm100


@dataclass(frozen=True)
class _Chunk:
    start: int
    end: int
    blocks: int
    max_query: int
    cu_q: torch.Tensor
    lengths: torch.Tensor
    prefixes: torch.Tensor
    indices: torch.Tensor
    native_plan: dict


@lru_cache(maxsize=4)
def _plans(
    table: torch.Tensor,
    segments: tuple[tuple[int, int, int], ...],
    prefix_lengths: tuple[int, ...],
    chunk_rows: int,
) -> tuple[_Chunk, ...]:
    planner, _ = _native_api()
    device = table.device
    total = sum(length for _, _, length in segments)
    chunks = []
    for start in range(0, total, chunk_rows):
        end = min(start + chunk_rows, total)
        pieces = []
        segment_start = 0
        for request, begin, length in segments:
            lo, hi = max(start, segment_start), min(end, segment_start + length)
            if lo < hi:
                pieces.append(
                    (
                        request,
                        prefix_lengths[request] + begin + lo - segment_start,
                        hi - lo,
                    )
                )
            segment_start += length
        query_lens = [length for _, _, length in pieces]
        prefixes = [offset for _, offset, _ in pieces]
        key_lens = [offset + length for _, offset, length in pieces]
        blocks = (max(key_lens) + 127) // 128
        cu_q = [0]
        for length in query_lens:
            cu_q.append(cu_q[-1] + length)
        indices = torch.cat(
            [
                table[request, : (key_len + 127) // 128]
                for (request, _, _), key_len in zip(pieces, key_lens)
            ]
        ).contiguous()
        native_plan = planner(
            torch.tensor(query_lens, dtype=torch.int32),
            torch.tensor(key_lens, dtype=torch.int32),
            4,
            num_kv_heads=1,
            qo_offset=torch.tensor(prefixes, dtype=torch.int32),
            page_size=128,
            output_maxscore=True,
            causal=True,
            num_kv_splits=1,
            device=device,
        )
        if (
            native_plan["max_k_tiles"] != triton.cdiv(blocks, 128) * 128
            or native_plan["orig_num_qo_heads"] != 4
            or native_plan["num_kv_splits"] != 1
            or native_plan["MM-SA-Nv"]
        ):
            raise RuntimeError("M3.1 native IndexScore plan has incompatible geometry")
        chunks.append(
            _Chunk(
                start,
                end,
                blocks,
                max(query_lens),
                torch.tensor(cu_q, dtype=torch.int32, device=device),
                torch.tensor(key_lens, dtype=torch.int32, device=device),
                torch.tensor(prefixes, dtype=torch.int32, device=device),
                indices,
                native_plan,
            )
        )
    return tuple(chunks)


def m31_native_index_topk(
    index_query: torch.Tensor,
    index_views,
    index_table: torch.Tensor,
    positions: torch.Tensor,
    segments: list[tuple[int, int, int]],
    prefix_lengths: tuple[int, ...],
    *,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor | None:
    """Return selected blocks, or None when the bounded native route is unavailable."""
    native = _native_api()
    pages = index_views.index_k.shape[0]
    if (
        native is None
        or index_query.dtype != torch.float8_e4m3fn
        or index_query.shape[1:] != (4, 128)
        or not index_query.is_contiguous()
        or index_table.dtype != torch.int32
        or index_table.shape[0] != len(prefix_lengths)
        or pages * 16384 + 4 * 640 * min(index_query.shape[0], 16384) * 4
        > _MAX_SCRATCH_BYTES
    ):
        return None
    chunks = _plans(index_table, tuple(segments), prefix_lengths, 16384)
    scratch = max(
        4 * c.native_plan["max_k_tiles"] * (c.end - c.start) * 4 for c in chunks
    )
    if (pages + 1) * 16384 + scratch > _MAX_SCRATCH_BYTES:
        return None
    staged = torch.empty(
        (pages + 1, 1, 128, 128), dtype=torch.float8_e4m3fn, device=index_query.device
    )
    staged[pages].zero_()
    _stage_pages[(pages,)](
        index_views.index_k,
        index_views.index_k_scale.view(torch.uint8),
        staged,
        pages,
        index_views.index_k.stride(0),
        index_views.index_k_scale.stride(0),
        num_warps=4,
    )
    topk = torch.empty(
        (4, index_query.shape[0], 16), dtype=torch.int32, device=index_query.device
    )
    _, score_native = native
    for chunk in chunks:
        rows = chunk.end - chunk.start
        shape = (4, chunk.native_plan["max_k_tiles"], rows)
        native_scores = torch.full(
            shape, float("-inf"), dtype=torch.float32, device=index_query.device
        )
        attention, returned = score_native(
            index_query[chunk.start : chunk.end],
            staged,
            staged,
            chunk.native_plan,
            kv_indices=chunk.indices,
            max_score=native_scores,
            sm_scale=1.0,
            q_scale=1.0,
            k_scale=1.0,
            output_maxscore=True,
            output_o=False,
        )
        if attention is not None or returned.data_ptr() != native_scores.data_ptr():
            raise RuntimeError("M3.1 native IndexScore output ownership changed")
        scores = torch.empty(
            (4, rows, chunk.blocks), dtype=torch.float32, device=index_query.device
        )
        _copy_scores[
            (
                triton.cdiv(chunk.max_query, 32),
                chunk.cu_q.numel() - 1,
                4 * triton.cdiv(chunk.blocks, 32),
            )
        ](
            native_scores,
            scores,
            chunk.cu_q,
            chunk.lengths,
            chunk.prefixes,
            rows,
            chunk.blocks,
            *native_scores.stride(),
            scores.stride(0),
            scores.stride(1),
            init_blocks,
            local_blocks,
            num_warps=4,
        )
        m31_topk16(
            scores,
            (positions[chunk.start : chunk.end] + 1).to(torch.int32),
            topk[:, chunk.start : chunk.end, :],
        )
    return topk

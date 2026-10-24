"""Top-16 page selection for MiniMax-M3.1 sparse decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk16(
    score_ptr,
    lens_ptr,
    out_ptr,
    score_h,
    score_b,
    out_h,
    out_b,
    MAX_BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    blocks = (tl.load(lens_ptr + batch) + 127) // 128
    col = tl.arange(0, TILE)
    score = tl.load(
        score_ptr + head * score_h + batch * score_b + col,
        mask=(col < MAX_BLOCKS) & (col < blocks),
        other=float("-inf"),
    )
    selected = tl.full((16,), 2147483647, tl.int32)
    for i in tl.static_range(16):
        value, index = tl.max(
            score, 0, return_indices=True, return_indices_tie_break_left=True
        )
        index = tl.where(value != float("-inf"), index, 2147483647)
        selected = tl.where(tl.arange(0, 16) == i, index, selected)
        score = tl.where(col == index, float("-inf"), score)
    selected = tl.sort(selected, descending=False)
    selected = tl.where(selected == 2147483647, -1, selected)
    tl.store(
        out_ptr + head * out_h + batch * out_b + tl.arange(0, 16),
        selected,
    )


@triton.jit
def _topk16_chunks(
    score_ptr,
    lens_ptr,
    candidates_ptr,
    score_h,
    score_b,
    cand_h,
    cand_b,
    cand_c,
    MAX_BLOCKS: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    chunk = tl.program_id(2)
    valid_blocks = (tl.load(lens_ptr + batch) + 127) // 128
    col = chunk * 1024 + tl.arange(0, 1024)
    score = tl.load(
        score_ptr + head * score_h + batch * score_b + col,
        mask=(col < MAX_BLOCKS) & (col < valid_blocks),
        other=float("-inf"),
    )
    selected = tl.full((16,), 2147483647, tl.int32)
    for i in tl.static_range(16):
        value, local = tl.max(
            score, 0, return_indices=True, return_indices_tie_break_left=True
        )
        index = tl.where(
            value != float("-inf"),
            tl.sum(tl.where(tl.arange(0, 1024) == local, col, 0)),
            2147483647,
        )
        selected = tl.where(tl.arange(0, 16) == i, index, selected)
        score = tl.where(tl.arange(0, 1024) == local, float("-inf"), score)
    selected = tl.sort(selected, descending=False)
    tl.store(
        candidates_ptr
        + head * cand_h
        + batch * cand_b
        + chunk * cand_c
        + tl.arange(0, 16),
        selected,
    )


@triton.jit
def _topk16_merge(
    score_ptr,
    lens_ptr,
    candidates_ptr,
    out_ptr,
    score_h,
    score_b,
    cand_h,
    cand_b,
    out_h,
    out_b,
    MAX_BLOCKS: tl.constexpr,
    CHUNKS: tl.constexpr,
    TILE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    valid_blocks = (tl.load(lens_ptr + batch) + 127) // 128
    pos = tl.arange(0, TILE)
    ids = tl.load(
        candidates_ptr + head * cand_h + batch * cand_b + pos,
        mask=pos < CHUNKS * 16,
        other=2147483647,
    )
    valid = (ids >= 0) & (ids < MAX_BLOCKS) & (ids < valid_blocks)
    score = tl.load(
        score_ptr + head * score_h + batch * score_b + ids,
        mask=valid,
        other=float("-inf"),
    )
    selected = tl.full((16,), 2147483647, tl.int32)
    for i in tl.static_range(16):
        value, local = tl.max(
            score, 0, return_indices=True, return_indices_tie_break_left=True
        )
        index = tl.where(
            value != float("-inf"), tl.sum(tl.where(pos == local, ids, 0)), 2147483647
        )
        selected = tl.where(tl.arange(0, 16) == i, index, selected)
        score = tl.where(pos == local, float("-inf"), score)
    selected = tl.sort(selected, descending=False)
    tl.store(
        out_ptr + head * out_h + batch * out_b + tl.arange(0, 16),
        tl.where(selected == 2147483647, -1, selected),
    )


def m31_topk16(scores: torch.Tensor, seq_lens: torch.Tensor, out: torch.Tensor) -> None:
    if scores.ndim != 3 or scores.shape[0] != 4 or scores.dtype != torch.float32:
        raise ValueError("M3.1 scores must be float32 [4,batch,blocks]")
    if seq_lens.shape != (scores.shape[1],) or seq_lens.dtype != torch.int32:
        raise ValueError("M3.1 lengths must be int32 [batch]")
    if out.shape != (4, scores.shape[1], 16) or out.dtype != torch.int32:
        raise ValueError("M3.1 topk output must be int32 [4,batch,16]")
    if scores.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("M3.1 score/topk inner dimensions must be contiguous")
    blocks = scores.shape[-1]
    if blocks <= 0:
        raise ValueError("M3.1 topk requires at least one page")
    if scores.shape[1] == 0:
        return
    if blocks > 4096:
        chunks = triton.cdiv(blocks, 1024)
        candidates = torch.empty(
            (4, scores.shape[1], chunks, 16), dtype=torch.int32, device=scores.device
        )
        _topk16_chunks[(scores.shape[1], 4, chunks)](
            scores,
            seq_lens,
            candidates,
            scores.stride(0),
            scores.stride(1),
            candidates.stride(0),
            candidates.stride(1),
            candidates.stride(2),
            blocks,
            num_warps=1,
        )
        _topk16_merge[(scores.shape[1], 4)](
            scores,
            seq_lens,
            candidates,
            out,
            scores.stride(0),
            scores.stride(1),
            candidates.stride(0),
            candidates.stride(1),
            out.stride(0),
            out.stride(1),
            blocks,
            chunks,
            triton.next_power_of_2(chunks * 16),
            num_warps=1,
        )
        return
    _topk16[(scores.shape[1], 4)](
        scores,
        seq_lens,
        out,
        scores.stride(0),
        scores.stride(1),
        out.stride(0),
        out.stride(1),
        blocks,
        max(16, triton.next_power_of_2(blocks)),
        num_warps=1,
    )

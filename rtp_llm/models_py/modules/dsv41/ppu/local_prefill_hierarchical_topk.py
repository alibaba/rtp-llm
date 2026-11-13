"""Local eager K512 metadata fusion, around unchanged raw BF16 selectors.

The caller owns the default-off rollout gate and proves the start0 business
domain. Scores must already have received the original full-tile sanitizer.
This module neither sanitizes nor finite-filters; the original wrapper must
canonicalize the remapped output and apply its unchanged finite finish.
"""

from __future__ import annotations

import torch
import triton as tr
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.ppu.topk import (
    local_device_supported as _device_supported,
    overlaps as _overlap,
    raw_tensor as _raw,
)


K = 512
KEY_TILE = 256
GATHER_TILE = 512


@tr.jit(do_not_specialize=["WIDTH", "SCORE_STRIDE", "KEY_STRIDE"])
def _bf16_max8_keys(
    Scores, Ends, Keys, BlockEnds,
    WIDTH, SCORE_STRIDE, KEY_STRIDE,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    key = tl.program_id(1) * TILE + tl.arange(0, TILE)
    lane = tl.arange(0, 8)
    end = tl.minimum(tl.maximum(tl.load(Ends + row), 0), WIDTH)
    position = key[:, None].to(tl.int64) * 8 + lane[None, :]
    valid = (key[:, None] < WIDTH // 8) & (position < end)
    value = tl.load(Scores + row * SCORE_STRIDE + position, valid, other=-float("inf")).to(tl.float32)
    # BF16 -> FP32 is exact, max performs no summation, and every result is
    # an input BF16 value (or -Inf). No FP16 block key is introduced.
    maximum = tl.max(value, axis=1)
    tl.store(Keys + row * KEY_STRIDE + key, maximum.to(tl.bfloat16), key < KEY_STRIDE)
    if tl.program_id(1) == 0:
        tl.store(BlockEnds + row, (end + 7) // 8)


@tr.jit(do_not_specialize=["WIDTH", "SCORE_STRIDE"])
def _bf16_gather_block_candidates(
    Scores, Ends, Blocks, Candidates, CandidateEnds,
    WIDTH, SCORE_STRIDE, TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = tl.load(Blocks + row * 512 + column // 8).to(tl.int64)
    position = block * 8 + column % 8
    end = tl.minimum(tl.maximum(tl.load(Ends + row), 0), WIDTH)
    valid = (block >= 0) & (block < WIDTH // 8) & (position >= 0) & (position < end)
    value = tl.load(Scores + row * SCORE_STRIDE + position, valid, other=-float("inf"))
    tl.store(Candidates + row * 4096 + column, value)
    if tl.program_id(1) == 0:
        # Padding is -Inf, so it can occupy only slots later removed by the
        # original finite finish; raw +Inf slots are never discarded here.
        tl.store(CandidateEnds + row, 4096)


@tr.jit(do_not_specialize=["WIDTH"])
def _remap_block_candidate_slots(Chosen, Blocks, Ends, Out, WIDTH):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, 512)
    column = tl.load(Chosen + row * 512 + slot)
    valid = (column >= 0) & (column < 4096)
    block = tl.load(Blocks + row * 512 + column // 8, valid, other=-1).to(tl.int64)
    position = block * 8 + column % 8
    end = tl.minimum(tl.maximum(tl.load(Ends + row), 0), WIDTH)
    valid = valid & (block >= 0) & (block < WIDTH // 8) & (position >= 0) & (position < end)
    tl.store(Out + row * 512 + slot, tl.where(valid, position, -1).to(tl.int32))


def is_supported(scores, ends, out=None):
    """Metadata-only admission, without reading scores or GPU length values."""
    if not (
        not torch.is_grad_enabled() and _raw(scores) and _raw(ends)
        and scores.dtype == torch.bfloat16 and scores.ndim == 2
        and 1 <= scores.shape[0] <= 4096 and 32768 <= scores.shape[1] <= 131072
        and scores.shape[1] % 8 == 0 and scores.stride(1) == 1
        and scores.stride(0) >= scores.shape[1] and scores.stride(0) % 8 == 0
        and scores.data_ptr() % 16 == 0
        and ends.device == scores.device and ends.dtype == torch.int32
        and ends.shape == (scores.shape[0],) and ends.is_contiguous()
        and not _overlap(scores, ends)
    ):
        return False
    if out is not None and not (
        _raw(out) and out.device == scores.device and out.dtype == torch.int32
        and out.shape == (scores.shape[0], K) and out.is_contiguous()
        and out.data_ptr() % 16 == 0
        and not _overlap(out, scores) and not _overlap(out, ends)
    ):
        return False
    return _device_supported(scores)


def select(scores, starts, ends, out):
    """Return raw remapped K512, or None before an unsupported launch.

    ``starts`` is the original bounds=None wrapper's all-zero vector; its
    values are a caller proof, not certified by this metadata-only gate.
    Once admitted, allocation/import/launch errors propagate without retry.
    """
    if out is None or not is_supported(scores, ends, out):
        return None
    if not (
        _raw(starts) and starts.device == scores.device and starts.dtype == torch.int32
        and starts.shape == (scores.shape[0],) and starts.is_contiguous()
        and not any(_overlap(starts, value) for value in (scores, ends, out))
    ):
        return None
    from rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer import topk_bf16

    rows, width = scores.shape
    key_width = width // 8
    key_stride = tr.cdiv(key_width, 8) * 8
    key_bank = torch.empty((rows, key_stride), dtype=torch.bfloat16, device=scores.device)
    keys = key_bank[:, :key_width]
    block_ends = torch.empty_like(ends)
    blocks = torch.empty((rows, K), dtype=torch.int32, device=scores.device)
    candidates = torch.empty((rows, 4096), dtype=torch.bfloat16, device=scores.device)
    candidate_ends = torch.empty_like(ends)
    chosen = torch.empty_like(out)
    _bf16_max8_keys[(rows, tr.cdiv(key_stride, KEY_TILE))](
        scores, ends, key_bank, block_ends, width, scores.stride(0), key_stride, KEY_TILE, num_warps=4,
    )
    topk_bf16(keys, starts, block_ends, blocks)
    _bf16_gather_block_candidates[(rows, 4096 // GATHER_TILE)](
        scores, ends, blocks, candidates, candidate_ends, width, scores.stride(0), GATHER_TILE, num_warps=4,
    )
    topk_bf16(candidates, starts, candidate_ends, chosen)
    _remap_block_candidate_slots[(rows,)](chosen, blocks, ends, out, width, num_warps=4)
    return out

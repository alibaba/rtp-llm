"""UT for the fused CompressorFP8.prepare_metadata slot builder.

The decode/speculative path flattens request-major ``[B, S]`` metadata into
``positions[N]`` and ``b_idx[N]``.  This test compares the default-on fused
Triton path against the Python reference path for both CSA/indexer
``ratio=4`` and HCA ``ratio=128`` shapes.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _fused_compressor_meta_triton
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _compressor_meta_fused_enabled,
)

DEVICE = "cuda"
STATE_EB = 256
requires_cuda = unittest.skipUnless(
    torch.cuda.is_available() and _fused_compressor_meta_triton._TRITON_AVAILABLE,
    "CUDA/Triton unavailable",
)


@contextmanager
def _env(value: Optional[str]):
    prev = os.environ.get("DSV4_FUSED_PREPARE")
    if value is None:
        os.environ.pop("DSV4_FUSED_PREPARE", None)
    else:
        os.environ["DSV4_FUSED_PREPARE"] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DSV4_FUSED_PREPARE", None)
        else:
            os.environ["DSV4_FUSED_PREPARE"] = prev


def _make_positions(
    start_pos: torch.Tensor, q_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(start_pos.numel())
    pos_2d = start_pos.view(bsz, 1) + torch.arange(
        q_len, dtype=torch.long, device=start_pos.device
    ).view(1, q_len)
    positions = pos_2d.reshape(-1).contiguous()
    b_idx = torch.arange(bsz, dtype=torch.long, device=start_pos.device)
    return positions, b_idx.repeat_interleave(q_len).contiguous()


def _shell(
    *,
    ratio: int,
    state_bt: torch.Tensor,
    kv_bt: Optional[torch.Tensor],
    kv_eb: int,
    pool_rows: int = 0,
) -> CompressorFP8:
    cmp = CompressorFP8.__new__(CompressorFP8)
    object.__setattr__(cmp, "compress_ratio", ratio)
    object.__setattr__(cmp, "_state_block_table", state_bt)
    object.__setattr__(cmp, "_state_eb", STATE_EB)
    object.__setattr__(cmp, "_state_tokens_per_block", STATE_EB)
    object.__setattr__(cmp, "_kv_block_table", kv_bt)
    object.__setattr__(cmp, "_kv_eb", kv_eb)
    object.__setattr__(cmp, "_kv_cache_sharded", False)
    object.__setattr__(cmp, "_cp_ctx", None)
    if pool_rows > 0 and kv_eb > 0:
        assert pool_rows % kv_eb == 0
        pool = torch.empty(
            (pool_rows // kv_eb, kv_eb, 1), dtype=torch.uint8, device=DEVICE
        )
    else:
        pool = None
    object.__setattr__(cmp, "_kv_pool_3d", pool)
    return cmp


def _assert_meta_equal(py, fused):
    assert torch.equal(py.positions, fused.positions)
    assert torch.equal(py.b_idx, fused.b_idx)
    assert torch.equal(py.state_slots, fused.state_slots)
    assert torch.equal(py.kv_slots, fused.kv_slots)
    assert torch.equal(py.token_to_req, fused.token_to_req)
    assert py.is_batched == fused.is_batched


def _compare_default_fused_to_python(
    *,
    ratio: int,
    q_len: int,
    start_pos_values: list[int],
    state_bt: torch.Tensor,
    kv_bt: Optional[torch.Tensor],
    kv_eb: int,
    pool_rows: int = 0,
):
    positions, b_idx = _make_positions(
        torch.tensor(start_pos_values, dtype=torch.long, device=DEVICE), q_len
    )
    cmp = _shell(
        ratio=ratio,
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=kv_eb,
        pool_rows=pool_rows,
    )

    seq_start_per_req = positions.view(len(start_pos_values), q_len)[:, 0].to(
        torch.int32
    )
    cu_seq_per_req = torch.arange(
        0,
        (len(start_pos_values) + 1) * q_len,
        q_len,
        dtype=torch.int32,
        device=DEVICE,
    )

    with _env("0"):
        py = cmp.prepare_metadata(
            positions,
            b_idx,
            is_batched=q_len > 1,
            seq_start_per_req=seq_start_per_req,
            cu_seq_per_req=cu_seq_per_req,
        )

    with _env(None):
        assert _compressor_meta_fused_enabled()
        fused = cmp.prepare_metadata(
            positions,
            b_idx,
            is_batched=q_len > 1,
            seq_start_per_req=seq_start_per_req,
            cu_seq_per_req=cu_seq_per_req,
        )

    _assert_meta_equal(py, fused)
    return fused


def test_default_enabled():
    with _env(None):
        assert _compressor_meta_fused_enabled()
    with _env("0"):
        assert not _compressor_meta_fused_enabled()


@requires_cuda
def test_ratio4_batched_speculative_q_len_gt_1():
    state_bt = torch.tensor(
        [
            [1, 2, 0, 4],
            [0, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_bt = torch.tensor(
        [
            [1, 2, 3, 4],
            [0, 6, 7, 8],
            [9, 10, 11, 12],
            [999, 14, 15, 16],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    _compare_default_fused_to_python(
        ratio=4,
        q_len=3,
        start_pos_values=[1, 126, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=64,
        # Small enough that block_id=999 is rejected by the overflow guard.
        pool_rows=64 * 64,
    )


@requires_cuda
def test_ratio128_batched_speculative_q_len_gt_1():
    state_bt = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 0, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_bt = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [0, 10, 11, 12],
            [999, 14, 15, 16],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    _compare_default_fused_to_python(
        ratio=128,
        q_len=3,
        start_pos_values=[125, 253, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=2,
        pool_rows=2 * 64,
    )


@requires_cuda
def test_no_kv_context_writes_negative_kv_slots():
    state_bt = torch.tensor(
        [[1, 2], [3, 0]],
        dtype=torch.int32,
        device=DEVICE,
    )
    _compare_default_fused_to_python(
        ratio=4,
        q_len=3,
        start_pos_values=[1, 255],
        state_bt=state_bt,
        kv_bt=None,
        kv_eb=0,
    )


def _ratio4_bt():
    return (
        torch.tensor(
            [[1, 2, 0, 4], [0, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        torch.tensor(
            [[1, 2, 3, 4], [0, 6, 7, 8], [9, 10, 11, 12], [999, 14, 15, 16]],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


def _ratio128_bt():
    return (
        torch.tensor(
            [[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [0, 10, 11, 12], [999, 14, 15, 16]],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


@requires_cuda
def test_ratio4_q_len_1():
    state_bt, kv_bt = _ratio4_bt()
    _compare_default_fused_to_python(
        ratio=4,
        q_len=1,
        start_pos_values=[1, 126, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=64,
        pool_rows=64 * 64,
    )


@requires_cuda
def test_ratio4_q_len_2():
    state_bt, kv_bt = _ratio4_bt()
    _compare_default_fused_to_python(
        ratio=4,
        q_len=2,
        start_pos_values=[1, 126, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=64,
        pool_rows=64 * 64,
    )


@requires_cuda
def test_ratio4_q_len_4():
    state_bt, kv_bt = _ratio4_bt()
    _compare_default_fused_to_python(
        ratio=4,
        q_len=4,
        start_pos_values=[1, 126, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=64,
        pool_rows=64 * 64,
    )


@requires_cuda
def test_ratio128_q_len_1():
    state_bt, kv_bt = _ratio128_bt()
    _compare_default_fused_to_python(
        ratio=128,
        q_len=1,
        start_pos_values=[125, 253, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=2,
        pool_rows=2 * 64,
    )


@requires_cuda
def test_ratio128_q_len_2():
    state_bt, kv_bt = _ratio128_bt()
    _compare_default_fused_to_python(
        ratio=128,
        q_len=2,
        start_pos_values=[125, 253, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=2,
        pool_rows=2 * 64,
    )


@requires_cuda
def test_ratio128_q_len_4():
    state_bt, kv_bt = _ratio128_bt()
    _compare_default_fused_to_python(
        ratio=128,
        q_len=4,
        start_pos_values=[125, 253, 255, 1021],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=2,
        pool_rows=2 * 64,
    )


@requires_cuda
def test_ratio128_q_len_gt_ratio():
    state_bt, kv_bt = _ratio128_bt()
    kv_bt = kv_bt.clone()
    kv_bt[2, 0] = -1
    meta = _compare_default_fused_to_python(
        ratio=128,
        q_len=129,
        start_pos_values=[0, 127, 255, 0],
        state_bt=state_bt,
        kv_bt=kv_bt,
        kv_eb=2,
        pool_rows=2 * 64,
    )
    assert int(meta.kv_slots[127].item()) == 2  # row0: boundary in block 0
    assert int(meta.kv_slots[386].item()) == 20  # row2: crosses into block 1
    assert int(meta.kv_slots[258].item()) == -1  # row2: negative block sentinel
    assert int(meta.kv_slots[514].item()) == -1  # row3: block_id 999 over pool_rows


if __name__ == "__main__":
    test_default_enabled()
    test_ratio4_q_len_1()
    test_ratio4_q_len_2()
    test_ratio4_batched_speculative_q_len_gt_1()
    test_ratio4_q_len_4()
    test_ratio128_q_len_1()
    test_ratio128_q_len_2()
    test_ratio128_batched_speculative_q_len_gt_1()
    test_ratio128_q_len_4()
    test_ratio128_q_len_gt_ratio()
    test_no_kv_context_writes_negative_kv_slots()

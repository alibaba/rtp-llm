"""UT for the fused CompressorFP8.prepare_metadata slot builder.

The decode/speculative path flattens request-major ``[B, S]`` metadata into
``positions[N]`` and ``b_idx[N]``.  This test compares the fused
Triton path against the Python reference path for both CSA/indexer
``ratio=4`` and HCA ``ratio=128`` shapes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _fused_compressor_meta_triton
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8, CompressorMeta

DEVICE = "cuda"
STATE_EB = 256
requires_cuda = unittest.skipUnless(
    torch.cuda.is_available() and _fused_compressor_meta_triton._TRITON_AVAILABLE,
    "CUDA/Triton unavailable",
)


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
    state_eb: int = STATE_EB,
    state_tokens_per_block: int = STATE_EB,
    kv_tokens_per_block: int = 0,
    kv_owner_tokens_per_block: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> CompressorFP8:
    cmp = CompressorFP8.__new__(CompressorFP8)
    object.__setattr__(cmp, "compress_ratio", ratio)
    object.__setattr__(cmp, "_state_block_table", state_bt)
    object.__setattr__(cmp, "_state_eb", state_eb)
    object.__setattr__(cmp, "_state_tokens_per_block", state_tokens_per_block)
    object.__setattr__(cmp, "_kv_block_table", kv_bt)
    object.__setattr__(cmp, "_kv_eb", kv_eb)
    kv_tpb = kv_tokens_per_block or (kv_eb * ratio if kv_eb > 0 else 0)
    object.__setattr__(cmp, "_kv_tokens_per_block", kv_tpb)
    object.__setattr__(
        cmp,
        "_kv_owner_tokens_per_block",
        kv_owner_tokens_per_block or kv_tpb,
    )
    object.__setattr__(cmp, "_kv_cache_sharded", cp_size > 1)
    object.__setattr__(
        cmp,
        "_cp_ctx",
        None if cp_size <= 1 else SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank),
    )
    object.__setattr__(cmp, "_state_pool_3d", None)
    if pool_rows > 0 and kv_eb > 0:
        assert pool_rows % kv_eb == 0
        pool = torch.empty(
            (pool_rows // kv_eb, kv_eb, 1), dtype=torch.uint8, device=DEVICE
        )
    else:
        pool = None
    object.__setattr__(cmp, "_kv_pool_view", pool)
    return cmp


def _assert_meta_equal(py, fused):
    assert torch.equal(py.positions, fused.positions)
    assert torch.equal(py.b_idx, fused.b_idx)
    assert torch.equal(py.state_slots, fused.state_slots)
    assert torch.equal(py.kv_slots, fused.kv_slots)
    assert torch.equal(py.token_to_req, fused.token_to_req)
    assert py.is_batched == fused.is_batched


def _prepare_python_reference(
    cmp: CompressorFP8,
    positions: torch.Tensor,
    b_idx: torch.Tensor,
    *,
    q_len: int,
    start_pos_values: list[int],
) -> CompressorMeta:
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
    seq_end_per_req = seq_start_per_req.to(torch.long) + (
        cu_seq_per_req[1:] - cu_seq_per_req[:-1]
    ).to(torch.long)
    return CompressorMeta(
        positions=positions,
        b_idx=b_idx,
        state_slots=cmp._compute_state_slot_mapping(
            positions,
            b_idx,
            seq_end_per_req,
        ),
        kv_slots=cmp._compute_kv_slot_mapping(positions, b_idx),
        token_to_req=b_idx.to(torch.int32),
        has_prefix=any(v > 0 for v in start_pos_values),
        is_batched=q_len > 1,
        seq_start_per_req=seq_start_per_req,
        cu_seq_per_req=cu_seq_per_req,
    )


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

    py = _prepare_python_reference(
        cmp,
        positions,
        b_idx,
        q_len=q_len,
        start_pos_values=start_pos_values,
    )
    seq_start_per_req = py.seq_start_per_req
    cu_seq_per_req = torch.arange(
        0,
        (len(start_pos_values) + 1) * q_len,
        q_len,
        dtype=torch.int32,
        device=DEVICE,
    )

    fused = cmp.prepare_metadata(
        positions,
        b_idx,
        has_prefix=any(v > 0 for v in start_pos_values),
        is_batched=q_len > 1,
        seq_start_per_req=seq_start_per_req,
        cu_seq_per_req=cu_seq_per_req,
    )

    _assert_meta_equal(py, fused)
    return fused


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
def test_cp_sharded_slot_mapping_matches_python_reference_exact():
    starts = torch.tensor([3, 253, 767], dtype=torch.long, device=DEVICE)
    lengths = (13, 10, 20)
    positions = torch.cat(
        [torch.arange(s, s + n, device=DEVICE) for s, n in zip(starts, lengths)]
    ).to(torch.long)
    b_idx = torch.cat(
        [torch.full((n,), b, device=DEVICE) for b, n in enumerate(lengths)]
    ).to(torch.long)
    cu_seq = torch.tensor(
        [0, lengths[0], lengths[0] + lengths[1], sum(lengths)],
        dtype=torch.int64,
        device=DEVICE,
    )
    state_bt = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 0, 8], [9, 10, 11, 12]],
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_bt = torch.tensor(
        [[13, 14, 15, 16], [17, 0, 19, 20], [21, 22, 23, 24]],
        dtype=torch.int32,
        device=DEVICE,
    )

    for ratio, state_eb, kv_eb, kv_tpb in (
        (4, 3, 16, 64),
        (128, 65, 1, 128),
    ):
        for cp_rank in range(4):
            cmp = _shell(
                ratio=ratio,
                state_bt=state_bt,
                kv_bt=kv_bt,
                kv_eb=kv_eb,
                state_eb=state_eb,
                state_tokens_per_block=256,
                kv_tokens_per_block=kv_tpb,
                kv_owner_tokens_per_block=256,
                cp_size=4,
                cp_rank=cp_rank,
            )
            seq_end = starts + torch.tensor(lengths, device=DEVICE)
            expected_state = cmp._compute_state_slot_mapping(
                positions, b_idx, seq_end
            )
            expected_kv = cmp._compute_kv_slot_mapping(positions, b_idx)
            got = cmp.prepare_metadata(
                positions,
                b_idx,
                has_prefix=True,
                is_batched=True,
                seq_start_per_req=starts,
                cu_seq_per_req=cu_seq,
            )
            assert torch.equal(got.state_slots, expected_state)
            assert torch.equal(got.kv_slots, expected_kv)
            assert torch.equal(got.token_to_req, b_idx.to(torch.int32))


@requires_cuda
def test_cp_runtime_shapes_reuse_one_triton_specialization():
    kernel = _fused_compressor_meta_triton._compressor_slot_mapping_kernel
    kernel.device_caches.clear()
    state_bt = torch.ones((4, 8), dtype=torch.int32, device=DEVICE)
    kv_bt = torch.ones((4, 8), dtype=torch.int32, device=DEVICE)

    def launch(batch_size: int, n: int, ratio: int, cp_rank: int) -> None:
        positions = torch.arange(n, dtype=torch.long, device=DEVICE)
        b_idx = torch.arange(n, device=DEVICE, dtype=torch.long) % batch_size
        counts = torch.bincount(b_idx, minlength=batch_size).to(torch.int32)
        cu_seq = torch.cat(
            [
                torch.zeros(1, device=DEVICE, dtype=torch.int64),
                counts.to(torch.int64).cumsum(0),
            ]
        )
        starts = torch.arange(batch_size, device=DEVICE, dtype=torch.int64) * 257
        _fused_compressor_meta_triton.fused_compressor_slot_mapping(
            positions,
            b_idx,
            state_bt[:batch_size],
            3 if ratio == 4 else 65,
            kv_bt[:batch_size],
            16 if ratio == 4 else 1,
            ratio,
            starts,
            cu_seq,
            256,
            pool_rows=4096,
            kv_tokens_per_block=64 if ratio == 4 else 128,
            cp_size=4,
            cp_rank=cp_rank,
            kv_owner_tokens_per_block=256,
        )
        torch.cuda.synchronize()

    launch(1, 7, 4, 0)
    cache_count = sum(len(cache) for cache in kernel.device_caches.values())
    launch(4, 43, 128, 3)
    assert sum(len(cache) for cache in kernel.device_caches.values()) == cache_count


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
    test_cp_sharded_slot_mapping_matches_python_reference_exact()
    test_cp_runtime_shapes_reuse_one_triton_specialization()

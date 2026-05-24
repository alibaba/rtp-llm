"""UT for the fused CompressorFP8.prepare_metadata slot builder.

The decode/speculative path flattens request-major ``[B, S]`` metadata into
``positions[N]`` and ``b_idx[N]``.  This test compares the default-on fused
Triton path against the Python reference path for both CSA/indexer
``ratio=4`` and HCA ``ratio=128`` shapes.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._fused_compressor_meta_triton import (
    fused_compressor_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8


DEVICE = "cuda"
STATE_EB = 256


def _make_positions(start_pos: torch.Tensor, q_len: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    object.__setattr__(cmp, "_kv_block_table", kv_bt)
    object.__setattr__(cmp, "_kv_eb", kv_eb)
    if pool_rows > 0 and kv_eb > 0:
        assert pool_rows % kv_eb == 0
        pool = torch.empty((pool_rows // kv_eb, kv_eb, 1), dtype=torch.uint8, device=DEVICE)
    else:
        pool = None
    object.__setattr__(cmp, "_kv_pool_3d", pool)
    return cmp



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

    py_state = cmp._compute_state_slot_mapping(positions, b_idx)
    py_kv = cmp._compute_kv_slot_mapping(positions, b_idx)
    py_t2r = b_idx.to(torch.int32)

    fu_state, fu_kv, fu_t2r = fused_compressor_slot_mapping(
        positions,
        b_idx,
        state_bt,
        cmp._state_eb,
        kv_bt,
        kv_eb,
        cmp.compress_ratio,
        pool_rows=pool_rows,
    )

    assert torch.equal(py_state, fu_state), (
        f"state_slots mismatch:\npy={py_state}\nfu={fu_state}"
    )
    assert torch.equal(py_kv, fu_kv), (
        f"kv_slots mismatch:\npy={py_kv}\nfu={fu_kv}"
    )
    assert torch.equal(py_t2r, fu_t2r), (
        f"token_to_req mismatch:\npy={py_t2r}\nfu={fu_t2r}"
    )



def test_ratio4_batched_speculative_q_len_gt_1():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio128_batched_speculative_q_len_gt_1():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_no_kv_context_writes_negative_kv_slots():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio4_q_len_1():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio4_q_len_2():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio4_q_len_4():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio128_q_len_1():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio128_q_len_2():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio128_q_len_4():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
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


def test_ratio128_q_len_gt_ratio():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
    state_bt, kv_bt = _ratio128_bt()
    _compare_default_fused_to_python(
        ratio=128,
        q_len=129,
        start_pos_values=[0, 127],
        state_bt=state_bt[:2],
        kv_bt=kv_bt[:2],
        kv_eb=2,
        pool_rows=2 * 64,
    )


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

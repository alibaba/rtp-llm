"""Real CP2 sparse state and FULL compressed destination writers."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8.sparse_save import (
    SparseSaveTargets,
    iter_sparse_full_kv_slots,
    iter_sparse_save_slots,
)

_WORLD_SIZE = 2


def _check_cp2_state_writer(rank: int, device: torch.device) -> None:
    entries, width = 128, 1024
    pool = torch.full((8, entries, 2 * width), -19.0, device=device)
    pool[1].fill_(17 + rank)
    pool[6].copy_(pool[1])
    pool[7].copy_(pool[1])
    source = pool[1].clone()
    positions = torch.arange(1025, 1030, dtype=torch.int64, device=device)
    request_ids = torch.zeros_like(positions)
    torch.manual_seed(10 + rank)
    kv = torch.randn((5, width), dtype=torch.bfloat16, device=device) * 0.1
    score = torch.randn((5, width), dtype=torch.bfloat16, device=device) * 0.1
    ape = torch.zeros((4, width), dtype=torch.float32, device=device)
    runtime_slots = torch.full_like(positions, -1)
    if rank == 0:
        runtime_slots = 6 * entries + positions % (entries * _WORLD_SIZE)
    targets = SparseSaveTargets(
        torch.tensor([[1030]], dtype=torch.int64),
        torch.tensor([[7]], dtype=torch.int32),
    )
    save_slots = list(
        iter_sparse_save_slots(
            targets,
            positions,
            request_ids,
            entries_per_block=entries,
            tokens_per_block=512,
            runtime_block_table=torch.tensor(
                [[0, 0, 6, 0]], dtype=torch.int32, device=device
            ),
            cp_size=_WORLD_SIZE,
            cp_rank=rank,
        )
    )[0][1]
    expected_slots = torch.full_like(positions, -1)
    if rank == 0:
        expected_slots = 7 * entries + positions % (entries * _WORLD_SIZE)
    torch.testing.assert_close(save_slots, expected_slots, rtol=0, atol=0)
    run_save_partial_states(kv, score, ape, positions, pool, runtime_slots, 4)
    run_save_partial_states(kv, score, ape, positions, pool, save_slots, 4)
    torch.cuda.synchronize(device)

    expected = source.clone()
    if rank == 0:
        rows = positions % entries
        expected[rows, :width] = kv.float()
        expected[rows, width:] = score.float()
    torch.testing.assert_close(pool[1], source, rtol=0, atol=0)
    torch.testing.assert_close(pool[6], expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[7], expected, rtol=0, atol=0)

    gathered = [torch.empty_like(pool[7]) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, pool[7].contiguous())
    if rank == 0:
        torch.testing.assert_close(gathered[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            gathered[1], torch.full_like(source, 18), rtol=0, atol=0
        )


def _check_cp2_full_kv_writer(rank: int, device: torch.device) -> None:
    ratio, physical, kernel, entries = 4, 256, 64, 16
    pool = torch.zeros((32, entries, 584), dtype=torch.uint8, device=device)
    pool[20:24].fill_(37)  # immutable page A for G1300
    pool[24:28].copy_(pool[20:24])  # runtime C
    pool[28:32].copy_(pool[20:24])  # independent D
    source = pool[20:24].clone()
    before = pool[28:32].clone()
    positions = torch.arange(1300, 1320, dtype=torch.int64, device=device)
    req = torch.zeros_like(positions)
    table = torch.zeros((1, 12), dtype=torch.int32, device=device)
    table[0, 8] = 24
    targets = SparseSaveTargets(
        torch.tensor([[1320]], dtype=torch.int64),
        torch.tensor([[7]], dtype=torch.int32),
    )
    slots = list(
        iter_sparse_full_kv_slots(
            targets,
            positions,
            req,
            physical_tokens_per_block=physical,
            kernel_tokens_per_block=kernel,
            entries_per_kernel_block=entries,
            ratio=ratio,
            runtime_block_table=table,
            cp_size=_WORLD_SIZE,
            cp_rank=rank,
        )
    )[0][1]
    expected_slots = torch.full_like(positions, -1)
    if rank == 1:
        valid = ((positions + 1) % ratio) == 0
        expected_slots = torch.where(
            valid, 28 * entries + (positions % kernel) // ratio, -1
        )
    torch.testing.assert_close(slots, expected_slots, rtol=0, atol=0)

    torch.manual_seed(20261018)
    kv_raw = torch.randn((28, 1024), dtype=torch.bfloat16, device=device) * 0.1
    score_raw = torch.randn((28, 1024), dtype=torch.bfloat16, device=device) * 0.1
    cos_sin, _ = build_cos_sin_cache(
        torch.ones((2048, 32), dtype=torch.complex64, device=device)
    )
    run_fused_compress_kv_write(
        torch.zeros((2, 128, 2048), dtype=torch.float32, device=device),
        torch.zeros_like(positions, dtype=torch.int32),
        positions,
        torch.full_like(positions, -1),
        torch.ones((1, 1), dtype=torch.int32, device=device),
        torch.ones(512, dtype=torch.bfloat16, device=device),
        1e-6,
        cos_sin,
        pool,
        slots,
        kv_raw,
        score_raw,
        torch.zeros((4, 1024), dtype=torch.float32, device=device),
        1292,
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=1,
        seq_start_per_req=torch.tensor([1292], dtype=torch.int64, device=device),
        cu_seq_per_req=torch.tensor([0, 28], dtype=torch.int64, device=device),
        state_tokens_per_block=512,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(pool[20:24], source, rtol=0, atol=0)
    if rank == 0:
        torch.testing.assert_close(pool[28:32], before, rtol=0, atol=0)
    else:
        changed = pool[28].flatten()
        prior = before[0].flatten()
        for entry in range(5, 10):
            assert not torch.equal(
                changed[entry * 576 : (entry + 1) * 576],
                prior[entry * 576 : (entry + 1) * 576],
            )
            assert not torch.equal(
                changed[entries * 576 + entry * 8 : entries * 576 + (entry + 1) * 8],
                prior[entries * 576 + entry * 8 : entries * 576 + (entry + 1) * 8],
            )
        for entry in range(5):
            torch.testing.assert_close(
                changed[entry * 576 : (entry + 1) * 576],
                prior[entry * 576 : (entry + 1) * 576],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                changed[entries * 576 + entry * 8 : entries * 576 + (entry + 1) * 8],
                prior[entries * 576 + entry * 8 : entries * 576 + (entry + 1) * 8],
                rtol=0,
                atol=0,
            )
    gathered = [torch.empty_like(pool[28]) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, pool[28].contiguous())
    if rank == 0:
        torch.testing.assert_close(gathered[0], before[0], rtol=0, atol=0)
        assert not torch.equal(gathered[1], before[0])


def _worker(rank: int, init_file: str) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=120),
        )
        _check_cp2_state_writer(rank, device)
        _check_cp2_full_kv_writer(rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class SparseCPWritersDistributedTest(unittest.TestCase):
    def test_cp2_state_and_full_destinations(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")
        with tempfile.TemporaryDirectory(prefix="dsv4_cp_writers_") as temp_dir:
            mp.spawn(
                _worker,
                args=(os.path.join(temp_dir, "nccl_file_store"),),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()

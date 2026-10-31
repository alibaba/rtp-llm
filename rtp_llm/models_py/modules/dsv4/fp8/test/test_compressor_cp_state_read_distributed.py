"""Real two-rank G1025 prefix-history read across fixed-state slots."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    quantize_and_insert_k_cache_cp_byte_sliced,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    compute_swa_slot_mapping_from_positions,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    _build_cp_full_state_read_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.sparse_save import (
    SparseSaveTargets,
    iter_sparse_save_slots,
)

_WORLD_SIZE = 2
_LOCAL_ROWS = 128
_FULL_BLOCK_TOKENS = 512
_FULL_RING_ROWS = _LOCAL_ROWS * _WORLD_SIZE
_HIDDEN = 4


def _expected_row(position: int) -> torch.Tensor:
    return torch.tensor(
        [position, position + 10000, -position, 0.25 * position], dtype=torch.float32
    )


def _check_cp2_swa_sparse_writer(rank: int, device: torch.device) -> None:
    """Both byte owners preserve A→D history while writing the new suffix."""
    entries, stride, blocks = 128, 74880, 8
    local_bytes = stride // _WORLD_SIZE
    torch.manual_seed(20261018)
    old_k = torch.randn((20, 512), dtype=torch.bfloat16, device=device) * 0.1
    new_k = torch.randn((20, 512), dtype=torch.bfloat16, device=device) * 0.1
    reference_raw = torch.zeros((blocks, stride), dtype=torch.uint8, device=device)
    reference_view = reference_raw.as_strided((blocks, entries, 584), (stride, 584, 1))
    old_slots = 5 * entries + torch.arange(20, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(old_k, reference_view, old_slots)
    raw = reference_raw[:, rank * local_bytes : (rank + 1) * local_bytes].contiguous()
    raw[6].copy_(raw[5])  # runtime C copied from immutable A
    raw[7].copy_(raw[5])  # independent save D copied from immutable A
    source = raw[5].clone()
    reference_raw[6].copy_(reference_raw[5])
    reference_raw[7].copy_(reference_raw[5])

    positions = torch.arange(1300, 1320, dtype=torch.int32, device=device)
    request_ids = torch.zeros_like(positions)
    runtime_table = torch.tensor([[0, 0, 0, 0, 0, 6]], dtype=torch.int32, device=device)
    runtime_slots = compute_swa_slot_mapping_from_positions(
        block_table=runtime_table,
        req_id_per_token=request_ids,
        positions=positions,
        seq_lens=torch.tensor([1320], dtype=torch.int32, device=device),
        num_tokens=20,
        pool_entries_per_block=entries,
        tokens_per_block_for_block_table=256,
        ring_entries=entries,
    )
    torch.testing.assert_close(
        runtime_slots, 6 * entries + positions.to(torch.int64) % entries, rtol=0, atol=0
    )
    targets = SparseSaveTargets(
        torch.tensor([[1320]], dtype=torch.int64),
        torch.tensor([[7]], dtype=torch.int32),
    )
    save_slots = list(
        iter_sparse_save_slots(
            targets,
            positions.to(torch.int64),
            request_ids,
            entries_per_block=entries,
            tokens_per_block=256,
            runtime_block_table=runtime_table,
        )
    )[0][1]
    torch.testing.assert_close(
        save_slots, 7 * entries + positions.to(torch.int64) % entries, rtol=0, atol=0
    )
    for name, slots in (("runtime", runtime_slots), ("save", save_slots)):
        compaction = build_cp_byte_sliced_slot_compaction(
            slots,
            full_entries_per_block=entries,
            num_blocks=blocks,
            validation_site=f"test.cp2.swa.{name}",
            negative_mode="skip_minus_one",
        )
        quantize_and_insert_k_cache_cp_byte_sliced(
            new_k,
            raw,
            slots,
            full_entries_per_block=entries,
            cp_rank=rank,
            cp_size=_WORLD_SIZE,
            compaction=compaction,
        )
        quantize_and_insert_k_cache(new_k, reference_view, slots)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(raw[5], source, rtol=0, atol=0)
    for block in (6, 7):
        torch.testing.assert_close(
            raw[block],
            reference_raw[block, rank * local_bytes : (rank + 1) * local_bytes],
            rtol=0,
            atol=0,
        )
    gathered = [torch.empty_like(raw[7]) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, raw[7].contiguous())
    torch.testing.assert_close(torch.cat(gathered), reference_raw[7], rtol=0, atol=0)


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
        collective_torch._group_map.clear()
        collective_torch._group_map[Group.DP_AND_TP] = dist.group.WORLD
        collective_torch._parallelism_config = SimpleNamespace(
            tp_size=_WORLD_SIZE, dp_size=1, world_size=_WORLD_SIZE
        )
        collective_torch._initialized = True

        pool = torch.full((13, _LOCAL_ROWS, _HIDDEN), -777.0, device=device)
        # G1025 crosses logical 512-token blocks 1 and 2. CSA's CP2 state
        # ring holds 256 rows: rank 0 owns offsets 0..127, rank 1 128..255.
        for logical_block, physical_id, ring_start in ((1, 11, 768), (2, 12, 1024)):
            for local_row in range(_LOCAL_ROWS):
                position = ring_start + rank * _LOCAL_ROWS + local_row
                pool[physical_id, local_row].copy_(_expected_row(position).to(device))
        source = pool.clone()
        table = torch.tensor([[0, 11, 12, 0]], dtype=torch.int32, device=device)
        read_cache, read_table = _build_cp_full_state_read_cache(
            pool,
            table,
            cp_size=_WORLD_SIZE,
            seq_start_per_req=torch.tensor([1025], dtype=torch.int64, device=device),
            token_count=8,
            state_tokens_per_block=_FULL_BLOCK_TOKENS,
        )
        torch.cuda.synchronize(device)
        if tuple(read_cache.shape) != (3, _FULL_RING_ROWS, _HIDDEN):
            raise AssertionError(
                f"rank {rank} read cache shape {tuple(read_cache.shape)}"
            )
        if read_table[0].cpu().tolist() != [0, 1, 2, 0]:
            raise AssertionError(f"rank {rank} cross-slot read table differs")
        for read_id, ring_start in ((1, 768), (2, 1024)):
            for offset in range(_FULL_RING_ROWS):
                position = ring_start + offset
                torch.testing.assert_close(
                    read_cache[read_id, offset].cpu(),
                    _expected_row(position),
                    rtol=0,
                    atol=0,
                )
        # The actual compressor history for G1025 spans 1018..1024,
        # including rank 1's previous block and rank 0's new block.
        history = []
        for position in range(1018, 1025):
            logical_block = position // _FULL_BLOCK_TOKENS
            offset = position % _FULL_RING_ROWS
            read_id = int(read_table[0, logical_block % 4].item())
            history.append(read_cache[read_id, offset].cpu())
        torch.testing.assert_close(
            torch.stack(history),
            torch.stack([_expected_row(position) for position in range(1018, 1025)]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(pool, source, rtol=0, atol=0)
        _check_cp2_swa_sparse_writer(rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        collective_torch._group_map.clear()
        collective_torch._parallelism_config = None
        collective_torch._initialized = False


class CompressorCPStateReadDistributedTest(unittest.TestCase):
    def test_cp2_g1025_cross_slot_history_from_real_nccl(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")
        with tempfile.TemporaryDirectory(prefix="dsv4_cp_state_read_") as temp_dir:
            mp.spawn(
                _worker,
                args=(os.path.join(temp_dir, "nccl_file_store"),),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()

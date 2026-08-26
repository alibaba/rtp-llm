"""Two-rank integration test for fused CP indexer-K gather and restore."""

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
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
    build_indexer_cp_chunk_plan,
    prepare_assemble_indexer_k_async,
    start_assemble_indexer_k_async,
    wait_assemble_indexer_k_async,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_gather_triton import (
    try_gather_indexer_k_to_padded,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)

_WORLD_SIZE = 2
_BLOCK_SIZE = 4
_PER_REQ_TOTAL_LENS = (13, 10, 20)


def _expected_payload() -> tuple[torch.Tensor, torch.Tensor]:
    k_rows = []
    scales = []
    dim_offsets = torch.arange(INDEXER_HEAD_DIM, dtype=torch.int64)
    for req_id, req_len in enumerate(_PER_REQ_TOTAL_LENS):
        for token_idx in range(req_len):
            k_rows.append(
                ((dim_offsets + req_id * 37 + token_idx * 11) % 251).to(torch.uint8)
            )
            scales.append(req_id * 100.0 + token_idx + 0.25)
    k_bytes = torch.stack(k_rows).contiguous()
    scale_bytes = (
        torch.tensor(scales, dtype=torch.float32)
        .view(torch.uint8)
        .reshape(-1, 4)
        .contiguous()
    )
    return k_bytes, scale_bytes


def _owned_tokens(req_len: int, rank: int) -> list[int]:
    return [
        token_idx
        for token_idx in range(req_len)
        if (token_idx // _BLOCK_SIZE) % _WORLD_SIZE == rank
    ]


def _rank_cache_payload(
    rank: int, expected_k: torch.Tensor, expected_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    owned_blocks_per_req = []
    for req_len in _PER_REQ_TOTAL_LENS:
        logical_blocks = (req_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        owned_blocks_per_req.append(
            sum(block_idx % _WORLD_SIZE == rank for block_idx in range(logical_blocks))
        )

    max_owned_blocks = max(owned_blocks_per_req)
    block_table = torch.full(
        (len(_PER_REQ_TOTAL_LENS), max_owned_blocks), -1, dtype=torch.int32
    )
    next_physical_block = 0
    for req_id, block_count in enumerate(owned_blocks_per_req):
        if block_count > 0:
            block_table[req_id, :block_count] = torch.arange(
                next_physical_block,
                next_physical_block + block_count,
                dtype=torch.int32,
            )
            next_physical_block += block_count

    pool = torch.full(
        (max(next_physical_block, 1), _BLOCK_SIZE, INDEXER_ENTRY_BYTES),
        0xCD,
        dtype=torch.uint8,
    )
    req_global_start = 0
    actual_lens = []
    for req_id, req_len in enumerate(_PER_REQ_TOTAL_LENS):
        owned = _owned_tokens(req_len, rank)
        actual_lens.append(len(owned))
        for token_idx in owned:
            logical_block = token_idx // _BLOCK_SIZE
            local_block = logical_block // _WORLD_SIZE
            token_in_block = token_idx % _BLOCK_SIZE
            physical_block = int(block_table[req_id, local_block])
            block_bytes = pool[physical_block].reshape(-1)
            k_start = token_in_block * INDEXER_HEAD_DIM
            block_bytes[k_start : k_start + INDEXER_HEAD_DIM].copy_(
                expected_k[req_global_start + token_idx]
            )
            scale_start = _BLOCK_SIZE * INDEXER_HEAD_DIM + token_in_block * 4
            block_bytes[scale_start : scale_start + 4].copy_(
                expected_scale[req_global_start + token_idx]
            )
        req_global_start += req_len
    return pool, block_table, tuple(actual_lens)


def _install_world_as_tp_group() -> None:
    collective_torch._group_map.clear()
    collective_torch._group_map[Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=_WORLD_SIZE,
        dp_size=1,
        world_size=_WORLD_SIZE,
    )
    collective_torch._initialized = True


def _clear_collective_torch_state() -> None:
    collective_torch._group_map.clear()
    collective_torch._parallelism_config = None
    collective_torch._initialized = False


def _fused_gather_allgather_restore_worker(
    rank: int, world_size: int, init_file: str
) -> None:
    if world_size != _WORLD_SIZE:
        raise AssertionError(f"expected world_size={_WORLD_SIZE}, got {world_size}")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["DSV4_TRAP_INVALID_KV_ACCESS"] = "0"

    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        _install_world_as_tp_group()

        expected_k, expected_scale = _expected_payload()
        pool_cpu, block_table_cpu, expected_actual_lens = _rank_cache_payload(
            rank, expected_k, expected_scale
        )
        per_req_total_lens = torch.tensor(
            _PER_REQ_TOTAL_LENS, dtype=torch.int64, device=device
        )
        cp_ctx = SimpleNamespace(cp_size=world_size, cp_rank=rank)
        plan = build_indexer_cp_chunk_plan(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req_total_lens,
            block_size=_BLOCK_SIZE,
            owner_block_size=_BLOCK_SIZE,
            total_kv_len=sum(_PER_REQ_TOTAL_LENS),
            device=device,
        )
        actual_lens = tuple(plan.per_req_actual_local_kv_lens.cpu().tolist())
        if actual_lens != expected_actual_lens:
            raise AssertionError(
                f"rank {rank} actual lens mismatch: "
                f"{actual_lens} != {expected_actual_lens}"
            )

        pool_source = pool_cpu.to(device=device)
        pool = torch.empty_like(pool_source)
        block_table = block_table_cpu.to(device=device)
        local_q = torch.empty(
            (plan.total_local_T, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        local_scale = torch.empty(
            (plan.total_local_T, 4), dtype=torch.uint8, device=device
        )
        total_tokens = sum(_PER_REQ_TOTAL_LENS)
        out_q = torch.empty(
            (total_tokens, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        out_scale = torch.empty((total_tokens, 4), dtype=torch.uint8, device=device)
        out_q.view(torch.uint8).fill_(0xA5)
        out_scale.fill_(0xA5)

        producer_stream = torch.cuda.Stream(device=device)
        gather_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        producer_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(producer_stream):
            pool.copy_(pool_source)
            fused = try_gather_indexer_k_to_padded(
                pool,
                block_table,
                plan.per_req_local_kv_lens,
                plan.per_req_actual_local_kv_lens,
                local_q,
                local_scale,
                total_actual_tokens=plan.total_actual_local_T,
            )
            if not fused:
                raise AssertionError(f"rank {rank} fused gather was not selected")
            handle = start_assemble_indexer_k_async(
                plan=plan,
                local_k_quant=local_q,
                local_k_scale=local_scale,
                out_k_quant=out_q,
                out_k_scale=out_scale,
                stream=gather_stream,
            )
        if handle is None:
            raise AssertionError(f"rank {rank} async NCCL gather did not start")

        prepare_assemble_indexer_k_async(handle, stream=post_stream)
        wait_assemble_indexer_k_async(handle)
        torch.cuda.synchronize(device)

        actual_q = out_q.view(torch.uint8).cpu()
        actual_scale = out_scale.cpu()
        if not torch.equal(actual_q, expected_k):
            raise AssertionError(f"rank {rank} restored K bytes differ")
        if not torch.equal(actual_scale, expected_scale):
            raise AssertionError(f"rank {rank} restored scale bytes differ")

        local_q_bytes = local_q.view(torch.uint8).cpu()
        local_scale_bytes = local_scale.cpu()
        padded_lens = tuple(plan.per_req_local_kv_lens.cpu().tolist())
        if not any(
            padded_len > actual_len
            for padded_len, actual_len in zip(padded_lens, expected_actual_lens)
        ):
            raise AssertionError(f"rank {rank} did not exercise a padding tail")
        padded_start = 0
        for req_id, (padded_len, actual_len) in enumerate(
            zip(padded_lens, expected_actual_lens)
        ):
            tail = slice(padded_start + actual_len, padded_start + padded_len)
            if int(local_q_bytes[tail].count_nonzero()) != 0:
                raise AssertionError(f"rank {rank} request {req_id} K tail is not zero")
            if int(local_scale_bytes[tail].count_nonzero()) != 0:
                raise AssertionError(
                    f"rank {rank} request {req_id} scale tail is not zero"
                )
            padded_start += padded_len
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
        _clear_collective_torch_state()


class IndexerCPPaddedGatherDistributedTest(unittest.TestCase):
    def test_two_rank_side_stream_allgather_restore_byte_exact(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")

        with tempfile.TemporaryDirectory(prefix="dsv4_indexer_cp_dist_") as temp_dir:
            init_file = os.path.join(temp_dir, "nccl_file_store")
            mp.spawn(
                _fused_gather_allgather_restore_worker,
                args=(_WORLD_SIZE, init_file),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()

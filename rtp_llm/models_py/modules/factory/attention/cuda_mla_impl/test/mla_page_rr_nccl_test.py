"""Mandatory TP2 CUDA/NCCL gate for MLA page-RR cache write and restore."""

from __future__ import annotations

import multiprocessing as mp
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashMLAPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
)
from rtp_llm.ops import KvCacheDataType, NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager

_WORLD_SIZE = 2
_PAGE_TOKENS = 128
_KV_LORA_RANK = 512
_ROPE_HEAD_DIM = 64
_FP8_KV_SCALE = 0.5


class _CanonicalPrefixCapture:
    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: SimpleNamespace,
        layer_id: int,
        *,
        canonical_prefix_kv: torch.Tensor,
    ) -> torch.Tensor:
        return canonical_prefix_kv


def _canonical_rows(
    device: torch.device, prefix_lens
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = sum(prefix_lens)
    token_ids = torch.arange(token_count, dtype=torch.int64, device=device).unsqueeze(1)
    compressed_columns = torch.arange(
        _KV_LORA_RANK, dtype=torch.int64, device=device
    ).unsqueeze(0)
    rope_columns = torch.arange(
        _ROPE_HEAD_DIM, dtype=torch.int64, device=device
    ).unsqueeze(0)
    compressed_kv = torch.remainder(token_ids * 7 + compressed_columns, 251).to(
        torch.bfloat16
    )
    k_pe = torch.remainder(token_ids * 11 + rope_columns + 3, 241).to(torch.bfloat16)
    return compressed_kv, k_pe


def _local_block_table(rank: int, device: torch.device) -> torch.Tensor:
    tables = (
        ((7, 2), (9, 4)),
        ((6, 3), (8, -1)),
    )
    return torch.tensor(tables[rank], dtype=torch.int32, device=device)


def _tp2_page_rr_worker(rank: int, init_port: int, physical_page_tokens: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    parallelism = ParallelismConfig()
    parallelism.world_rank = rank
    parallelism.world_size = _WORLD_SIZE
    parallelism.local_world_size = _WORLD_SIZE
    parallelism.local_rank = rank
    parallelism.tp_size = _WORLD_SIZE
    parallelism.dp_size = 1
    base_port = init_port + 11
    init_distributed_environment(
        parallelism,
        nccl_comm_config=NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        ),
        nccl_init_port=init_port,
        backend="nccl",
        timeout=60,
    )
    try:
        prefix_lens = (3 * physical_page_tokens + 1, 2 * physical_page_tokens + 1)
        subpages = physical_page_tokens // _PAGE_TOKENS
        adapter = MlaPageRRCacheAdapter(
            page_tokens=physical_page_tokens,
            kernel_page_tokens=_PAGE_TOKENS,
            shard_size=_WORLD_SIZE,
            shard_rank=rank,
        )
        physical_table = _local_block_table(rank, device)
        block_table = (
            (
                physical_table[..., None] * subpages
                + torch.arange(subpages, device=device)
            )
            .flatten(1)
            .to(torch.int32)
        )
        block_table.masked_fill_(
            physical_table.repeat_interleave(subpages, dim=1) < 0, -1
        )
        adapter.validate_block_table_capacity(block_table, prefix_lens)

        positions = torch.cat(
            [
                torch.arange(prefix_len, dtype=torch.int32, device=device)
                for prefix_len in prefix_lens
            ]
        )
        batch_indices = torch.cat(
            [
                torch.full((prefix_len,), request_idx, dtype=torch.int32, device=device)
                for request_idx, prefix_len in enumerate(prefix_lens)
            ]
        )
        slot_mapping = adapter.slot_mapping(
            positions,
            batch_indices,
            block_table,
        )
        expected_owned = sum(
            min(
                physical_page_tokens,
                max(0, prefix_len - global_page * physical_page_tokens),
            )
            for prefix_len in prefix_lens
            for global_page in range(
                rank,
                (prefix_len + physical_page_tokens - 1) // physical_page_tokens,
                _WORLD_SIZE,
            )
        )
        assert int((slot_mapping >= 0).sum().item()) == expected_owned

        compressed_kv, k_pe = _canonical_rows(device, prefix_lens)
        expected = torch.cat((compressed_kv, k_pe), dim=1)
        payload_features = _KV_LORA_RANK + _ROPE_HEAD_DIM
        for raw_dtype, cache_dtype, fp8_compute, kv_scale in (
            (torch.bfloat16, KvCacheDataType.BASE, False, 1.0),
            (torch.float8_e4m3fn, KvCacheDataType.FP8, True, _FP8_KV_SCALE),
        ):
            raw_cache = torch.full(
                (10 * subpages, _PAGE_TOKENS, payload_features),
                -123.0,
                dtype=raw_dtype,
                device=device,
            )
            kv_cache = SimpleNamespace(kv_cache_base=raw_cache)
            MlaKVCacheWriteOp(
                cache_dtype,
                fp8_compute=fp8_compute,
                kv_scale=kv_scale,
            ).forward(
                compressed_kv,
                k_pe,
                kv_cache,
                SimpleNamespace(slot_mapping=slot_mapping),
                slot_mapping_override=slot_mapping,
            )
            torch.cuda.synchronize(device)

            with patch.object(
                collective_torch,
                "all_gather_into",
                wraps=collective_torch.all_gather_into,
            ) as gather:
                empty = adapter.read_prefix(raw_cache, block_table, (0, 0))
                assert empty.shape == (0, payload_features)
                assert gather.call_count == 0

                impl = object.__new__(MlaFlashMLAPrefillImpl)
                impl.fmha_impl = _CanonicalPrefixCapture()
                impl.fmha_params = SimpleNamespace(prefix_lens_host=prefix_lens)
                impl.page_rr_cache_adapter = adapter
                impl.attn_inputs = SimpleNamespace(
                    kv_cache_kernel_block_id_device=block_table
                )
                impl.attn_configs = SimpleNamespace(
                    kv_lora_rank=_KV_LORA_RANK,
                    rope_head_dim=_ROPE_HEAD_DIM,
                    mla_fp8_compute=fp8_compute,
                    mla_fp8_kv_scale=kv_scale,
                )
                restored = impl.compute_prefill_context(
                    torch.empty((0,), dtype=torch.bfloat16, device=device),
                    torch.empty((0,), dtype=torch.bfloat16, device=device),
                    torch.empty((0,), dtype=torch.bfloat16, device=device),
                    kv_cache,
                    0,
                )
                assert gather.call_count == 1

            if fp8_compute:
                expected_dequantized = (
                    expected.float()
                    .div(_FP8_KV_SCALE)
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                    .to(torch.bfloat16)
                    .mul(_FP8_KV_SCALE)
                )
                assert restored.dtype == torch.bfloat16
                torch.testing.assert_close(
                    restored, expected_dequantized, rtol=0, atol=0
                )
            else:
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)

        # Request 1 has three global pages. Rank 1 therefore contributes one
        # real page plus a padded local tail slot; pack/restore must never read
        # the -1 block-table entry.
        if rank == 1:
            assert int(block_table[1, subpages].item()) == -1
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
    finally:
        destroy_distributed_environment()


class MlaPageRRNCCLTest(unittest.TestCase):
    def test_tp2_bf16_and_fp8_owner_write_gather_and_canonical_restore(self) -> None:
        self._run_topology(128)

    def test_tp2_split_pages_bf16_and_fp8_write_gather_and_restore(self) -> None:
        self._run_topology(1024)

    def test_tp2_512_physical_pages_bf16_and_fp8_write_gather_and_restore(self) -> None:
        self._run_topology(512)

    def test_tp2_8192_physical_pages_bf16_and_fp8_write_gather_and_restore(
        self,
    ) -> None:
        self._run_topology(8192)

    def _run_topology(self, physical_page_tokens: int) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            raise RuntimeError(
                "mla_page_rr_nccl_test is a mandatory two-GPU CUDA/NCCL gate"
            )
        mp.set_start_method("spawn", force=True)
        port_manager = PortManager()
        ports, locks = port_manager.get_consecutive_ports(1)
        started_processes = []
        try:
            processes = [
                mp.Process(
                    target=_tp2_page_rr_worker,
                    args=(rank, ports[0], physical_page_tokens),
                    name=f"mla-page-rr-rank-{rank}",
                )
                for rank in range(_WORLD_SIZE)
            ]
            for process in processes:
                process.start()
                started_processes.append(process)
            for process in processes:
                process.join(timeout=120)
                if process.is_alive():
                    self.fail(f"{process.name} timed out")
                self.assertEqual(process.exitcode, 0, process.name)
        finally:
            for process in started_processes:
                if process.is_alive():
                    process.terminate()
            for process in started_processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()

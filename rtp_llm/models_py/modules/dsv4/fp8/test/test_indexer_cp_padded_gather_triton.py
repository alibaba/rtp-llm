"""Exactness and JIT-shape tests for the fused CP indexer-K gather."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
    copy_actual_indexer_k_to_padded,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_gather_triton import (
    _cp_gather_indexer_k_to_padded_kernel,
    try_gather_indexer_k_to_padded,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    HEAD_DIM,
    _restore_dequantize_scatter_packed_k_cache_flat_kernel,
    try_restore_dequantize_scatter_packed_k_cache_flat,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _triton_cache_size(kernel_fn) -> int:
    return sum(
        len(kernel_cache) for kernel_cache, *_ in kernel_fn.device_caches.values()
    )


class IndexerCPPaddedGatherTritonTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or not hasattr(
            rtp_llm_ops, "cp_gather_indexer_k_quant_cache"
        ):
            self.skipTest("CUDA and cp_gather_indexer_k_quant_cache are required")
        self.device = torch.device("cuda")
        torch.manual_seed(20260821)

    def _make_pool(
        self, num_blocks: int, block_size: int, block_padding: int
    ) -> torch.Tensor:
        block_bytes = block_size * INDEXER_ENTRY_BYTES
        storage = torch.empty(
            (num_blocks, block_bytes + block_padding),
            dtype=torch.uint8,
            device=self.device,
        )
        pool = torch.as_strided(
            storage,
            (num_blocks, block_size, INDEXER_ENTRY_BYTES),
            (block_bytes + block_padding, INDEXER_ENTRY_BYTES, 1),
        )
        pool.copy_(
            torch.randint(
                0,
                256,
                pool.shape,
                dtype=torch.uint8,
                device=self.device,
            )
        )
        return pool

    def _run_exact_case(
        self,
        *,
        padded_lens: list[int],
        actual_lens: list[int],
        block_size: int,
        lens_dtype: torch.dtype,
        block_padding: int = 0,
    ) -> None:
        self.assertEqual(len(padded_lens), len(actual_lens))
        batch_size = len(padded_lens)
        blocks_per_request = [
            (length + block_size - 1) // block_size for length in actual_lens
        ]
        max_blocks = max(max(blocks_per_request), 1)
        num_cache_blocks = 1 + sum(blocks_per_request)
        pool = self._make_pool(num_cache_blocks, block_size, block_padding)
        block_table = torch.full(
            (batch_size, max_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        physical_block = 1
        for request, request_blocks in enumerate(blocks_per_request):
            if request_blocks:
                block_table[request, :request_blocks] = torch.arange(
                    physical_block,
                    physical_block + request_blocks,
                    dtype=torch.int32,
                    device=self.device,
                )
                physical_block += request_blocks

        padded = torch.tensor(
            padded_lens, dtype=lens_dtype, device=self.device
        ).contiguous()
        actual = torch.tensor(
            actual_lens, dtype=lens_dtype, device=self.device
        ).contiguous()
        total_padded = sum(padded_lens)
        total_actual = sum(actual_lens)

        ref_q = torch.zeros(
            (total_padded, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        ref_scale = torch.zeros(
            (total_padded, 4), dtype=torch.uint8, device=self.device
        )
        if total_actual:
            actual_cu = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            actual_cu[1:] = torch.cumsum(actual.to(torch.int32), dim=0)
            compact_q = torch.empty(
                (total_actual, INDEXER_HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            compact_scale = torch.empty(
                (total_actual, 4), dtype=torch.uint8, device=self.device
            )
            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                pool,
                compact_q,
                compact_scale,
                block_table,
                actual_cu,
            )
            plan = SimpleNamespace(
                total_local_T=total_padded,
                total_actual_local_T=total_actual,
                per_req_local_kv_lens=padded,
                per_req_actual_local_kv_lens=actual,
            )
            copy_actual_indexer_k_to_padded(
                plan=plan,
                actual_k_quant=compact_q,
                actual_k_scale=compact_scale,
                padded_k_quant=ref_q,
                padded_k_scale=ref_scale,
            )

        fused_q = torch.empty_like(ref_q)
        fused_scale = torch.empty_like(ref_scale)
        with mock.patch.dict(
            os.environ,
            {"DSV4_TRAP_INVALID_KV_ACCESS": "0"},
        ):
            fused = try_gather_indexer_k_to_padded(
                pool,
                block_table,
                padded,
                actual,
                fused_q,
                fused_scale,
                total_actual_tokens=total_actual,
            )
        self.assertTrue(fused)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_q.view(torch.uint8), ref_q.view(torch.uint8)))
        self.assertTrue(torch.equal(fused_scale, ref_scale))

        padded_start = 0
        for padded_len, actual_len in zip(padded_lens, actual_lens):
            tail = slice(padded_start + actual_len, padded_start + padded_len)
            self.assertEqual(int(fused_q.view(torch.uint8)[tail].count_nonzero()), 0)
            self.assertEqual(int(fused_scale[tail].count_nonzero()), 0)
            padded_start += padded_len

    def test_matches_old_gather_and_scatter_byte_exact(self) -> None:
        cases = (
            ([64], [7], 64, torch.int64, 0),
            ([8, 4], [5, 0], 4, torch.int64, 0),
            ([64, 64, 64, 64], [2, 2, 2, 2], 64, torch.int64, 256),
            ([8, 16, 8, 8], [0, 1, 8, 7], 8, torch.int64, 128),
            ([8, 4, 8], [3, 0, 7], 4, torch.int64, 64),
            ([4] * 5, [1, 4, 0, 3, 2], 4, torch.int64, 0),
            ([4] * 8, [0, 1, 2, 3, 4, 1, 0, 4], 4, torch.int64, 0),
            ([4] * 63, [i % 5 for i in range(63)], 4, torch.int64, 0),
            ([4, 4, 4, 4], [0, 0, 0, 0], 4, torch.int64, 0),
        )
        for padded, actual, block_size, dtype, block_padding in cases:
            with self.subTest(
                batch_size=len(padded),
                actual=actual,
                block_size=block_size,
                dtype=dtype,
                block_padding=block_padding,
            ):
                self._run_exact_case(
                    padded_lens=padded,
                    actual_lens=actual,
                    block_size=block_size,
                    lens_dtype=dtype,
                    block_padding=block_padding,
                )

    def test_unsupported_input_leaves_outputs_untouched(self) -> None:
        pool = torch.zeros(
            (2, 4, INDEXER_ENTRY_BYTES), dtype=torch.uint8, device=self.device
        )
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=self.device)
        padded = torch.tensor([2], dtype=torch.int64, device=self.device)
        actual = torch.tensor([1], dtype=torch.int64, device=self.device)
        out_q = torch.full(
            (2, INDEXER_HEAD_DIM),
            1.0,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        out_scale = torch.full((2, 4), 0xA5, dtype=torch.uint8, device=self.device)
        q_before = out_q.view(torch.uint8).clone()
        scale_before = out_scale.clone()
        self.assertFalse(
            try_gather_indexer_k_to_padded(
                pool,
                block_table,
                padded.to(torch.int32),
                actual.to(torch.int32),
                out_q,
                out_scale,
                total_actual_tokens=1,
            )
        )
        self.assertTrue(torch.equal(out_q.view(torch.uint8), q_before))
        self.assertTrue(torch.equal(out_scale, scale_before))

    def test_current_stream_orders_cache_write_before_gather(self) -> None:
        block_size = 4
        pool = torch.zeros(
            (2, block_size, INDEXER_ENTRY_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        source = torch.randint(
            0, 256, pool.shape, dtype=torch.uint8, device=self.device
        )
        block_table = torch.ones((1, 1), dtype=torch.int32, device=self.device)
        padded = torch.tensor([4], dtype=torch.int64, device=self.device)
        actual = torch.tensor([4], dtype=torch.int64, device=self.device)
        out_q = torch.empty(
            (4, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        out_scale = torch.empty((4, 4), dtype=torch.uint8, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream), mock.patch.dict(
            os.environ,
            {"DSV4_TRAP_INVALID_KV_ACCESS": "0"},
        ):
            pool.copy_(source)
            self.assertTrue(
                try_gather_indexer_k_to_padded(
                    pool,
                    block_table,
                    padded,
                    actual,
                    out_q,
                    out_scale,
                    total_actual_tokens=4,
                )
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        expected_q = (
            source[1]
            .reshape(-1)[: block_size * INDEXER_HEAD_DIM]
            .reshape(block_size, INDEXER_HEAD_DIM)
        )
        scale_offset = block_size * INDEXER_HEAD_DIM
        expected_scale = (
            source[1]
            .reshape(-1)[scale_offset : scale_offset + block_size * 4]
            .reshape(block_size, 4)
        )
        self.assertTrue(torch.equal(out_q.view(torch.uint8), expected_q))
        self.assertTrue(torch.equal(out_scale, expected_scale))

    def test_hot_path_is_one_kernel_without_torch_scatter_ops(self) -> None:
        pool = torch.zeros(
            (8, 64, INDEXER_ENTRY_BYTES), dtype=torch.uint8, device=self.device
        )
        block_table = torch.arange(1, 5, dtype=torch.int32, device=self.device).reshape(
            4, 1
        )
        padded = torch.full((4,), 64, dtype=torch.int64, device=self.device)
        actual = torch.full((4,), 2, dtype=torch.int64, device=self.device)
        out_q = torch.empty(
            (256, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        out_scale = torch.empty((256, 4), dtype=torch.uint8, device=self.device)

        def run() -> None:
            self.assertTrue(
                try_gather_indexer_k_to_padded(
                    pool,
                    block_table,
                    padded,
                    actual,
                    out_q,
                    out_scale,
                    total_actual_tokens=8,
                )
            )

        run()
        run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        torch.cuda.synchronize()

        events = list(prof.key_averages())
        fused = [
            event
            for event in events
            if "_cp_gather_indexer_k_to_padded_kernel" in event.key
        ]
        self.assertEqual(sum(event.count for event in fused), 1)
        keys = {event.key for event in events}
        for forbidden in (
            "aten::zeros",
            "aten::arange",
            "aten::repeat_interleave",
            "aten::cumsum",
            "aten::index_select",
            "aten::sub",
            "aten::add",
            "aten::index_copy_",
            "cp_gather_indexer_k_quant_cache",
        ):
            self.assertNotIn(forbidden, keys)

    def test_warmup_shapes_cover_production_shapes_without_recompile(self) -> None:
        _restore_dequantize_scatter_packed_k_cache_flat_kernel.device_caches.clear()
        _cp_gather_indexer_k_to_padded_kernel.device_caches.clear()
        with mock.patch.dict(
            os.environ,
            {"DSV4_TRAP_INVALID_KV_ACCESS": "0"},
        ):
            warm_lens = torch.full((4,), 2, dtype=torch.int32, device=self.device)
            self.assertTrue(
                try_restore_dequantize_scatter_packed_k_cache_flat(
                    torch.empty(
                        (4, 2, HEAD_DIM), dtype=torch.bfloat16, device=self.device
                    ),
                    torch.zeros(
                        (8, ENTRY_BYTES), dtype=torch.uint8, device=self.device
                    ),
                    torch.arange(8, dtype=torch.int64, device=self.device),
                    warm_lens,
                    0,
                )
            )
            torch.cuda.synchronize()
            pool_cache_after_warmup = _triton_cache_size(
                _restore_dequantize_scatter_packed_k_cache_flat_kernel
            )
            self.assertEqual(pool_cache_after_warmup, 1)

            prod_lens = torch.full((4,), 2, dtype=torch.int32, device=self.device)
            self.assertTrue(
                try_restore_dequantize_scatter_packed_k_cache_flat(
                    torch.empty(
                        (4, 11, HEAD_DIM), dtype=torch.bfloat16, device=self.device
                    ),
                    torch.zeros(
                        (1024, ENTRY_BYTES), dtype=torch.uint8, device=self.device
                    ),
                    torch.arange(8, dtype=torch.int64, device=self.device),
                    prod_lens,
                    0,
                )
            )
            torch.cuda.synchronize()
            self.assertEqual(
                _triton_cache_size(
                    _restore_dequantize_scatter_packed_k_cache_flat_kernel
                ),
                pool_cache_after_warmup,
            )

            strided_out = torch.empty_strided(
                (4, 11, HEAD_DIM),
                (11 * 513 + 1, 513, 1),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.assertTrue(
                try_restore_dequantize_scatter_packed_k_cache_flat(
                    strided_out,
                    torch.zeros(
                        (1024, ENTRY_BYTES), dtype=torch.uint8, device=self.device
                    ),
                    torch.arange(8, dtype=torch.int64, device=self.device),
                    prod_lens,
                    0,
                )
            )
            torch.cuda.synchronize()
            self.assertEqual(
                _triton_cache_size(
                    _restore_dequantize_scatter_packed_k_cache_flat_kernel
                ),
                pool_cache_after_warmup,
            )

            divisible_lens = torch.full((4,), 4, dtype=torch.int32, device=self.device)
            self.assertTrue(
                try_restore_dequantize_scatter_packed_k_cache_flat(
                    torch.empty(
                        (4, 17, HEAD_DIM), dtype=torch.bfloat16, device=self.device
                    ),
                    torch.zeros(
                        (1024, ENTRY_BYTES), dtype=torch.uint8, device=self.device
                    ),
                    torch.arange(16, dtype=torch.int64, device=self.device),
                    divisible_lens,
                    0,
                )
            )
            torch.cuda.synchronize()
            self.assertEqual(
                _triton_cache_size(
                    _restore_dequantize_scatter_packed_k_cache_flat_kernel
                ),
                pool_cache_after_warmup,
            )

            warm_pool = torch.zeros(
                (2, 4, INDEXER_ENTRY_BYTES), dtype=torch.uint8, device=self.device
            )
            warm_bt = torch.zeros((4, 1), dtype=torch.int32, device=self.device)
            warm_padded = torch.full((4,), 2, dtype=torch.int64, device=self.device)
            warm_actual = torch.ones(4, dtype=torch.int64, device=self.device)
            self.assertTrue(
                try_gather_indexer_k_to_padded(
                    warm_pool,
                    warm_bt,
                    warm_padded,
                    warm_actual,
                    torch.empty(
                        (8, INDEXER_HEAD_DIM),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    ),
                    torch.empty((8, 4), dtype=torch.uint8, device=self.device),
                    total_actual_tokens=4,
                )
            )
            torch.cuda.synchronize()
            indexer_cache_after_warmup = _triton_cache_size(
                _cp_gather_indexer_k_to_padded_kernel
            )
            self.assertEqual(indexer_cache_after_warmup, 1)

            self.assertTrue(
                try_gather_indexer_k_to_padded(
                    warm_pool,
                    warm_bt[:3],
                    warm_padded[:3],
                    warm_actual[:3],
                    torch.empty(
                        (6, INDEXER_HEAD_DIM),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    ),
                    torch.empty((6, 4), dtype=torch.uint8, device=self.device),
                    total_actual_tokens=3,
                )
            )
            torch.cuda.synchronize()
            self.assertEqual(
                _triton_cache_size(_cp_gather_indexer_k_to_padded_kernel),
                indexer_cache_after_warmup,
            )

            prod_pool = torch.zeros(
                (1024, 64, INDEXER_ENTRY_BYTES),
                dtype=torch.uint8,
                device=self.device,
            )
            prod_bt = torch.zeros((4, 128), dtype=torch.int32, device=self.device)
            prod_padded = torch.full((4,), 64, dtype=torch.int64, device=self.device)
            prod_actual = torch.full((4,), 2, dtype=torch.int64, device=self.device)
            self.assertTrue(
                try_gather_indexer_k_to_padded(
                    prod_pool,
                    prod_bt,
                    prod_padded,
                    prod_actual,
                    torch.empty(
                        (256, INDEXER_HEAD_DIM),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    ),
                    torch.empty((256, 4), dtype=torch.uint8, device=self.device),
                    total_actual_tokens=8,
                )
            )
            torch.cuda.synchronize()
            self.assertEqual(
                _triton_cache_size(_cp_gather_indexer_k_to_padded_kernel),
                indexer_cache_after_warmup,
            )

            for expected_cache_size, (batch_block, runtime_batch) in enumerate(
                ((8, 5), (64, 63)), start=2
            ):
                pool = torch.zeros(
                    (2, 4, INDEXER_ENTRY_BYTES),
                    dtype=torch.uint8,
                    device=self.device,
                )
                block_table = torch.zeros(
                    (batch_block, 1), dtype=torch.int32, device=self.device
                )
                padded = torch.full(
                    (batch_block,), 2, dtype=torch.int64, device=self.device
                )
                actual = torch.ones(batch_block, dtype=torch.int64, device=self.device)
                self.assertTrue(
                    try_gather_indexer_k_to_padded(
                        pool,
                        block_table,
                        padded,
                        actual,
                        torch.empty(
                            (2 * batch_block, INDEXER_HEAD_DIM),
                            dtype=torch.float8_e4m3fn,
                            device=self.device,
                        ),
                        torch.empty(
                            (2 * batch_block, 4),
                            dtype=torch.uint8,
                            device=self.device,
                        ),
                        total_actual_tokens=batch_block,
                    )
                )
                torch.cuda.synchronize()
                cache_after_bucket_warmup = _triton_cache_size(
                    _cp_gather_indexer_k_to_padded_kernel
                )
                self.assertEqual(cache_after_bucket_warmup, expected_cache_size)

                self.assertTrue(
                    try_gather_indexer_k_to_padded(
                        pool,
                        block_table[:runtime_batch],
                        padded[:runtime_batch],
                        actual[:runtime_batch],
                        torch.empty(
                            (2 * runtime_batch, INDEXER_HEAD_DIM),
                            dtype=torch.float8_e4m3fn,
                            device=self.device,
                        ),
                        torch.empty(
                            (2 * runtime_batch, 4),
                            dtype=torch.uint8,
                            device=self.device,
                        ),
                        total_actual_tokens=runtime_batch,
                    )
                )
                torch.cuda.synchronize()
                self.assertEqual(
                    _triton_cache_size(_cp_gather_indexer_k_to_padded_kernel),
                    cache_after_bucket_warmup,
                )


if __name__ == "__main__":
    unittest.main()

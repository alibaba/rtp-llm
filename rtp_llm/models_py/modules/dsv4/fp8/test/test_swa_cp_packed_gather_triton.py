"""Exactness and launch-count tests for direct CP SWA packed gather."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    _gather_k_cache_packed_to_flat_kernel,
    gather_k_cache_packed,
    try_gather_k_cache_packed_to_flat,
)


def _triton_cache_size(kernel_fn) -> int:
    return sum(
        len(kernel_cache) for kernel_cache, *_ in kernel_fn.device_caches.values()
    )


class SwaCPPackedGatherTritonTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = torch.device("cuda")
        torch.manual_seed(20260826)

    def _make_pool(
        self, num_blocks: int, block_size: int, block_padding: int
    ) -> torch.Tensor:
        block_bytes = block_size * ENTRY_BYTES
        storage = torch.randint(
            0,
            256,
            (num_blocks, block_bytes + block_padding),
            dtype=torch.uint8,
            device=self.device,
        )
        return torch.as_strided(
            storage,
            (num_blocks, block_size, ENTRY_BYTES),
            (block_bytes + block_padding, ENTRY_BYTES, 1),
        )

    def _make_case(
        self,
        padded_lens: list[int],
        actual_lens: list[int],
        block_size: int,
        *,
        block_padding: int = 0,
        table_padding: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.assertEqual(len(padded_lens), len(actual_lens))
        blocks_per_request = [
            (length + block_size - 1) // block_size for length in actual_lens
        ]
        max_blocks = max(blocks_per_request, default=0)
        num_cache_blocks = max(1, 1 + sum(blocks_per_request))
        pool = self._make_pool(num_cache_blocks, block_size, block_padding)
        table_storage = torch.full(
            (len(padded_lens), max_blocks + table_padding),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        block_table = table_storage[:, :max_blocks]
        physical_ids = (
            torch.randperm(
                sum(blocks_per_request), dtype=torch.int64, device=self.device
            ).to(torch.int32)
            + 1
        )
        physical_offset = 0
        for request, request_blocks in enumerate(blocks_per_request):
            if request_blocks:
                block_table[request, :request_blocks] = physical_ids[
                    physical_offset : physical_offset + request_blocks
                ]
                physical_offset += request_blocks
        padded = torch.tensor(
            padded_lens, dtype=torch.int32, device=self.device
        ).contiguous()
        actual = torch.tensor(
            actual_lens, dtype=torch.int32, device=self.device
        ).contiguous()
        return pool, block_table, padded, actual

    def _old_path_reference(
        self,
        pool: torch.Tensor,
        block_table: torch.Tensor,
        padded: torch.Tensor,
        actual: torch.Tensor,
        padded_lens: list[int],
        *,
        block_size: int,
    ) -> torch.Tensor:
        max_padded = max(padded_lens, default=0)
        packed = torch.zeros(
            (len(padded_lens), max_padded, ENTRY_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        if bool(torch.any(actual > 0)):
            gather_k_cache_packed(
                packed,
                pool,
                actual,
                None,
                block_table.contiguous(),
                block_size,
                0,
            )
        rows = [packed[i, :length] for i, length in enumerate(padded_lens)]
        if not rows or sum(padded_lens) == 0:
            return torch.empty(
                (0, ENTRY_BYTES), dtype=torch.uint8, device=self.device
            )
        return torch.cat(rows, dim=0)

    def _run_exact_case(
        self,
        padded_lens: list[int],
        actual_lens: list[int],
        block_size: int,
        *,
        block_padding: int = 0,
        table_padding: int = 0,
    ) -> None:
        pool, block_table, padded, actual = self._make_case(
            padded_lens,
            actual_lens,
            block_size,
            block_padding=block_padding,
            table_padding=table_padding,
        )
        expected = self._old_path_reference(
            pool,
            block_table,
            padded,
            actual,
            padded_lens,
            block_size=block_size,
        )
        out = torch.empty_like(expected)
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CP_DIRECT_FLAT_PACK": "1",
                "DSV4_TRAP_INVALID_KV_ACCESS": "0",
            },
        ):
            self.assertTrue(
                try_gather_k_cache_packed_to_flat(
                    out,
                    pool,
                    block_table,
                    padded,
                    actual,
                    block_size=block_size,
                    has_actual_tokens=any(actual_lens),
                )
            )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, expected))

        flat_start = 0
        for padded_len, actual_len in zip(padded_lens, actual_lens):
            tail = out[flat_start + actual_len : flat_start + padded_len]
            self.assertEqual(int(tail.count_nonzero()), 0)
            flat_start += padded_len

    def test_matches_old_gather_pack_byte_exact(self) -> None:
        cases = (
            ([64], [7], 64, 0, 0),
            ([300], [257], 256, 128, 1),
            ([8, 4], [5, 0], 4, 0, 0),
            ([2, 2, 2, 2], [1, 2, 0, 1], 1, 256, 3),
            (
                [8, 4, 8, 4, 8, 4, 8, 4, 8, 4],
                [3, 0, 7, 4, 2, 1, 8, 0, 5, 3],
                2,
                64,
                2,
            ),
            ([0, 4, 0, 8, 0], [0, 3, 0, 7, 0], 4, 128, 2),
            ([4] * 63, [i % 5 for i in range(63)], 4, 128, 1),
            ([4] * 64, [(i * 3) % 5 for i in range(64)], 4, 0, 0),
            ([4, 4, 4, 4], [0, 0, 0, 0], 4, 0, 1),
        )
        for padded, actual, block_size, block_padding, table_padding in cases:
            with self.subTest(
                batch_size=len(padded),
                block_size=block_size,
                block_padding=block_padding,
                table_padding=table_padding,
            ):
                self._run_exact_case(
                    padded,
                    actual,
                    block_size,
                    block_padding=block_padding,
                    table_padding=table_padding,
                )

    def test_unsupported_or_disabled_input_leaves_output_untouched(self) -> None:
        pool, block_table, padded, actual = self._make_case([4], [2], 2)
        out = torch.full(
            (4, ENTRY_BYTES), 0xA5, dtype=torch.uint8, device=self.device
        )
        before = out.clone()
        self.assertFalse(
            try_gather_k_cache_packed_to_flat(
                out,
                pool,
                block_table,
                padded.to(torch.int64),
                actual,
                block_size=2,
                has_actual_tokens=True,
            )
        )
        self.assertTrue(torch.equal(out, before))
        with mock.patch.dict(os.environ, {"DSV4_CP_DIRECT_FLAT_PACK": "0"}):
            self.assertFalse(
                try_gather_k_cache_packed_to_flat(
                    out,
                    pool,
                    block_table,
                    padded,
                    actual,
                    block_size=2,
                    has_actual_tokens=True,
                )
            )
        self.assertTrue(torch.equal(out, before))

        pool65, table65, padded65, actual65 = self._make_case(
            [1] * 65, [1] * 65, 1
        )
        out65 = torch.full(
            (65, ENTRY_BYTES), 0xA5, dtype=torch.uint8, device=self.device
        )
        before65 = out65.clone()
        self.assertFalse(
            try_gather_k_cache_packed_to_flat(
                out65,
                pool65,
                table65,
                padded65,
                actual65,
                block_size=1,
                has_actual_tokens=True,
            )
        )
        self.assertTrue(torch.equal(out65, before65))

        self.assertFalse(
            try_gather_k_cache_packed_to_flat(
                out,
                pool,
                block_table,
                padded,
                actual,
                block_size=1,
                has_actual_tokens=True,
            )
        )
        self.assertTrue(torch.equal(out, before))

    def test_empty_output_is_supported_without_launch(self) -> None:
        pool, block_table, padded, actual = self._make_case([0], [0], 4)
        out = torch.empty((0, ENTRY_BYTES), dtype=torch.uint8, device=self.device)
        with mock.patch.dict(os.environ, {"DSV4_CP_DIRECT_FLAT_PACK": "1"}):
            self.assertTrue(
                try_gather_k_cache_packed_to_flat(
                    out,
                    pool,
                    block_table,
                    padded,
                    actual,
                    block_size=4,
                    has_actual_tokens=False,
                )
            )

    def test_current_stream_orders_cache_write_before_gather(self) -> None:
        pool, block_table, padded, actual = self._make_case([4], [4], 2)
        source = torch.randint(
            0, 256, pool.shape, dtype=torch.uint8, device=self.device
        )
        out = torch.empty((4, ENTRY_BYTES), dtype=torch.uint8, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream), mock.patch.dict(
            os.environ,
            {
                "DSV4_CP_DIRECT_FLAT_PACK": "1",
                "DSV4_TRAP_INVALID_KV_ACCESS": "0",
            },
        ):
            pool.copy_(source)
            self.assertTrue(
                try_gather_k_cache_packed_to_flat(
                    out,
                    pool,
                    block_table,
                    padded,
                    actual,
                    block_size=2,
                    has_actual_tokens=True,
                )
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        expected = self._old_path_reference(
            pool, block_table, padded, actual, [4], block_size=2
        )
        self.assertTrue(torch.equal(out, expected))

    def test_hot_path_is_one_kernel_without_torch_pack_ops(self) -> None:
        padded_lens = [8, 4, 8, 4, 8, 4, 8, 4, 8, 4]
        actual_lens = [3, 0, 7, 4, 2, 1, 8, 0, 5, 3]
        pool, block_table, padded, actual = self._make_case(
            padded_lens, actual_lens, 2
        )
        out = torch.empty(
            (sum(padded_lens), ENTRY_BYTES), dtype=torch.uint8, device=self.device
        )

        def run() -> None:
            self.assertTrue(
                try_gather_k_cache_packed_to_flat(
                    out,
                    pool,
                    block_table,
                    padded,
                    actual,
                    block_size=2,
                    has_actual_tokens=True,
                )
            )

        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CP_DIRECT_FLAT_PACK": "1",
                "DSV4_TRAP_INVALID_KV_ACCESS": "0",
            },
        ):
            run()
            run()
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                run()
            torch.cuda.synchronize()

        events = list(prof.key_averages())
        fused = [
            event
            for event in events
            if "_gather_k_cache_packed_to_flat_kernel" in event.key
        ]
        self.assertEqual(sum(event.count for event in fused), 1)
        keys = {event.key for event in events}
        for forbidden in (
            "aten::zeros",
            "aten::arange",
            "aten::repeat_interleave",
            "aten::cumsum",
            "aten::index_select",
            "aten::index",
        ):
            self.assertNotIn(forbidden, keys)

    def test_same_batch_production_shapes_do_not_recompile(self) -> None:
        _gather_k_cache_packed_to_flat_kernel.device_caches.clear()
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CP_DIRECT_FLAT_PACK": "1",
                "DSV4_TRAP_INVALID_KV_ACCESS": "0",
            },
        ):
            for index, (
                padded_lens,
                actual_lens,
                block_size,
                block_padding,
            ) in enumerate(
                (
                    ([4] * 9, [1, 2, 3, 4, 1, 2, 3, 4, 1], 1, 0),
                    ([4] * 10, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2], 1, 0),
                    (
                        [64] * 10,
                        [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],
                        4,
                        256,
                    ),
                )
            ):
                pool, block_table, padded, actual = self._make_case(
                    padded_lens,
                    actual_lens,
                    block_size,
                    block_padding=block_padding,
                    table_padding=index + 1,
                )
                out = torch.empty(
                    (sum(padded_lens), ENTRY_BYTES),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.assertTrue(
                    try_gather_k_cache_packed_to_flat(
                        out,
                        pool,
                        block_table,
                        padded,
                        actual,
                        block_size=block_size,
                        has_actual_tokens=True,
                    )
                )
                torch.cuda.synchronize()
                if index == 0:
                    cache_after_warmup = _triton_cache_size(
                        _gather_k_cache_packed_to_flat_kernel
                    )
                    self.assertEqual(cache_after_warmup, 1)
        self.assertEqual(
            _triton_cache_size(_gather_k_cache_packed_to_flat_kernel),
            cache_after_warmup,
        )


if __name__ == "__main__":
    unittest.main()

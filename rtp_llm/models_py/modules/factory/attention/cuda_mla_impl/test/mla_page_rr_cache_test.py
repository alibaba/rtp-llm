"""CPU layout/routing references plus supplemental single-GPU checks."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
    build_mla_page_rr_slot_mapping,
)


class MlaPageRRSlotMappingTest(unittest.TestCase):
    PAGE_TOKENS = 128
    SHARD_SIZE = 8

    def _block_table(self, rank: int, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [
                [100 + rank * 10, 101 + rank * 10],
                [200 + rank * 10, 201 + rank * 10],
            ],
            dtype=dtype,
        )

    def _reference(
        self,
        positions: torch.Tensor,
        batch_indices: torch.Tensor,
        block_table: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        slots = []
        for position, batch_idx in zip(positions.tolist(), batch_indices.tolist()):
            global_page = position // self.PAGE_TOKENS
            if global_page % self.SHARD_SIZE != rank:
                slots.append(-1)
                continue
            local_page = global_page // self.SHARD_SIZE
            block_id = int(block_table[batch_idx, local_page])
            slots.append(block_id * self.PAGE_TOKENS + position % self.PAGE_TOKENS)
        return torch.tensor(slots, dtype=torch.int64)

    def test_all_ranks_match_boundary_mixed_batch_reference(self) -> None:
        positions = torch.tensor(
            [
                0,
                127,
                128,
                255,
                7 * 128 + 127,
                8 * 128,
                8 * 128 + 127,
                15 * 128 + 17,
                2 * 128 + 3,
                6 * 128 + 127,
                9 * 128 + 9,
            ],
            dtype=torch.int32,
        )
        batch_indices = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32
        )
        owned_by_rank = []
        owned_pages_by_rank = []

        for rank in range(self.SHARD_SIZE):
            with self.subTest(rank=rank):
                block_table = self._block_table(rank, dtype=torch.int32)
                actual = build_mla_page_rr_slot_mapping(
                    positions,
                    batch_indices,
                    block_table,
                    self.PAGE_TOKENS,
                    self.SHARD_SIZE,
                    rank,
                )
                expected = self._reference(positions, batch_indices, block_table, rank)
                self.assertEqual(actual.dtype, torch.int64)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                owned_by_rank.append(actual >= 0)
                owned_pages_by_rank.append(
                    {
                        (int(batch_idx), int(position) // self.PAGE_TOKENS)
                        for position, batch_idx, slot in zip(
                            positions.tolist(),
                            batch_indices.tolist(),
                            actual.tolist(),
                        )
                        if slot >= 0
                    }
                )

        ownership_count = torch.stack(owned_by_rank).sum(dim=0)
        torch.testing.assert_close(
            ownership_count,
            torch.ones_like(ownership_count),
            rtol=0,
            atol=0,
        )
        replicated_pages = {
            (int(batch_idx), int(position) // self.PAGE_TOKENS)
            for position, batch_idx in zip(positions.tolist(), batch_indices.tolist())
        }
        self.assertEqual(set().union(*owned_pages_by_rank), replicated_pages)
        for left_rank in range(self.SHARD_SIZE):
            for right_rank in range(left_rank + 1, self.SHARD_SIZE):
                self.assertTrue(
                    owned_pages_by_rank[left_rank].isdisjoint(
                        owned_pages_by_rank[right_rank]
                    )
                )

    def test_partial_terminal_leaves_unused_tail_ranks_unwritten(self) -> None:
        positions = torch.tensor([0, 127, 128, 129], dtype=torch.int64)
        batch_indices = torch.zeros(4, dtype=torch.int64)
        owner_tables = {
            0: torch.tensor([[30]], dtype=torch.int64),
            1: torch.tensor([[31]], dtype=torch.int64),
        }

        actual_by_rank = []
        for rank in range(self.SHARD_SIZE):
            block_table = owner_tables.get(
                rank, torch.tensor([[-1]], dtype=torch.int64)
            )
            actual_by_rank.append(
                build_mla_page_rr_slot_mapping(
                    positions,
                    batch_indices,
                    block_table,
                    self.PAGE_TOKENS,
                    self.SHARD_SIZE,
                    rank,
                )
            )

        torch.testing.assert_close(
            actual_by_rank[0],
            torch.tensor([30 * 128, 30 * 128 + 127, -1, -1]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_by_rank[1],
            torch.tensor([-1, -1, 31 * 128, 31 * 128 + 1]),
            rtol=0,
            atol=0,
        )
        for rank in range(2, self.SHARD_SIZE):
            torch.testing.assert_close(
                actual_by_rank[rank],
                torch.full((4,), -1, dtype=torch.int64),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_mapping_matches_cpu_reference(self) -> None:
        positions = torch.tensor(
            [0, 127, 128, 8 * 128 + 23, 15 * 128 + 127], dtype=torch.int32
        )
        batch_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        block_table = self._block_table(0, dtype=torch.int32)
        expected = self._reference(positions, batch_indices, block_table, rank=0)

        actual = build_mla_page_rr_slot_mapping(
            positions.cuda(),
            batch_indices.cuda(),
            block_table.cuda(),
            self.PAGE_TOKENS,
            self.SHARD_SIZE,
            0,
        )

        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_owner_page_requires_allocated_local_block(self) -> None:
        for unused_block in (-1, 0):
            with self.subTest(unused_block=unused_block):
                with self.assertRaisesRegex(RuntimeError, "owner page.*null or unused"):
                    build_mla_page_rr_slot_mapping(
                        torch.tensor([128], dtype=torch.int32),
                        torch.tensor([0], dtype=torch.int32),
                        torch.tensor([[unused_block]], dtype=torch.int32),
                        self.PAGE_TOKENS,
                        self.SHARD_SIZE,
                        1,
                    )

    def test_owner_page_requires_sufficient_table_width(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "block table width"):
            build_mla_page_rr_slot_mapping(
                torch.tensor([8 * 128], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([[17]], dtype=torch.int64),
                self.PAGE_TOKENS,
                self.SHARD_SIZE,
                0,
            )

    def test_metadata_contract_fails_fast(self) -> None:
        positions = torch.tensor([0], dtype=torch.int32)
        batch_indices = torch.tensor([0], dtype=torch.int32)
        block_table = torch.tensor([[1]], dtype=torch.int32)

        invalid_calls = (
            (
                "positions must be 1D",
                (positions.view(1, 1), batch_indices, block_table, 128, 8, 0),
            ),
            (
                "batch_indices must be 1D",
                (positions, batch_indices.view(1, 1), block_table, 128, 8, 0),
            ),
            (
                "same length",
                (positions.repeat(2), batch_indices, block_table, 128, 8, 0),
            ),
            (
                "local_block_table must be 2D",
                (positions, batch_indices, block_table.view(-1), 128, 8, 0),
            ),
            (
                "same dtype",
                (positions.to(torch.int64), batch_indices, block_table, 128, 8, 0),
            ),
            (
                "integer dtype",
                (
                    positions.float(),
                    batch_indices.float(),
                    block_table.float(),
                    128,
                    8,
                    0,
                ),
            ),
            (
                "same device",
                (positions, batch_indices, block_table.to("meta"), 128, 8, 0),
            ),
        )
        for message, args in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    build_mla_page_rr_slot_mapping(*args)

    def test_position_and_batch_values_fail_fast(self) -> None:
        block_table = torch.tensor([[1]], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "positions must be non-negative"):
            build_mla_page_rr_slot_mapping(
                torch.tensor([-1]),
                torch.tensor([0]),
                block_table,
                128,
                8,
                0,
            )
        with self.assertRaisesRegex(RuntimeError, "batch index out of range"):
            build_mla_page_rr_slot_mapping(
                torch.tensor([0]),
                torch.tensor([1]),
                block_table,
                128,
                8,
                0,
            )


class MlaPageRRCapacityTest(unittest.TestCase):
    def test_adapter_validates_rank_local_capacity(self) -> None:
        adapter = MlaPageRRCacheAdapter(
            page_tokens=4,
            shard_size=2,
            shard_rank=1,
        )
        adapter.validate_block_table_capacity(
            torch.empty((2, 1), dtype=torch.int32),
            (5, 8),
        )
        with self.assertRaisesRegex(RuntimeError, "rank-local block table"):
            adapter.validate_block_table_capacity(
                torch.empty((2, 1), dtype=torch.int32),
                (13, 8),
            )


if __name__ == "__main__":
    unittest.main()

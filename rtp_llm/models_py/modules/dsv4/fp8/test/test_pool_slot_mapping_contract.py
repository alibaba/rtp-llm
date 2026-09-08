from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
)


def _scalar_slots(
    block_table: list[list[int]],
    positions: list[int],
    entries: int,
    tokens_per_block: int,
    ring: int,
) -> list[int]:
    """Independent scalar oracle; does not call the production mapper."""
    batch = len(block_table)
    q_len = len(positions) // batch
    result: list[int] = []
    for flat_index, position in enumerate(positions):
        request = flat_index // q_len
        if position < 0:
            result.append(-1)
            continue
        row = position // tokens_per_block
        if row >= len(block_table[request]):
            result.append(-1)
            continue
        block_id = block_table[request][row]
        result.append(-1 if block_id <= 0 else block_id * entries + position % ring)
    return result


class PoolSlotMappingContractTest(unittest.TestCase):
    def test_scalar_oracle_slot_and_block_boundaries(self) -> None:
        table = [[1, 3], [5, 7]]
        positions = [0, 127, 128, 255, 256, 511, 512, -1]
        expected = _scalar_slots(table, positions, 128, 256, 128)
        actual = compute_kv_pool_slot_mapping(
            torch.tensor(table, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            pool_entries_per_block=128,
            pool_tokens_per_block=256,
            ring_entries=128,
        )
        self.assertEqual(actual.tolist(), expected)
        self.assertEqual(expected, [128, 255, 128, 255, 896, 1023, -1, -1])

    def test_oob_does_not_alias_positive_last_column(self) -> None:
        table = torch.tensor([[11, 999]], dtype=torch.int64)
        actual = compute_kv_pool_slot_mapping(
            table,
            torch.tensor([511, 512, 768], dtype=torch.int64),
            pool_entries_per_block=128,
            pool_tokens_per_block=256,
            ring_entries=128,
        )
        self.assertEqual(actual.tolist(), [999 * 128 + 127, -1, -1])

    def test_reserved_null_and_valid_mask(self) -> None:
        actual = compute_kv_pool_slot_mapping(
            torch.tensor([[0, -1]], dtype=torch.int32),
            torch.tensor([0, 256], dtype=torch.int32),
            pool_entries_per_block=128,
            pool_tokens_per_block=256,
            ring_entries=128,
            valid_mask=torch.tensor([True, False], dtype=torch.bool),
        )
        self.assertEqual(actual.tolist(), [-1, -1])

    def test_rejects_non_integer_indices(self) -> None:
        good_table = torch.tensor([[1]], dtype=torch.int32)
        good_pos = torch.tensor([0], dtype=torch.int32)
        for table_dtype in (torch.float32, torch.bool):
            with self.subTest(table_dtype=table_dtype), self.assertRaises(TypeError):
                compute_kv_pool_slot_mapping(
                    good_table.to(table_dtype), good_pos, 128, 256, 128
                )
        for pos_dtype in (torch.float32, torch.bool):
            with self.subTest(pos_dtype=pos_dtype), self.assertRaises(TypeError):
                compute_kv_pool_slot_mapping(
                    good_table, good_pos.to(pos_dtype), 128, 256, 128
                )

    def test_rejects_non_bool_or_cross_device_valid_mask(self) -> None:
        table = torch.tensor([[1]], dtype=torch.int32)
        pos = torch.tensor([0], dtype=torch.int32)
        with self.assertRaisesRegex(TypeError, "valid_mask must be bool"):
            compute_kv_pool_slot_mapping(
                table,
                pos,
                128,
                256,
                128,
                valid_mask=torch.tensor([1], dtype=torch.int32),
            )
        meta_mask = torch.empty((1,), dtype=torch.bool, device="meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            compute_kv_pool_slot_mapping(
                table, pos, 128, 256, 128, valid_mask=meta_mask
            )

    def test_rejects_cross_device_indices_and_ring_larger_than_pool(self) -> None:
        table = torch.tensor([[1]], dtype=torch.int32)
        meta_pos = torch.empty((1,), dtype=torch.int32, device="meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            compute_kv_pool_slot_mapping(table, meta_pos, 128, 256, 128)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            compute_kv_pool_slot_mapping(
                table,
                torch.tensor([0], dtype=torch.int32),
                pool_entries_per_block=128,
                pool_tokens_per_block=256,
                ring_entries=129,
            )

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_table"):
            compute_kv_pool_slot_mapping(
                torch.empty((1, 0), dtype=torch.int32),
                torch.empty((0,), dtype=torch.int32),
                1,
                1,
                1,
            )
        with self.assertRaisesRegex(ValueError, "abs_pos must be 1D"):
            compute_kv_pool_slot_mapping(
                torch.tensor([[1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                1,
                1,
                1,
            )


if __name__ == "__main__":
    unittest.main()

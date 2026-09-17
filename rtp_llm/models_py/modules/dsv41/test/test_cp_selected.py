"""Exact compact CP bytes, owner errors and changing page maps on one GPU."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.cp import _selected_local_rows


class CPSelectedRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell CUDA is required")

    @staticmethod
    def fixture(entries, entry_bytes, rank, limit=1048576, salt=11):
        capacity = (limit + 8 * entries - 1) // (8 * entries)
        virtual = torch.arange(capacity, device="cuda", dtype=torch.int64)
        physical = (virtual + 17 * rank) % capacity + 1
        table_storage = torch.full(
            (1, 2 * capacity), -123, dtype=torch.int32, device="cuda"
        )
        table = table_storage[:, ::2]
        table.copy_(physical[None, :])
        backing = torch.full(
            (capacity + 1, entries * entry_bytes + 32),
            197,
            dtype=torch.uint8,
            device="cuda",
        )
        pool = backing[:, : entries * entry_bytes]
        position = (virtual[:, None] * 8 + rank) * entries + torch.arange(
            entries, device="cuda"
        )
        columns = torch.arange(entry_bytes, device="cuda")
        payload = ((position[:, :, None] * 73 + columns * 29 + salt) % 256).to(
            torch.uint8
        )
        pool[physical] = payload.flatten(1)
        return pool, table

    @staticmethod
    def expected(positions, entries, entry_bytes, rank, salt=11):
        positions = positions.cpu().long()
        columns = torch.arange(entry_bytes)
        result = ((positions[:, None] * 73 + columns * 29 + salt) % 256).to(torch.uint8)
        owner = (positions >= 0) & ((positions // entries) % 8 == rank)
        result[~owner] = 0
        return result

    @torch.inference_mode()
    def test_exact_long_context_bytes_and_noncontiguous_metadata(self):
        positions = (
            torch.arange(1021, device="cuda", dtype=torch.int64) * 1877
        ) % 1048576
        positions[::17] = -1
        positions[1] = 1048575
        positions[2] = -9
        backing = torch.stack((positions, torch.full_like(positions, -11)), dim=1)
        wanted = backing[:, 0]
        for entries, entry_bytes in ((64, 288), (128, 288), (128, 68)):
            for rank in range(8):
                with self.subTest(entries=entries, entry_bytes=entry_bytes, rank=rank):
                    pool, table = self.fixture(entries, entry_bytes, rank)
                    actual = _selected_local_rows(
                        pool, table, wanted, entries, entry_bytes, rank
                    )
                    torch.testing.assert_close(
                        actual.cpu(),
                        self.expected(wanted, entries, entry_bytes, rank),
                        rtol=0,
                        atol=0,
                    )
                    with patch(
                        "torch.cuda.get_device_capability", lambda *a, **k: (9, 0)
                    ):
                        legacy = _selected_local_rows(
                            pool, table, wanted, entries, entry_bytes, rank
                        )
                    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)

    @torch.inference_mode()
    def test_missing_owner_pages_read_as_zero_rows(self):
        entries, entry_bytes, rank = 64, 288, 3
        pool, table = self.fixture(entries, entry_bytes, rank, limit=2048)
        wanted = torch.tensor(
            [3 * entries, 4 * entries, -1], dtype=torch.int64, device="cuda"
        )
        for invalid in (0, -1, pool.shape[0], pool.shape[0] + 101):
            table[0, 0] = invalid
            with self.subTest(page=invalid):
                actual = _selected_local_rows(
                    pool, table, wanted, entries, entry_bytes, rank
                )
                # An owned row whose page is missing reads as zeros instead of
                # raising; the valid owner row keeps its payload.
                torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
                torch.testing.assert_close(
                    actual[1:].cpu(),
                    self.expected(wanted[1:], entries, entry_bytes, rank),
                    rtol=0,
                    atol=0,
                )
                with patch("torch.cuda.get_device_capability", lambda *a, **k: (9, 0)):
                    legacy = _selected_local_rows(
                        pool, table, wanted, entries, entry_bytes, rank
                    )
                torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        outside = torch.tensor([(table.shape[1] * 8 + rank) * entries], device="cuda")
        actual = _selected_local_rows(pool, table, outside, entries, entry_bytes, rank)
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    @torch.inference_mode()
    def test_repeated_calls_read_current_metadata_and_empty_input(self):
        entries, entry_bytes, rank = 128, 68, 7
        pool, table = self.fixture(entries, entry_bytes, rank, limit=4096)
        wanted = torch.tensor([7 * entries + 3, -1], device="cuda", dtype=torch.int32)
        first = _selected_local_rows(pool, table, wanted, entries, entry_bytes, rank)
        torch.testing.assert_close(
            first.cpu(), self.expected(wanted, entries, entry_bytes, rank)
        )
        table[0, 0], table[0, 1] = table[0, 1].clone(), table[0, 0].clone()
        second = _selected_local_rows(pool, table, wanted, entries, entry_bytes, rank)
        changed = wanted.long().clone()
        changed[0] += 8 * entries
        torch.testing.assert_close(
            second.cpu(), self.expected(changed, entries, entry_bytes, rank)
        )
        wanted.fill_(-1)
        self.assertEqual(
            torch.count_nonzero(
                _selected_local_rows(pool, table, wanted, entries, entry_bytes, rank)
            ).item(),
            0,
        )
        empty = _selected_local_rows(
            pool, table, wanted[:0], entries, entry_bytes, rank
        )
        self.assertEqual(empty.shape, (0, entry_bytes))


if __name__ == "__main__":
    unittest.main()

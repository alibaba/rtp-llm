import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_metadata_triton import (
    begin_accepted_pair_resolution,
)


class PLECommitMetadataTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU test must execute on CUDA")

    def fixture(self, batch=4, query=3, page_size=16, table_cols=8, pool_rows=64):
        prefixes = torch.arange(batch, device="cuda", dtype=torch.int64) * page_size
        state_table = torch.arange(
            1, batch * table_cols + 1, device="cuda", dtype=torch.int32
        ).reshape(batch, table_cols) % (pool_rows - 1)
        state_table = state_table + 1
        ctx_table = torch.arange(
            1, batch * table_cols + 1, device="cuda", dtype=torch.int32
        ).reshape(batch, table_cols) % (pool_rows - 1)
        ctx_table = ctx_table + 1
        # Distinct valid physical blocks on each row's committed page.
        for row in range(batch):
            state_table[row, row] = 1 + row
            ctx_table[row, row] = 1 + batch + row
        stage = SimpleNamespace(
            prefixes=prefixes,
            state_inputs=SimpleNamespace(kv_cache_block_id_device=state_table),
            ctx_inputs=SimpleNamespace(kv_cache_block_id_device=ctx_table),
            state_pool=torch.zeros(pool_rows, 2, device="cuda"),
            ctx_pool=torch.zeros(pool_rows, 2, device="cuda"),
            page_size=page_size,
        )
        transaction = SimpleNamespace(
            stages={0: stage},
            query_len=query,
            batch_size=batch,
            prefixes=prefixes,
        )
        return transaction, state_table, ctx_table, prefixes

    def accept(self, values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    def begin(self, transaction, accept_len):
        with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            return begin_accepted_pair_resolution(transaction, accept_len)

    def test_valid_resolution_matches_reference(self):
        transaction, state_table, ctx_table, prefixes = self.fixture()
        accept_len = self.accept([1, 2, 3, 1])
        resolution = self.begin(transaction, accept_len)
        self.assertIsNotNone(resolution)
        # Enqueue unrelated GPU work between begin and verify: the deferred
        # verdict must stay correct with interleaved launches.
        dummy = torch.zeros(1024, device="cuda")
        dummy.fill_(1)
        resolution.verify()
        accepted, pending = resolution.accepted, resolution.pending
        stage, state_blocks, ctx_blocks = pending[0]
        pages = (prefixes + accept_len.to(torch.long) - 1) // stage.page_size
        rows = torch.arange(accept_len.numel(), device="cuda")
        torch.testing.assert_close(
            accepted.cpu(), accept_len.to(torch.long).cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            state_blocks.cpu(),
            state_table[rows, pages].to(torch.long).cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ctx_blocks.cpu(),
            ctx_table[rows, pages].to(torch.long).cpu(),
            rtol=0,
            atol=0,
        )
        torch.cuda.synchronize()
        self.assertEqual(int(dummy.sum().item()), 1024)

    def test_accept_range_rejected_with_clamped_payload(self):
        transaction, _, _, _ = self.fixture()
        accept_len = self.accept([1, 99, 1, 1])
        resolution = self.begin(transaction, accept_len)
        self.assertIsNotNone(resolution)
        # Speculative gathers consume the payload before verify(): the stored
        # values must stay inside the valid bounds even for invalid inputs.
        clamped = resolution.accepted.cpu()
        upper = transaction.query_len
        self.assertTrue(bool(((clamped >= 1) & (clamped <= upper)).all()))
        with self.assertRaisesRegex(RuntimeError, "accept_len must be in"):
            resolution.verify()

    def test_duplicate_destination_rejected(self):
        transaction, state_table, _, _ = self.fixture()
        state_table[0, 0] = state_table[1, 1]
        accept_len = self.accept([1, 1, 2, 3])
        resolution = self.begin(transaction, accept_len)
        self.assertIsNotNone(resolution)
        clamped_blocks = resolution.pending[0][1].cpu()
        pool_rows = transaction.stages[0].state_pool.shape[0]
        self.assertTrue(
            bool(((clamped_blocks >= 0) & (clamped_blocks < pool_rows)).all())
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate target destination"):
            resolution.verify()

    def test_block_out_of_pool_rejected(self):
        transaction, state_table, _, _ = self.fixture()
        state_table[0, 0] = 64
        accept_len = self.accept([1, 1, 2, 3])
        resolution = self.begin(transaction, accept_len)
        self.assertIsNotNone(resolution)
        with self.assertRaisesRegex(RuntimeError, "exceeds its pool"):
            resolution.verify()

    def test_unsupported_device_disables_fused_path(self):
        transaction, _, _, _ = self.fixture()
        accept_len = self.accept([1, 2, 3, 1])
        with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
            self.assertIsNone(begin_accepted_pair_resolution(transaction, accept_len))

    def test_non_contiguous_accept_falls_back(self):
        transaction, _, _, _ = self.fixture()
        accept_len = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], device="cuda", dtype=torch.int32
        )[:, 0]
        self.assertFalse(accept_len.is_contiguous())
        resolution = self.begin(transaction, accept_len)
        self.assertIsNone(resolution)

    def test_repeated_resolution_reuses_channel(self):
        transaction, _, _, _ = self.fixture()
        for _ in range(3):
            resolution = self.begin(transaction, self.accept([1, 2, 3, 1]))
            self.assertIsNotNone(resolution)
            resolution.verify()


if __name__ == "__main__":
    unittest.main()

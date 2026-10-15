import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.cache_views import PLECacheViews


class PLECommitDestinationTest(unittest.TestCase):
    def _devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_strided_tables_and_large_batch_fallback(self):
        for device in self._devices():
            for batch in (1, 17, 1025):
                with self.subTest(device=device, batch=batch):
                    table = torch.arange(
                        1, batch * 8 + 1, device=device, dtype=torch.int32
                    ).reshape(batch, 8)[:, ::2]
                    pages = torch.arange(batch, device=device) % 4
                    inputs = SimpleNamespace(kv_cache_block_id_device=table)
                    blocks, host = PLECacheViews._commit_physical_blocks(
                        inputs, "state", pages, batch * 8 + 1, torch.device(device)
                    )
                    expected = table[torch.arange(batch, device=device), pages].long()
                    torch.testing.assert_close(blocks, expected, rtol=0, atol=0)
                    self.assertEqual(host, expected.tolist())

    def test_invalid_destinations_rejected_without_mutating_table(self):
        for device in self._devices():
            for page, block, message in (
                (-1, 1, "invalid logical page"),
                (4, 1, "exceeds its block table"),
                (0, 0, "unallocated"),
                (0, 5, "exceeds its pool"),
            ):
                with self.subTest(device=device, page=page, block=block):
                    table = torch.ones((2, 4), device=device, dtype=torch.int32)
                    table[0, 0] = block
                    before = table.clone()
                    pages = torch.tensor([page, 0], device=device)
                    with self.assertRaisesRegex(RuntimeError, message):
                        PLECacheViews._commit_physical_blocks(
                            SimpleNamespace(kv_cache_block_id_device=table),
                            "state",
                            pages,
                            5,
                            torch.device(device),
                        )
                    torch.testing.assert_close(table, before, rtol=0, atol=0)

    def test_empty_columns_rejected_before_lookup(self):
        for device in self._devices():
            with self.subTest(device=device):
                with self.assertRaisesRegex(RuntimeError, "block table|empty"):
                    PLECacheViews._commit_physical_blocks(
                        SimpleNamespace(
                            kv_cache_block_id_device=torch.empty(
                                (2, 0), device=device, dtype=torch.int32
                            )
                        ),
                        "state",
                        torch.zeros(2, device=device, dtype=torch.long),
                        5,
                        torch.device(device),
                    )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

# Load the real helper without attention/__init__.py, which imports the
# compiled engine. These tensor-only regressions also run on CPU developer
# machines, whereas the existing operator tests exercise the full ROCm path.
_spec = importlib.util.spec_from_file_location(
    "rtp_paged_prefill_metadata",
    Path(__file__).resolve().parents[1] / "paged_prefill_metadata.py",
)
_metadata = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _metadata
_spec.loader.exec_module(_metadata)
prepare_paged_prefill_metadata = _metadata.prepare_paged_prefill_metadata
sanitize_and_pad_block_table = _metadata.sanitize_and_pad_block_table


class PagedPrefillMetadataTest(unittest.TestCase):
    def test_padding_retains_valid_ids_and_covers_speculative_prefetch(self):
        for block_size in (8, 16, 32, 64):
            with self.subTest(block_size=block_size):
                lengths = torch.tensor(
                    [1, block_size + 1, 3 * block_size], dtype=torch.int32
                )
                table = torch.tensor(
                    [[9, -1, -1], [4, 7, -1], [3, 5, 8]], dtype=torch.int32
                )
                before = table.clone()
                actual, _ = sanitize_and_pad_block_table(
                    table, lengths, block_size, 3 * block_size
                )
                self.assertEqual(
                    actual.shape[1], 3 + (128 + block_size - 1) // block_size
                )
                self.assertEqual(actual[0].tolist(), [9] * actual.shape[1])
                self.assertEqual(actual[1].tolist(), [4] + [7] * (actual.shape[1] - 1))
                self.assertEqual(actual[2, :3].tolist(), [3, 5, 8])
                self.assertTrue(torch.equal(table, before))

    def test_valid_invalid_ids_are_not_hidden(self):
        table = torch.tensor([[2, -1, 7]], dtype=torch.int32)
        actual, _ = sanitize_and_pad_block_table(table, torch.tensor([48]), 16, 48)
        self.assertEqual(actual[0, :3].tolist(), [2, -1, 7])

    def test_uses_existing_lengths_and_prepares_dummy_inputs_once(self):
        lengths = torch.tensor([17, 33], dtype=torch.int32)
        cu = torch.tensor([0, 17, 50], dtype=torch.int32)
        table = torch.tensor([[2, 4, -1], [8, 10, 12]], dtype=torch.int32)
        params = prepare_paged_prefill_metadata(cu, table, 16, 33, seqlen_k=lengths)
        self.assertIs(params.seqlen_k, lengths)
        self.assertEqual(params.kv_indptr.tolist(), [0, 0, 0])
        self.assertEqual(params.kv_page_indices.tolist(), [0])
        self.assertIsNone(params.descale)

    def test_new_batches_and_cache_groups_do_not_reuse_sanitized_tables(self):
        table = torch.tensor([[3, -1, -1]], dtype=torch.int32)
        old = prepare_paged_prefill_metadata(torch.tensor([0, 1]), table, 16, 1)
        old_table = old.block_table.clone()
        table[0] = torch.tensor([6, 9, -1])
        new = prepare_paged_prefill_metadata(torch.tensor([0, 17]), table, 16, 17)
        self.assertTrue(torch.equal(old.block_table, old_table))
        self.assertNotEqual(old.block_table.data_ptr(), new.block_table.data_ptr())
        self.assertEqual(new.block_table[0, :3].tolist(), [6, 9, 9])

    def test_fp8_scale_is_batch_owned(self):
        params = prepare_paged_prefill_metadata(
            torch.tensor([0, 16]), torch.tensor([[1]]), 16, 16, fp8=True
        )
        self.assertEqual(params.descale.dtype, torch.float32)
        self.assertEqual(params.descale.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()

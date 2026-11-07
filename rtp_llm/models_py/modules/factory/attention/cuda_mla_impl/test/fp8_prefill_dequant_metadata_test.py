"""Virtual FP8 cache pages cover the original canonical rows exactly once."""
import unittest
import torch
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import (
    fp8_prefill_dequant_metadata,
    fp8_prefill_dequant_pages,
)


class Fp8PrefillDequantMetadataTest(unittest.TestCase):
    def test_virtual_pages_preserve_source_and_destination_row_addresses(self):
        for rows in (0, 1, 128, 1023, 1024, 1088, 8192, 65535, 65536):
            with self.subTest(rows=rows):
                table, lengths, starts = fp8_prefill_dequant_metadata(rows, "cpu")
                pages = table.shape[0]
                self.assertEqual(pages, fp8_prefill_dequant_pages(rows))
                self.assertEqual(sum(t.numel() * t.element_size() for t in (table, lengths, starts)), pages * 12)
                self.assertTrue(all(t.dtype == torch.int32 for t in (table, lengths, starts)))
                page_size = rows // pages
                source = torch.arange(rows).view(pages, page_size)
                result = torch.full((rows,), -1, dtype=torch.int64)
                coverage = torch.zeros(rows, dtype=torch.int32)
                for page in range(pages):
                    length, start = lengths[page].item(), starts[page].item()
                    result[start:start + length] = source[table[page, 0], :length]
                    coverage[start:start + length] += 1
                torch.testing.assert_close(result, torch.arange(rows), rtol=0, atol=0)
                torch.testing.assert_close(coverage, torch.ones(rows, dtype=torch.int32), rtol=0, atol=0)

    def test_small_and_unaligned_chunks_keep_the_original_metadata(self):
        for rows in (0, 1, 128, 1023, 1025, 65535):
            table, lengths, starts = fp8_prefill_dequant_metadata(rows, "cpu")
            self.assertEqual(table.tolist(), [[0]])
            self.assertEqual(lengths.tolist(), [rows])
            self.assertEqual(starts.tolist(), [0])

    def test_64k_launch_has_1024_independent_virtual_pages(self):
        table, lengths, starts = fp8_prefill_dequant_metadata(65536, "cpu")
        self.assertEqual(tuple(table.shape), (1024, 1))
        self.assertEqual(lengths.unique().tolist(), [64])
        self.assertEqual(starts[0].item(), 0)
        self.assertEqual(starts[-1].item() + lengths[-1].item(), 65536)


if __name__ == "__main__":
    unittest.main()

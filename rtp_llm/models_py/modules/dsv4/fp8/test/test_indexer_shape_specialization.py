"""Request geometry must not create an unbounded set of Triton variants."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_quant_triton as quant
from rtp_llm.models_py.modules.dsv4.fp8 import _topk_canonical_triton as topk


def compiled_variants(kernel):
    return sum(len(cache) for cache, *_ in kernel.device_caches.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexerShapeSpecializationTest(unittest.TestCase):
    def test_canonical_topk_varied_geometry_and_ties(self):
        topk._canonical_topk_kernel.device_caches.clear()
        for i, width in enumerate((67, 71, 73, 79, 83, 89, 97, 101, 113, 127)):
            k = 17 + i
            # Include padded row strides, exact ties, empty and short windows.
            scores = torch.randint(0, 7, (4, width + i), device="cuda").float()
            scores = scores[:, :width]
            starts = torch.tensor([0, 3, 5, 7], device="cuda", dtype=torch.int32)
            ends = torch.tensor(
                [width, width - 2, 5, 9], device="cuda", dtype=torch.int32
            )
            output = torch.full((4, k + i), -1, device="cuda", dtype=torch.int32)[:, :k]
            expected = torch.full_like(output, -1)
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                values = scores[row, start:end]
                n = min(k, end - start)
                output[row, :n] = values.topk(n).indices.int()
                selected = values.argsort(descending=True, stable=True)[:n]
                expected[row, :n] = selected.sort().values.int()
            self.assertTrue(
                topk.canonicalize_topk_if_supported(scores, output, ends, starts)
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        # Contiguous and padded rows are bounded layout choices, not variants
        # for every request width, row stride or top-k length.
        self.assertLessEqual(compiled_variants(topk._canonical_topk_kernel), 2)

    def test_canonical_topk_bounds_and_padding(self):
        scores = torch.tensor([[1, 4, 2, 3]], device="cuda", dtype=torch.float32)
        # Index 4 is inside the reported window but outside the allocation.
        output = torch.tensor([[1, 3, 4, -1]], device="cuda", dtype=torch.int32)
        ends = torch.tensor([8], device="cuda", dtype=torch.int32)
        self.assertTrue(topk.canonicalize_topk_if_supported(scores, output, ends))
        self.assertEqual(output.tolist(), [[1, 3, -1, -1]])
        output.fill_(-1)
        self.assertTrue(topk.canonicalize_topk_if_supported(scores, output, ends))
        self.assertEqual(output.tolist(), [[-1, -1, -1, -1]])

    def test_indexer_gather_varied_offsets_lengths_and_strides(self):
        quant._gather_indexer_k_fp32_kernel.device_caches.clear()
        block = 64
        for i, count in enumerate((1, 3, 5, 17, 31, 64, 67, 127, 191, 257)):
            blocks = 8 + i
            keys = torch.randn(blocks * block, 128, device="cuda", dtype=torch.bfloat16)
            pool = torch.empty((blocks, block, 132), device="cuda", dtype=torch.uint8)
            slots = torch.arange(blocks * block, device="cuda", dtype=torch.int64)
            quant.quantize_indexer_k(keys, slots, pool)
            table = torch.arange(blocks, device="cuda", dtype=torch.int32)
            table = torch.stack((table, table.flip(0)))
            table[1, 1] = -1
            storage = torch.zeros(
                (2, blocks * (i + 1)), device="cuda", dtype=torch.int32
            )
            storage[:, :: i + 1] = table
            table = storage[:, :: i + 1]
            start = i * 7
            actual, valid = quant.gather_indexer_k_fp32(
                pool, table, start, count, block
            )
            positions = torch.arange(start, start + count, device="cuda")
            ids = table[:, positions // block]
            expected_valid = (ids >= 0) & (ids < blocks)
            expected_slots = ids.clamp(0, blocks - 1).long() * block + positions % block
            expected = quant.dequantize_indexer_k(pool, expected_slots.flatten())
            torch.testing.assert_close(actual, expected.view_as(actual), rtol=0, atol=0)
            self.assertTrue(torch.equal(valid, expected_valid))
        self.assertEqual(compiled_variants(quant._gather_indexer_k_fp32_kernel), 1)


if __name__ == "__main__":
    unittest.main()

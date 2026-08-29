import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import canonical_topk


class Sm120SparseMlaCanonicalTest(unittest.TestCase):
    def test_none_length_counts_non_negative_slots(self):
        indices = torch.tensor([[11, -1, 13, -1], [-1, 7, 8, 9]], dtype=torch.int64)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.dtype, torch.int32)
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(canonical[0, 1].item(), -1)

    def test_short_explicit_length_is_preserved(self):
        indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([2]), (4, 8))
        self.assertEqual(canonical.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(lengths.tolist(), [2])

    def test_unsupported_width_padding_keeps_invalid_slots(self):
        indices = torch.tensor([[4, -1, 9]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.shape, (1, 4))
        self.assertEqual(canonical.tolist(), [[4, -1, 9, -1]])
        self.assertEqual(lengths.tolist(), [2])


if __name__ == "__main__":
    unittest.main()

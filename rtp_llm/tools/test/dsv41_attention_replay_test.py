"""Check logical sparse positions across the two attention storage layouts."""

import unittest

import torch

from rtp_llm.tools.dsv41_attention_replay import aligned_sparse_inputs


class AttentionReplayAlignmentTest(unittest.TestCase):
    def setUp(self):
        kv = torch.arange(3 * 512).reshape(1, 3, 512).bfloat16()
        self.reference = {
            "kv": kv,
            "indices": torch.tensor([[[0, -1, -1], [0, 1, 2]]]),
        }
        self.captured = {
            "kv": kv[0, [2, 0, 1]].unsqueeze(1),
            "indices": torch.tensor([[[1, -1, -1]], [[0, 1, 2]]]),
            "lengths": torch.tensor([1, 3]),
        }

    def test_layout_alignment_preserves_set_and_reports_order(self):
        kv, selections = aligned_sparse_inputs(self.reference, self.captured, 2)
        self.assertTrue(kv["bitwise_equal"])
        self.assertTrue(all(row["set_equal"] for row in selections))
        self.assertTrue(selections[0]["order_equal"])
        self.assertFalse(selections[1]["order_equal"])
        self.assertEqual(selections[1]["native_aligned"], [2, 0, 1])

    def test_selected_position_mismatch_is_rejected(self):
        self.captured["indices"][0, 0, 0] = 2
        with self.assertRaisesRegex(AssertionError, "position set differs"):
            aligned_sparse_inputs(self.reference, self.captured, 2)

    def test_invalid_indices_and_lengths_are_rejected(self):
        for invalid in (-2, 3):
            with self.subTest(index=invalid):
                self.captured["indices"][0, 0, 0] = invalid
                with self.assertRaisesRegex(AssertionError, "position set differs"):
                    aligned_sparse_inputs(self.reference, self.captured, 2)
        self.captured["indices"][0, 0, 0] = 1
        self.captured["lengths"][0] = 4
        with self.assertRaisesRegex(ValueError, "row length"):
            aligned_sparse_inputs(self.reference, self.captured, 2)

    def test_equal_duplicate_sets_are_rejected(self):
        self.reference["indices"][0, 1] = torch.tensor([0, 0, 2])
        self.captured["indices"][1, 0] = torch.tensor([1, 1, 0])
        with self.assertRaisesRegex(AssertionError, "position set differs"):
            aligned_sparse_inputs(self.reference, self.captured, 2)


if __name__ == "__main__":
    unittest.main()

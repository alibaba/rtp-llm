"""Check logical sparse positions across the two attention storage layouts."""

import unittest

import torch

from rtp_llm.tools.dsv41_attention_replay import (
    aligned_sparse_inputs,
    history_identities,
)


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

    def test_decode_query_is_independent_of_full_swa_ring(self):
        kv = torch.arange(131 * 512).reshape(1, 131, 512).bfloat16()
        reference = {
            "kv": kv,
            "indices": torch.tensor([[[0, 1, 2, 3, 4, 5, 128, 129, 130, -1]]]),
        }
        captured = {
            "kv": torch.cat((kv[0, 128:], kv[0, :128])).unsqueeze(1),
            "indices": torch.tensor([[[3, 4, 5, 6, 7, 8, 0, 1, 2, -1]]]),
            "lengths": torch.tensor([10]),
        }
        values, selections = aligned_sparse_inputs(reference, captured, 128)
        self.assertTrue(values["bitwise_equal"])
        self.assertEqual(len(selections), 1)
        self.assertTrue(selections[0]["set_equal"])
        self.assertTrue(selections[0]["order_equal"])

    def test_query_count_and_swa_segment_admission(self):
        for rows in (0, -1, 1.5):
            with self.subTest(window_rows=rows):
                with self.assertRaisesRegex(ValueError, "SWA segment"):
                    aligned_sparse_inputs(self.reference, self.captured, rows)
        captured = {**self.captured, "lengths": torch.tensor([1])}
        with self.assertRaisesRegex(ValueError, "query row count"):
            aligned_sparse_inputs(self.reference, captured, 2)


class AttentionReplayHistoryTest(unittest.TestCase):
    def traces(self, steps=((0, (10, 20, 30, 40, 50)), (5, (60,)), (6, (70,)))):
        return [
            {"metadata": {"start_pos": start, "input_ids": [list(ids)]}}
            for start, ids in steps
        ]

    def test_contiguous_history_and_matching_native_cache_spans(self):
        reference = self.traces()
        native = [
            {
                "metadata": {
                    "input_ids": trace["metadata"]["input_ids"][0],
                    "cache_spans": [[1, 1, 1, 1, start, span]],
                }
            }
            for trace, start, span in zip(reference, (0, 5, 6), (5, 1, 1))
        ]
        identities = history_identities(reference, native)
        self.assertEqual([entry["start_pos"] for entry in identities], [0, 5, 6])
        self.assertEqual(identities[-1]["input_ids"], [70])

    def test_gaps_overlap_and_resets_are_rejected(self):
        for start in (0, 4, 6):
            with self.subTest(start=start):
                with self.assertRaisesRegex(ValueError, "gap, overlap or reset"):
                    history_identities(
                        self.traces(((0, (10, 20, 30, 40, 50)), (start, (60,))))
                    )
        with self.assertRaisesRegex(ValueError, "gap, overlap or reset"):
            history_identities(self.traces(((5, (60,)),)))

    def test_prefill_and_decode_shapes_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "initial prefill"):
            history_identities([])
        for steps in (((0, (1,)),), ((0, (1, 2)), (2, (3, 4)))):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "single-token decode"):
                    history_identities(self.traces(steps))
        with self.assertRaisesRegex(ValueError, "first SWA ring"):
            history_identities(self.traces(((0, tuple(range(128))), (128, (1,)))))

    def test_divergent_current_or_earlier_prefix_is_rejected(self):
        reference = self.traces()
        for step in range(3):
            native = self.traces()
            native[step]["metadata"]["input_ids"][0][-1] += 1
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "divergent consumed prefixes"):
                    history_identities(reference, native)
        with self.assertRaisesRegex(ValueError, "different step counts"):
            history_identities(reference, reference[:2])


if __name__ == "__main__":
    unittest.main()

"""Compressed-indexer cold start must retain the uncompressed live tail."""

import unittest

import torch

from rtp_llm.models_py.modules.indexer_grouping import (
    append_incomplete_tail_indices,
    expand_indexer_group_indices,
    pad_empty_indexer_history,
)


class GlmSparseColdStartContractTest(unittest.TestCase):
    def test_empty_history_matches_sentinel_history_and_preserves_live_tail(self):
        lengths = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        for shape in ((4, 0), (4, 1, 0)):
            with self.subTest(shape=shape):
                history = pad_empty_indexer_history(
                    torch.empty(shape, dtype=torch.int32), 512
                )
                self.assertEqual(history.shape, (*shape[:-1], 512))
                history = history.reshape(4, 512)
                actual = append_incomplete_tail_indices(
                    expand_indexer_group_indices(
                        history, 4, raw_sequence_lengths=lengths
                    ),
                    lengths,
                    4,
                )
                self.assertEqual(actual.shape, (4, 2051))
                self.assertTrue(torch.all(actual[:, :2048] == -1))
                torch.testing.assert_close(
                    actual[:, 2048:],
                    torch.tensor(
                        [[-1, -1, -1], [0, -1, -1], [0, 1, -1], [0, 1, 2]],
                        dtype=torch.int32,
                    ),
                )

    def test_empty_batch_retains_semantic_width(self):
        history = pad_empty_indexer_history(torch.empty((0, 0), dtype=torch.int32), 512)
        lengths = torch.empty(0, dtype=torch.int32)
        actual = append_incomplete_tail_indices(
            expand_indexer_group_indices(history, 4, raw_sequence_lengths=lengths),
            lengths,
            4,
        )
        self.assertEqual(actual.shape, (0, 2051))

    def test_nonempty_widths_are_left_for_consumer_validation(self):
        for width in (1, 511, 512, 2051):
            with self.subTest(width=width):
                history = torch.arange(width, dtype=torch.int32).reshape(1, width)
                self.assertIs(pad_empty_indexer_history(history, 512), history)


if __name__ == "__main__":
    unittest.main()

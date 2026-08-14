from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.model_desc.kimi_k3_checkpoint import (
    join_packed_conv_outputs,
    packed_checkpoint_layout,
)


class PackedCheckpointLayoutTest(unittest.TestCase):
    def test_multi_sequence_tail_offsets(self) -> None:
        counts, offsets = packed_checkpoint_layout([1306, 1269, 1871], 512)
        self.assertEqual(counts, [3, 3, 4])
        self.assertEqual(offsets, [0, 3, 6, 10])

    def test_exact_pages_have_no_extra_checkpoint(self) -> None:
        counts, offsets = packed_checkpoint_layout([512, 1024, 1536], 512)
        self.assertEqual(counts, [1, 2, 3])
        self.assertEqual(offsets, [0, 1, 3, 6])

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "interval"):
            packed_checkpoint_layout([64], 0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            packed_checkpoint_layout([64, 0], 64)

    def test_multi_sequence_preserves_output_target_storage(self) -> None:
        output_target = torch.empty((5, 3))
        first = output_target[:2]
        second = output_target[2:]
        first.fill_(1)
        second.fill_(2)

        output = join_packed_conv_outputs([first, second], output_target)

        self.assertEqual(output.data_ptr(), output_target.data_ptr())
        torch.testing.assert_close(output[:2], torch.ones((2, 3)))
        torch.testing.assert_close(output[2:], torch.full((3, 3), 2.0))

    def test_multi_sequence_without_target_is_concatenated(self) -> None:
        first = torch.ones((2, 3))
        second = torch.full((3, 3), 2.0)

        output = join_packed_conv_outputs([first, second], None)

        torch.testing.assert_close(output[:2], first)
        torch.testing.assert_close(output[2:], second)


if __name__ == "__main__":
    unittest.main()

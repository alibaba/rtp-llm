import unittest

import torch

from rtp_llm.models_py.model_desc.bert import (
    pad_flat_hidden_states,
    unpad_padded_hidden_states,
)


class BertDenseMaskedAttentionTest(unittest.TestCase):
    def test_pad_and_unpad_hidden_states_preserve_token_order(self):
        hidden_states = torch.arange(6 * 3, dtype=torch.float32).view(6, 3)
        input_lengths = torch.tensor([2, 1, 3], dtype=torch.int32)

        padded = pad_flat_hidden_states(hidden_states, input_lengths)

        self.assertEqual((3, 3, 3), tuple(padded.shape))
        torch.testing.assert_close(padded[0, :2], hidden_states[:2])
        torch.testing.assert_close(padded[1, :1], hidden_states[2:3])
        torch.testing.assert_close(padded[2, :3], hidden_states[3:6])
        torch.testing.assert_close(padded[0, 2], torch.zeros(3))
        torch.testing.assert_close(padded[1, 1:], torch.zeros(2, 3))

        restored = unpad_padded_hidden_states(padded, input_lengths)

        torch.testing.assert_close(restored, hidden_states)


if __name__ == "__main__":
    unittest.main()

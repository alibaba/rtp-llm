"""Semantic examples for K3's shifted draft input contract."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.kimi_k3_mtp_contract import (
    mtp_positions,
    restore_shifted_media_tokens,
)


class KimiK3MtpContractTest(unittest.TestCase):
    def test_shifted_media_stays_inside_each_request(self):
        # A: target [10, image(2), 11] -> draft [image(2), 11, 12].
        # B begins with its image, so the first feature has no draft row.
        ids = torch.tensor([-101, -102, 11, -201, 22, 21], dtype=torch.int32)
        boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)
        result = restore_shifted_media_tokens(
            ids,
            [torch.zeros(2, 4), torch.zeros(2, 4)],
            torch.tensor([1, 3], dtype=torch.int32),
            boundaries,
            media_token_id=99,
            cu_seqlens_host=boundaries,
        )
        self.assertEqual(result.tolist(), [99, 99, 11, 99, 22, 21])
        self.assertEqual(ids.tolist(), [-101, -102, 11, -201, 22, 21])

    def test_shifted_media_handles_prefix_truncated_feature(self):
        # Only the tail of a three-row feature remains in this request window.
        ids = torch.tensor([-102, 11, 12], dtype=torch.int32)
        result = restore_shifted_media_tokens(
            ids,
            [torch.zeros(3, 4)],
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            media_token_id=99,
        )
        self.assertEqual(result.tolist(), [99, 11, 12])

    def test_media_metadata_must_not_spill_into_next_request(self):
        ids = torch.tensor([-101, 10, 20, 21], dtype=torch.int32)
        boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "crosses a request boundary"):
            restore_shifted_media_tokens(
                ids,
                [torch.zeros(3, 4)],
                torch.tensor([1], dtype=torch.int32),
                boundaries,
                media_token_id=99,
            )
        with self.assertRaisesRegex(ValueError, "matching host locations"):
            restore_shifted_media_tokens(
                ids, [torch.zeros(1, 4)], None, boundaries, media_token_id=99
            )

    def test_positions_use_decode_cache_length_and_prefill_prefix(self):
        inputs = SimpleNamespace(
            input_ids=torch.arange(4),
            combo_position_ids=None,
            attention_inputs=SimpleNamespace(
                is_prefill=False,
                input_lengths=torch.tensor([1, 3], dtype=torch.int32),
                sequence_lengths=torch.tensor([9], dtype=torch.int32),
                prefix_lengths=torch.tensor([4], dtype=torch.int32),
            ),
        )
        self.assertEqual(mtp_positions(inputs).tolist(), [9, 4, 5, 6])
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.input_lengths = torch.tensor([2, 2])
        inputs.attention_inputs.prefix_lengths = torch.tensor([5, 10])
        self.assertEqual(mtp_positions(inputs).tolist(), [5, 6, 10, 11])
        inputs.combo_position_ids = torch.tensor([8, 9, 10, 11])
        self.assertIs(mtp_positions(inputs), inputs.combo_position_ids)
        inputs.combo_position_ids = torch.tensor([8, 9])
        with self.assertRaisesRegex(ValueError, "one absolute position"):
            mtp_positions(inputs)


if __name__ == "__main__":
    unittest.main()

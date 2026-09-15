import unittest

import torch

from rtp_llm.ops.fused_rope_kvcache_op import (
    _decode_uses_sequence_lengths,
    _get_prefill_position_ids_keyword,
)


class FusedRopeKVCacheCompatTest(unittest.TestCase):
    def test_uses_position_ids_for_existing_kernel_api(self):
        def prefill_op(*, position_ids=None) -> torch.Tensor:
            return position_ids

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "position_ids"
        )

    def test_uses_cp_position_ids_for_cuda13_arm_kernel_api(self):
        def prefill_op(*, cp_position_ids=None) -> torch.Tensor:
            return cp_position_ids

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "cp_position_ids"
        )

    def test_uses_legacy_keyword_for_generic_kwargs_api(self):
        def prefill_op(**kwargs) -> torch.Tensor:
            return kwargs["position_ids"]

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "position_ids"
        )

    def test_decode_uses_sequence_lengths_for_existing_kernel_api(self):
        def decode_op(qkv, position_ids, sequence_lengths, batch_size):
            return qkv

        self.assertTrue(_decode_uses_sequence_lengths(decode_op))

    def test_decode_omits_sequence_lengths_for_cuda13_arm_kernel_api(self):
        def decode_op(qkv, position_ids, batch_size):
            return qkv

        self.assertFalse(_decode_uses_sequence_lengths(decode_op))

    def test_decode_rejects_unknown_kernel_api(self):
        def decode_op(qkv, position_ids):
            return qkv

        with self.assertRaisesRegex(RuntimeError, "batch_size is missing"):
            _decode_uses_sequence_lengths(decode_op)


if __name__ == "__main__":
    unittest.main()

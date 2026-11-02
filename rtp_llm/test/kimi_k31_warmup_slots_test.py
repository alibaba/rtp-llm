"""Reserved block zero is legal only with an explicit startup-warmup flag."""
import copy
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
    build_mla_page_rr_slot_mapping,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


class DsaWarmupSlotsTest(unittest.TestCase):
    def slots(self, table, **kwargs):
        return build_mla_page_rr_slot_mapping(
            torch.tensor([0, 127, 1024, 2048], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32), table,
            1024, 2, 0, kernel_page_tokens=128, **kwargs)

    def test_real_request_still_rejects_reserved_zero(self):
        with self.assertRaisesRegex(RuntimeError, "null or unused"):
            self.slots(torch.zeros(1, 16, dtype=torch.int32))

    def test_only_warmup_accepts_zero_and_preserves_owner_mapping(self):
        actual = self.slots(torch.zeros(1, 16, dtype=torch.int32),
                            allow_reserved_block_zero=True)
        torch.testing.assert_close(actual, torch.tensor([0, 127, -1, 0]), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "null or unused"):
            self.slots(torch.full((1, 16), -1, dtype=torch.int32),
                       allow_reserved_block_zero=True)

    def test_padding_does_not_require_any_block(self):
        actual = self.slots(torch.zeros(1, 16, dtype=torch.int32),
                            valid_token_mask=torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(actual, torch.full((4,), -1), rtol=0, atol=0)

    def test_shared_attention_flag_defaults_false_and_survives_tag_copy(self):
        inputs = PyAttentionInputs()
        self.assertFalse(inputs.is_warmup)
        inputs.is_warmup = True
        self.assertTrue(copy.copy(inputs).is_warmup)

    def test_kda_warmup_uses_private_real_state_slot_and_executes_temporal_path(self):
        from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaPrefill
        from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import plan_state_sequences
        inputs = PyAttentionInputs()
        inputs.prefix_lengths = torch.tensor([128], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = torch.zeros(1, 2, dtype=torch.int32)
        cache = torch.full((2, 16), 7, dtype=torch.uint8)
        self.assertIs(KdaPrefill._warmup_cache_inputs(inputs, cache)[0], inputs)
        with self.assertRaisesRegex(ValueError, 'missing a cached prefix state'):
            KdaPrefill._validate_prefix_state(inputs, 128)
        inputs.is_warmup = True
        with self.assertRaisesRegex(ValueError, 'missing a cached prefix state'):
            KdaPrefill._validate_prefix_state(inputs, 128)
        view, scratch = KdaPrefill._warmup_cache_inputs(inputs, cache)
        KdaPrefill._validate_prefix_state(view, 128)
        self.assertNotEqual(scratch.data_ptr(),cache.data_ptr())
        self.assertEqual(scratch.dtype,cache.dtype)
        self.assertFalse(scratch.count_nonzero())
        self.assertTrue((cache == 7).all())
        self.assertFalse(inputs.kv_cache_kernel_block_id.count_nonzero())
        for prefix in (0,128):
            sequences = plan_state_sequences([0,64],[prefix],
                view.kv_cache_kernel_block_id.tolist(),128)
            self.assertEqual(len(sequences[0].segments),1)
            self.assertEqual(sequences[0].segments[0].cache_block,1)

        from rtp_llm.models_py.modules.hybrid.query_layout import AttentionInputView
        proxy = AttentionInputView(inputs, physical_token_count=64)
        copied, _ = KdaPrefill._warmup_cache_inputs(proxy, cache)
        self.assertIs(copied.original, inputs)
        self.assertEqual(copied.physical_token_count,64)
        self.assertFalse(proxy.kv_cache_kernel_block_id.count_nonzero())

        inputs.kv_cache_kernel_block_id = torch.tensor([[0,0],[0,-1]],dtype=torch.int32)
        mapped, scratch = KdaPrefill._warmup_cache_inputs(inputs,cache)
        torch.testing.assert_close(mapped.kv_cache_kernel_block_id,
            torch.tensor([[1,1],[2,-1]],dtype=torch.int32),rtol=0,atol=0)
        self.assertEqual(scratch.shape[0],3)


if __name__ == "__main__":
    unittest.main()

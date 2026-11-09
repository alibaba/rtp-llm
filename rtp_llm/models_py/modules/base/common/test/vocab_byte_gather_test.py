import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.fp8_embedding import Fp8VocabEmbedding
from rtp_llm.models_py.triton_kernels.common.fp8_vocab_gather import raw_rows
from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import vocab_rows
from rtp_llm.models_py.triton_kernels.common.test.vocab_gather_fixture import (
    VocabByteGatherTestFixture,
)
from rtp_llm.ops import RoleType

MODULE = "rtp_llm.models_py.modules.base.common.fp8_embedding"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VocabByteGatherTest(VocabByteGatherTestFixture, unittest.TestCase):

    def test_module_collects_scales_once_and_uses_byte_tp_consumer(self):
        module = Fp8VocabEmbedding(
            self.weights[0], 2, self.weights[0].device, self.scales[:1]
        )
        calls = []
        ids = torch.arange(1024, device="cuda", dtype=torch.int32) % 257
        mask = ids % 5 != 0

        def gather(local, group):
            calls.append(local.dtype)
            if local.dtype == torch.float32:
                return self.scales.clone()
            self.assertEqual(local.dtype, torch.uint8)
            return torch.cat([local, raw_rows(ids, self.weights[1])])

        with patch(MODULE + ".all_gather", side_effect=gather):
            module.prepare_byte_gather()
            module.prepare_byte_gather()
            actual = module.forward_hyper(ids, branches=4, text_tokens_mask=mask)
        self.assertEqual(calls, [torch.float32, torch.uint8])
        self.assertTrue(
            torch.equal(
                actual.view(torch.uint8), self.reference(ids, mask).view(torch.uint8)
            )
        )
        self.assertEqual(module.weight.data_ptr(), self.weights[0].data_ptr())

    def test_small_and_unqualified_branches_keep_bf16_transport(self):
        module = Fp8VocabEmbedding(
            self.weights[0], 2, self.weights[0].device, self.scales[:1]
        )
        module.column_scales = self.scales
        for rows, branches in ((4, 4), (1024, 1), (12289, 4)):
            ids = torch.arange(rows, device="cuda", dtype=torch.int64) % 257

            def gather(local, group):
                self.assertEqual(local.dtype, torch.bfloat16)
                return torch.cat(
                    [local, vocab_rows(ids, self.weights[1], self.scales[1:])]
                )

            with patch(MODULE + ".all_gather", side_effect=gather):
                actual = module.forward_hyper(ids, branches=branches)
            self.assertTrue(torch.equal(actual, self.reference(ids, branches=branches)))

    def test_real_role_enum_prepares_prefill_and_fusion_scales(self):
        for role in (RoleType.PREFILL, RoleType.PDFUSION, "PREFILL", "PDFUSION"):
            module = Fp8VocabEmbedding(
                self.weights[0], 2, self.weights[0].device, self.scales[:1]
            )
            with patch(MODULE + ".all_gather", return_value=self.scales) as gather:
                module.prepare_byte_gather(role=role)
                module.prepare_byte_gather(role=role)
            self.assertEqual(gather.call_count, 1)
            self.assertTrue(torch.equal(module.column_scales, self.scales))

    def test_real_decode_role_keeps_initialization_collectives_disabled(self):
        for role in (RoleType.DECODE, "DECODE"):
            module = Fp8VocabEmbedding(
                self.weights[0], 2, self.weights[0].device, self.scales[:1]
            )
            with patch(
                MODULE + ".all_gather", side_effect=AssertionError("decode init")
            ):
                module.prepare_byte_gather(role=role)
            self.assertIsNone(module.column_scales)

    def test_prepare_rejects_capture_and_bad_scale_gather(self):
        module = Fp8VocabEmbedding(
            self.weights[0], 2, self.weights[0].device, self.scales[:1]
        )
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                module.prepare_byte_gather()
        with patch(MODULE + ".all_gather", return_value=self.scales[:1]):
            with self.assertRaisesRegex(ValueError, "column scales"):
                module.prepare_byte_gather()
        self.assertIsNone(module.column_scales)


if __name__ == "__main__":
    unittest.main()

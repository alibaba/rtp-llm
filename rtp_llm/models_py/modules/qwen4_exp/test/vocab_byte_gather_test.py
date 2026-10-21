import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.fp8_embedding import Fp8VocabEmbedding
from rtp_llm.models_py.triton_kernels.common.fp8_vocab_gather import (
    dequant_hyper,
    raw_rows,
)
from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import (
    gather_hyper_rows,
    vocab_rows,
)
from rtp_llm.ops import RoleType

MODULE = "rtp_llm.models_py.modules.base.common.fp8_embedding"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VocabByteGatherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(578)
        self.weights = [
            torch.randn(257, 1280, device="cuda").to(torch.float8_e4m3fn)
            for _ in range(2)
        ]
        self.scales = torch.tensor([0.119713, 0.27351], device="cuda")

    def reference(self, ids, mask=None, branches=4):
        local = [
            vocab_rows(ids, w, self.scales[i : i + 1], mask)
            for i, w in enumerate(self.weights)
        ]
        return gather_hyper_rows(torch.cat(local), ids.numel(), 1280, 2, branches)

    def bytes_consumer(self, ids, mask=None, branches=4):
        gathered = torch.cat([raw_rows(ids, w) for w in self.weights])
        return dequant_hyper(
            gathered, self.scales, ids.numel(), 1280, 2, branches, mask
        )

    def test_rank_scales_masks_invalid_tokens_and_bf16_rounding(self):
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor(
                [-1, 0, 256, 0, 257, 0, 17, 0, 3, 0, 8, 0], device="cuda", dtype=dtype
            )[::2]
            mask = torch.tensor(
                [1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 0], device="cuda", dtype=torch.int32
            )[::2]
            for scales in (
                [0.119713, 0.27351],
                [0.0, 1.0],
                [float("inf"), float("nan")],
            ):
                self.scales.copy_(torch.tensor(scales, device="cuda"))
                for selected_mask in (None, mask):
                    for branches in (1, 4):
                        actual = self.bytes_consumer(ids, selected_mask, branches)
                        expected = self.reference(ids, selected_mask, branches)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), expected.view(torch.uint8)
                            )
                        )
        self.assertEqual(self.bytes_consumer(ids[:0]).shape, (0, 10240))

    def test_live_graph_reads_ids_masks_bytes_and_all_rank_scales(self):
        ids = torch.randint(0, 257, (1024,), device="cuda", dtype=torch.int32)
        mask = torch.ones_like(ids, dtype=torch.bool)
        for _ in range(3):
            self.bytes_consumer(ids, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.bytes_consumer(ids, mask)
        for gain in (0.01, 1.0, 3.0):
            ids.random_(0, 257)
            mask.copy_(ids % 3 != 0)
            self.weights[0].fill_(gain)
            self.weights[1].fill_(gain * 2)
            self.scales.copy_(torch.tensor([gain * 0.17, gain * 0.31], device="cuda"))
            graph.replay()
            self.assertTrue(
                torch.equal(
                    actual.view(torch.uint8),
                    self.reference(ids, mask).view(torch.uint8),
                )
            )
        graph.reset()

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

    def test_invalid_layouts_fail_before_kernel(self):
        ids = torch.arange(4, device="cuda", dtype=torch.int32)
        for bad in (ids.cpu(), ids.float(), ids.reshape(2, 2)):
            with self.assertRaises(ValueError):
                raw_rows(bad, self.weights[0])
        gathered = torch.cat([raw_rows(ids, w) for w in self.weights])
        for bad in (self.scales.cpu(), self.scales[:1], self.scales.to(torch.bfloat16)):
            with self.assertRaises(ValueError):
                dequant_hyper(gathered, bad, 4, 1280, 2, 4)
        with self.assertRaises(ValueError):
            dequant_hyper(gathered, self.scales, 4, 1280, 2, 4, ids[:1])


if __name__ == "__main__":
    unittest.main()

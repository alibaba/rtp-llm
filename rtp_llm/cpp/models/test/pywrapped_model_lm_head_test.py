import unittest
from unittest.mock import Mock

import torch
from torch import nn

from rtp_llm.cpp.models.test.libth_pywrapped_model_lm_head_test import run_lm_head
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import clear_comm_ops, register_comm_ops


class ProjectionModel(GptModelBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.nvfp4_lm_head = nn.Linear(
            8, 16, bias=False, device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad():
            self.nvfp4_lm_head.weight.copy_(
                torch.arange(128, device="cuda").reshape(16, 8) / 128
            )
        self.projected_rows = []

    def forward(self, inputs, fmha_impl=None):
        raise AssertionError("This fixture exercises C++ output processing only")

    def compute_quantized_logits(self, hidden):
        self.projected_rows.append(hidden.shape[0])
        self.assert_outside_graph()
        return super().compute_quantized_logits(hidden)

    @staticmethod
    def assert_outside_graph():
        if torch.cuda.is_current_stream_capturing():
            raise AssertionError("LM head must run after the model graph replay")


class PyWrappedModelLmHeadTest(unittest.TestCase):
    def setUp(self):
        self.model = ProjectionModel()
        self.hidden = (
            torch.arange(48, device="cuda").reshape(6, 8).to(torch.bfloat16) / 16
        )
        self.indexes = torch.tensor([5, 1], dtype=torch.int32, pin_memory=True)
        # Dispatch sentinel; the callback's projection is a deterministic BF16
        # reference. Packed codes/scales/GEMM have a separate SM120 test.
        self.packed = torch.empty((16, 4), dtype=torch.uint8, device="cuda")

    def test_prefill_all_logits_and_decode_share_cpp_processing(self):
        for packed in (False, True):
            for need_all in (False, True):
                for has_context in (False, True):
                    with (
                        self.subTest(
                            packed=packed, need_all=need_all, has_context=has_context
                        ),
                        torch.no_grad(),
                    ):
                        kernel = (
                            self.packed if packed else self.model.nvfp4_lm_head.weight
                        )
                        result = run_lm_head(
                            self.model,
                            kernel,
                            self.hidden,
                            self.indexes,
                            has_context_request=has_context,
                            need_all_logits=need_all,
                            output_vocab_size=13,
                        )
                        # The dense C++ GEMM accumulates into FP32. The packed
                        # projection returns BF16 before the shared output path.
                        all_logits = (
                            self.model.nvfp4_lm_head(self.hidden).float()
                            if packed
                            else self.hidden.float()
                            @ self.model.nvfp4_lm_head.weight.float().t()
                        )[:, :13]
                        selected = has_context and not need_all
                        expected = (
                            all_logits[[5, 1]] if selected or need_all else all_logits
                        )
                        torch.testing.assert_close(result["logits"], expected)
                        self.assertEqual(result["logits"].dtype, torch.float32)
                        torch.testing.assert_close(
                            result["hidden_states"],
                            self.hidden[[5, 1]] if selected else self.hidden,
                        )
                        torch.testing.assert_close(
                            result["all_hidden_states"], self.hidden
                        )
                        if need_all:
                            torch.testing.assert_close(result["all_logits"], all_logits)
                        else:
                            self.assertIsNone(result["all_logits"])
                        if packed:
                            self.assertEqual(
                                self.model.projected_rows[-1], 2 if selected else 6
                            )
                        else:
                            self.assertEqual(self.model.projected_rows, [])

    def test_cp_selected_rows_use_same_projection_and_vocab_crop(self):
        selected = self.hidden[[5, 1]]
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                selected,
                self.indexes,
                already_selected=True,
                output_vocab_size=13,
            )
            expected = self.model.nvfp4_lm_head(selected).float()[:, :13]
        torch.testing.assert_close(result["logits"], expected)
        torch.testing.assert_close(result["all_hidden_states"], selected)
        self.assertEqual(self.model.projected_rows, [2])

    def test_tp_joins_bf16_vocabulary_shards_before_fp32(self):
        communicated = []

        def gather(receivers, mode, senders, inplace):
            # Emulate the peer shard while exercising the actual C++ gather
            # adapter, rank-major layout conversion, crop and FP32 conversion.
            buffer = receivers[0]
            local = buffer.reshape(2, 2, 16)[0].clone()
            communicated.append(local)
            buffer.reshape(2, 2, 16).copy_(torch.stack((local, local + 16)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    self.packed,
                    self.hidden,
                    self.indexes,
                    tp_size=2,
                    output_vocab_size=29,
                )
                local = self.model.nvfp4_lm_head(self.hidden[[5, 1]])
            self.assertEqual(len(communicated), 1)
            self.assertEqual(communicated[0].dtype, torch.bfloat16)
            expected = torch.cat((local, local + 16), dim=1)[:, :29].float()
            torch.testing.assert_close(result["logits"], expected)
        finally:
            clear_comm_ops()

    def test_row_selected_all_logits_gather(self):
        # need_all_logits with explicit row indexes: the local lm head runs the full GEMM
        # (bitwise-identical local logits), then only [lm rows ++ scoring rows] are kept
        # for the (TP) gather.
        scoring = torch.tensor([0, 2, 3], dtype=torch.int32, pin_memory=True)
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                self.hidden,
                self.indexes,
                need_all_logits=True,
                all_logits_row_indexes=scoring,
                output_vocab_size=13,
            )
        full = self.model.nvfp4_lm_head(self.hidden).float()[:, :13]
        # rows = indexes([5,1]) ++ scoring([0,2,3]) = [5,1,0,2,3]
        expected_all = full[[5, 1, 0, 2, 3]]
        # Selected rows are claimed bitwise-identical to the unselected path:
        # exact equality, no tolerance.
        torch.testing.assert_close(result["all_logits"], expected_all, rtol=0, atol=0)
        torch.testing.assert_close(result["logits"], full[[5, 1]], rtol=0, atol=0)
        self.assertEqual(result["all_logits_lm_rows"], 2)
        # hidden_states/all_hidden_states keep the legacy full-sequence contract.
        torch.testing.assert_close(result["hidden_states"], self.hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            result["all_hidden_states"], self.hidden, rtol=0, atol=0
        )
        # The projection ran on all 6 token rows (full GEMM, select-after-mm).
        self.assertEqual(self.model.projected_rows[-1], 6)

    def test_row_selected_empty_indexes_keep_only_lm_rows(self):
        # A defined-but-empty row set still selects the row-gather path: the gather
        # then ships only the lm_output_indexes rows.
        scoring = torch.tensor([], dtype=torch.int32, pin_memory=True)
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                self.hidden,
                self.indexes,
                need_all_logits=True,
                all_logits_row_indexes=scoring,
                output_vocab_size=13,
            )
        full = self.model.nvfp4_lm_head(self.hidden).float()[:, :13]
        torch.testing.assert_close(result["all_logits"], full[[5, 1]], rtol=0, atol=0)
        torch.testing.assert_close(result["logits"], full[[5, 1]], rtol=0, atol=0)
        self.assertEqual(result["all_logits_lm_rows"], 2)

    def test_row_selected_tp_gather_sends_only_needed_rows(self):
        gathered_shapes = []

        def gather(receivers, mode, senders, inplace):
            buffer = receivers[0]
            rows = buffer.shape[0]
            gathered_shapes.append(tuple(buffer.shape))
            local = buffer.reshape(2, rows, 16)[0].clone()
            buffer.reshape(2, rows, 16).copy_(torch.stack((local, local + 16)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            scoring = torch.tensor([4], dtype=torch.int32, pin_memory=True)
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    self.packed,
                    self.hidden,
                    self.indexes,
                    need_all_logits=True,
                    all_logits_row_indexes=scoring,
                    tp_size=2,
                    output_vocab_size=29,
                )
            # 2 lm rows + 1 scoring row -> one 3-row gather, not 6 rows.
            self.assertEqual(gathered_shapes, [(3, 32)])
            expected = torch.cat(
                (
                    self.model.nvfp4_lm_head(self.hidden[[5, 1, 4]]),
                    self.model.nvfp4_lm_head(self.hidden[[5, 1, 4]]) + 16,
                ),
                dim=1,
            )[:, :29].float()
            torch.testing.assert_close(result["all_logits"], expected, rtol=0, atol=0)
            torch.testing.assert_close(result["logits"], expected[:2], rtol=0, atol=0)
            self.assertEqual(result["all_logits_lm_rows"], 2)
        finally:
            clear_comm_ops()

    def test_invalid_python_projection_is_rejected_at_boundary(self):
        self.model.compute_quantized_logits = lambda hidden: hidden
        with self.assertRaisesRegex(RuntimeError, "BF16 local logits"):
            run_lm_head(self.model, self.packed, self.hidden, self.indexes)

    def test_non_bf16_projection_is_rejected_before_tp_gather(self):
        gather = Mock()
        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            for dtype in (torch.float16, torch.float32):
                for already_selected in (False, True):
                    with self.subTest(dtype=dtype, already_selected=already_selected):
                        self.model.compute_quantized_logits = (
                            lambda hidden: torch.zeros(
                                (hidden.shape[0], 16), device=hidden.device, dtype=dtype
                            )
                        )
                        with self.assertRaisesRegex(RuntimeError, "BF16 local logits"):
                            run_lm_head(
                                self.model,
                                self.packed,
                                self.hidden,
                                self.indexes,
                                already_selected=already_selected,
                                tp_size=2,
                            )
            gather.assert_not_called()
        finally:
            clear_comm_ops()

    def test_public_bf16_projection_callback_is_supported(self):
        class PublicProjection:
            def initialize(inner, resources):
                return True

            def forward(inner, inputs, fmha_impl=None):
                raise AssertionError("Only projection should be called")

            def compute_quantized_logits(inner, hidden):
                return self.model.compute_quantized_logits(hidden)

        with torch.no_grad():
            result = run_lm_head(
                PublicProjection(), self.packed, self.hidden, self.indexes
            )
            expected = self.model.compute_quantized_logits(self.hidden[[5, 1]]).float()
        torch.testing.assert_close(result["logits"], expected)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock

import torch
from torch import nn

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import clear_comm_ops, register_comm_ops
from rtp_llm.cpp.models.test.libth_pywrapped_model_lm_head_test import run_lm_head


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

    def test_tp_joins_vocabulary_shards_before_fp32(self):
        for tp_size in (2, 4):
            for packed in (False, True):
                for already_selected, has_context, need_all in (
                    (False, True, False),
                    (False, False, False),
                    (False, True, True),
                    (True, True, False),
                ):
                    with self.subTest(
                        tp_size=tp_size,
                        packed=packed,
                        already_selected=already_selected,
                        has_context=has_context,
                        need_all=need_all,
                    ):
                        self.check_tp_logits(
                            tp_size, packed, already_selected, has_context, need_all
                        )

    def test_tp_gather_accepts_noncontiguous_projection(self):
        project = self.model.compute_quantized_logits

        def strided_projection(hidden):
            local = project(hidden)
            result = torch.stack((local, local), dim=-1)[..., 0]
            self.assertFalse(result.is_contiguous())
            return result

        self.model.compute_quantized_logits = strided_projection
        self.check_tp_logits(2, True, False, True, False)

    def check_tp_logits(self, tp_size, packed, already_selected, has_context, need_all):
        communicated = []
        selected = already_selected or (has_context and not need_all)
        projected_hidden = self.hidden[[5, 1]] if selected else self.hidden
        rows = projected_hidden.shape[0]

        def gather(receivers, mode, senders, inplace):
            # The production adapter sends the local shard separately. The
            # receive buffer has no initialized rank slice before the gather.
            self.assertEqual(mode, 0)  # ParallelMode::TP
            self.assertFalse(inplace)
            self.assertEqual(len(receivers), 1)
            self.assertEqual(len(senders), 1)
            local = senders[0]
            buffer = receivers[0]
            self.assertEqual(tuple(local.shape), (rows, 16))
            self.assertEqual(tuple(buffer.shape), (tp_size * rows, 16))
            self.assertTrue(local.is_contiguous())
            self.assertTrue(buffer.is_contiguous())
            self.assertEqual(local.dtype, buffer.dtype)
            communicated.append(local.clone())
            buffer.reshape(tp_size, rows, 16).copy_(
                torch.stack([local + 16 * rank for rank in range(tp_size)])
            )

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    self.packed if packed else self.model.nvfp4_lm_head.weight,
                    self.hidden[[5, 1]] if already_selected else self.hidden,
                    self.indexes,
                    has_context_request=has_context,
                    need_all_logits=need_all,
                    already_selected=already_selected,
                    tp_size=tp_size,
                    output_vocab_size=16 * tp_size - 3,
                )
                local = (
                    self.model.nvfp4_lm_head(projected_hidden)
                    if packed
                    else projected_hidden.float()
                    @ self.model.nvfp4_lm_head.weight.float().t()
                )
            self.assertEqual(len(communicated), 1)
            self.assertEqual(
                communicated[0].dtype, torch.bfloat16 if packed else torch.float32
            )
            torch.testing.assert_close(communicated[0], local)
            all_logits = torch.cat(
                [local + 16 * rank for rank in range(tp_size)], dim=1
            )[:, : 16 * tp_size - 3].float()
            expected = all_logits[[5, 1]] if need_all else all_logits
            torch.testing.assert_close(result["logits"], expected)
            if need_all:
                torch.testing.assert_close(result["all_logits"], all_logits)
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

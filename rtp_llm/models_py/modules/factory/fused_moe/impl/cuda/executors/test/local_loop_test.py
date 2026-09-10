import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    normalize_moe_w13_gate_up,
    split_moe_w13_gate_up,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
    _validate_topk_indices,
)


class _WeightedIdentity(nn.Module):
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return x * weights


class LocalLoopExecutorTest(unittest.TestCase):
    @staticmethod
    def _make_executor() -> LocalLoopExecutor:
        executor = LocalLoopExecutor.__new__(LocalLoopExecutor)
        nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(max_tokens_per_rank=4096, dim=2, ep_size=1)
        executor.experts = nn.ModuleList([_WeightedIdentity()])
        return executor

    def test_eager_sums_duplicate_slots_for_the_same_expert(self):
        executor = LocalLoopExecutor.__new__(LocalLoopExecutor)
        nn.Module.__init__(executor)
        executor.experts = nn.ModuleList([_WeightedIdentity(), _WeightedIdentity()])
        x = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.2]])
        indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
        y = torch.zeros_like(x)

        result = executor._forward_eager(
            x,
            weights,
            indices,
            y,
            local_start=0,
            local_end=2,
        )

        expected = torch.tensor([[2.0, 3.0], [3.5, 4.9]])
        torch.testing.assert_close(result, expected)

    def test_topk_index_validation_rejects_negative_and_upper_bound(self):
        for invalid in (-1, 2):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "outside"):
                    _validate_topk_indices(
                        torch.tensor([[0, invalid]], dtype=torch.long),
                        num_experts=2,
                    )

    def test_output_buffers_are_private_and_sized_to_live_tokens(self):
        first_executor = self._make_executor()
        second_executor = self._make_executor()
        indices = torch.zeros((2, 1), dtype=torch.long)
        weights = torch.ones((2, 1))

        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            first_output = first_executor._forward_into_buf(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                weights,
                indices,
                local_start=0,
                local_end=1,
            )
            first_snapshot = first_output.clone()
            second_output = second_executor._forward_into_buf(
                torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                weights,
                indices,
                local_start=0,
                local_end=1,
            )

        torch.testing.assert_close(first_output, first_snapshot)
        self.assertNotEqual(first_output.data_ptr(), second_output.data_ptr())
        self.assertEqual(first_output.shape, (2, 2))
        self.assertEqual(second_output.shape, (2, 2))
        buffer_ref = weakref.ref(
            first_output if first_output._base is None else first_output._base
        )
        del first_output
        self.assertIsNone(buffer_ref())

    def test_next_call_starts_zeroed_and_does_not_overwrite_live_output(self):
        executor = self._make_executor()
        x = torch.ones(2, 2)
        indices = torch.zeros(2, 1, dtype=torch.long)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            first = executor._forward_into_buf(x, torch.ones(2, 1), indices, 0, 1)
            second = executor._forward_into_buf(x, torch.zeros(2, 1), indices, 0, 1)
        self.assertNotEqual(second.data_ptr(), first.data_ptr())
        torch.testing.assert_close(first, x)
        torch.testing.assert_close(second, torch.zeros_like(x))

    def test_w13_layout_contract_maps_both_producer_orders_to_gate_up(self):
        up = torch.full((2, 2, 4), 3, dtype=torch.int8)
        gate = torch.full((2, 2, 4), 1, dtype=torch.int8)
        up_scale = torch.full((2, 2, 4), 13, dtype=torch.uint8)
        gate_scale = torch.full((2, 2, 4), 11, dtype=torch.uint8)

        for layout, weights, scales in (
            ("gate_up", (gate, up), (gate_scale, up_scale)),
            ("up_gate", (up, gate), (up_scale, gate_scale)),
        ):
            with self.subTest(layout=layout):
                w13 = torch.cat(weights, dim=-2)
                s13 = torch.cat(scales, dim=-2)
                split_gate, split_up = split_moe_w13_gate_up(w13, 2, layout)
                torch.testing.assert_close(split_gate, gate)
                torch.testing.assert_close(split_up, up)
                split_gate_scale, split_up_scale = split_moe_w13_gate_up(s13, 2, layout)
                self.assertFalse(split_gate_scale.is_contiguous())
                self.assertFalse(split_up_scale.is_contiguous())
                torch.testing.assert_close(split_gate_scale, gate_scale)
                torch.testing.assert_close(split_up_scale, up_scale)

                normalized_w13, normalized_s13 = normalize_moe_w13_gate_up(
                    w13, s13, 2, layout
                )
                torch.testing.assert_close(
                    normalized_w13, torch.cat((gate, up), dim=-2)
                )
                torch.testing.assert_close(
                    normalized_s13, torch.cat((gate_scale, up_scale), dim=-2)
                )
                if hasattr(torch, "float8_e8m0fnu"):
                    _, normalized_ue8m0 = normalize_moe_w13_gate_up(
                        w13, s13.view(torch.float8_e8m0fnu), 2, layout
                    )
                    self.assertEqual(normalized_ue8m0.dtype, torch.float8_e8m0fnu)
                    torch.testing.assert_close(
                        normalized_ue8m0.view(torch.uint8), normalized_s13
                    )

    def test_w13_layout_contract_rejects_invalid_layout(self):
        w13 = torch.ones((1, 4, 1), dtype=torch.int8)
        s13 = torch.ones((1, 4, 1), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "moe_w1_layout"):
            split_moe_w13_gate_up(w13, 2, "invalid")
        with self.assertRaisesRegex(ValueError, "moe_w1_layout"):
            normalize_moe_w13_gate_up(w13, s13, 2, "invalid")

    def test_w13_layout_contract_validates_weight_and_scale_rows(self):
        for layout in ("gate_up", "up_gate"):
            for rows in (3, 6):
                with self.subTest(layout=layout, rows=rows):
                    w13 = torch.ones((1, 4, 1), dtype=torch.int8)
                    s13 = torch.ones((1, 4, 1), dtype=torch.uint8)
                    wrong_w13 = torch.ones((1, rows, 1), dtype=torch.int8)
                    wrong_s13 = torch.ones((1, rows, 1), dtype=torch.uint8)
                    with self.assertRaisesRegex(ValueError, "rows, expected 4"):
                        split_moe_w13_gate_up(wrong_w13, 2, layout)
                    with self.assertRaisesRegex(ValueError, "rows, expected 4"):
                        normalize_moe_w13_gate_up(wrong_w13, s13, 2, layout)
                    with self.assertRaisesRegex(ValueError, "rows, expected 4"):
                        normalize_moe_w13_gate_up(w13, wrong_s13, 2, layout)


if __name__ == "__main__":
    unittest.main()

"""Shared wo_a output layout and arithmetic without accelerator libraries."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.grouped_projection import (
    batched_wo_a,
    project_wo_a,
)
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUPTPCLinear,
    PPUResidentBF16Linear,
)


def assert_bits(test, actual, expected):
    test.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


class WoaEntryTest(unittest.TestCase):
    def test_grouped_protocol_keeps_phase_kwargs_and_ignores_bf16_policy(self):
        value = torch.randn(3, 2, 16)
        projected = torch.randn(3, 2, 4)
        fallback = Mock(side_effect=AssertionError("PTPC must retain its provider"))

        class Grouped:
            forward_grouped = Mock(return_value=projected)

            @property
            def weight(self):
                raise AssertionError("entry must not reinterpret PTPC weights")

        op = Grouped()
        for prefill, kwargs in ((False, {}), (True, {"prefill": True})):
            self.assertIs(
                project_wo_a(value, op, prefill=prefill, bf16_projection=fallback),
                projected,
            )
            op.forward_grouped.assert_called_once_with(value, **kwargs)
            op.forward_grouped.reset_mock()
        fallback.assert_not_called()

    def test_legacy_grouped_protocol_needs_no_new_keyword(self):
        value = torch.randn(2, 2, 16)

        class Grouped:
            def forward_grouped(self, actual):
                return actual

        self.assertIs(project_wo_a(value, Grouped(), bf16_projection=Mock()), value)

    @torch.inference_mode()
    def test_real_ptpc_keeps_quantized_math_on_empty_and_strided_inputs(self):
        torch.manual_seed(9101)
        op = PPUPTPCLinear(
            torch.randn(8, 16).to(torch.float8_e4m3fn), torch.rand(8, 1) + 0.01
        )
        op.prepare_grouped(2, 4)
        fallback = Mock(side_effect=AssertionError("PTPC cannot fall through to BF16"))
        for rows in (0, 1, 8, 24, 32):
            value = torch.randn(rows * 2, 2, 16).bfloat16()[::2]
            for prefill in (False, True):
                expected = op.forward_grouped(value, prefill=prefill)
                actual = project_wo_a(
                    value, op, prefill=prefill, bf16_projection=fallback
                )
                assert_bits(self, actual, expected)
        fallback.assert_not_called()

    def test_bf16_policy_receives_original_view_and_resident_weight(self):
        value = torch.randn(6, 2, 16)[1:5:2]
        weight = torch.randn(8, 16)
        expected = torch.randn(2, 2, 4)
        callback = Mock(return_value=expected)
        actual = project_wo_a(
            value,
            SimpleNamespace(weight=weight),
            bf16_projection=callback,
            prefill=True,
        )
        self.assertIs(actual, expected)
        self.assertIs(callback.call_args.args[0], value)
        self.assertIs(callback.call_args.args[1], weight)
        callback.assert_called_once()

    def test_provider_failures_propagate_without_bf16_retry(self):
        value = torch.randn(1, 2, 16)
        fallback = Mock(
            side_effect=AssertionError("must not retry after provider failure")
        )
        provider = Mock(side_effect=RuntimeError("selected PTPC failed"))
        with self.assertRaisesRegex(RuntimeError, "selected PTPC failed"):
            project_wo_a(
                value,
                SimpleNamespace(forward_grouped=provider),
                bf16_projection=fallback,
            )
        provider.assert_called_once_with(value)
        fallback.assert_not_called()


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_WOA_ENTRY_DEVICE") == "1", "explicit PPU device test"
)
class WoaEntryDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertIn("ZW-", torch.cuda.get_device_name())

    @torch.inference_mode()
    def test_native_ptpc_phase_selection_and_changed_input_graph(self):
        torch.manual_seed(9102)
        op = PPUPTPCLinear(
            torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            torch.rand(8192, 1, device="cuda", dtype=torch.float32) * 0.01 + 0.001,
        )
        op.prepare_grouped(8, 1024)
        self.assertIsNotNone(op._group_gemm)
        self.assertIsNotNone(op._group_direct_gemm)
        fallback = Mock(side_effect=AssertionError("PTPC cannot use BF16 policy"))
        with patch.dict(
            os.environ,
            {"RTP_V41_PPU_PTPC_FUSION": "1", "RTP_V41_PPU_PTPC_WOA_DIRECT": "1"},
        ):
            for rows, prefill in (
                (1, False),
                (8, False),
                (24, False),
                (32, False),
                (144, True),
            ):
                value = torch.randn(rows, 8, 4096, device="cuda", dtype=torch.bfloat16)
                caller = torch.cuda.Stream()
                caller.wait_stream(torch.cuda.current_stream())
                provider = op._group_direct_gemm
                with patch.object(op, "_group_direct_gemm", wraps=provider) as direct:
                    with torch.cuda.stream(caller):
                        expected = op.forward_grouped(value, prefill=prefill)
                        actual = project_wo_a(
                            value, op, prefill=prefill, bf16_projection=fallback
                        )
                        assert_bits(self, actual, expected)
                    caller.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=caller):
                        captured = project_wo_a(
                            value, op, prefill=prefill, bf16_projection=fallback
                        )
                    for magnitude in (0.0, 0.125, 2.0):
                        with torch.cuda.stream(caller):
                            value.copy_(torch.randn_like(value) * magnitude)
                            graph.replay()
                            expected = op.forward_grouped(value, prefill=prefill)
                            assert_bits(self, captured, expected)
                    caller.synchronize()
                    self.assertEqual(direct.call_count > 0, prefill)
        fallback.assert_not_called()

    @torch.inference_mode()
    def test_native_bf16_batched_policy_changed_input_graph(self):
        torch.manual_seed(9103)
        op = PPUResidentBF16Linear(
            torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
        )
        policy = lambda value, weight: batched_wo_a(value, weight, 8, 1024)
        for rows in (1, 8, 24, 32):
            value = torch.randn(rows, 8, 4096, device="cuda", dtype=torch.bfloat16)
            expected = policy(value, op.weight)
            assert_bits(self, project_wo_a(value, op, bf16_projection=policy), expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = project_wo_a(value, op, bf16_projection=policy)
            for magnitude in (0.0, 0.125, 2.0):
                value.copy_(torch.randn_like(value) * magnitude)
                graph.replay()
                assert_bits(self, captured, policy(value, op.weight))


class GroupedProjectionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_decode_verify_and_proposal_rows(self):
        torch.manual_seed(4112)
        groups, rank, width = 8, 7, 16
        for rows in (0, 1, 8, 24, 32, 140):
            with self.subTest(rows=rows):
                # Small integer sums are exactly representable in BF16, so
                # independent per-group linears can assert byte equality.
                output = torch.randint(-2, 3, (rows, groups, width)).bfloat16()
                weight = torch.randint(-2, 3, (groups * rank, width)).bfloat16()
                before = output.clone(), weight.clone()
                expected = torch.stack(
                    [
                        F.linear(output[:, g], weight[g * rank : (g + 1) * rank])
                        for g in range(groups)
                    ],
                    dim=1,
                )
                with patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
                    actual = batched_wo_a(output, weight, groups, rank)
                self.assertEqual(bmm.call_count, 1)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.shape, (rows, groups, rank))
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.flatten(1).stride(), (groups * rank, 1))
                self.assertTrue(torch.equal(output, before[0]))
                self.assertTrue(torch.equal(weight, before[1]))

    @torch.inference_mode()
    def test_nonzero_storage_offsets_are_views(self):
        output = torch.ones((4, 2, 16), dtype=torch.bfloat16)[1:3]
        weight = torch.ones((8, 16), dtype=torch.bfloat16)[2:6]
        with patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
            actual = batched_wo_a(output, weight, 2, 2)
        left, right = bmm.call_args.args
        self.assertTrue(torch._C._is_alias_of(left, output))
        self.assertTrue(torch._C._is_alias_of(right, weight))
        self.assertTrue(torch._C._is_alias_of(bmm.call_args.kwargs["out"], actual))
        self.assertTrue(torch.equal(actual, torch.full((2, 2, 2), 16).bfloat16()))


if __name__ == "__main__":
    unittest.main()

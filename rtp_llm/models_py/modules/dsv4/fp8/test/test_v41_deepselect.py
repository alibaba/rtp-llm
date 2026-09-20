"""Direct BF16 DeepSelect contract tests; no FP32 scorer in the tested path."""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as selector
from rtp_llm.ops.compute_ops import rtp_llm_ops


class DeepSelectGateTest(unittest.TestCase):
    def test_cpu_fallback_without_cuda_query(self):
        logits = torch.empty((2, 1024), dtype=torch.bfloat16)
        ends = torch.zeros(2, dtype=torch.int32)
        with mock.patch.object(selector, "_device_supported") as device_query:
            self.assertIsNone(selector.try_select_sparse_tokens(logits, ends))
        device_query.assert_not_called()

    def test_planning_availability_and_explicit_disable(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            rtp_llm_ops, "deepselect_bf16_available", return_value=True, create=True
        ), mock.patch.object(selector, "_device_supported", return_value=True):
            self.assertTrue(selector.is_available(torch.device("cuda:0")))
            self.assertFalse(selector.is_available(torch.device("cpu")))
            with mock.patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "0"}):
                self.assertFalse(selector.is_available(torch.device("cuda:0")))
        with mock.patch.object(
            rtp_llm_ops, "deepselect_bf16_available", return_value=False, create=True
        ), mock.patch.object(selector, "_device_supported") as device_query:
            self.assertFalse(selector.is_available(torch.device("cuda:0")))
            device_query.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeepSelectCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("SM100/SM103 required")
        available = getattr(rtp_llm_ops, "deepselect_bf16_available", None)
        if available is None or not available():
            raise unittest.SkipTest("DeepSelect native binding is not built")

    def setUp(self):
        self.env_patch = mock.patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "1"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def assert_valid_selection(self, logits, ends, actual):
        """Compare score multisets, allowing legal BF16 cutoff ties/order."""
        self.assertIsNotNone(actual)
        self.assertEqual(actual.dtype, torch.int32)
        actual_cpu = actual.cpu().to(torch.int64)
        logits_cpu = logits.cpu().float()
        ends_cpu = ends.cpu().clamp(0, logits.shape[1])
        for row, end in enumerate(ends_cpu.tolist()):
            indices = actual_cpu[row]
            valid = indices >= 0
            self.assertEqual(valid.sum().item(), min(end, 512), f"row {row}")
            self.assertTrue((indices[valid] < end).all().item())
            self.assertTrue((indices[~valid] == -1).all().item())
            self.assertEqual(indices[valid].unique().numel(), valid.sum().item())
            selected = logits_cpu[row, indices[valid]].sort().values
            expected = logits_cpu[row, :end].topk(min(end, 512)).values.sort().values
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_finite_negative_ties_short_clamped_and_strided(self):
        # 173 rows exercise occupancy-two on both SM100 and SM103. Views test
        # independent row strides and aligned nonzero storage offsets.
        torch.manual_seed(417)
        rows, width = 173, 16384
        storage = torch.randn((rows, width + 512), device="cuda", dtype=torch.bfloat16)
        logits = storage[:, 8 : width + 8]
        logits[::4].abs_().neg_()
        logits[1::4].fill_(-3)
        logits[2::4, 4000:] = -float("inf")
        ends = (
            torch.tensor(
                [-(2**31), -1, 0, 1, 511, 512, 513, 8191, 16384, 2**31 - 1],
                device="cuda",
                dtype=torch.int32,
            )
            .repeat(18)[:rows]
            .contiguous()
        )
        out_storage = torch.full((rows, 528), 12345, device="cuda", dtype=torch.int32)
        output = out_storage[:, 8:520]
        actual = selector.try_select_sparse_tokens(logits, ends, output)
        self.assertIs(actual, output)
        self.assert_valid_selection(logits, ends, actual)
        self.assertTrue((out_storage[:, :8] == 12345).all().item())
        self.assertTrue((out_storage[:, 520:] == 12345).all().item())

    def test_one_wave_and_width_variants(self):
        for width in (512, 1024, 16384, 32768):
            with self.subTest(width=width):
                torch.manual_seed(width)
                logits = torch.randn((7, width), device="cuda", dtype=torch.bfloat16)
                ends = torch.tensor(
                    [0, 1, 511, 512, 513, width - 1, width],
                    device="cuda",
                    dtype=torch.int32,
                )
                self.assert_valid_selection(
                    logits, ends, selector.try_select_sparse_tokens(logits, ends)
                )

    def test_nonfinite_contract_and_nan_beyond_end(self):
        logits = torch.randn((7, 16384), device="cuda", dtype=torch.bfloat16)
        ends = torch.tensor(
            [16384, 600, 513, 511, 8, 600, 16384], device="cuda", dtype=torch.int32
        )
        logits[0, 100] = float("nan")
        logits[1, 599] = -float("nan")
        logits[2, 0] = float("nan")
        logits[3, 4] = float("nan")
        logits[4, 0] = float("inf")
        logits[5, 600:] = float("nan")
        logits[6, :4] = float("inf")
        logits[6, 1000:] = -float("inf")
        out = torch.full((7, 512), 9876, device="cuda", dtype=torch.int32)
        selector.try_select_sparse_tokens(logits, ends, out)
        self.assertTrue((out[:3] == -1).all().item())
        # Short rows do not scan scores. Fused remap is responsible for filtering
        # these selected nonfinite values, and for all +/-inf in long rows.
        torch.testing.assert_close(
            out[3, :511], torch.arange(511, device="cuda", dtype=torch.int32)
        )
        self.assertEqual(out[3, 511].item(), -1)
        self.assert_valid_selection(logits[4:], ends[4:], out[4:])

    def test_cuda_graph_replay_uses_device_bounds(self):
        logits = torch.randn((4, 16384), device="cuda", dtype=torch.bfloat16)
        ends = torch.full((4,), 16384, device="cuda", dtype=torch.int32)
        out = torch.empty((4, 512), device="cuda", dtype=torch.int32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                selector.try_select_sparse_tokens(logits, ends, out)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selector.try_select_sparse_tokens(logits, ends, out)
        ends.copy_(
            torch.tensor([0, 17, 600, 2**31 - 1], device="cuda", dtype=torch.int32)
        )
        logits.neg_()
        graph.replay()
        self.assert_valid_selection(logits, ends, out)

    def test_extreme_bf16_and_all_nonfinite_rows(self):
        logits = torch.zeros((173, 16384), device="cuda", dtype=torch.bfloat16)
        logits[0].fill_(-float("inf"))
        logits[1].fill_(float("inf"))
        logits[2].fill_(float("nan"))
        # BF16 extrema and subnormals exercise the signed ordering without
        # reducing the score tensor to FP32 in the selector.
        bits = torch.arange(1, 128, device="cuda", dtype=torch.int16)
        tiny = bits.view(torch.bfloat16)
        logits[3, :127] = tiny
        logits[3, 127:254] = -tiny
        logits[4, :700] = torch.finfo(torch.bfloat16).min
        logits[4, 700:1400] = torch.finfo(torch.bfloat16).max
        ends = torch.full((173,), 16384, device="cuda", dtype=torch.int32)
        out = selector.try_select_sparse_tokens(logits, ends)
        self.assertTrue((out[2] == -1).all().item())
        self.assert_valid_selection(logits[:2], ends[:2], out[:2])
        self.assert_valid_selection(logits[3:], ends[3:], out[3:])

    def test_metadata_gates_and_execution_errors(self):
        logits = torch.zeros((2, 1024), device="cuda", dtype=torch.bfloat16)
        ends = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
        self.assertTrue(selector.is_supported(logits, ends))
        for invalid in (logits.float(), logits[:, :513], logits[:, 1:], logits[:, ::2]):
            self.assertIsNone(selector.try_select_sparse_tokens(invalid, ends))
        self.assertIsNone(
            selector.try_select_sparse_tokens(logits, ends.to(torch.int64))
        )
        bad_out = torch.empty((2, 513), device="cuda", dtype=torch.int32)[:, :512]
        self.assertIsNone(selector.try_select_sparse_tokens(logits, ends, bad_out))
        with mock.patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "0"}):
            self.assertIsNone(selector.try_select_sparse_tokens(logits, ends))
        with mock.patch.object(
            rtp_llm_ops, "deepselect_bf16", side_effect=RuntimeError("native failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                selector.try_select_sparse_tokens(logits, ends)
        with self.assertRaises(RuntimeError):
            rtp_llm_ops.deepselect_bf16(
                logits.float(),
                ends,
                torch.empty((2, 512), device="cuda", dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()

"""Direct BF16 DeepSelect contract tests; no FP32 scorer in the tested path."""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_deepselect as selector
from rtp_llm.ops.compute_ops import rtp_llm_ops


class DeepSelectGateTest(unittest.TestCase):
    def test_cpu_fallback_without_cuda_query(self):
        logits = torch.empty((2, 1024), dtype=torch.bfloat16)
        ends = torch.zeros(2, dtype=torch.int32)
        with mock.patch.object(selector, "_device_supported") as device_query:
            self.assertIsNone(selector.try_select_sparse_tokens(logits, ends))
        device_query.assert_not_called()

    def test_planning_availability_and_missing_native_binding(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            rtp_llm_ops, "deepselect_bf16_available", return_value=True, create=True
        ), mock.patch.object(selector, "_device_supported", return_value=True):
            self.assertTrue(selector.is_available(torch.device("cuda:0")))
            self.assertFalse(selector.is_available(torch.device("cpu")))
            with mock.patch.object(
                rtp_llm_ops, "deepselect_bf16_available", return_value=False
            ):
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
        with mock.patch.object(selector, "is_available", return_value=False):
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

    def assert_dense_selection(self, logits, ends, output, *, filter_finite):
        """Compare against the same BF16 values cast losslessly to FP32."""
        scores = logits.float()
        bounds = ends.clamp(0, logits.shape[1])
        columns = torch.arange(logits.shape[1], device=logits.device)
        reference_scores = scores.masked_fill(
            columns[None] >= bounds[:, None], -torch.inf
        )
        expected_values = reference_scores.topk(512, dim=-1).values
        if filter_finite:
            actual_values = scores.gather(1, output.long().clamp_min(0))
            self.assertTrue(
                ((output == -1) | ((output >= 0) & (output < bounds[:, None])))
                .all()
                .item()
            )
            self.assertTrue((actual_values.isfinite() | (output == -1)).all().item())
            expected_values = expected_values.masked_fill(
                ~expected_values.isfinite(), -torch.inf
            )
            actual_values = actual_values.masked_fill(output < 0, -torch.inf)
            torch.testing.assert_close(
                (output >= 0).sum(-1),
                expected_values.isfinite().sum(-1),
                rtol=0,
                atol=0,
            )
        else:
            # Raw callers must filter before publishing. Bounds still initialize
            # every slot, and NaNs consume legal selected slots instead of
            # invalidating the complete row as in the sparse entry point.
            self.assertTrue(
                ((output == -1) | ((output >= 0) & (output < bounds[:, None])))
                .all()
                .item()
            )
            torch.testing.assert_close(
                (output >= 0).sum(-1), bounds.clamp_max(512).long(), rtol=0, atol=0
            )
            actual_values = scores.gather(1, output.long().clamp_min(0)).masked_fill(
                output < 0, -torch.inf
            )
            actual_values = torch.where(actual_values.isnan(), torch.inf, actual_values)
            expected_values = torch.where(
                expected_values.isnan(), torch.inf, expected_values
            )
        ordered = output.masked_fill(output < 0, logits.shape[1]).sort(-1).values
        self.assertFalse(
            ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != logits.shape[1]))
            .any()
            .item()
        )
        torch.testing.assert_close(
            actual_values.sort(-1).values,
            expected_values.sort(-1).values,
            rtol=0,
            atol=0,
        )

    def test_dense_nan_inf_ties_short_and_strided(self):
        for rows, width in ((7, 512), (7, 1024), (173, 16384), (173, 32768)):
            with self.subTest(rows=rows, width=width):
                torch.manual_seed(512 + width)
                backing = torch.randn(
                    (rows, width + 512), device="cuda", dtype=torch.bfloat16
                )
                logits = backing[:, 8 : width + 8]
                ends = (
                    torch.tensor(
                        [
                            -(2**31),
                            -1,
                            0,
                            1,
                            17,
                            511,
                            512,
                            513,
                            width - 1,
                            width,
                            2**31 - 1,
                        ],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    .repeat((rows + 10) // 11)[:rows]
                    .contiguous()
                )
                if rows == 7:
                    ends.copy_(
                        torch.tensor(
                            [0, 1, 17, 511, 512, 513, width],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                logits[:, 0] = torch.nan
                logits[:, 1] = -torch.nan
                logits[:, 2] = torch.inf
                logits[:, 3] = -torch.inf
                # Exercise NaNs both in the initial window and permuted scan.
                logits[:, width // 2] = torch.nan
                logits[:, -1] = -torch.nan
                if rows > 7:
                    logits[11].fill_(torch.nan)
                    logits[12].fill_(-torch.nan)
                    logits[13].fill_(torch.inf)
                    logits[14].fill_(-torch.inf)
                    logits[15].fill_(-3)
                    logits[16].zero_()
                    logits[17, :700] = torch.finfo(torch.bfloat16).max
                    logits[17, 700:1400] = torch.finfo(torch.bfloat16).min
                    ends[11:18] = width
                    # NaNs outside a short/long row's end must not affect it.
                    logits[18:20].zero_()
                    ends[18:20] = torch.tensor(
                        [17, 600], device="cuda", dtype=torch.int32
                    )
                    logits[18, 17:] = torch.nan
                    logits[19, 600:] = torch.nan
                before = logits.clone()
                out_storage = torch.full(
                    (rows, 528), 12345, device="cuda", dtype=torch.int32
                )
                output = out_storage[:, 8:520]
                for filtering in (True, False):
                    rtp_llm_ops.deepselect_bf16_dense(logits, ends, output, filtering)
                    self.assert_dense_selection(
                        logits, ends, output, filter_finite=filtering
                    )
                    self.assertTrue((out_storage[:, :8] == 12345).all().item())
                    self.assertTrue((out_storage[:, 520:] == 12345).all().item())
                torch.testing.assert_close(
                    logits, before, rtol=0, atol=0, equal_nan=True
                )

    def test_dense_and_sparse_nan_contracts_remain_distinct(self):
        logits = (
            torch.arange(16384, device="cuda", dtype=torch.float32)
            .repeat(3, 1)
            .to(torch.bfloat16)
        )
        logits[0, 100] = torch.nan
        logits[1, -1] = -torch.nan
        logits[2, 4] = torch.nan
        ends = torch.tensor([16384, 16384, 17], device="cuda", dtype=torch.int32)
        sparse = selector.try_select_sparse_tokens(logits, ends)
        dense = torch.empty((3, 512), device="cuda", dtype=torch.int32)
        rtp_llm_ops.deepselect_bf16_dense(logits, ends, dense, True)
        self.assertTrue((sparse[:2] == -1).all().item())
        self.assertEqual(sparse[2, 4].item(), 4)
        self.assertTrue(((dense[:2] >= 0).sum(-1) == 511).all().item())
        self.assertEqual((dense[2] >= 0).sum().item(), 16)
        self.assert_dense_selection(logits, ends, dense, filter_finite=True)

    def test_dense_cuda_graph_replay_uses_updated_scores_and_bounds(self):
        logits = torch.randn((7, 16384), device="cuda", dtype=torch.bfloat16)
        ends = torch.full((7,), 16384, device="cuda", dtype=torch.int32)
        output = torch.empty((7, 512), device="cuda", dtype=torch.int32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                rtp_llm_ops.deepselect_bf16_dense(logits, ends, output, True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rtp_llm_ops.deepselect_bf16_dense(logits, ends, output, True)
        for lengths in ([0, 17, 511, 512, 513, 16000, 2**31 - 1], [600] * 7):
            ends.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            logits.neg_()
            logits[:, 3] = torch.nan
            graph.replay()
            self.assert_dense_selection(logits, ends, output, filter_finite=True)

    def test_dense_native_rejects_bad_dtype_and_aliasing(self):
        logits = torch.empty((2, 1024), device="cuda", dtype=torch.bfloat16)
        ends = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
        output = torch.empty((2, 512), device="cuda", dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            rtp_llm_ops.deepselect_bf16_dense(logits.float(), ends, output, True)
        with self.assertRaises(RuntimeError):
            rtp_llm_ops.deepselect_bf16_dense(
                logits, ends, logits.view(torch.int32), True
            )


if __name__ == "__main__":
    unittest.main()

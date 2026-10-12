"""The prefill K512 boundary, including finite filtering on short rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
from rtp_llm.ops.compute_ops import rtp_llm_ops


def reference(logits, visible, k=512):
    columns = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(
        columns[None] >= visible[:, None].clamp(0, logits.shape[1]), -torch.inf
    )
    values, indices = masked.topk(k, dim=-1)
    return torch.where(values.isfinite(), indices, -1).int()


def select_row_prefill_candidate(logits, visible, backend, k=512):
    """Legacy benchmark candidates stay in tests, outside serving dispatch."""
    rows, width = logits.shape
    ends = visible.clamp(0, width).to(torch.int32).contiguous()
    starts = torch.zeros_like(ends)
    output = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    rtp_llm_ops.dsv4_top_k_per_row_prefill(
        logits,
        starts,
        ends,
        output,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        backend == "radix",
    )
    topk._prefill_topk_finite_kernel[(rows,)](logits, ends, output, logits.stride(0), k)
    return output


def assert_equivalent(case, got, expected, logits, visible):
    """Exact value multisets and unique valid members; tie membership may differ."""
    case.assertEqual(got.dtype, torch.int32)
    case.assertEqual(got.shape, expected.shape)
    width = logits.shape[1]
    ends = visible.clamp(0, width)
    case.assertTrue(((got == -1) | ((got >= 0) & (got < ends[:, None]))).all().item())
    ordered = got.masked_fill(got < 0, width).sort(dim=-1).values
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != width)
    case.assertFalse(duplicate.any().item())
    actual_values = logits.gather(1, got.long().clamp_min(0)).masked_fill(
        got < 0, -torch.inf
    )
    expected_values = logits.gather(1, expected.long().clamp_min(0)).masked_fill(
        expected < 0, -torch.inf
    )
    torch.testing.assert_close(
        actual_values.sort(-1).values, expected_values.sort(-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        (got >= 0).sum(-1), (expected >= 0).sum(-1), rtol=0, atol=0
    )


class V41PrefillTopKCPU(unittest.TestCase):
    def fixture(self):
        logits = SimpleNamespace(
            is_cuda=True,
            dtype=torch.float32,
            ndim=2,
            shape=(7, 1024),
            stride=lambda axis: (1280, 1)[axis],
            device=torch.device("cuda:0"),
        )
        visible = SimpleNamespace(
            device=logits.device,
            ndim=1,
            numel=lambda: 7,
            dtype=torch.int64,
        )
        return logits, visible

    def test_native_availability_and_test_only_reference_hook(self):
        logits, visible = self.fixture()
        with patch.object(topk, "_TOPK_V3_OK", True):
            self.assertTrue(topk.is_supported(logits, visible))
        with patch.object(topk, "_TOPK_V3_OK", False):
            self.assertIsNone(topk.try_select_tokens(logits, visible))

    def test_support_fallbacks(self):
        with patch.object(topk, "_TOPK_V3_OK", True):
            for change in (
                {"is_cuda": False},
                {"dtype": torch.bfloat16},
                {"shape": (0, 1024)},
                {"shape": (7, 511)},
                {"stride": lambda axis: (2048, 2)[axis]},
            ):
                logits, visible = self.fixture()
                vars(logits).update(change)
                self.assertFalse(topk.is_supported(logits, visible))
            logits, visible = self.fixture()
            self.assertFalse(topk.is_supported(logits, visible, 1024))
            visible.dtype = torch.float32
            self.assertFalse(topk.is_supported(logits, visible))

    def test_execution_errors_propagate(self):
        with patch.object(topk, "is_supported", return_value=True), patch.object(
            topk, "_select_tokens", side_effect=RuntimeError("launch failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failure"):
                topk.try_select_tokens(None, None)

    def test_raw_selection_requires_explicit_completion_buffers(self):
        with patch.object(topk, "is_supported", return_value=True), patch.object(
            topk, "_select_tokens", side_effect=AssertionError("must not launch")
        ):
            self.assertIsNone(topk.try_select_tokens(None, None, filter_finite=False))

    def test_finite_entry_raw_publication_and_older_binary_fallback(self):
        logits = torch.zeros((2, 1024))
        starts = torch.zeros((2,), dtype=torch.int32)
        ends = torch.full((2,), 1024, dtype=torch.int32)
        out = torch.empty((2, 512), dtype=torch.int32)
        workspace = object()
        for available, filtering in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            legacy, fused, finish = Mock(), Mock(), Mock()
            ops = SimpleNamespace(topk_v3=legacy)
            if available:
                ops.dsv41_topk_v3_finite = fused
            with patch.object(topk, "rtp_llm_ops", ops), patch.object(
                topk, "_get_topk_workspace", return_value=workspace
            ), patch.object(topk, "finish_tokens", finish):
                self.assertIs(
                    topk._select_tokens(
                        logits,
                        ends,
                        bounds=(starts, ends),
                        out=out,
                        filter_finite=filtering,
                    ),
                    out,
                )
            self.assertEqual(fused.call_count, int(available and filtering))
            self.assertEqual(legacy.call_count, int(not (available and filtering)))
            self.assertEqual(finish.call_count, int(filtering and not available))
            selected = fused if available and filtering else legacy
            selected.assert_called_once_with(logits, ends, out, workspace, 512, 1024)
            if filtering and not available:
                finish.assert_called_once_with(logits, ends, out)

    def test_finite_entry_failure_is_not_retried(self):
        logits = torch.zeros((2, 1024))
        ends = torch.full((2,), 1024, dtype=torch.int32)
        out = torch.empty((2, 512), dtype=torch.int32)
        ops = SimpleNamespace(
            topk_v3=Mock(),
            dsv41_topk_v3_finite=Mock(side_effect=RuntimeError("fused launch")),
        )
        with patch.object(topk, "rtp_llm_ops", ops), patch.object(
            topk, "_get_topk_workspace", return_value=None
        ), patch.object(topk, "finish_tokens") as finish:
            with self.assertRaisesRegex(RuntimeError, "fused launch"):
                topk._select_tokens(logits, ends, bounds=(ends, ends), out=out)
        ops.topk_v3.assert_not_called()
        finish.assert_not_called()


@unittest.skipUnless(
    torch.cuda.is_available() and hasattr(rtp_llm_ops, "topk_v3"),
    "CUDA topk_v3 required",
)
class V41PrefillTopKCUDA(unittest.TestCase):
    def test_strided_short_long_and_clamped_bounds(self):
        for width in (512, 769, 4099, 16384, 32768, 65536):
            with self.subTest(width=width):
                backing = torch.randn((13, width + 259), device="cuda")
                logits = backing[:, 1 : width + 1]
                bounds = torch.tensor(
                    [
                        -9,
                        0,
                        1,
                        17,
                        511,
                        512,
                        513,
                        769,
                        width - 1,
                        width,
                        width + 99,
                        2**40,
                        37,
                    ],
                    device="cuda",
                )
                noncontiguous = torch.stack((bounds, bounds), dim=1)[:, 0]
                logits[:, ::7] = -torch.inf
                got = topk.try_select_tokens(logits, noncontiguous)
                self.assertIsNotNone(got)
                assert_equivalent(self, got, reference(logits, bounds), logits, bounds)

    def test_nan_inf_ties_and_all_masked_rows(self):
        width = 4099
        logits = torch.randn((17, width + 257), device="cuda")[:, :width]
        visible = torch.tensor(
            [0, 1, 17, 511, 512, 513, 1024] + [width] * 10,
            dtype=torch.int32,
            device="cuda",
        )
        logits[:, 0] = float("nan")
        logits[:, 1] = -float("nan")
        logits[:, 2] = torch.inf
        logits[:, 3] = -torch.inf
        logits[:, 4:8] = torch.tensor(
            [
                0.0,
                -0.0,
                torch.finfo(torch.float32).max,
                -torch.finfo(torch.float32).max,
            ],
            device="cuda",
        )
        logits[8].fill_(0)
        logits[9].fill_(-torch.inf)
        logits[10].fill_(torch.inf)
        logits[11].fill_(-float("nan"))
        logits[12, ::3] = -torch.inf
        for row, count in enumerate((513, 1025, 2049, 3073), start=13):
            logits[row, :count] = -float("nan")
        got = topk.try_select_tokens(logits, visible)
        assert_equivalent(self, got, reference(logits, visible), logits, visible)

    def test_existing_row_prefill_candidates_on_finite_scores(self):
        if not hasattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill"):
            self.skipTest("row-prefill kernel unavailable")
        logits = torch.randn((7, 32768 + 256), device="cuda")[:, :32768]
        visible = torch.tensor([0, 17, 511, 512, 513, 16383, 32768], device="cuda")
        logits[:, ::13] = -torch.inf
        expected = reference(logits, visible)
        for backend in ("insertion", "radix"):
            with self.subTest(backend=backend):
                got = select_row_prefill_candidate(logits, visible, backend)
                assert_equivalent(self, got, expected, logits, visible)

    def test_real_cp_shapes(self):
        for rows, width in ((4096, 16384), (2048, 32768), (1024, 65536)):
            with self.subTest(rows=rows, width=width):
                logits = torch.randn((rows, width + 256), device="cuda")[:, :width]
                # Later CP chunks have large global positions despite small local M.
                visible = torch.arange(
                    width - rows + 1, width + 1, device="cuda", dtype=torch.int32
                )
                got = topk.try_select_tokens(logits, visible)
                # Reference a spread across the whole chunk without another full
                # M*N materialization; the candidate still runs at the real shape.
                sample = torch.linspace(0, rows - 1, 37, device="cuda").long()
                selected_logits, selected_visible = logits[sample], visible[sample]
                assert_equivalent(
                    self,
                    got[sample],
                    reference(selected_logits, selected_visible),
                    selected_logits,
                    selected_visible,
                )

    def test_cuda_graph_replay_with_changed_bounds_and_scores(self):
        logits = torch.randn((7, 2304), device="cuda")[:, :2048]
        visible = torch.tensor([0, 17, 511, 512, 513, 1024, 2048], device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                topk.try_select_tokens(logits, visible)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            result = topk.try_select_tokens(logits, visible)
        logits.normal_()
        logits[:, 0] = torch.inf
        visible.copy_(torch.tensor([2048, 1024, 513, 512, 511, 17, -1], device="cuda"))
        graph.replay()
        assert_equivalent(self, result, reference(logits, visible), logits, visible)


if __name__ == "__main__":
    unittest.main()

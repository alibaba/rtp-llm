"""The prefill K512 boundary, including finite filtering on short rows."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_default_enabled_and_explicit_disable(self):
        logits, visible = self.fixture()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            topk, "_topk_v3_enabled", return_value=True
        ):
            self.assertTrue(topk.is_supported(logits, visible))
            with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_TOPK": "0"}):
                self.assertIsNone(topk.try_select_tokens(logits, visible))

    def test_support_fallbacks(self):
        with patch.object(topk, "_topk_v3_enabled", return_value=True):
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


@unittest.skipUnless(
    torch.cuda.is_available() and hasattr(rtp_llm_ops, "topk_v3"),
    "CUDA topk_v3 required",
)
class V41PrefillTopKCUDA(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ, {"DSV41_FUSED_PREFILL_TOPK": "1", "DSV4_TOPK_V3": "1"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

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
                got = topk._select_tokens(logits, visible, backend=backend)
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

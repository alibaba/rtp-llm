import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa import cub_topk as cub
from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import try_select_qsa_tokens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaCubTopkTest(unittest.TestCase):
    def test_dense_unmasked_prefill_dispatch_owns_visibility_mask(self):
        for columns in (4096, 16384):
            # Match the actual 8K-token Prefill rows, including score
            # values beyond each row's causal boundary that must be ignored.
            scores = torch.randn(8192, columns, device="cuda")
            lengths = torch.arange(8192, device="cuda", dtype=torch.int32) * 7
            lengths.clamp_(max=columns * 4)
            scores[0].zero_()
            scores[1, ::3] = float("nan")
            scores.view(torch.int32)[2, ::5] = -2147483648
            lengths[:3] = columns * 4
            for finite in (True, False):
                expected = self.reference(scores, lengths, finite)
                with patch.object(
                    torch,
                    "topk",
                    side_effect=AssertionError("CUB did not dispatch"),
                ):
                    actual = try_select_qsa_tokens(
                        scores,
                        lengths,
                        compress_ratio=4,
                        token_budget=2048,
                        scores_are_masked=False,
                        finite_only=finite,
                    )
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_large_width_preserves_exact_order_and_live_graph_metadata(self):
        for columns in (4096, 6144, 8192, 16384):
            scores = torch.randn(65, columns, device="cuda")
            lengths = torch.arange(65, device="cuda", dtype=torch.int32) * 941
            lengths.clamp_(max=columns * 4)
            scores[0].zero_()
            scores[1].copy_(torch.randint(-3, 4, (columns,), device="cuda"))
            scores[2, ::5] = float("nan")
            scores.view(torch.int32)[3, ::7] = -4194303
            lengths[:4] = columns * 4
            actual = cub.try_cub_qsa_tokens(scores, lengths)
            self.assertIsNotNone(actual)
            torch.testing.assert_close(
                actual, self.reference(scores, lengths), rtol=0, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = cub.try_cub_qsa_tokens(scores, lengths)
            scores.normal_()
            lengths.clamp_(max=4099)
            graph.replay()
            torch.testing.assert_close(
                replayed, self.reference(scores, lengths), rtol=0, atol=0
            )
            graph.reset()

    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 pinned Torch ordering")
        torch.manual_seed(143201)

    @staticmethod
    def reference(scores, lengths, finite=True):
        masked = scores.masked_fill(
            torch.arange(scores.shape[1], device=scores.device)[None, :]
            >= lengths[:, None] // 4,
            float("-inf"),
        )
        values, indices = torch.topk(masked, 512, dim=-1)
        valid = torch.isfinite(values) if finite else values > float("-inf")
        tokens = (
            torch.where(
                valid[..., None],
                indices[..., None] * 4 + torch.arange(4, device=scores.device),
                -1,
            )
            .flatten(1)
            .int()
        )
        tail = lengths[:, None] // 4 * 4 + torch.arange(3, device=scores.device)
        return torch.cat(
            (tokens, torch.where(tail < lengths[:, None], tail, -1).int()), 1
        )

    def test_native_bits_ties_nan_signs_ragged_and_partial_columns(self):
        for rows in (1, 3, 4, 8, 16, 20, 24, 28, 32):
            for columns in (2047, 2048, 2144):
                scores = torch.randn(rows, columns, device="cuda")
                lengths = torch.arange(rows, device="cuda", dtype=torch.int32) * 379
                lengths[0] = columns * 4 - 1
                for kind in (
                    "random",
                    "equal",
                    "ties",
                    "nonfinite",
                    "negative_nan",
                    "signed_zero",
                ):
                    scores.normal_()
                    if kind == "equal":
                        scores.zero_()
                    if kind == "ties":
                        scores.copy_(torch.randint(-3, 4, scores.shape, device="cuda"))
                    if kind == "nonfinite":
                        scores[:, ::3] = float("inf")
                        scores[:, 1::5] = float("nan")
                        scores[:, 2::7] = float("-inf")
                    if kind == "negative_nan":
                        scores[:, ::5] = float("nan")
                        scores.view(torch.int32)[:, ::7] = -4194303
                    if kind == "signed_zero":
                        scores.zero_()
                        scores.view(torch.int32)[:, ::2] = -2147483648
                    for finite in (True, False):
                        actual = cub.try_cub_qsa_tokens(
                            scores, lengths, finite_only=finite
                        )
                        self.assertIsNotNone(
                            actual,
                            "Required CUDA compiler or CUB candidate unavailable",
                        )
                        torch.testing.assert_close(
                            actual,
                            self.reference(scores, lengths, finite),
                            rtol=0,
                            atol=0,
                        )

    def test_two_graphs_live_lengths_nan_masks_and_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for columns in (2047, 2048, 2144):
                scores = torch.randn(4, columns, device="cuda")
                lengths = torch.full(
                    (4,), columns * 4, device="cuda", dtype=torch.int32
                )
                for _ in range(3):
                    cub.try_cub_qsa_tokens(scores, lengths)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = cub.try_cub_qsa_tokens(scores, lengths)
                for step in range(6):
                    scores.normal_()
                    scores[0, ::5] = float("nan")
                    scores.view(torch.int32)[0, ::7] = -4194303
                    lengths.copy_(
                        torch.tensor(
                            [129 + step, 4099, columns * 4 - 1, 7],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    graph.replay()
                    torch.testing.assert_close(
                        actual, self.reference(scores, lengths), rtol=0, atol=0
                    )
                graph.reset()
        torch.cuda.current_stream().wait_stream(stream)

    def test_owner_masked_contract_and_unqualified_fallbacks(self):
        scores = torch.randn(4, 2048, device="cuda")
        lengths = torch.tensor([129, 4099, 8191, 7], device="cuda", dtype=torch.int32)
        scores.masked_fill_(
            torch.arange(2048, device="cuda")[None, :] >= lengths[:, None] // 4,
            float("-inf"),
        )
        actual = try_select_qsa_tokens(
            scores, lengths, compress_ratio=4, token_budget=2048, scores_are_masked=True
        )
        torch.testing.assert_close(
            actual, self.reference(scores, lengths), rtol=0, atol=0
        )
        self.assertIsNone(cub.try_cub_qsa_tokens(scores[:, ::2], lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.double(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.requires_grad_(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.detach().cpu(), lengths.cpu()))

    def test_unprimed_capture_or_failed_build_keeps_original_selector(self):
        scores = torch.randn(4, 2048, device="cuda")
        lengths = torch.full((4,), 8192, device="cuda", dtype=torch.int32)
        with (
            patch.object(cub, "_MODULE", None),
            patch.object(cub, "_BUILD_FAILED", False),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            self.assertIsNone(cub.try_cub_qsa_tokens(scores, lengths))
        with (
            patch.object(cub, "_MODULE", None),
            patch.object(cub, "_BUILD_FAILED", True),
        ):
            actual = try_select_qsa_tokens(
                scores,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                scores_are_masked=True,
            )
        torch.testing.assert_close(
            actual, self.reference(scores, lengths), rtol=0, atol=0
        )

    @staticmethod
    def _mtp3_target_lengths(rows, generation=0):
        # Include fewer than 512 blocks, all four tail widths and full capacity.
        pattern = torch.tensor(
            [0, 1, 2, 3, 4, 7, 8, 129, 2044, 2045, 2050, 4095, 4099, 8191, 8192],
            device="cuda",
            dtype=torch.int32,
        )
        return pattern.roll(generation).repeat((rows + 14) // 15)[:rows].contiguous()

    @staticmethod
    def _fill_mtp3_target_scores(scores, kind):
        scores.normal_()
        if kind == "equal":
            scores.zero_()
        elif kind == "ties":
            scores.copy_(torch.randint(-3, 4, scores.shape, device=scores.device))
        elif kind == "nonfinite":
            scores[:, ::3] = float("inf")
            scores[:, 1::5] = float("nan")
            scores[:, 2::7] = float("-inf")
        elif kind == "negative_nan":
            scores[:, ::5] = float("nan")
            scores.view(torch.int32)[:, ::7] = -4194303
        elif kind == "signed_zero":
            scores.zero_()
            scores.view(torch.int32)[:, ::2] = -2147483648

    def test_mtp3_target_opt_in_preserves_default_and_adjacent_domains(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        with patch.dict("os.environ", RTP_QWEN4_TARGET_TOPK_IMPL="invalid"):
            with self.assertRaisesRegex(ValueError, "must be native or cub"):
                cub.is_supported(None, None)
        with patch.dict(
            "os.environ",
            RTP_QWEN4_TARGET_TOPK_IMPL="native",
        ):
            for rows in (64, 256):
                scores = torch.randn(rows, 2048, device="cuda")
                lengths = self._mtp3_target_lengths(rows)
                self.assertFalse(cub.is_supported(scores, lengths))
            for rows in (1, 16, 32):
                scores = torch.randn(rows, 2048, device="cuda")
                self.assertTrue(
                    cub.is_supported(scores, self._mtp3_target_lengths(rows))
                )
        with patch.dict(
            "os.environ",
            RTP_QWEN4_TARGET_TOPK_IMPL="cub",
        ):
            for rows, columns in (
                (63, 2048),
                (65, 2048),
                (255, 2048),
                (257, 2048),
                (64, 2047),
                (64, 2049),
                (64, 2144),
                (256, 2144),
            ):
                scores = torch.randn(rows, columns, device="cuda")
                lengths = self._mtp3_target_lengths(rows).clamp(max=columns * 4)
                self.assertFalse(cub.is_supported(scores, lengths))
            for rows in (64, 256):
                scores = torch.randn(rows, 2048, device="cuda")
                lengths = self._mtp3_target_lengths(rows)
                self.assertTrue(cub.is_supported(scores, lengths))
                with patch.object(torch.version, "git_version", "unqualified-runtime"):
                    self.assertFalse(cub.is_supported(scores, lengths))

    def test_mtp3_target_all_tokens_match_pinned_torch_without_native_fallback(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        with patch.dict(
            "os.environ",
            RTP_QWEN4_TARGET_TOPK_IMPL="cub",
        ):
            for rows in (64, 256):
                scores = torch.empty(rows, 2048, device="cuda")
                lengths = self._mtp3_target_lengths(rows)
                for kind in (
                    "random",
                    "equal",
                    "ties",
                    "nonfinite",
                    "negative_nan",
                    "signed_zero",
                ):
                    self._fill_mtp3_target_scores(scores, kind)
                    for finite in (True, False):
                        expected = self.reference(scores, lengths, finite)
                        with patch.object(
                            torch,
                            "topk",
                            side_effect=AssertionError(
                                "MTP3 target CUB did not dispatch"
                            ),
                        ):
                            actual = try_select_qsa_tokens(
                                scores,
                                lengths,
                                compress_ratio=4,
                                token_budget=2048,
                                scores_are_masked=False,
                                finite_only=finite,
                            )
                        self.assertIsNotNone(actual)
                        self.assertEqual(actual.shape, (rows, 2051))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mtp3_target_owner_masked_scores_keep_all_tail_tokens(self):
        with patch.dict(
            "os.environ",
            RTP_QWEN4_TARGET_TOPK_IMPL="cub",
        ):
            for rows in (64, 256):
                scores = torch.randn(rows, 2048, device="cuda")
                lengths = self._mtp3_target_lengths(rows)
                scores.masked_fill_(
                    torch.arange(2048, device="cuda")[None, :] >= lengths[:, None] // 4,
                    float("-inf"),
                )
                expected = self.reference(scores, lengths)
                with patch.object(
                    torch,
                    "topk",
                    side_effect=AssertionError(
                        "MTP3 owner-masked CUB did not dispatch"
                    ),
                ):
                    actual = try_select_qsa_tokens(
                        scores,
                        lengths,
                        compress_ratio=4,
                        token_budget=2048,
                        scores_are_masked=True,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mtp3_target_two_live_graphs_reuse_metadata_on_nondefault_stream(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        states = []
        try:
            with (
                torch.cuda.stream(stream),
                patch.dict(
                    "os.environ",
                    RTP_QWEN4_TARGET_TOPK_IMPL="cub",
                ),
            ):
                for rows in (64, 256):
                    scores = torch.randn(rows, 2048, device="cuda")
                    lengths = self._mtp3_target_lengths(rows)
                    for _ in range(3):
                        self.assertIsNotNone(cub.try_cub_qsa_tokens(scores, lengths))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = cub.try_cub_qsa_tokens(scores, lengths)
                    self.assertIsNotNone(actual)
                    states.append((scores, lengths, graph, actual))
                for step, kind in enumerate(
                    (
                        "random",
                        "ties",
                        "nonfinite",
                        "negative_nan",
                        "signed_zero",
                        "equal",
                    )
                ):
                    first_output = None
                    for index, (scores, lengths, graph, actual) in enumerate(states):
                        self._fill_mtp3_target_scores(scores, kind)
                        lengths.copy_(self._mtp3_target_lengths(scores.shape[0], step))
                        expected = self.reference(scores, lengths)
                        graph.replay()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        repeated = actual.clone()
                        graph.replay()
                        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
                        if index == 0:
                            first_output = actual.clone()
                        else:
                            torch.testing.assert_close(
                                states[0][3], first_output, rtol=0, atol=0
                            )
        finally:
            for _, _, graph, _ in states:
                graph.reset()
            torch.cuda.current_stream().wait_stream(stream)

    def test_mtp3_target_unprimed_capture_and_failed_build_keep_native_selector(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        for rows in (64, 256):
            scores = torch.randn(rows, 2048, device="cuda")
            lengths = self._mtp3_target_lengths(rows)
            # Warm the unchanged Torch/expansion graph operations, not CUB.
            with patch.dict(
                "os.environ",
                RTP_QWEN4_TARGET_TOPK_IMPL="native",
            ):
                for _ in range(3):
                    try_select_qsa_tokens(
                        scores,
                        lengths,
                        compress_ratio=4,
                        token_budget=2048,
                        scores_are_masked=False,
                    )
            with (
                patch.dict(
                    "os.environ",
                    RTP_QWEN4_TARGET_TOPK_IMPL="cub",
                ),
                patch.object(cub, "_MODULE", None),
                patch.object(cub, "_BUILD_FAILED", False),
            ):
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        unavailable = cub.try_cub_qsa_tokens(scores, lengths)
                        actual = try_select_qsa_tokens(
                            scores,
                            lengths,
                            compress_ratio=4,
                            token_budget=2048,
                            scores_are_masked=False,
                        )
                    self.assertIsNone(unavailable)
                    self._fill_mtp3_target_scores(scores, "negative_nan")
                    lengths.copy_(self._mtp3_target_lengths(rows, 3))
                    graph.replay()
                    torch.testing.assert_close(
                        actual, self.reference(scores, lengths), rtol=0, atol=0
                    )
                finally:
                    graph.reset()
            with (
                patch.dict(
                    "os.environ",
                    RTP_QWEN4_TARGET_TOPK_IMPL="cub",
                ),
                patch.object(cub, "_MODULE", None),
                patch.object(cub, "_BUILD_FAILED", True),
            ):
                self.assertIsNone(cub.try_cub_qsa_tokens(scores, lengths))
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=False,
                )
                torch.testing.assert_close(
                    actual, self.reference(scores, lengths), rtol=0, atol=0
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaCubPackedDraftTopkTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 pinned Torch ordering")
        torch.manual_seed(20261020)

    def test_live_ragged_maps_ties_nonfinite_signed_zero(self):
        from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import draft_rows

        for batch in range(1, 9):
            rows = batch * 4
            scores = torch.randn(rows, 2144, device="cuda")
            lengths = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
            prefix = torch.full_like(lengths, 8190)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4

            def call():
                _, _, _, _, visible = draft_rows(lengths, prefix, cu, rows)
                return cub.try_cub_draft_qsa_tokens(
                    scores, visible.flatten(), lengths, cu
                )

            self.assertIsNotNone(call())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = call()
            for generation in range(32):
                requests = torch.arange(batch, device="cuda", dtype=torch.int32)
                lengths.copy_((requests + generation) % 5)
                cu[0] = 0
                cu[1:].copy_(lengths.cumsum(0))
                prefix.copy_((requests * 379 + generation * 19) % 8192)
                scores.normal_()
                kind = generation % 6
                if kind == 1:
                    scores.zero_()
                if kind == 2:
                    scores.copy_(torch.randint(-3, 4, scores.shape, device="cuda"))
                if kind == 3:
                    scores[:, ::3] = float("inf")
                    scores[:, 1::5] = float("nan")
                    scores[:, 2::7] = float("-inf")
                if kind == 4:
                    scores.view(torch.int32)[:, ::7] = -4194303
                if kind == 5:
                    scores.zero_()
                    scores.view(torch.int32)[:, ::2] = -2147483648
                _, inverse, valid, _, visible = draft_rows(lengths, prefix, cu, rows)
                fixed = QsaCubTopkTest.reference(scores, visible.flatten())
                expected = torch.where(
                    valid[:, None], fixed.index_select(0, inverse), -1
                )
                graph.replay()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph.reset()

    def test_unqualified_metadata_and_unprimed_capture_fall_back(self):
        scores = torch.randn(4, 2144, device="cuda")
        visible = torch.full((4,), 8192, device="cuda", dtype=torch.int32)
        lengths = torch.tensor([4], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        for args in (
            (scores, visible, lengths.long(), cu),
            (scores, visible, lengths.cpu(), cu),
            (scores, visible, lengths, cu[:1]),
            (scores, visible, lengths.repeat(2), cu),
            (scores[:, :2048].contiguous(), visible, lengths, cu),
        ):
            self.assertIsNone(cub.try_cub_draft_qsa_tokens(*args))
        with (
            patch.object(cub, "_MODULE", None),
            patch.object(cub, "_BUILD_FAILED", False),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            self.assertIsNone(
                cub.try_cub_draft_qsa_tokens(scores, visible, lengths, cu)
            )


if __name__ == "__main__":
    unittest.main()

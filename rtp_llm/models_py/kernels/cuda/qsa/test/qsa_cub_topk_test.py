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

    def test_device_length_buckets_refresh_boundaries_and_preserve_native_bits(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                for columns in (8191, 8192):
                    scores = torch.empty(16, columns, device="cuda")
                    pattern = torch.tensor(
                        [
                            0,
                            1,
                            2,
                            3,
                            511 * 4 + 3,
                            512 * 4,
                            513 * 4 + 1,
                            2048 * 4 + 2,
                            2064 * 4 + 3,
                            4095 * 4 + 1,
                            4096 * 4,
                            4096 * 4 + 3,
                            4097 * 4,
                            columns * 4 + 3,
                            -1,
                            (columns + 1) * 4 + 1,
                        ],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    lengths = pattern.clone()
                    short = pattern.clamp(min=0, max=2064 * 4 + 3)
                    empty = torch.zeros_like(lengths)
                    column_ids = torch.arange(columns, device="cuda")[None]
                    for finite in (True, False):
                        scores.normal_()
                        self.assertIsNotNone(
                            cub.try_cub_qsa_tokens(
                                scores,
                                lengths,
                                finite_only=finite,
                                device_length_buckets=True,
                            )
                        )
                        module = cub._load_module(scores.device)
                        graph = torch.cuda.CUDAGraph()
                        try:
                            with (
                                patch.object(
                                    module,
                                    "qsa_cub_topk_length_buckets",
                                    wraps=module.qsa_cub_topk_length_buckets,
                                ) as bucket,
                                patch.object(
                                    torch,
                                    "topk",
                                    side_effect=AssertionError("CUB did not dispatch"),
                                ) as topk,
                            ):
                                with torch.cuda.graph(graph, stream=stream):
                                    actual = try_select_qsa_tokens(
                                        scores,
                                        lengths,
                                        compress_ratio=4,
                                        token_budget=2048,
                                        finite_only=finite,
                                        scores_are_masked=False,
                                        device_length_buckets=True,
                                    )
                                bucket.assert_called_once()
                                topk.assert_not_called()
                            for step, kind in enumerate(
                                (
                                    "random",
                                    "equal",
                                    "ties",
                                    "nonfinite",
                                    "negative_nan",
                                    "signed_zero",
                                )
                            ):
                                for refreshed in (
                                    pattern.roll(step),
                                    short,
                                    empty,
                                    pattern.roll(step),
                                ):
                                    self._fill_selection_scores(scores, kind)
                                    lengths.copy_(refreshed)
                                    scores.masked_fill_(
                                        column_ids >= lengths[:, None] // 4,
                                        float("nan") if step % 2 else 1.0e30,
                                    )
                                    legacy = cub.try_cub_qsa_tokens(
                                        scores, lengths, finite_only=finite
                                    )
                                    native = self.reference(scores, lengths, finite)
                                    valid = (lengths >= 0) & (lengths // 4 <= columns)
                                    actual.fill_(-99)
                                    graph.replay()
                                    # Raw invalid lengths retain legacy behavior;
                                    # supported rows match pinned Torch exactly.
                                    torch.testing.assert_close(
                                        actual, legacy, rtol=0, atol=0
                                    )
                                    torch.testing.assert_close(
                                        actual[valid], native[valid], rtol=0, atol=0
                                    )
                        finally:
                            graph.reset()
        finally:
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
        module = cub._load_module(scores.device)
        for rows, columns in ((4, 2047), (4, 2048), (4, 6144), (33, 8191), (33, 8192)):
            adjacent = torch.randn(rows, columns, device="cuda")
            visible = torch.arange(rows, device="cuda", dtype=torch.int32) * 379
            visible.clamp_(max=columns * 4)
            visible[0] = columns * 4 - 1
            adjacent.masked_fill_(
                torch.arange(columns, device="cuda")[None] >= visible[:, None] // 4,
                float("nan"),
            )
            expected = self.reference(adjacent, visible)
            with (
                patch.object(
                    module,
                    "qsa_cub_topk_length_buckets",
                    side_effect=AssertionError("unexpected length-bucket entry"),
                ) as bucket,
                patch.object(
                    module,
                    "qsa_cub_init_length_buckets",
                    side_effect=AssertionError("unexpected length-bucket init"),
                ) as initialize,
                patch.object(
                    module, "qsa_cub_topk", wraps=module.qsa_cub_topk
                ) as original,
            ):
                selected = cub.try_cub_qsa_tokens(
                    adjacent, visible, device_length_buckets=True
                )
                original.assert_called_once()
                bucket.assert_not_called()
                initialize.assert_not_called()
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        self.assertIsNone(cub.try_cub_qsa_tokens(scores[:, ::2], lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.double(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.requires_grad_(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.detach().cpu(), lengths.cpu()))

    def test_unprimed_capture_or_failed_build_keeps_original_selector(self):
        for columns, options in (
            (2048, {}),
            (8191, {"device_length_buckets": True}),
            (8192, {"device_length_buckets": True}),
        ):
            scores = torch.randn(4, columns, device="cuda")
            lengths = torch.tensor(
                [3, 129, columns * 2 + 3, columns * 4], device="cuda", dtype=torch.int32
            )
            scores.masked_fill_(
                torch.arange(columns, device="cuda")[None] >= lengths[:, None] // 4,
                float("nan") if options else float("-inf"),
            )
            expected = self.reference(scores, lengths)
            with (
                patch.object(cub, "_MODULE", None),
                patch.object(cub, "_BUILD_FAILED", False),
                patch("torch.cuda.is_current_stream_capturing", return_value=True),
            ):
                self.assertIsNone(cub.try_cub_qsa_tokens(scores, lengths, **options))
            with (
                patch.object(cub, "_MODULE", None),
                patch.object(cub, "_BUILD_FAILED", True),
            ):
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=not bool(options),
                    **options,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if options:
                from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
                    select_qsa_paged_tokens,
                )

                self.assertTrue(cub.is_supported(scores, lengths))
                module = cub._load_module(scores.device)
                self.assertIsNotNone(module)
                native_cub = cub.try_cub_qsa_tokens
                column_ids = torch.arange(columns, device="cuda")[None]
                long_lengths = torch.tensor(
                    [columns * 4, 4097 * 4 + 3, 4096 * 4, 2064 * 4 + 1],
                    device="cuda",
                    dtype=torch.int32,
                )
                short_lengths = torch.tensor(
                    [3, 129, 2048 * 4 + 2, 2064 * 4 + 3],
                    device="cuda",
                    dtype=torch.int32,
                )
                empty_lengths = torch.zeros_like(lengths)

                def consume():
                    return select_qsa_paged_tokens(
                        scores,
                        lengths,
                        compress_ratio=4,
                        token_budget=2048,
                        validate_lengths=False,
                        scores_are_masked=False,
                        device_length_buckets=True,
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())

                def capture_fallback(*, use_original_cub=False):
                    def observe_cub(*args, **kwargs):
                        self.assertTrue(kwargs.get("device_length_buckets"))
                        result = native_cub(*args, **kwargs)
                        if use_original_cub:
                            self.assertIsNotNone(result)
                        else:
                            self.assertIsNone(result)
                        return result

                    graph = torch.cuda.CUDAGraph()
                    try:
                        with (
                            patch.object(
                                cub, "try_cub_qsa_tokens", side_effect=observe_cub
                            ) as cub_selector,
                            patch.object(
                                module, "qsa_cub_topk", wraps=module.qsa_cub_topk
                            ) as original,
                            patch.object(
                                module,
                                "qsa_cub_topk_length_buckets",
                                side_effect=AssertionError("unready bucket entry"),
                            ) as bucket,
                            patch.object(torch, "topk", wraps=torch.topk) as topk,
                        ):
                            with torch.cuda.graph(graph, stream=stream):
                                replayed = consume()
                            cub_selector.assert_called_once()
                            bucket.assert_not_called()
                            if use_original_cub:
                                original.assert_called_once()
                                topk.assert_not_called()
                            else:
                                original.assert_not_called()
                                topk.assert_called_once()
                        for step, refreshed in enumerate(
                            (long_lengths, short_lengths, empty_lengths, long_lengths)
                        ):
                            lengths.copy_(refreshed)
                            scores.normal_()
                            scores.masked_fill_(
                                column_ids >= lengths[:, None] // 4,
                                float("nan") if step % 2 else 1.0e30,
                            )
                            replayed.fill_(-99)
                            graph.replay()
                            torch.testing.assert_close(
                                replayed,
                                self.reference(scores, lengths),
                                rtol=0,
                                atol=0,
                            )
                    finally:
                        graph.reset()

                try:
                    with torch.cuda.stream(stream):
                        # Compile/warm every operation in the complete fallback
                        # before entering any real capture below.
                        with (
                            patch.object(cub, "_MODULE", None),
                            patch.object(cub, "_BUILD_FAILED", True),
                        ):
                            for _ in range(3):
                                consume()
                        self.assertIsNotNone(native_cub(scores, lengths))
                        with (
                            patch.object(
                                module,
                                "qsa_cub_init",
                                side_effect=AssertionError(
                                    "capture attempted old init"
                                ),
                            ) as initialize_original,
                            patch.object(
                                module,
                                "qsa_cub_init_length_buckets",
                                side_effect=AssertionError("capture attempted init"),
                            ) as initialize,
                            patch(
                                "torch.utils.cpp_extension.load_inline",
                                side_effect=AssertionError("capture attempted build"),
                            ) as build,
                        ):
                            for failed in (False, True):
                                with (
                                    patch.object(cub, "_MODULE", None),
                                    patch.object(cub, "_BUILD_FAILED", failed),
                                ):
                                    capture_fallback()
                            with patch.object(cub, "_INITIALIZED_DEVICES", set()):
                                capture_fallback()
                            with (
                                patch.object(cub, "_LENGTH_BUCKET_DEVICES", set()),
                                patch.object(
                                    cub, "_LENGTH_BUCKET_FAILED_DEVICES", set()
                                ),
                            ):
                                capture_fallback(use_original_cub=True)
                            initialize_original.assert_not_called()
                            initialize.assert_not_called()
                            build.assert_not_called()
                        with (
                            patch.object(cub, "_LENGTH_BUCKET_DEVICES", set()),
                            patch.object(cub, "_LENGTH_BUCKET_FAILED_DEVICES", set()),
                            patch.object(
                                module,
                                "qsa_cub_init_length_buckets",
                                side_effect=RuntimeError("init rejected"),
                            ) as initialize,
                        ):
                            with self.assertLogs(cub._LOGGER, level="WARNING"):
                                failed_init_output = consume()
                            torch.testing.assert_close(
                                failed_init_output,
                                self.reference(scores, lengths),
                                rtol=0,
                                atol=0,
                            )
                            initialize.assert_called_once()
                            initialize.reset_mock()
                            capture_fallback(use_original_cub=True)
                            initialize.assert_not_called()
                            torch.testing.assert_close(
                                cub.try_cub_qsa_tokens(scores, lengths),
                                self.reference(scores, lengths),
                                rtol=0,
                                atol=0,
                            )
                finally:
                    torch.cuda.current_stream().wait_stream(stream)

    @staticmethod
    def _selection_lengths(rows, generation=0):
        # Include fewer than 512 blocks, all four tail widths and full capacity.
        pattern = torch.tensor(
            [0, 1, 2, 3, 4, 7, 8, 129, 2044, 2045, 2050, 4095, 4099, 8191, 8192],
            device="cuda",
            dtype=torch.int32,
        )
        return pattern.roll(generation).repeat((rows + 14) // 15)[:rows].contiguous()

    @staticmethod
    def _fill_selection_scores(scores, kind):
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

    def test_default_dispatch_preserves_target_fallback_and_capacity_guards(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        for rows, columns in (
            (0, 2048),
            (33, 2047),
            (64, 2048),
            (256, 2048),
            (64, 2144),
            (64, 2046),
            (1, 16385),
            (8193, 2145),
        ):
            scores = torch.empty(rows, columns, device="cuda")
            lengths = self._selection_lengths(rows).clamp(max=columns * 4)
            self.assertFalse(cub.is_supported(scores, lengths), (rows, columns))
        for rows, columns in (
            (1, 2048),
            (16, 2048),
            (32, 2048),
            (32, 2176),
            (32, 4352),
            (64, 2145),
            (64, 4096),
        ):
            scores = torch.empty(rows, columns, device="cuda")
            lengths = self._selection_lengths(rows)
            self.assertTrue(cub.is_supported(scores, lengths), (rows, columns))
            with patch.object(torch.version, "git_version", "unqualified-runtime"):
                self.assertFalse(cub.is_supported(scores, lengths))

    def test_padded_score_widths_match_torch_in_default_domain(self):
        for rows in (2, 16, 32):
            for columns in list(range(2047, 2177)) + list(range(4095, 4353)):
                scores = torch.randn(rows, columns, device="cuda")
                lengths = torch.arange(rows, device="cuda", dtype=torch.int32) * 379
                lengths[0] = columns * 4 - 1
                for kind in ("random", "ties", "nonfinite", "signed_zero"):
                    self._fill_selection_scores(scores, kind)
                    for finite in (True, False):
                        actual = cub.try_cub_qsa_tokens(
                            scores, lengths, finite_only=finite
                        )
                        self.assertIsNotNone(actual, (rows, columns, kind))
                        torch.testing.assert_close(
                            actual,
                            self.reference(scores, lengths, finite),
                            rtol=0,
                            atol=0,
                        )

    def test_default_all_tokens_match_pinned_torch_without_native_fallback(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        for rows in (16, 32):
            scores = torch.empty(rows, 2048, device="cuda")
            lengths = self._selection_lengths(rows)
            for kind in (
                "random",
                "equal",
                "ties",
                "nonfinite",
                "negative_nan",
                "signed_zero",
            ):
                self._fill_selection_scores(scores, kind)
                for finite in (True, False):
                    expected = self.reference(scores, lengths, finite)
                    with patch.object(
                        torch,
                        "topk",
                        side_effect=AssertionError("Default CUB did not dispatch"),
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

    def test_default_owner_masked_scores_keep_all_tail_tokens(self):
        for rows in (16, 32):
            scores = torch.randn(rows, 2048, device="cuda")
            lengths = self._selection_lengths(rows)
            scores.masked_fill_(
                torch.arange(2048, device="cuda")[None, :] >= lengths[:, None] // 4,
                float("-inf"),
            )
            expected = self.reference(scores, lengths)
            with patch.object(
                torch,
                "topk",
                side_effect=AssertionError("Owner-masked CUB did not dispatch"),
            ):
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=True,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_default_two_live_graphs_reuse_metadata_on_nondefault_stream(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        states = []
        try:
            with torch.cuda.stream(stream):
                for rows in (16, 32):
                    scores = torch.randn(rows, 2048, device="cuda")
                    lengths = self._selection_lengths(rows)
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
                        self._fill_selection_scores(scores, kind)
                        lengths.copy_(self._selection_lengths(scores.shape[0], step))
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

    def test_default_unprimed_capture_and_failed_build_keep_native_selector(self):
        self.assertEqual(torch.version.git_version, cub._TORCH_GIT)
        for rows in (16, 32):
            scores = torch.randn(rows, 2048, device="cuda")
            lengths = self._selection_lengths(rows)
            # Warm the fallback graph operations without initializing CUB.
            with (
                patch.object(cub, "_BUILD_FAILED", True),
                patch.object(cub, "_MODULE", None),
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
                    self._fill_selection_scores(scores, "negative_nan")
                    lengths.copy_(self._selection_lengths(rows, 3))
                    graph.replay()
                    torch.testing.assert_close(
                        actual, self.reference(scores, lengths), rtol=0, atol=0
                    )
                finally:
                    graph.reset()
            with (
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

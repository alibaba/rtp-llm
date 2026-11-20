import unittest

import torch
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)

from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import try_select_qsa_tokens
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.topk_reference import (
    assert_qsa_selection_close,
    select_qsa_paged_tokens as reference_topk,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTopkTritonTest(unittest.TestCase):
    def test_prefill_policy_keeps_positive_infinity_and_selected_values(self):
        for rows, columns in ((4, 1024), (127, 513), (4096, 1024), (12288, 2048)):
            torch.manual_seed(7381 + rows)
            scores = torch.randn(rows, columns, device="cuda")
            scores[0] = 0
            scores[1, :7] = float("nan")
            scores[2, 0] = float("inf")
            scores[3] = float("-inf")
            lengths = torch.randint(
                0, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
            )
            lengths[2] = columns * 4
            complete = lengths // 4
            masked = scores.masked_fill(
                torch.arange(columns, device="cuda")[None, :] >= complete[:, None],
                float("-inf"),
            )
            values, indices = masked.topk(min(512, columns), dim=-1)
            chosen = torch.where(values > float("-inf"), indices, -1)
            tokens = torch.where(
                chosen[:, :, None] >= 0,
                chosen[:, :, None] * 4 + torch.arange(4, device="cuda"),
                -1,
            )
            expected = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
            expected[:, : tokens.shape[1] * 4] = tokens.flatten(1).int()
            tail = complete[:, None] * 4 + torch.arange(3, device="cuda")
            expected[:, 2048:] = torch.where(tail < lengths[:, None], tail, -1)
            for premasked in (False, True):
                actual = try_select_qsa_tokens(
                    masked if premasked else scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=premasked,
                    finite_only=False,
                )
                assert_qsa_selection_close(actual, expected, scores, lengths)

    def reference(self, scores, lengths, **options):
        return reference_topk(
            scores, lengths, compress_ratio=4, token_budget=2048, **options
        )

    def test_selected_values_for_ties_nonfinite_values_and_ragged_lengths(self):
        torch.manual_seed(821)
        for rows, columns in (
            (1, 1),
            (7, 31),
            (17, 513),
            (32, 1024),
            (127, 2048),
            (4093, 1024),
        ):
            with self.subTest(rows=rows, columns=columns):
                scores = torch.randn(rows, columns, device="cuda")
                scores[0] = 0
                if rows > 1:
                    scores[1, : min(5, columns)] = float("nan")
                    scores[2, 0] = float("inf")
                    scores[3] = float("-inf")
                lengths = torch.randint(
                    0, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
                )
                expected = self.reference(scores, lengths)
                actual = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                self.assertIsNotNone(actual)
                assert_qsa_selection_close(actual, expected, scores, lengths)

    def test_graph_replay_refreshes_strided_scores_and_lengths(self):
        scores = torch.randn(17, 2048, device="cuda")[:, ::2]
        lengths = torch.full((34,), 4096, device="cuda", dtype=torch.int32)[::2]
        try_select_qsa_tokens(scores, lengths, compress_ratio=4, token_budget=2048)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
        for step in range(3):
            scores.copy_(torch.randn_like(scores))
            scores[0] = 0
            lengths.copy_(torch.arange(17, device="cuda", dtype=torch.int32) * 7 + step)
            graph.replay()
            assert_qsa_selection_close(
                actual, self.reference(scores, lengths), scores, lengths
            )

    def test_paged_producer_mask_preserves_selected_values_and_tail(self):
        from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
            qsa_paged_indexer_score,
        )

        torch.manual_seed(2701)
        for rows, columns in ((1, 31), (4, 1024), (16, 2048), (28, 8192)):
            q = torch.randn(rows, 1, 4, 128, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(rows, 4, device="cuda")
            pages = (columns + 31) // 32
            pool = torch.randn(
                (pages + 1) * 32, 128, device="cuda", dtype=torch.bfloat16
            )
            table = (
                torch.arange(1, pages + 1, device="cuda", dtype=torch.int32)
                .expand(rows, -1)
                .contiguous()
            )
            lengths = torch.randint(
                0, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
            )
            compressed = (lengths // 4).unsqueeze(1)
            scores = qsa_paged_indexer_score(
                q,
                weight,
                pool,
                table,
                compressed,
                block_size=32,
                max_ctx_len=columns,
                validate_block_table=False,
            )
            for special in (None, 0.0, float("nan"), float("inf"), float("-inf")):
                if special is not None:
                    visible = (
                        torch.arange(columns, device="cuda").unsqueeze(0)
                        < lengths.unsqueeze(1) // 4
                    )
                    scores.copy_(
                        torch.where(
                            visible, torch.full_like(scores, special), float("-inf")
                        )
                    )
                expected = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                actual = select_qsa_paged_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    validate_lengths=False,
                    scores_are_masked=True,
                )
                assert_qsa_selection_close(actual, expected, scores, lengths)

    def test_premasked_graph_replays_new_history_and_reordered_rows(self):
        rows, columns = 16, 2048
        scores = torch.empty(rows, columns, device="cuda")
        lengths = torch.full((rows,), 4096, device="cuda", dtype=torch.int32)

        def refresh(step):
            lengths.copy_(
                torch.arange(rows, device="cuda", dtype=torch.int32).flip(0) * 127
                + step
            )
            scores.copy_(
                torch.randn_like(scores).masked_fill(
                    torch.arange(columns, device="cuda").unsqueeze(0)
                    >= lengths.unsqueeze(1) // 4,
                    float("-inf"),
                )
            )

        refresh(0)
        try_select_qsa_tokens(
            scores, lengths, compress_ratio=4, token_budget=2048, scores_are_masked=True
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_select_qsa_tokens(
                scores,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                scores_are_masked=True,
            )
        for step in (1, 2, 3, 4096):
            refresh(step)
            graph.replay()
            expected = try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
            assert_qsa_selection_close(actual, expected, scores, lengths)

    def test_premasked_selection_preserves_float_values(self):
        for rows in (1, 3, 4, 8, 16, 20, 24, 28, 32):
            for columns in (1024, 2048):
                generator = torch.Generator(device="cuda").manual_seed(
                    2716 + rows + columns
                )
                bits = torch.randint(
                    0,
                    2**32,
                    (rows, columns),
                    device="cuda",
                    dtype=torch.int64,
                    generator=generator,
                ).to(torch.int32)
                scores = bits.view(torch.float32)
                lengths = torch.arange(rows, device="cuda", dtype=torch.int32)
                lengths = (lengths * 127 + columns * 2 + 3).clamp(max=columns * 4)
                scores.masked_fill_(
                    torch.arange(columns, device="cuda").unsqueeze(0)
                    >= lengths.unsqueeze(1) // 4,
                    float("-inf"),
                )
                # The default path remains the independent native primitive.
                expected = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=True,
                )
                assert_qsa_selection_close(actual, expected, scores, lengths)

    def test_premasked_signed_zero_and_mixed_ties(self):
        for columns in (1024, 2048):
            lengths = torch.full(
                (4,), columns * 4 - 1, device="cuda", dtype=torch.int32
            )
            for scores in (
                torch.zeros(4, columns, device="cuda"),
                torch.randint(-4, 5, (4, columns), device="cuda").float(),
            ):
                scores[:, ::2] *= -1.0
                scores[:, -1] = float("-inf")
                expected = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=True,
                )
                assert_qsa_selection_close(actual, expected, scores, lengths)

    def test_premasked_graph_switches_native_nan_selection_and_sort(self):
        for columns in (1024, 2048):
            scores = torch.randn(4, columns, device="cuda")
            lengths = torch.full(
                (4,), columns * 4 - 1, device="cuda", dtype=torch.int32
            )
            scores[:, -1] = float("-inf")
            try_select_qsa_tokens(
                scores,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                scores_are_masked=True,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = try_select_qsa_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    scores_are_masked=True,
                )
            for kind in ("negative-nan", "finite", "positive-nan", "zeros"):
                scores.normal_()
                if kind == "negative-nan":
                    scores.view(torch.int32)[0, 0] = 0xFF8F1586 - 2**32
                    scores.view(torch.int32)[1, 2] = 0x7F8EB130
                elif kind == "positive-nan":
                    scores[:, 3] = float("nan")
                elif kind == "zeros":
                    scores.zero_()
                    scores[:, ::2] *= -1.0
                scores[:, -1] = float("-inf")
                expected = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                graph.replay()
                assert_qsa_selection_close(actual, expected, scores, lengths)

    @unittest.skipUnless(
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0),
        "SM90 native TopK",
    )
    def test_graph_capacity_above_8192_replays_runtime_lengths(self):
        for columns in (8447, 8448):
            scores = torch.randn(8, columns, device="cuda")
            lengths = torch.full((8,), 8195, device="cuda", dtype=torch.int32)
            try_select_qsa_tokens(scores, lengths, compress_ratio=4, token_budget=2048)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
            for length in (0, 2051, 8195, columns * 4 + 3):
                lengths.fill_(length)
                scores.normal_()
                scores[0].zero_()
                graph.replay()
                expected = self.reference(scores, lengths)
                assert_qsa_selection_close(actual, expected, scores, lengths)

    @unittest.skipUnless(
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0),
        "Hopper native TopK",
    )
    def test_native_strides_broadcasts_and_odd_widths(self):
        from unittest.mock import patch

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        for columns in (31, 513, 8447, 65537):
            for layout in ("padded", "columns", "broadcast"):
                backing = torch.randn(7, columns * 2 + 1, device="cuda")
                if layout == "padded":
                    scores = backing[:, 1 : columns + 1]
                elif layout == "columns":
                    scores = backing[:, 1::2]
                else:
                    scores = backing[:1, 1 : columns + 1].expand(7, -1)
                storage = torch.empty(14, device="cuda", dtype=torch.int32)
                lengths = storage[::2]
                lengths.copy_(
                    torch.linspace(0, columns * 4 + 3, 7, device="cuda").int()
                )
                with patch.object(
                    rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
                ) as native:
                    actual = try_select_qsa_tokens(
                        scores, lengths, compress_ratio=4, token_budget=2048
                    )
                    self.assertEqual(native.call_count, 1)
                    if scores.stride(1) == 1:
                        self.assertEqual(
                            native.call_args[0][0].data_ptr(), scores.data_ptr()
                        )
                assert_qsa_selection_close(
                    actual, self.reference(scores, lengths), scores, lengths
                )
        scores = torch.randn(1, 1, device="cuda").expand(7, 513)
        lengths = torch.tensor([2055], device="cuda", dtype=torch.int32).expand(7)
        actual = try_select_qsa_tokens(
            scores, lengths, compress_ratio=4, token_budget=2048
        )
        assert_qsa_selection_close(
            actual, self.reference(scores, lengths), scores, lengths
        )

    def test_empty_inputs_and_floating_type_fallbacks(self):
        for rows, columns in ((0, 513), (7, 0), (1, 1)):
            scores = torch.randn(rows, columns, device="cuda")
            lengths = torch.full(
                (rows,), columns * 4 + 3, device="cuda", dtype=torch.int32
            )
            actual = select_qsa_paged_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
            assert_qsa_selection_close(
                actual, self.reference(scores, lengths), scores, lengths
            )
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            scores = torch.randn(7, 513, device="cuda", dtype=dtype)
            lengths = torch.full((7,), 2055, device="cuda", dtype=torch.int32)
            actual = select_qsa_paged_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
            torch.testing.assert_close(
                actual, self.reference(scores, lengths), atol=0, rtol=0
            )

    @unittest.skipUnless(
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0),
        "Hopper native TopK",
    )
    def test_rows_above_previous_prefill_limit(self):
        scores = torch.zeros(32769, 1, device="cuda")
        lengths = torch.full((32769,), 7, device="cuda", dtype=torch.int32)
        actual = try_select_qsa_tokens(
            scores, lengths, compress_ratio=4, token_budget=2048
        )
        self.assertIsNotNone(actual)
        assert_qsa_selection_close(
            actual, self.reference(scores, lengths), scores, lengths
        )

    @unittest.skipUnless(
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0),
        "Hopper native TopK",
    )
    def test_precomputed_lengths_graph_reuses_live_metadata(self):
        from unittest.mock import patch

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        for columns in (512, 8447, 8448):
            scores = torch.randn(8, columns, device="cuda")
            lengths = torch.full((8,), columns * 4, device="cuda", dtype=torch.int32)
            compressed = (lengths // 4).view(2, 4)

            def select():
                return select_qsa_paged_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    validate_lengths=False,
                    compressed_lengths=compressed,
                )

            with (
                patch(
                    "torch.div",
                    side_effect=AssertionError("redundant length conversion"),
                ),
                patch.object(
                    rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
                ) as native,
            ):
                self.assertIsNotNone(select())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = select()
                self.assertEqual(native.call_count, 2)
                self.assertIs(native.call_args[0][1], compressed)
            for step in range(4):
                lengths.copy_(
                    (
                        torch.arange(8, device="cuda", dtype=torch.int32) * 1373 + step
                    ).clamp(max=columns * 4 + 3)
                )
                compressed.copy_((lengths // 4).view(2, 4))
                scores.normal_()
                scores[0].zero_()
                graph.replay()
                assert_qsa_selection_close(
                    actual, self.reference(scores, lengths), scores, lengths
                )
            graph.reset()

    @unittest.skipUnless(
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0),
        "Hopper native TopK",
    )
    def test_precomputed_lengths_fallback_and_validation(self):
        from unittest.mock import patch

        scores = torch.randn(8, 1024, device="cuda")
        lengths = torch.arange(8, device="cuda", dtype=torch.int32) * 541 + 3
        compressed = lengths // 4
        strided = torch.empty(16, device="cuda", dtype=torch.int32)[::2]
        strided.copy_(compressed)
        for hint in (
            None,
            compressed.long(),
            compressed.float(),
            compressed.cpu(),
            compressed[:1],
            strided,
        ):
            with patch("torch.div", wraps=torch.div) as divide:
                actual = select_qsa_paged_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    validate_lengths=False,
                    compressed_lengths=hint,
                )
                divide.assert_called_once()
            assert_qsa_selection_close(
                actual, self.reference(scores, lengths), scores, lengths
            )
        # Validated calls derive counts from the actual token lengths even if
        # a caller supplies an outdated hint.
        actual = select_qsa_paged_tokens(
            scores,
            lengths,
            compress_ratio=4,
            token_budget=2048,
            compressed_lengths=torch.zeros_like(compressed),
        )
        assert_qsa_selection_close(
            actual, self.reference(scores, lengths), scores, lengths
        )

    def test_validation_fallback_and_unvalidated_negative_lengths(self):
        scores = torch.randn(7, 31, device="cuda")
        lengths = torch.tensor(
            [-2147483648, -7, -1, 0, 1, 124, 2147483647],
            device="cuda",
            dtype=torch.int32,
        )
        expected = self.reference(scores, lengths, validate_lengths=False)
        actual = select_qsa_paged_tokens(
            scores,
            lengths,
            compress_ratio=4,
            token_budget=2048,
            validate_lengths=False,
        )
        assert_qsa_selection_close(actual, expected, scores, lengths)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_qsa_paged_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
        with self.assertRaisesRegex(ValueError, "do not cover"):
            select_qsa_paged_tokens(
                scores,
                torch.full_like(lengths, 128),
                compress_ratio=4,
                token_budget=2048,
            )
        for unsupported in (scores.double(), scores[:, :0], scores.cpu()):
            self.assertIsNone(
                try_select_qsa_tokens(
                    unsupported,
                    lengths.to(unsupported.device),
                    compress_ratio=4,
                    token_budget=2048,
                )
            )
        for ratio, budget in ((2, 2048), (4, 128)):
            self.assertIsNone(
                try_select_qsa_tokens(
                    scores, lengths, compress_ratio=ratio, token_budget=budget
                )
            )


if __name__ == "__main__":
    unittest.main()

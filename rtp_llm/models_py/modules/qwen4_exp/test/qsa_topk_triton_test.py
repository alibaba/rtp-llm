import unittest

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    try_select_qsa_tokens,
)
from rtp_llm.models_py.modules.qwen4_exp.test.topk_reference import (
    select_qsa_paged_tokens as reference_topk,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTopkTritonTest(unittest.TestCase):
    def test_prefill_policy_keeps_positive_infinity_and_native_order(self):
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
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def reference(self, scores, lengths, **options):
        return reference_topk(
            scores, lengths, compress_ratio=4, token_budget=2048, **options
        )

    def test_exact_native_order_for_ties_nonfinite_values_and_ragged_lengths(self):
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
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

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
            torch.testing.assert_close(
                actual, self.reference(scores, lengths), atol=0, rtol=0
            )

    def test_paged_producer_mask_preserves_native_order_and_tail(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
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
                self.assertTrue(torch.equal(actual, expected))

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
            self.assertTrue(torch.equal(actual, expected))

    def test_premasked_radix_key_preserves_float_bits_and_native_indices(self):
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
                self.assertTrue(torch.equal(actual, expected))

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
                self.assertTrue(torch.equal(actual, expected))

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
                self.assertTrue(torch.equal(actual, expected))

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
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
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

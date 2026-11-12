import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import try_select_qsa_tokens
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.topk_reference import (
    select_qsa_paged_tokens as reference_topk,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTopkTritonTest(unittest.TestCase):
    def reference(self, scores, lengths, **options):
        return reference_topk(
            scores, lengths, compress_ratio=4, token_budget=2048, **options
        )

    def test_paged_producer_mask_preserves_native_order_and_tail(self):
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

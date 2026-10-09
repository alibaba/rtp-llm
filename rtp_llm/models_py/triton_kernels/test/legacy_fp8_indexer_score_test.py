"""Compare both legacy cache layouts against FP32 scores on GB200."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    legacy_fp8_mqa_logits,
    legacy_fp8_paged_mqa_logits,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class LegacyFP8IndexerScoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4021)
        torch.backends.cuda.matmul.allow_tf32 = False

    def reference(self, q, k, scale, weight):
        return (
            torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
            * weight.unsqueeze(-1)
        ).sum(1) * scale

    def test_ragged_fp32_weights_scales_and_mask(self):
        for heads, queries, keys in [(1, 1, 3), (7, 5, 67), (64, 17, 131)]:
            with self.subTest(heads=heads, queries=queries, keys=keys):
                q = torch.randn(queries, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.rand(keys, device="cuda") * 0.01371 + 0.000123
                weights = torch.randn(queries, heads, 2, device="cuda")[..., 0]
                starts = torch.arange(queries, device="cuda", dtype=torch.int32) % keys
                ends = torch.minimum(
                    starts + keys // 2 + 1, torch.full_like(starts, keys)
                )
                actual = legacy_fp8_mqa_logits(q, (k, scales), weights, starts, ends)
                expected = self.reference(q, k, scales, weights)
                columns = torch.arange(keys, device="cuda")
                expected.masked_fill_(
                    (columns < starts[:, None]) | (columns >= ends[:, None]),
                    -float("inf"),
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_paged_next_n_permutation_and_context_lengths(self):
        batch, next_n, heads, page, pages = 3, 2, 64, 16, 12
        q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        keys = torch.randn(pages, page, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device="cuda") * 0.00457 + 0.000019
        packed = torch.empty(pages, page, 1, 132, dtype=torch.uint8, device="cuda")
        packed[:, :, 0, :128] = keys.view(torch.uint8)
        packed[:, :, 0, 128:] = scales.unsqueeze(-1).contiguous().view(torch.uint8)
        table = torch.randperm(pages, device="cuda").to(torch.int32).reshape(batch, 4)
        lens = torch.tensor(
            [[0, 3], [31, 42], [63, 64]], device="cuda", dtype=torch.int32
        )
        weights = torch.randn(batch * next_n, heads, device="cuda")
        actual = legacy_fp8_paged_mqa_logits(q, packed, weights, lens, table, 64)
        for b in range(batch):
            physical = table[b].long()
            k = keys[physical].reshape(-1, 128)
            scale = scales[physical].reshape(-1)
            expected = self.reference(
                q[b], k, scale, weights[b * next_n : (b + 1) * next_n]
            )
            expected.masked_fill_(
                torch.arange(64, device="cuda") >= lens[b, :, None], -float("inf")
            )
            torch.testing.assert_close(
                actual[b * next_n : (b + 1) * next_n], expected, rtol=2e-5, atol=2e-5
            )


if __name__ == "__main__":
    unittest.main()

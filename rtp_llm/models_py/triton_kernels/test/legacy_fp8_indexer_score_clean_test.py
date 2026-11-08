"""clean_logits coverage for the legacy FP8 paged MQA kernel.

Sibling of legacy_fp8_indexer_score_test.py, which is left unchanged.
This file exercises ``clean_logits=True`` — out-of-range logits
must come back as ``-inf`` rather than ``0.0`` — with the same independent
reference math as the original test.
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    legacy_fp8_paged_mqa_logits,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class LegacyFP8IndexerScoreCleanLogitsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4021)
        torch.backends.cuda.matmul.allow_tf32 = False

    def reference(self, q, k, scale, weight):
        return (
            torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
            * weight.unsqueeze(-1)
        ).sum(1) * scale

    def test_paged_clean_logits_masks_out_of_range_with_neg_inf(self):
        batch, next_n, heads, page, pages = 3, 2, 64, 16, 12
        q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        keys = torch.randn(pages, page, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device="cuda") * 0.00457 + 0.000019
        # The fused pool is block-planar, matching the legacy DeepGEMM
        # contract the kernel preserves: each page stores page_size*128 K
        # bytes followed by page_size*4 scale bytes. The [pages, page, 1,
        # 132] shape is the nominal container DeepGEMM indexes internally.
        packed = torch.empty(pages, page * 132, dtype=torch.uint8, device="cuda")
        packed[:, : page * 128] = keys.view(torch.uint8).reshape(pages, page * 128)
        packed[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, page * 4)
        packed = packed.view(pages, page, 1, 132)
        table = torch.randperm(pages, device="cuda").to(torch.int32).reshape(batch, 4)
        lens = torch.tensor(
            [[0, 3], [31, 42], [63, 64]], device="cuda", dtype=torch.int32
        )
        weights = torch.randn(batch * next_n, heads, device="cuda")
        actual = legacy_fp8_paged_mqa_logits(
            q, packed, weights, lens, table, 64, clean_logits=True
        )
        for b in range(batch):
            physical = table[b].long()
            k = keys[physical].reshape(-1, 128)
            scale = scales[physical].reshape(-1)
            expected = self.reference(
                q[b], k, scale, weights[b * next_n : (b + 1) * next_n]
            )
            expected.masked_fill_(
                torch.arange(64, device="cuda") >= lens[b, :, None],
                -float("inf"),
            )
            torch.testing.assert_close(
                actual[b * next_n : (b + 1) * next_n],
                expected,
                rtol=2e-5,
                atol=2e-5,
            )


if __name__ == "__main__":
    unittest.main()

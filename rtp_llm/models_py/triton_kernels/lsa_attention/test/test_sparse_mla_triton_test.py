"""GPU tests: Triton LSA kernels vs the PyTorch reference.

Covers the three goal scenarios at kernel granularity:

- prefill-like shapes (many query rows share one KV buffer),
- decode-like shapes (few rows, long slot sets),
- spec-decoding shapes (``next_n`` query rows per request in one call),
- padding (``-1`` slots) and empty rows.

Run directly:

    python3 test_sparse_mla_triton_test.py

or through Bazel (GPU pool tag ``A10_CU13`` for this host class).
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.indexer_triton import (
    indexer_scores_triton,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    indexer_scores,
    sparse_mla_attention,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    sparse_mla_fwd,
    sparse_mla_fwd_split,
)

CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for the Triton kernels")
class TestSparseMlaTriton(unittest.TestCase):
    """Kernel-level agreement with the reference implementation."""

    def _check(self, t_count: int, heads: int, dq: int, dv: int, n_count: int, slots: int,
               pad_last_row: bool = False, seed: int = 0) -> None:
        torch.manual_seed(seed)
        device = "cuda"
        q_abs = (torch.randn(t_count, heads, dq) * 0.15).to(device).bfloat16()
        kv = (torch.randn(n_count, dq) * 0.15).to(device).bfloat16()
        idx = torch.randint(0, n_count, (t_count, 1, slots), device=device)
        if pad_last_row:
            idx[-1] = -1

        out_ref, lse_ref = sparse_mla_attention(
            q_abs.float(), kv.float(), idx, kv_lora_rank=dv
        )
        out_tri, lse_tri = sparse_mla_fwd(
            q_abs, kv, idx, kv_lora_rank=dv, out_dtype=torch.float32
        )

        valid = lse_ref[:, 0] != float("-inf")
        self.assertTrue(torch.allclose(out_tri[valid], out_ref[valid], atol=2e-2, rtol=2e-2))
        self.assertTrue(torch.allclose(lse_tri[valid], lse_ref[valid], atol=5e-3, rtol=1e-3))
        if pad_last_row:
            self.assertTrue(torch.all(lse_tri[-1] == float("-inf")))
            self.assertTrue(torch.all(out_tri[-1] == 0))

    def test_prefill_shapes(self) -> None:
        # chunked prefill: 2048 tokens/step at full index_topk budget.
        self._check(t_count=128, heads=32, dq=576, dv=512, n_count=8192, slots=2048, seed=1)

    def test_decode_shapes(self) -> None:
        # decode batch of 32 requests, one token each.
        self._check(t_count=32, heads=32, dq=576, dv=512, n_count=65536, slots=2048, seed=2)

    def test_spec_decoding_shapes(self) -> None:
        # MTP verify: 16 requests x 6 speculative tokens in one call.
        self._check(t_count=96, heads=32, dq=576, dv=512, n_count=65536, slots=2048, seed=3)

    def test_padding_and_empty_rows(self) -> None:
        self._check(t_count=17, heads=32, dq=576, dv=512, n_count=4096, slots=2048,
                    pad_last_row=True, seed=4)

    def test_split_kv_matches_reference(self) -> None:
        """Split-KV partials + merge must reproduce the plain kernel/reference."""
        for t_count, slots, splits in ((32, 2048, 8), (17, 512, 4), (1, 2048, 8)):
            with self.subTest(t_count=t_count, splits=splits):
                torch.manual_seed(t_count + splits)
                device = "cuda"
                q_abs = (torch.randn(t_count, 32, 576) * 0.15).to(device).bfloat16()
                kv = (torch.randn(8192, 576) * 0.15).to(device).bfloat16()
                idx = torch.randint(0, 8192, (t_count, 1, slots), device=device)
                idx[-1] = -1  # fully empty row -> zero output, -inf LSE
                idx[0, :, ::4] = -1  # partial padding on the first row

                out_ref, lse_ref = sparse_mla_attention(
                    q_abs.float(), kv.float(), idx, kv_lora_rank=512
                )
                out_split, lse_split = sparse_mla_fwd_split(
                    q_abs, kv, idx, kv_lora_rank=512, out_dtype=torch.float32,
                    splits=splits, block_n=32, num_stages=2,
                )
                valid = lse_ref[:, 0] != float("-inf")
                self.assertTrue(
                    torch.allclose(out_split[valid], out_ref[valid], atol=2e-2, rtol=2e-2)
                )
                self.assertTrue(
                    torch.allclose(lse_split[valid], lse_ref[valid], atol=5e-3, rtol=1e-3)
                )
                self.assertTrue(torch.all(lse_split[-1] == float("-inf")))

    def test_small_odd_shapes(self) -> None:
        for t_count, slots in ((1, 8), (5, 33), (7, 129), (33, 513)):
            with self.subTest(t_count=t_count, slots=slots):
                self._check(t_count=t_count, heads=16, dq=576, dv=512, n_count=2048,
                            slots=slots, seed=t_count + slots)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for the Triton kernels")
class TestIndexerTriton(unittest.TestCase):
    def test_scores_match_reference(self) -> None:
        torch.manual_seed(5)
        device = "cuda"
        t_count, heads, dim, n_count = 64, 16, 128, 1024
        q_idx = (torch.randn(t_count, heads, dim) * 0.1).to(device).bfloat16()
        k_idx = (torch.randn(n_count, dim) * 0.1).to(device).bfloat16()
        weights = torch.randn(t_count, heads, device=device)

        ref = indexer_scores(q_idx.float(), k_idx.float(), weights)
        got = indexer_scores_triton(q_idx, k_idx, weights)
        self.assertTrue(torch.allclose(got, ref, atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()
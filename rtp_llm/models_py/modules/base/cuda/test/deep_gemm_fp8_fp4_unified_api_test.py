"""Smoke test for the unified ``deep_gemm.fp8_fp4_{paged_,}mqa_logits`` API.

The wire-up change in ``indexer_op.py`` collapsed the old branches:

  * ``deep_gemm.fp8_paged_mqa_logits(q_fp8, ...)``
  * ``deep_gemm.fp8_mqa_logits(q_fp8, kv, ...)``

into the unified API:

  * ``deep_gemm.fp8_fp4_paged_mqa_logits((q_fp8, None), ...)``
  * ``deep_gemm.fp8_fp4_mqa_logits((q_fp8, None), kv, ...)``

This test pins the minimum invariant that does not depend on reproducing
production cache-byte layouts: the unified entries accept FP8 mode (i.e.
``q[1] == None``), return tensors of the expected shape/dtype, and do not
crash. Numerical end-to-end equivalence with the legacy entries is verified
by the FP4 indexer smoke targets — comparing the two entries here would
require re-deriving the exact byte layout written by
``indexer_k_quant_and_cache``, which is what the smoke target already checks.

Blackwell SM100+ only — the unified entry is Blackwell-only in v1.
"""

from unittest import SkipTest, TestCase, main

import torch


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


def _have_deep_gemm_unified() -> bool:
    try:
        import deep_gemm  # noqa: F401
    except Exception:
        return False
    import deep_gemm as _dg

    return all(
        hasattr(_dg, name)
        for name in (
            "fp8_fp4_paged_mqa_logits",
            "fp8_fp4_mqa_logits",
            "get_paged_mqa_logits_metadata",
            "get_num_sms",
        )
    )


class DeepGemmFp8Fp4UnifiedApiTest(TestCase):
    HEAD_DIM = 128
    NUM_HEADS = 32  # block_q = 128 / 32 = 4 satisfies fp8_mqa_logits alignment
    PAGE_SIZE = 64

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not _is_blackwell():
            raise SkipTest("Unified fp8_fp4 entry is Blackwell SM100+ only")
        if not _have_deep_gemm_unified():
            raise SkipTest("deep_gemm missing unified fp8_fp4_* entries")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)

    def _build_fp8_paged_kv_cache(self, num_blocks: int) -> torch.Tensor:
        """Build a paged KV cache with properly-formatted FP8 data + per-token
        fp32 scale. The exact production byte layout (per-block scale section
        offset) is checked by the smoke target; here we just need bytes that
        parse to finite floats so the kernel doesn't NaN out.
        """
        from deep_gemm.utils import per_token_cast_to_fp8

        total_tokens = num_blocks * self.PAGE_SIZE
        kv_bf16 = torch.randn(
            total_tokens, self.HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        k_fp8, k_sf = per_token_cast_to_fp8(
            kv_bf16, use_ue8m0=False, gran_k=self.HEAD_DIM, use_packed_ue8m0=False
        )
        cache = torch.empty(
            total_tokens,
            self.HEAD_DIM + 4,
            dtype=torch.uint8,
            device=self.device,
        )
        cache[:, : self.HEAD_DIM] = k_fp8.view(torch.uint8)
        cache[:, self.HEAD_DIM :] = k_sf.view(-1, 1).contiguous().view(torch.uint8)
        return cache.view(num_blocks, self.PAGE_SIZE, 1, self.HEAD_DIM + 4)

    # ------------------------------------------------------------------
    # paged: decode-style entry. ``q`` carries a leading next_n=1 dim.
    # ------------------------------------------------------------------
    def test_paged_fp8_mode_accepts_none_qscale(self):
        import deep_gemm

        batch = 4
        seq_len = 192
        pages_per_seq = (seq_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        total_pages = batch * pages_per_seq

        q_bf16 = torch.randn(
            batch,
            1,
            self.NUM_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)

        kv_cache = self._build_fp8_paged_kv_cache(total_pages)

        weights = torch.randn(
            batch, self.NUM_HEADS, device=self.device, dtype=torch.float32
        )
        block_table = torch.arange(
            total_pages, dtype=torch.int32, device=self.device
        ).view(batch, pages_per_seq)
        context_lens = torch.full(
            (batch, 1), seq_len, dtype=torch.int32, device=self.device
        )
        sched_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, self.PAGE_SIZE, deep_gemm.get_num_sms()
        )
        max_ctx = pages_per_seq * self.PAGE_SIZE

        # FP8 mode of the unified entry: q_scale=None, weights are
        # pre-multiplied by q_scale in production (via the head gate).
        out = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp8, None),
            kv_cache,
            weights,
            context_lens,
            block_table,
            sched_meta,
            max_ctx,
            clean_logits=False,
        )

        self.assertEqual(out.shape[0], batch)
        # Per-(batch, position) logit; the trailing axis is max_ctx in v1.
        self.assertEqual(out.dim(), 2)
        # Properly-formatted FP8 input must give finite logits — NaN/inf
        # would indicate the unified entry mis-interpreted the cache.
        self.assertTrue(
            torch.isfinite(out).any().item(),
            "unified paged entry produced no finite logits — kernel mis-call",
        )

    # ------------------------------------------------------------------
    # ragged: prefill-style entry. ``q`` has no next_n dim.
    # ------------------------------------------------------------------
    def test_ragged_fp8_mode_accepts_none_qscale(self):
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8

        seqlens = [24, 40, 16]
        total_k = sum(seqlens)
        total_q = total_k

        q_bf16 = torch.randn(
            total_q,
            self.NUM_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)

        k_bf16 = torch.randn(
            total_k, self.HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        k_fp8, k_scale = per_token_cast_to_fp8(
            k_bf16, use_ue8m0=False, gran_k=self.HEAD_DIM, use_packed_ue8m0=False
        )
        kv = (k_fp8, k_scale.view(-1))

        weights = torch.randn(
            total_q, self.NUM_HEADS, device=self.device, dtype=torch.float32
        )
        ks_list, ke_list, cursor = [], [], 0
        for n in seqlens:
            for row in range(n):
                ks_list.append(cursor)
                ke_list.append(cursor + row + 1)  # causal upper bound
            cursor += n
        ks = torch.tensor(ks_list, dtype=torch.int32, device=self.device)
        ke = torch.tensor(ke_list, dtype=torch.int32, device=self.device)

        out = deep_gemm.fp8_fp4_mqa_logits(
            (q_fp8, None), kv, weights, ks, ke, clean_logits=False
        )

        self.assertEqual(out.shape[0], total_q)
        self.assertTrue(
            torch.isfinite(out).any().item(),
            "unified ragged entry produced no finite logits — kernel mis-call",
        )


if __name__ == "__main__":
    main()

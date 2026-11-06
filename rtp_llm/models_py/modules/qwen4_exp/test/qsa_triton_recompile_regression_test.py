"""New prefill sizes must reuse existing Triton kernel variants.

Warm tile choices and scalar alignment classes before probing unseen lengths.
Cover paged QSA, fallback TopK, PLE tails, small MoE routing, and L2Norm.
"""

import unittest

import indexer_prefix_reuse_triton_test as prefix_reuse_test
import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa import (
    indexer_prefill_triton,
    sparse_paged_fmha,
)


def _variant_total(fn):
    """Compiled variant count across devices (triton>=3.6 ``device_caches``)."""
    total = 0
    caches = getattr(fn, "device_caches", None) or {}
    for cache in caches.values():
        entry = cache[0] if isinstance(cache, tuple) else cache
        total += len(entry)
    return total


def _jit_functions(module):
    """Every triton JITFunction reachable in ``module``'s namespace."""
    found = {}
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "device_caches") and hasattr(obj, "fn"):
            found[f"{module.__name__.rsplit('.', 1)[-1]}.{name}"] = obj
    return found


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefillTritonRecompileRegressionTest(unittest.TestCase):
    # Width divisible by 16 so the packed-selection strides stay in the
    # divisible-by-16 class for every probe query length.
    WIDTH = 1088
    Q_HEADS = 12
    KV_HEADS = 1
    HEAD_DIM = 256
    PAGE = 128
    CACHE_BLOCKS = 64

    def _run_writer(self, length):
        harness = prefix_reuse_test.IndexerPrefixReuseTritonTest()
        args = harness.inputs([512], [length])
        result = harness.write(args, args["kv"].clone(), args["state"].clone())
        self.assertIsNotNone(result, "fused writer declined the probe geometry")

    def _run_fmha(self, query_len):
        torch.manual_seed(7)
        width = self.WIDTH
        page = self.PAGE
        span = min((self.CACHE_BLOCKS - 1) * page, query_len + 64)
        q = torch.randn(
            1,
            self.Q_HEADS,
            query_len,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cache = torch.randn(
            self.CACHE_BLOCKS,
            2,
            self.KV_HEADS,
            page,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        pages = (span + page - 1) // page
        pool = torch.randperm(self.CACHE_BLOCKS - 1)[:pages] + 1
        block_table = pool.to(torch.int32).to("cuda").unsqueeze(0).contiguous()
        kv_lens = torch.full((1, query_len), span, dtype=torch.int32, device="cuda")
        selected = torch.full(
            (1, query_len, width), -1, dtype=torch.int32, device="cuda"
        )
        fill = min(width, span)
        picks = torch.randperm(span)[:fill].to(torch.int32).to("cuda")
        selected[0, :, :fill] = picks
        sparse_paged_fmha.sparse_paged_gqa_attn(
            q,
            cache,
            block_table,
            kv_lens,
            selected,
            page_size=page,
            kv_head_num=self.KV_HEADS,
        )

    def test_new_lengths_do_not_recompile(self):
        self.assert_reuses_variants(
            indexer_prefill_triton, self._run_writer, (1001,), (1002, 1005)
        )
        # Cover split counts 1/2/4/8 before varying lengths within each tile
        # and scalar-alignment class.
        self.assert_reuses_variants(
            sparse_paged_fmha,
            self._run_fmha,
            (5000, 1000, 100, 4),
            (5004, 1012, 116, 12),
        )

    def assert_reuses_variants(self, module, run, warmups, probes):
        kernels = _jit_functions(module)
        self.assertTrue(kernels)
        for shape in warmups:
            run(shape)
        torch.cuda.synchronize()
        before = {name: _variant_total(fn) for name, fn in kernels.items()}
        self.assertGreater(sum(before.values()), 0)
        for shape in probes:
            run(shape)
        torch.cuda.synchronize()
        after = {name: _variant_total(fn) for name, fn in kernels.items()}
        self.assertEqual(before, after, "new request sizes compiled new variants")

    def test_topk_fallback_reuses_column_buckets(self):
        from topk_reference import select_qsa_paged_tokens

        from rtp_llm.models_py.modules.qwen4_exp.qsa import qsa_topk_triton

        def run(columns):
            scores = torch.randn(17, columns, device="cuda")
            scores[0] = 0
            scores[1, :3] = float("nan")
            scores[2, 0] = float("inf")
            lengths = (
                torch.arange(17, device="cuda", dtype=torch.int32) * 127 + 3
            ).remainder(columns * 4 + 4)
            actual = qsa_topk_triton.try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
            expected = select_qsa_paged_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
            self.assertIsNotNone(actual)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Stay below the radix path while crossing two power-of-two tiles and
        # the 512-block TopK cap. Warm both integer alignment classes.
        self.assert_reuses_variants(
            qsa_topk_triton,
            run,
            (249, 256, 511, 512, 513, 528),
            (250, 251, 253, 509, 510, 514, 517, 530),
        )

    def test_ple_tail_lengths_reuse_variants(self):
        from ple_sm90_test_utils import reference_conv

        from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton

        if not ple_conv_triton._sm_supported(torch.device("cuda")):
            self.skipTest("PLE convolution requires SM90 or SM120")

        def run(length):
            for batch in (1, 2):
                for has_initial, has_residual in (
                    (False, False),
                    (True, False),
                    (True, True),
                ):
                    kw = {"device": "cuda", "dtype": torch.bfloat16}
                    gated = torch.randn(batch, length, 10240, **kw)
                    normed = torch.randn_like(gated)
                    weight = torch.randn(10240, 1, 4, **kw) * 0.1
                    initial = (
                        torch.randn(batch, 9, 10240, **kw) if has_initial else None
                    )
                    residual = torch.randn_like(gated) if has_residual else None
                    actual = ple_conv_triton.try_ple_conv(
                        gated, normed, initial, weight, 3, residual=residual
                    )
                    expected = reference_conv(gated, normed, initial, weight, residual)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        self.assert_reuses_variants(ple_conv_triton, run, (5, 16), (6, 7, 15, 127))

    def test_small_moe_routes_reuse_tile_buckets(self):
        from rtp_llm.models_py.modules.qwen4_exp.moe import moe_align_triton
        from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
            moe_align_block_size_torch,
        )

        def run(rows):
            ids = torch.randint(512, (rows, 10), device="cuda", dtype=torch.int32)
            actual = moe_align_triton.moe_align_decode(ids, 16, 512)
            expected = moe_align_block_size_torch(ids, 16, 512)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)

        self.assert_reuses_variants(
            moe_align_triton, run, (7, 8, 17, 24), (9, 11, 12, 18, 19, 21, 23, 25)
        )

    def test_small_l2norm_reuses_row_counts(self):
        import importlib

        l2norm = importlib.import_module("rtp_llm.models_py.triton_kernels.fla.l2norm")

        def run(rows):
            x = torch.randn(rows, 128, device="cuda")
            actual = l2norm.l2norm_fwd(x)
            expected = x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        self.assert_reuses_variants(
            l2norm, run, (4, 16), (8, 12, 20, 28, 60, 68, 124, 128)
        )


if __name__ == "__main__":
    unittest.main()

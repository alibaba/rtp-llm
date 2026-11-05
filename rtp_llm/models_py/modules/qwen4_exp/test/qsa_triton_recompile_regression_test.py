"""Regression: new request lengths must not recompile the serving kernels.

Timeline evidence (timelines/20261104-qwen4exp-qsaopt-di1-l20x): the
serving-path triton kernels keyed constexpr arguments on per-request geometry
(query lengths, block-table widths, table strides), so the first QSA layer of
every step recompiled 1-5 kernels for 60-590 ms each.  The fix keeps those
arguments as runtime arguments: triton specializes a plain runtime int only into
the bounded {equal-to-1, divisible-by-16, other} classes, so once a small warmup
covers those classes a brand-new request length compiles nothing.

This test pins that property.  The warmup runs one length per split
configuration (splits 1/2/4/8 with block_n 128/128/128/32) plus the fused
prefix-reuse writer; the probes then use NEW lengths chosen to stay inside the
already-warmed specialization classes (all non-divisible-by-16, never 1, and
table widths in the same cdiv bucket), and no qwen4_exp qsa triton kernel may
gain a compiled variant.
"""

import unittest

import torch

from indexer_prefix_reuse_triton_test import IndexerPrefixReuseTritonTest

from rtp_llm.models_py.modules.qwen4_exp.qsa import indexer_prefill_triton
from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_paged_fmha


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
class QsaTritonRecompileRegressionTest(unittest.TestCase):
    # Width divisible by 16 so the packed-selection strides stay in the
    # divisible-by-16 class for every probe query length.
    WIDTH = 1088
    Q_HEADS = 12
    KV_HEADS = 1
    HEAD_DIM = 256
    PAGE = 128
    CACHE_BLOCKS = 64

    def _run_writer(self, length):
        harness = IndexerPrefixReuseTritonTest()
        args = harness.inputs([512], [length])
        result = harness.write(args, args["kv"].clone(), args["state"].clone())
        self.assertIsNotNone(result, "fused writer declined the probe geometry")

    def _run_fmha(self, query_len):
        torch.manual_seed(7)
        width = self.WIDTH
        page = self.PAGE
        span = min((self.CACHE_BLOCKS - 1) * page, query_len + 64)
        q = torch.randn(
            1, self.Q_HEADS, query_len, self.HEAD_DIM,
            device="cuda", dtype=torch.bfloat16,
        )
        cache = torch.randn(
            self.CACHE_BLOCKS, 2, self.KV_HEADS, page, self.HEAD_DIM,
            device="cuda", dtype=torch.bfloat16,
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
        kernels = {}
        for module in (indexer_prefill_triton, sparse_paged_fmha):
            kernels.update(_jit_functions(module))
        self.assertTrue(kernels)

        # Warmup: cover every split configuration of the grouped split kernel
        # (base = batch*query_len*kv_heads crosses the 4096/512/64 thresholds)
        # and the fused prefix-reuse writer.
        self._run_writer(1001)
        for query_len in (5000, 1000, 100, 4):
            self._run_fmha(query_len)
        torch.cuda.synchronize()
        before = {name: _variant_total(fn) for name, fn in kernels.items()}

        # Probes: brand-new lengths inside the warmed specialization classes
        # (non-divisible-by-16, never 1, same table-width buckets).
        self._run_writer(1002)
        self._run_writer(1005)
        for query_len in (5004, 1012, 116, 12):
            self._run_fmha(query_len)
        torch.cuda.synchronize()
        after = {name: _variant_total(fn) for name, fn in kernels.items()}

        grown = {
            name: (before[name], after[name])
            for name in before
            if after[name] > before[name]
        }
        self.assertFalse(
            grown, "per-length triton recompile regression: " + str(grown)
        )


if __name__ == "__main__":
    unittest.main()

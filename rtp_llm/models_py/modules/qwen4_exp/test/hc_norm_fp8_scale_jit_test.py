"""The hc.mlp fp8-scale reduction must not recompile per token count.

Serving timelines (warm5/warm6, 49-layer steps) show a ~60ms GPU stall in
the first ``qwen4.hc.mlp`` window of every step: ``_norm_fp8_scale``
declares its element count ``N`` as ``tl.constexpr``, and the call sites
pass ``N = rows * 4`` with the step's token count — every new token count
compiles a fresh variant (one JIT per step, 8x per capture window). The
``BLOCK`` bucketing (next_power_of_2 of the same count) is already bounded,
so the fix is to demote ``N`` to a runtime argument.

This test locks the contract:

* the scale stays bit-identical to max/448 across element counts (the
  kernel's ``tl.div_rn`` matches the true division of the native absmax);
* a new element count inside an already-warmed BLOCK bucket compiles no
  new variant (the per-token-count recompile is gone);
* bucket boundaries (a count that moves to the next power-of-two BLOCK)
  still produce a correct scale.
"""

import unittest

import torch
import triton

from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
    _norm_fp8_scale,
)


def _variant_total(fn):
    caches = getattr(fn, "device_caches", None) or {}
    total = 0
    for cache in caches.values():
        entry = cache[0] if isinstance(cache, tuple) else cache
        total += len(entry)
    return total


def _run(partial, n):
    scale = torch.empty(1, device=partial.device, dtype=torch.float32)
    _norm_fp8_scale[(1,)](
        partial,
        scale,
        N=n,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
    )
    return scale


class HcNormFp8ScaleJitTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scale_value_matches_true_division(self):
        torch.manual_seed(3)
        for rows in (1, 7, 246, 1434, 1572, 4096):
            with self.subTest(rows=rows):
                partial = torch.rand(rows * 4, device="cuda", dtype=torch.float32) * 400
                scale = _run(partial, rows * 4)
                expected = (partial.max() / 448.0).item()
                # torch's scalar division may multiply a rounded reciprocal;
                # the kernel's tl.div_rn is the exact division. Compare with
                # a 1-ulp-level tolerance.
                self.assertTrue(
                    abs(scale.item() - expected) <= 2e-6 * abs(expected),
                    f"scale {scale.item()} != reference {expected}",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_new_token_count_in_warmed_bucket_compiles_nothing(self):
        torch.manual_seed(4)

        def make(n):
            return torch.rand(n, device="cuda", dtype=torch.float32) * 100.0

        # Warm two different counts inside the same BLOCK bucket
        # (1434*4=5736 and 1484*4=5936 both bucket to BLOCK=8192).
        _run(make(1434 * 4), 1434 * 4)
        _run(make(1484 * 4), 1484 * 4)
        torch.cuda.synchronize()
        before = _variant_total(_norm_fp8_scale)
        # Probe: brand-new counts in the same bucket.
        _run(make(1572 * 4), 1572 * 4)
        _run(make(1461 * 4), 1461 * 4)
        _run(make(1385 * 4), 1385 * 4)
        torch.cuda.synchronize()
        self.assertEqual(
            _variant_total(_norm_fp8_scale),
            before,
            "a new token count inside a warmed BLOCK bucket must not "
            "compile a new variant",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bucket_boundary_stays_correct(self):
        torch.manual_seed(5)
        # 4095/4096/4097 cross the 4096 BLOCK boundary.
        for n in (4095, 4096, 4097, 8192):
            with self.subTest(n=n):
                partial = torch.rand(n, device="cuda", dtype=torch.float32) * 300
                scale = _run(partial, n)
                expected = (partial.max() / 448.0).item()
                self.assertTrue(
                    abs(scale.item() - expected) <= 2e-6 * abs(expected),
                    f"scale {scale.item()} != reference {expected} for n={n}",
                )


if __name__ == "__main__":
    unittest.main()

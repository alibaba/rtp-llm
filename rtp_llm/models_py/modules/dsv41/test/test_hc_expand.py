"""The Prefill embedding must materialize independent, exact mHC lanes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.prefill._hc_expand_triton import expand_hc


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHCExpand(unittest.TestCase):
    def test_exact_independent_lanes_and_tail(self):
        with torch.inference_mode():
            for tokens in (1, 3, 128, 8082, 8193, 16384, 126980):
                source = torch.randint(
                    0, 32767, (tokens, 5120), device="cuda", dtype=torch.int16
                ).view(torch.bfloat16)
                actual = expand_hc(source)
                expected = source.unsqueeze(1).repeat(1, 4, 1)
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                actual[:, 0].zero_()
                self.assertTrue(
                    torch.equal(
                        actual[:, 1].view(torch.int16), source.view(torch.int16)
                    )
                )

    def test_no_recompile_across_token_counts(self):
        # tokens is a runtime argument: one compiled variant per int
        # specialization class (==1, %16==0, generic), not one per value.
        from rtp_llm.models_py.modules.dsv41.prefill import _hc_expand_triton

        kernel = _hc_expand_triton._expand_hc_kernel

        def cache_size():
            return sum(len(dc[0]) for dc in kernel.device_caches.values())

        with torch.inference_mode():
            for tokens in (1, 16, 17):  # cover all int specialization classes
                expand_hc(torch.zeros((tokens, 5120), device="cuda", dtype=torch.bfloat16))
            size_after_warmup = cache_size()
            self.assertGreater(size_after_warmup, 0)
            for tokens in (3, 32, 48, 1001, 126981):
                expand_hc(torch.zeros((tokens, 5120), device="cuda", dtype=torch.bfloat16))
            self.assertEqual(size_after_warmup, cache_size())


if __name__ == "__main__":
    unittest.main()

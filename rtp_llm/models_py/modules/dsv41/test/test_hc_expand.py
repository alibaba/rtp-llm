"""The Prefill embedding must materialize independent, exact mHC lanes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.prefill._hc_expand_triton import expand_hc


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHCExpand(unittest.TestCase):
    def test_exact_independent_lanes_and_tail(self):
        with torch.inference_mode():
            for tokens in (8193, 16384, 126980):
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


if __name__ == "__main__":
    unittest.main()

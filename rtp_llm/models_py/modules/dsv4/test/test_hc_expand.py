"""The Prefill embedding must materialize independent, exact mHC lanes.

R52 port of b394128ed4's test_hc_expand.py onto the dsv4 layout. Loads the
kernel module directly (importlib) so the test runs without the full rtp_llm
package / C++ ops import chain.
"""

import importlib.util
import unittest
from pathlib import Path

import torch


def _load():
    path = Path(__file__).resolve().parents[1] / "prefill" / "_hc_expand_triton.py"
    spec = importlib.util.spec_from_file_location("hc_expand_triton", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


expand_mod = _load()
expand_hc = expand_mod.expand_hc


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

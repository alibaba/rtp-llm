"""Tests for the (optional) self-written sparse MLA prefill kernel.

The custom kernel is shipped out-of-tree while it is being developed: it is a
``torch.utils.cpp_extension`` module exposing ``run(q, kv, idx, out, lse)``.
``sparse_mla_custom`` looks it up through ``LSA_CUSTOM_MLA_MODULE`` (a python
module name) and falls back to the FlashMLA/Triton path otherwise, so this test
skips on machines without it (e.g. local RTX 3090).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

try:
    from sparse_mla_custom import custom_mla_available, sparse_mla_custom_fwd
except Exception:  # pragma: no cover
    custom_mla_available = None
    sparse_mla_custom_fwd = None


class TestSparseMlaCustom(unittest.TestCase):
    def _case(self, T=4, N=4096, S=256, H=32, seed=0):
        torch.manual_seed(seed)
        q = (torch.randn(T, H, 576, device="cuda") * 0.1).bfloat16()
        kv = (torch.randn(N, 576, device="cuda") * 0.1).bfloat16()
        idx = torch.full((T, S), -1, dtype=torch.int32, device="cuda")
        pos = torch.arange(N - T, N, device="cuda")
        for r in range(T):
            p = int(pos[r])
            w = torch.arange(max(0, p - S + 1), p + 1, device="cuda")
            idx[r, : w.numel()] = w.int()
        return q, kv, idx

    def test_matches_torch_reference(self):
        if not (custom_mla_available and custom_mla_available()):
            self.skipTest("custom sparse MLA kernel not available in this environment")
        q, kv, idx = self._case()
        out, lse = sparse_mla_custom_fwd(q, kv, idx)
        k = kv[idx[0].long()].float()
        v = k[:, :512]
        sc = (q[0].float() @ k.t()) / math.sqrt(576.0)
        p = torch.softmax(sc, dim=-1)
        ref = p @ v
        rel = (out[0] - ref).abs().max().item() / max(1e-9, ref.abs().max().item())
        self.assertLess(rel, 2e-2, f"relative error too large: {rel}")

    def test_fallback_is_reported(self):
        """The availability probe must not raise, whatever the environment."""
        if custom_mla_available is None:
            self.skipTest("module import failed (recorded elsewhere)")
        self.assertIn(custom_mla_available(), (True, False))


if __name__ == "__main__":
    unittest.main()

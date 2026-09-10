"""Real HC heads must retain each model's construction-time chunk policy."""

import os
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires Torch for HC forward checks")
class HCHeadOptionsTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.hc import fallback_impl
        from rtp_llm.platforms.ppu.models.dsv4.ppu_module_provider import (
            PpuModuleProvider,
        )

        self.provider = PpuModuleProvider
        self.impl = fallback_impl
        torch.manual_seed(71)
        self.args = dict(
            fn=torch.randn(4, 32),
            base=torch.randn(4),
            scale=torch.ones(1),
            dim=8,
            hc_mult=4,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        self.x = torch.randn(11, 4, 8).to(torch.bfloat16)

    def build(self, options):
        return self.provider(options).build_hc_head(**self.args, tp_size=4, tp_rank=0)

    def run_head(self, head):
        with patch.object(
            self.impl, "_tp_linear_mixes", wraps=self.impl._tp_linear_mixes
        ) as mixes:
            result = head.head(self.x)
        return result, [call.args[1].shape[0] for call in mixes.call_args_list]

    def test_two_heads_retain_distinct_chunks_across_environment_changes(self):
        first = self.build({"DSV4_HC_FALLBACK_CHUNK_TOKENS": "3"})
        second = self.build({"DSV4_HC_FALLBACK_CHUNK_TOKENS": "5"})
        reference = self.impl.FallbackHCHead(**self.args, options={}).head(self.x)
        for env in ("1", "9"):
            with patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": env}):
                a, a_chunks = self.run_head(first)
                b, b_chunks = self.run_head(second)
            self.assertEqual(a_chunks, [3, 3, 3, 2])
            self.assertEqual(b_chunks, [5, 5, 1])
            torch.testing.assert_close(a, reference, rtol=0, atol=0)
            torch.testing.assert_close(b, reference, rtol=0, atol=0)

    def test_empty_snapshot_never_reads_conflicting_environment(self):
        with patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "1"}):
            head = self.build({})
            _, chunks = self.run_head(head)
        self.assertEqual(chunks, [11])

    def test_legacy_head_preserves_environment_policy(self):
        head = self.impl.FallbackHCHead(**self.args)
        with patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "4"}):
            _, chunks = self.run_head(head)
        self.assertEqual(chunks, [4, 4, 3])


if __name__ == "__main__":
    unittest.main()

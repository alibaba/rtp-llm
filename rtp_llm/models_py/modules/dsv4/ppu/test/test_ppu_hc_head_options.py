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
        from rtp_llm.models_py.modules.dsv4.ppu.hc import ppu_hc_head
        from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import (
            PpuModuleProvider,
        )

        self.provider = PpuModuleProvider
        self.impl = ppu_hc_head
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
        with patch.object(self.impl.F, "linear", wraps=self.impl.F.linear) as mixes:
            result = head.head(self.x)
        return result, [call.args[0].shape[0] for call in mixes.call_args_list]

    def test_empty_snapshot_never_reads_conflicting_environment(self):
        with patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "1"}):
            head = self.build({})
            _, chunks = self.run_head(head)
        self.assertEqual(chunks, [11])


if __name__ == "__main__":
    unittest.main()

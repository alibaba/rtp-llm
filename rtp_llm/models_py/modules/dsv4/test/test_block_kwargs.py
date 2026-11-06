"""The shared Block kwargs must stay constructible for every Block flavor.

``_block_kwargs`` feeds both the CUDA ``Block`` (whose ChunkedFp8Fp4MoeLayer
consumes the fused-MoE runtime kwargs) and the PPU ``PpuBlock`` (whose
explicit signature rejects them). These tests pin that split so a new kwarg
cannot break one platform silently.
"""

import unittest

from rtp_llm.models_py.modules.dsv4.transformer import V4Args, _block_kwargs

_FUSED_MOE_RUNTIME_KEYS = (
    "model_type",
    "parallelism_config",
    "moe_config",
    "enable_cuda_graph",
)


class BlockKwargsTest(unittest.TestCase):
    def test_default_includes_fused_moe_runtime_kwargs(self):
        kwargs = _block_kwargs(0, V4Args(), None)
        for key in _FUSED_MOE_RUNTIME_KEYS:
            self.assertIn(key, kwargs)

    def test_ppu_flavor_drops_fused_moe_runtime_kwargs(self):
        kwargs = _block_kwargs(0, V4Args(), None, include_moe_runtime=False)
        for key in _FUSED_MOE_RUNTIME_KEYS:
            self.assertNotIn(key, kwargs)

    def test_ppu_build_block_never_passes_fused_moe_runtime_kwargs(self):
        from rtp_llm.models_py.modules.dsv4.ppu import (
            transformer as ppu_transformer,
        )

        recorded = {}

        class _RecordingProvider:
            def build_block(self, factory, **kwargs):
                recorded.update(kwargs)
                recorded["factory"] = factory
                return object()

        ppu_transformer._build_block(
            0, V4Args(), None, platform_provider=_RecordingProvider()
        )

        self.assertIs(recorded["factory"], ppu_transformer.Block)
        for key in _FUSED_MOE_RUNTIME_KEYS:
            self.assertNotIn(key, recorded)
        # PPU-only plumbing must survive the filtering.
        self.assertIsNotNone(recorded["platform_provider"])


if __name__ == "__main__":
    unittest.main()

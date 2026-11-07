"""The PPU block builder must not receive fused-MoE runtime kwargs.

``ppu/transformer.py`` is packaged only for PPU builds, so this test loads
it from its runfiles path with the PPU-only sibling imports stubbed, the
same technique as ``test_platform_provider``; the CUDA13-lane kwargs
contract itself is covered by ``dsv4/test/test_block_kwargs.py``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv4.transformer import V4Args

ROOT = Path(__file__).parents[1]
PREFIX = "rtp_llm.models_py.modules.dsv4.ppu"

_FUSED_MOE_RUNTIME_KEYS = (
    "model_type",
    "parallelism_config",
    "moe_config",
    "enable_cuda_graph",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PpuBlockKwargsTest(unittest.TestCase):
    def setUp(self):
        # Keep the PPU-only construction leaves out of this contract test.
        self.stub = patch.dict(
            sys.modules,
            {
                PREFIX: types.ModuleType(PREFIX),
                PREFIX + ".block": SimpleNamespace(Block=object),
                PREFIX + ".hc.factory": SimpleNamespace(
                    build_hc_head=lambda *args, **kwargs: None
                ),
                PREFIX + ".tp_norm": SimpleNamespace(
                    tp_gather_hidden=lambda x: x, tp_rms_norm=lambda x: x
                ),
            },
        )
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.module = _load("ppu_contract_transformer", ROOT / "transformer.py")

    def test_ppu_build_block_omits_fused_moe_runtime_kwargs(self):
        recorded = {}

        class _RecordingProvider:
            def build_block(self, factory, **kwargs):
                recorded.update(kwargs)
                recorded["factory"] = factory
                return object()

        self.module._build_block(
            0, V4Args(), None, platform_provider=_RecordingProvider()
        )

        # The PPU Block is the module-level stub class; its packaged
        # MegaMoE never consumes the fused-MoE runtime kwargs.
        self.assertIs(recorded["factory"], object)
        for key in _FUSED_MOE_RUNTIME_KEYS:
            self.assertNotIn(key, recorded)
        # PPU-only plumbing must survive the filtering.
        self.assertIsNotNone(recorded["platform_provider"])


if __name__ == "__main__":
    unittest.main()

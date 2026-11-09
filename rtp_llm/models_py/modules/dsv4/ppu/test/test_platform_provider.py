"""Contracts for the actual instance-bound PPU construction providers."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).parents[1]
PREFIX = "rtp_llm.models_py.modules.dsv4.ppu"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlatformProviderTest(unittest.TestCase):
    def setUp(self):
        # Keep hardware/math leaves out of the construction-contract test.
        self.stub = patch.dict(sys.modules, {
            "torch": SimpleNamespace(bfloat16=object(), float32=object()),
            "rtp_llm.models_py.kernels.ppu.runtime": SimpleNamespace(
                PpuStreamPool=Mock(),
                native_row_gemv_enabled=Mock(return_value=False),
                require_ppu_op=Mock(),
            ),
            PREFIX + ".attention.attention_contract": SimpleNamespace(
                scheduler_key=Mock(), validate_dual_pool=Mock(), validate_swa_pool=Mock()),
        })
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.module = load("ppu_contract_provider", ROOT / "ppu_module_provider.py")
        self.manifest = load("ppu_contract_manifest", ROOT / "manifest.py")
        self.mods = patch.dict(sys.modules, {
            PREFIX + ".ppu_module_provider": self.module,
            PREFIX + ".manifest": self.manifest,
        })
        self.mods.start()
        self.addCleanup(self.mods.stop)
        self.decode = load("ppu_contract_decode", ROOT / "ppu_decode_provider.py")

    def test_options_are_owned_and_boolean_parsing_is_strict(self):
        options = {"flag": "yes"}
        provider = self.module.PpuModuleProvider(options)
        options["flag"] = "no"
        self.assertTrue(provider._bool("flag"))
        self.assertFalse(provider._bool("missing"))
        with self.assertRaises(TypeError):
            provider.execution_options["flag"] = "off"
        for value in ("1", "true", "on", "yes", " TRUE "):
            self.assertTrue(self.module.PpuModuleProvider({"flag": value})._bool("flag"))
        for value in ("0", "false", "off", "no"):
            self.assertFalse(self.module.PpuModuleProvider({"flag": value})._bool("flag"))
        with self.assertRaises(ValueError):
            self.module.PpuModuleProvider({"flag": "maybe"})._bool("flag")

    def test_attention_binds_identity_and_rejects_another_instance(self):
        provider = self.module.PpuModuleProvider({})
        ctor = Mock()
        with patch.dict(sys.modules, {PREFIX + ".attention.ppu_fp4_indexer": SimpleNamespace(PpuFP4Attention=ctor)}):
            self.assertIs(provider.build_attention(None, 7), ctor.return_value)
            ctor.assert_called_once_with(7, platform_provider=provider)
            with self.assertRaises(TypeError):
                provider.build_attention(None, platform_provider=object())

    def test_prefill_and_decode_keep_metadata_and_optional_fusion_contracts(self):
        prefill = self.module.PpuModuleProvider({})
        decode = self.decode.PpuDecodeProvider(self.manifest.DECODE_EXECUTION_OPTIONS)
        factory = Mock()
        for provider in (prefill, decode):
            self.assertIs(provider.build_decode_metadata(factory, 3, value=4), factory.return_value)
            factory.assert_called_with(3, value=4)
        self.assertIsNone(prefill.build_mtp_input_fusion(None, None, None, None))
        decode._decode_fusions = False
        self.assertIsNone(decode.build_mtp_input_fusion(None, None, None, None))
        ctor = Mock()
        decode._decode_fusions = True
        with patch.dict(sys.modules, {PREFIX + ".mtp.ppu_mtp_input": SimpleNamespace(PpuMtpInputFusion=ctor)}):
            self.assertIs(decode.build_mtp_input_fusion(1, 2, 3, 4), ctor.return_value)
            ctor.assert_called_once_with(1, 2, 3, 4)


if __name__ == "__main__":
    unittest.main()

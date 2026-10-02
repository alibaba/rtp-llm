"""The TokenSpeed backend is process-wide for a fixed runtime."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / "native_mla_prefill.py"


class NativeMlaPrefillBackendTest(unittest.TestCase):
    def test_reuses_backend_setup_for_bf16_and_fp8_instances(self):
        backend = types.ModuleType("tokenspeed_mla")
        backend.__path__ = []
        prefill = types.ModuleType("tokenspeed_mla.mla_prefill")
        token_speed = lambda **_kwargs: None
        prefill.tokenspeed_mla_prefill = token_speed
        cutlass = types.ModuleType("rtp_llm.models_py.utils.cutlass")
        setup = unittest.mock.Mock()
        cutlass.setup_cutlass_import_path = setup
        fake_modules = {
            "tokenspeed_mla": backend,
            "tokenspeed_mla.mla_prefill": prefill,
            "rtp_llm.models_py.utils.cutlass": cutlass,
        }
        with patch.dict(sys.modules, fake_modules), patch(
            "importlib.metadata.version", return_value="test-version"
        ) as get_version:
            spec = importlib.util.spec_from_file_location("k3_prefill_backend_under_test", SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            bf16 = module.KimiK3TokenspeedPrefill(fp8_compute=False)
            fp8 = module.KimiK3TokenspeedPrefill(fp8_compute=True)

        self.assertIs(bf16._run, token_speed)
        self.assertIs(fp8._run, token_speed)
        self.assertEqual(bf16.operand_dtype, torch.bfloat16)
        self.assertEqual(fp8.operand_dtype, torch.float8_e4m3fn)
        setup.assert_called_once_with()
        get_version.assert_called_once_with("tokenspeed-mla")


if __name__ == "__main__":
    unittest.main()

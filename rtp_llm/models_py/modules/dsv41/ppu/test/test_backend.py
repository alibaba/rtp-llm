"""Provider-entry tests: fail-loud semantics and capability probes."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41 import backend as b
from rtp_llm.models_py.modules.dsv41.ppu import backend as provider


class GetBackendTest(unittest.TestCase):
    def test_missing_framework_descriptor_base_is_rejected(self):
        from rtp_llm.models_py.modules.dsv41.ppu.weight import DeepSeekV41PPUWeight

        with patch.object(
            DeepSeekV41PPUWeight, "framework_base_available", return_value=False
        ):
            with self.assertRaisesRegex(
                RuntimeError, "framework weight descriptor base"
            ):
                provider.get_backend()

    def test_get_backend_returns_validated_backend(self):
        backend = provider.get_backend()
        self.assertIsInstance(backend, b.V41Backend)
        # validate() enforces name/strategy/ops/model_cls/weight_cls.
        self.assertIs(backend.validate(), backend)
        self.assertEqual(backend.name, "ppu")
        self.assertIsInstance(backend.model_cls, type)
        self.assertIsInstance(backend.weight_cls, type)
        from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel
        from rtp_llm.models_py.modules.dsv41.ppu.weight import DeepSeekV41PPUWeight

        self.assertIs(backend.model_cls, DeepSeekV41PPUModel)
        self.assertIs(backend.weight_cls, DeepSeekV41PPUWeight)
        for name in b.OP_NAMES:
            self.assertTrue(callable(getattr(backend.ops, name)), name)
        self.assertTrue(callable(backend.check_capabilities))

    def test_registry_entry_returns_integrated_backend(self):
        backend = b.get_v41_backend("ppu")
        self.assertIsInstance(backend.validate(), b.V41Backend)
        self.assertIn("ppu", b._backends)
        self.assertIs(b.get_v41_backend("ppu"), backend)

    def test_get_backend_fails_loudly_on_broken_device_surface(self):
        script = """
import sys, types
stub = types.ModuleType('rtp_llm.device.device_type')
stub.get_device_type = lambda: None
sys.modules['rtp_llm.device.device_type'] = stub
sys.path.insert(0, '.')
from rtp_llm.models_py.modules.dsv41.ppu import backend as provider
try:
    provider.get_backend()
except RuntimeError as error:
    assert 'not integrated' in str(error), error
else:
    raise AssertionError('partial device surface accepted')
print('ok')
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_get_ops_returns_contract_ops(self):
        ops = provider.get_ops(
            ep_size=1,
            num_routed_experts=8,
            num_local_experts=8,
            hidden_size=64,
            inter_dim=32,
        )
        for name in b.OP_NAMES:
            self.assertTrue(callable(getattr(ops, name)), name)


class CheckCapabilitiesTest(unittest.TestCase):
    @patch.dict("os.environ", {"RTP_V41_PPU_MOE_PRECISION": "w4a4"})
    def test_missing_production_fp4_api_fails_even_if_old_api_exists(self):
        modules = {
            name: SimpleNamespace(**{symbol: lambda: None for symbol in symbols})
            for name, symbols in provider.kernel_symbols()
        }
        del modules["deep_gemm"].m_grouped_gemm_fp4_fp4_bf16_nt_nopad
        modules["deep_gemm"].m_grouped_gemm_w4a16_nopad = lambda: None
        modules["deep_gemm"].preprocess_mxfp4_scales = lambda: None
        with patch(
            "rtp_llm.device.device_type.is_ppu", return_value=True
        ), patch.object(
            provider.importlib, "import_module", side_effect=modules.__getitem__
        ):
            with self.assertRaisesRegex(
                RuntimeError, "m_grouped_gemm_fp4_fp4_bf16_nt_nopad"
            ):
                provider.check_capabilities(None)

    def _is_ppu(self):
        from rtp_llm.device.device_type import is_ppu

        return is_ppu()

    def test_rejects_non_ppu_host(self):
        if self._is_ppu():
            self.skipTest("running on a PPU host")
        with self.assertRaises(RuntimeError) as ctx:
            provider.check_capabilities(None)
        self.assertIn("requires a PPU device", str(ctx.exception))

    def test_strategy_mismatch_is_rejected(self):
        class _Runtime:
            strategy = "some_other_strategy"

        if self._is_ppu():
            self.skipTest("running on a PPU host")
        with self.assertRaises(RuntimeError):
            provider.check_capabilities(_Runtime())


class ImportDisciplineTest(unittest.TestCase):
    def test_provider_module_imports_no_accelerator(self):
        script = (
            "import sys;"
            "import rtp_llm.models_py.modules.dsv41.ppu.backend as backend;"
            "bad = [m for m in ('deep_gemm', 'deep_ep', 'flash_mla', 'triton',"
            " 'tilelang') if m in sys.modules];"
            "assert not bad, bad;"
            "assert callable(backend.get_ops);"
            "print('ok')"
        )
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()

"""PPU fallback registration does not require a private backend or device."""

import sys
import types
import unittest
from unittest.mock import patch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.platforms.ppu.attention import register_attention


class AttentionRegistrationTest(unittest.TestCase):
    def test_public_fallbacks_and_optional_backend_priority(self):
        flashinfer = types.ModuleType("flashinfer_test")
        names = (
            "PyFlashinferPrefillImpl",
            "PyFlashinferHybridPrefillImpl",
            "PyFlashinferPagedPrefillImpl",
            "PyFlashinferDecodeImpl",
        )
        for name in names:
            setattr(flashinfer, name, type(name, (), {}))
        cp = types.ModuleType("cp_test")
        cp.CPFlashInferImpl = type("CPFlashInferImpl", (), {})
        prefix = "rtp_llm.models_py.modules.factory.attention."
        modules = {
            prefix + "cuda_impl.py_flashinfer_mha": flashinfer,
            prefix + "cuda_cp_impl.prefill_cp_flashinfer": cp,
        }
        with patch.dict(sys.modules, modules), patch(
            "rtp_llm.device.device_type.get_device_type", return_value=DeviceType.Ppu
        ):
            prefill, decode = [], []
            register_attention(prefill_mha_imps=prefill, decode_mha_imps=decode)
            self.assertEqual(
                prefill,
                [getattr(flashinfer, name) for name in names[:3]]
                + [cp.CPFlashInferImpl],
            )
            self.assertEqual(decode, [flashinfer.PyFlashinferDecodeImpl])
            optimized = object()
            prefill, decode = [optimized], [optimized]
            register_attention(prefill_mha_imps=prefill, decode_mha_imps=decode)
            self.assertIs(prefill[0], optimized)
            self.assertIs(decode[0], optimized)

    def test_other_devices_do_not_load_ppu_fallbacks(self):
        with patch(
            "rtp_llm.device.device_type.get_device_type", return_value=DeviceType.Cuda
        ):
            prefill, decode = [], []
            register_attention(prefill_mha_imps=prefill, decode_mha_imps=decode)
            self.assertEqual((prefill, decode), ([], []))


if __name__ == "__main__":
    unittest.main()

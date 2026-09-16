from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper


class TestFp8GemmNtDefaults(TestCase):
    def _call_with_device_mode(self, uses_e8m0: bool, explicit=None) -> bool:
        captured = {}

        def fake_impl(*args, **kwargs):
            captured.update(kwargs)

        with patch.object(deepgemm_wrapper, "_fp8_gemm_nt_impl", fake_impl), patch.object(
            deepgemm_wrapper,
            "is_deep_gemm_e8m0_used",
            return_value=uses_e8m0,
        ):
            deepgemm_wrapper.fp8_gemm_nt(
                (None, None),
                (None, None),
                None,
                disable_ue8m0_cast=explicit,
            )
        return captured["disable_ue8m0_cast"]

    def test_none_tracks_device_scale_format(self):
        self.assertFalse(self._call_with_device_mode(uses_e8m0=True))
        self.assertTrue(self._call_with_device_mode(uses_e8m0=False))

    def test_explicit_value_is_preserved(self):
        self.assertTrue(self._call_with_device_mode(uses_e8m0=True, explicit=True))
        self.assertFalse(self._call_with_device_mode(uses_e8m0=False, explicit=False))


if __name__ == "__main__":
    main()

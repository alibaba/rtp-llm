import importlib
import sys
import types
import unittest
from unittest import mock


class DeepGemmWrapperLazyTest(unittest.TestCase):
    def test_fp8_fp4_wrapper_lazily_resolves_symbol(self):
        fp8_fp4_impl = mock.Mock()
        fake_deep_gemm = types.SimpleNamespace(fp8_fp4_gemm_nt=fp8_fp4_impl)
        a = (object(), object())
        b = (object(), object())
        output = object()
        module_name = "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"
        sys.modules.pop(module_name, None)
        missing = object()
        previous_deep_gemm = sys.modules.get("deep_gemm", missing)
        sys.modules["deep_gemm"] = fake_deep_gemm
        try:
            deepgemm_wrapper = importlib.import_module(module_name)
            self.assertIsNone(deepgemm_wrapper._fp8_fp4_gemm_nt_impl)
        finally:
            if previous_deep_gemm is missing:
                sys.modules.pop("deep_gemm", None)
            else:
                sys.modules["deep_gemm"] = previous_deep_gemm

        with mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ), mock.patch.object(
            deepgemm_wrapper,
            "has_deep_gemm",
            return_value=True,
        ), mock.patch.object(
            deepgemm_wrapper, "_fp8_fp4_gemm_nt_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_require_sm100_packed_scale_for_fp8_fp4"
        ), mock.patch.object(
            deepgemm_wrapper, "is_deep_gemm_e8m0_used", return_value=True
        ):
            deepgemm_wrapper.fp8_fp4_gemm_nt(a, b, output)
            self.assertIs(deepgemm_wrapper._fp8_fp4_gemm_nt_impl, fp8_fp4_impl)

        fp8_fp4_impl.assert_called_once_with(
            a,
            b,
            output,
            None,
            recipe=None,
            recipe_a=None,
            recipe_b=None,
            compiled_dims="nk",
            disable_ue8m0_cast=False,
        )

    def test_required_symbols_reject_backend_import_failure(self):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        deepgemm_wrapper.has_deep_gemm.cache_clear()
        self.addCleanup(deepgemm_wrapper.has_deep_gemm.cache_clear)
        with mock.patch.object(
            deepgemm_wrapper, "has_module", return_value=True
        ), mock.patch.dict(sys.modules, {"deep_gemm": None}):
            self.assertFalse(
                deepgemm_wrapper.has_deep_gemm(("m_grouped_fp8_gemm_nt_masked",))
            )

    def test_required_symbols_reject_missing_grouped_kernel(self):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        fake_deep_gemm = types.SimpleNamespace(
            get_num_sms=mock.Mock(return_value=132),
            set_num_sms=mock.Mock(),
            m_grouped_fp8_gemm_nt_contiguous=mock.Mock(),
        )
        required_symbols = (
            "get_num_sms",
            "set_num_sms",
            "m_grouped_fp8_gemm_nt_contiguous",
            "m_grouped_fp8_gemm_nt_masked",
        )
        deepgemm_wrapper.has_deep_gemm.cache_clear()
        self.addCleanup(deepgemm_wrapper.has_deep_gemm.cache_clear)
        with mock.patch.object(
            deepgemm_wrapper, "has_module", return_value=True
        ), mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ), mock.patch.object(
            deepgemm_wrapper, "_get_num_sms_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_set_num_sms_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_m_grouped_fp8_gemm_nt_contiguous_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_m_grouped_fp8_gemm_nt_masked_impl", None
        ):
            self.assertFalse(deepgemm_wrapper.has_deep_gemm(required_symbols))

    def test_required_symbols_accept_complete_backend_and_configure_sms(self):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        get_num_sms = mock.Mock(return_value=132)
        set_num_sms = mock.Mock()
        grouped_contiguous = mock.Mock()
        grouped_masked = mock.Mock()
        fake_deep_gemm = types.SimpleNamespace(
            get_num_sms=get_num_sms,
            set_num_sms=set_num_sms,
            m_grouped_fp8_gemm_nt_contiguous=grouped_contiguous,
            m_grouped_fp8_gemm_nt_masked=grouped_masked,
        )
        required_symbols = (
            "get_num_sms",
            "set_num_sms",
            "m_grouped_fp8_gemm_nt_contiguous",
            "m_grouped_fp8_gemm_nt_masked",
        )
        deepgemm_wrapper.has_deep_gemm.cache_clear()
        self.addCleanup(deepgemm_wrapper.has_deep_gemm.cache_clear)
        with mock.patch.object(
            deepgemm_wrapper, "has_module", return_value=True
        ), mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ), mock.patch.object(
            deepgemm_wrapper, "_get_num_sms_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_set_num_sms_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_m_grouped_fp8_gemm_nt_contiguous_impl", None
        ), mock.patch.object(
            deepgemm_wrapper, "_m_grouped_fp8_gemm_nt_masked_impl", None
        ):
            self.assertTrue(deepgemm_wrapper.has_deep_gemm(required_symbols))
            with deepgemm_wrapper.configure_deep_gemm_num_sms(64):
                set_num_sms.assert_called_once_with(64)

        get_num_sms.assert_called_once_with()
        set_num_sms.assert_has_calls([mock.call(64), mock.call(132)])


if __name__ == "__main__":
    unittest.main()

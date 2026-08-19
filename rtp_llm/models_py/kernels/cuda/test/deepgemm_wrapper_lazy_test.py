import sys
import types
import unittest
from unittest import mock

from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper


class DeepGemmWrapperLazyTest(unittest.TestCase):
    def test_fp8_fp4_wrapper_lazily_resolves_symbol(self):
        fp8_fp4_impl = mock.Mock()
        fake_deep_gemm = types.SimpleNamespace(fp8_fp4_gemm_nt=fp8_fp4_impl)
        a = (object(), object())
        b = (object(), object())
        output = object()
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


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda import fp4_linear


class Fp4PrefillAutotuneTest(unittest.TestCase):
    def test_only_explicit_prefill_override_disables_autotune(self):
        packed = torch.zeros((16, 16), dtype=torch.uint8)
        scales = torch.ones((16, 2))
        scale = torch.tensor(1.0)
        activation = torch.ones((16, 32), dtype=torch.bfloat16)
        expected = torch.zeros((16, 16), dtype=torch.bfloat16)
        env = "RTP_LLM_FP4_PREFILL_AUTOTUNE"
        cases = (
            ("PREFILL", None, True),
            ("PREFILL", "0", False),
            ("PREFILL", "1", True),
            ("DECODE", "0", True),
        )
        for role, value, enabled in cases:
            with self.subTest(role=role, value=value), patch.dict(
                os.environ, ROLE_TYPE=role, RTP_LLM_FP4_GEMM_BACKEND="cute-dsl"
            ), patch.object(
                fp4_linear, "fp4_quantize", return_value=(packed, scales)
            ), patch.object(
                fp4_linear, "autotune", return_value=nullcontext()
            ) as tune, patch.object(
                fp4_linear, "mm_fp4", return_value=expected
            ) as gemm:
                if value is None:
                    os.environ.pop(env, None)
                else:
                    os.environ[env] = value
                linear = fp4_linear.CudaFp4GEMMLinear(
                    packed, scales, scale, weight_scale_2=scale
                )
                torch.testing.assert_close(linear(activation), expected, rtol=0, atol=0)
                tune.assert_called_once_with(tune_mode=enabled)
                self.assertEqual(gemm.call_args.kwargs["backend"], "cute-dsl")
                self.assertTrue(gemm.call_args.kwargs["use_nvfp4"])


if __name__ == "__main__":
    unittest.main()

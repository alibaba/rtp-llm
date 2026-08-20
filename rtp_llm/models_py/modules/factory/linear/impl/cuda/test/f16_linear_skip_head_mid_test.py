import os
import sys
import tempfile
import unittest

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}_{os.getpid()}"),
)
os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
)


class CudaF16LinearSkipHeadMidContractTest(unittest.TestCase):
    def test_invalid_split_is_rejected_before_deep_gemm_dispatch(self) -> None:
        inputs = torch.empty((1, 512), dtype=torch.bfloat16)
        checkpoint_weight = torch.empty((512, 12 * 256), dtype=torch.bfloat16)
        linear = CudaF16Linear(checkpoint_weight)

        with self.assertRaises(ValueError):
            linear.forward_skip_head_mid(inputs, (96, 64, 160))


class CudaF16LinearSkipHeadMidTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("bf16_gemm_nt_skip_head_mid requires SM100")
        import deep_gemm

        self.assertTrue(callable(deep_gemm.bf16_gemm_nt_skip_head_mid))
        if _LOCAL_DEEP_GEMM_PATH:
            self.assertTrue(deep_gemm.__file__.startswith(_LOCAL_DEEP_GEMM_PATH))

    def test_k3_projection_layout_and_values(self) -> None:
        torch.manual_seed(123)
        tokens = 257
        heads = 12
        k_dim = 512
        k_nope_dim = 128
        k_pe_dim = 64
        v_dim = 128
        logical_head_dim = k_nope_dim + v_dim
        physical_head_dim = k_nope_dim + k_pe_dim + v_dim

        inputs = torch.randn(
            (tokens, k_dim), device="cuda", dtype=torch.bfloat16
        )
        checkpoint_weight = torch.randn(
            (k_dim, heads * logical_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)

        actual = linear.forward_skip_head_mid(
            inputs, (k_nope_dim, k_pe_dim, v_dim)
        ).view(tokens, heads, physical_head_dim)
        caller_output = torch.empty(
            (tokens, heads * physical_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        caller_output_ptr = caller_output.data_ptr()
        reused = linear.forward_skip_head_mid(
            inputs,
            (k_nope_dim, k_pe_dim, v_dim),
            out=caller_output,
        )
        expected = linear(inputs).view(tokens, heads, logical_head_dim)

        torch.testing.assert_close(
            actual[..., :k_nope_dim],
            expected[..., :k_nope_dim],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual[..., k_nope_dim + k_pe_dim :],
            expected[..., k_nope_dim:],
            rtol=0,
            atol=0,
        )
        self.assertTrue(actual.is_contiguous())
        self.assertIs(reused, caller_output)
        self.assertEqual(reused.data_ptr(), caller_output_ptr)
        torch.testing.assert_close(reused.view_as(actual), actual, rtol=0, atol=0)
        self.assertEqual(linear.weight.stride(), (1, heads * logical_head_dim))

    def test_caller_output_contract_is_validated(self) -> None:
        inputs = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.randn(
            (512, 12 * 256), device="cuda", dtype=torch.bfloat16
        )
        linear = CudaF16Linear(checkpoint_weight)

        wrong_shape = torch.empty((1, 12 * 319), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "output must be contiguous"):
            linear.forward_skip_head_mid(
                inputs,
                (128, 64, 128),
                out=wrong_shape,
            )


if __name__ == "__main__":
    unittest.main()

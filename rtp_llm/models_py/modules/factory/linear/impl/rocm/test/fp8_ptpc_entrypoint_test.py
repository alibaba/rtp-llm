"""CPU dispatch contracts with mocked GPU primitives, not kernel validation."""

import importlib
import sys
import types
import unittest
from unittest import mock

import torch


class Fp8PtpcEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.aiter = types.ModuleType("aiter")
        self.aiter.gemm_a8w8_bpreshuffle = mock.Mock()
        self.aiter.hipb_mm = mock.Mock()
        self.aiter.hipb_create_extension = mock.Mock()
        gemm = types.ModuleType("aiter.ops.gemm_op_a8w8")
        self.cktile = gemm.gemm_a8w8_bpreshuffle_cktile = mock.Mock()
        quant = types.ModuleType("rtp_llm.models_py.kernels.rocm.fp8_kernel")
        self.quant = quant.rocm_per_token_quant_fp8 = mock.Mock()
        quant.rocm_per_token_group_quant_fp8 = mock.Mock()
        self.external = mock.patch.dict(
            sys.modules,
            {
                "aiter": self.aiter,
                "aiter.ops": types.ModuleType("aiter.ops"),
                "aiter.ops.gemm_op_a8w8": gemm,
                "rtp_llm.models_py.kernels.rocm.fp8_kernel": quant,
            },
        )
        self.external.start()
        self.addCleanup(self.external.stop)
        # Import the actual production module. No native binding or model
        # objects are replaced; only the GPU primitive boundary is mocked.
        name = "rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear"
        self.module = importlib.import_module(name)
        self.module = importlib.reload(self.module)

    def test_actual_export_is_resolved_by_newloader_preflight(self):
        from rtp_llm.models_py.quant_methods.fp8 import _rocm_ptpc_executor_available

        self.assertTrue(callable(self.module.run_rocm_fp8_ptpc_no_swizzle))
        self.assertTrue(_rocm_ptpc_executor_available("run_rocm_fp8_ptpc_no_swizzle"))
        self.assertFalse(_rocm_ptpc_executor_available("missing_executor"))

    def test_dispatch_boundaries_preserve_weight_scale_and_dtype(self):
        cases = (
            (1, 64, 128, True),
            (1, 64, 192, False),
            (511, 1537, 192, False),
            (512, 1536, 192, False),
            (512, 1537, 192, True),
            (1535, 64, 192, False),
            (1536, 64, 192, True),
        )
        for m, n, k, cktile in cases:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(m=m, n=n, k=k, dtype=dtype):
                    self.aiter.gemm_a8w8_bpreshuffle.reset_mock()
                    self.cktile.reset_mock()
                    x = torch.ones(m, k, dtype=dtype)
                    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
                    scale = torch.ones(n, 1)
                    qx = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
                    sx = torch.ones(m, 1)
                    self.quant.return_value = qx, sx
                    self.aiter.gemm_a8w8_bpreshuffle.return_value = torch.full(
                        (m, n), 3.0, dtype=torch.bfloat16
                    )
                    self.cktile.side_effect = lambda *args: args[-1].fill_(3)
                    actual = self.module.run_rocm_fp8_ptpc_no_swizzle(
                        x, weight, scale, n, torch.ones(n)
                    )
                    called = self.cktile if cktile else self.aiter.gemm_a8w8_bpreshuffle
                    unused = self.aiter.gemm_a8w8_bpreshuffle if cktile else self.cktile
                    called.assert_called_once()
                    unused.assert_not_called()
                    self.assertIs(called.call_args.args[1], weight)
                    self.assertIs(called.call_args.args[3], scale)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, torch.full_like(x[:, :1].expand(m, n), 4)
                    )
                    self.assertEqual(self.quant.call_args.args[0].dtype, torch.bfloat16)
                    self.assertEqual(self.quant.call_args.kwargs, {"eps": 1e-10})

    def test_class_and_module_owned_entrypoint_share_dispatch(self):
        linear = self.module.RocmFp8PTPCLinearNoSwizzle(
            weight=torch.zeros(32, 16, dtype=torch.float8_e4m3fn),
            weight_scales=torch.ones(1, 16),
            bias=torch.ones(16),
        )
        x = torch.ones(2, 32, dtype=torch.bfloat16)
        expected = torch.ones(2, 16, dtype=torch.bfloat16)
        with mock.patch.object(
            self.module, "run_rocm_fp8_ptpc_no_swizzle", return_value=expected
        ) as execute:
            self.assertIs(linear(x), expected)
        args = execute.call_args.args
        self.assertIs(args[0], x)
        self.assertIs(args[1], linear.weight)
        self.assertIs(args[2], linear.weight_scales)
        self.assertEqual(args[3], 16)
        self.assertIs(args[4], linear.bias)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Direct contract test for CudaFp8DeepGEMMLinear quantized-input reuse.

This file deliberately loads ``fp8_deepgemm_linear.py`` with tiny dependency
stubs, so it can run from a source tree without compiled RTP-LLM .so files.
It only verifies the Python-side contract added for DSV4 attention QKV input
quant reuse; real DeepGEMM numerical coverage remains in the Bazel tests.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def _install_stubs(calls):
    deepgemm_wrapper = types.ModuleType(
        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"
    )

    def fp8_gemm_nt(a, b, output, **kwargs):
        calls.append(("fp8_gemm_nt", a, b, kwargs))
        output.fill_(3.0)

    deepgemm_wrapper.fp8_gemm_nt = fp8_gemm_nt
    deepgemm_wrapper.has_deep_gemm = lambda: True
    deepgemm_wrapper.is_deep_gemm_e8m0_used = lambda: True

    fp8_kernel = types.ModuleType("rtp_llm.models_py.kernels.cuda.fp8_kernel")

    def sgl_per_token_group_quant_fp8(input, **kwargs):
        calls.append(("quant", input.shape, kwargs))
        return (
            torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device),
            torch.empty((input.shape[0], 1), dtype=torch.int32, device=input.device),
        )

    def create_per_token_group_quant_fp8_output_scale(**kwargs):
        x_shape = kwargs["x_shape"]
        return torch.empty((x_shape[0], 1), dtype=torch.int32, device=kwargs["device"])

    fp8_kernel.create_per_token_group_quant_fp8_output_scale = (
        create_per_token_group_quant_fp8_output_scale
    )
    fp8_kernel.requant_weight_ue8m0 = lambda *a, **k: None
    fp8_kernel.sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_fp8

    linear_pkg = types.ModuleType("rtp_llm.models_py.modules.factory.linear")

    class LinearBase(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    linear_pkg.LinearBase = LinearBase

    ops_mod = types.ModuleType("rtp_llm.ops")
    ops_mod.HWKernelConfig = object

    sys.modules["rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"] = deepgemm_wrapper
    sys.modules["rtp_llm.models_py.kernels.cuda.fp8_kernel"] = fp8_kernel
    sys.modules["rtp_llm.models_py.modules.factory.linear"] = linear_pkg
    sys.modules["rtp_llm.ops"] = ops_mod


def _load_module(calls):
    _install_stubs(calls)
    path = (
        Path(__file__).resolve().parents[1]
        / "fp8_deepgemm_linear.py"
    )
    spec = importlib.util.spec_from_file_location("fp8_deepgemm_linear_contract", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class QuantizedInputContractTest(unittest.TestCase):
    def _make_linear(self, mod, *, bias=False):
        layer = mod.CudaFp8DeepGEMMLinear.__new__(mod.CudaFp8DeepGEMMLinear)
        nn.Module.__init__(layer)
        layer.K = 4
        layer.N = 3
        layer.scale_ue8m0 = True
        layer.cached_scales = None
        layer.cached_scales_max_len = 0
        layer.weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
        layer.weight_scales = torch.empty((3, 1), dtype=torch.int32)
        layer.bias = torch.ones((3,), dtype=torch.bfloat16) if bias else None
        return layer

    def test_forward_uses_quantize_then_forward_quantized(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        x = torch.zeros((2, 4), dtype=torch.bfloat16)

        out = layer(x)

        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(calls[0][0], "quant")
        self.assertEqual(calls[1][0], "fp8_gemm_nt")
        self.assertEqual(calls[1][1][0].dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(calls[1][1][0].shape), (2, 4))
        self.assertEqual(calls[1][1][1].dtype, torch.int32)
        self.assertEqual(tuple(calls[1][1][1].shape), (2, 1))

    def test_forward_quantized_reuses_supplied_quant_tuple(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod, bias=True)
        x_fp8 = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((2, 1), dtype=torch.int32)

        out = layer.forward_quantized(x_fp8, x_scale)

        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "fp8_gemm_nt")
        self.assertIs(calls[0][1][0], x_fp8)
        self.assertIs(calls[0][1][1], x_scale)
        self.assertTrue(torch.all(out == torch.tensor(4.0, dtype=out.dtype)))

    def test_forward_quantized_respects_out_buffer(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        x_fp8 = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((2, 1), dtype=torch.int32)
        out = torch.empty((2, 3), dtype=torch.bfloat16)

        got = layer.forward_quantized(x_fp8, x_scale, out=out)

        self.assertIs(got, out)
        self.assertTrue(torch.all(out == torch.tensor(3.0, dtype=out.dtype)))


if __name__ == "__main__":
    unittest.main()

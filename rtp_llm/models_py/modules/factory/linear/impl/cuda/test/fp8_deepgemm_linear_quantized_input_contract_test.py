#!/usr/bin/env python3
"""Direct contracts for CudaFp8DeepGEMMLinear quantized-input paths.

This file deliberately loads ``fp8_deepgemm_linear.py`` with tiny dependency
stubs, so it can run from a source tree without compiled RTP-LLM .so files.
It verifies quantized-input reuse and fused packed-output dispatch; real
DeepGEMM numerical coverage remains in the SM100 Bazel test.
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

    def fp8_gemm_nt_skip_head_mid(a, b, output, head_splits, **kwargs):
        calls.append(("fp8_gemm_nt_skip_head_mid", a, b, head_splits, kwargs))
        left, middle, right = head_splits
        heads = output.shape[1] // (left + middle + right)
        packed = output.view(output.shape[0], heads, left + middle + right)
        packed[..., :left].fill_(5.0)
        packed[..., left + middle :].fill_(7.0)

    deepgemm_wrapper.fp8_gemm_nt = fp8_gemm_nt
    deepgemm_wrapper.fp8_gemm_nt_skip_head_mid = fp8_gemm_nt_skip_head_mid
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

    quantized_activation = types.ModuleType(
        "rtp_llm.models_py.modules.factory.linear.quantized_activation"
    )

    class QuantizedActivation:
        def __init__(self, values, scale_wire, bf16=None):
            self.values = values
            self.scale_wire = scale_wire
            self.bf16 = bf16

        @property
        def scales(self):
            return self.scale_wire.T[: self.shape[0]]

        @property
        def shape(self):
            return self.values.shape

        @property
        def ndim(self):
            return self.values.ndim

        @property
        def dtype(self):
            return self.values.dtype

        @property
        def device(self):
            return self.values.device

        @property
        def is_cuda(self):
            return self.values.is_cuda

        def dim(self):
            return self.values.dim()

    quantized_activation.QuantizedActivation = QuantizedActivation

    ops_mod = types.ModuleType("rtp_llm.ops")
    ops_mod.HWKernelConfig = object

    sys.modules["rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"] = deepgemm_wrapper
    sys.modules["rtp_llm.models_py.kernels.cuda.fp8_kernel"] = fp8_kernel
    sys.modules["rtp_llm.models_py.modules.factory.linear"] = linear_pkg
    sys.modules["rtp_llm.models_py.modules.factory.linear.quantized_activation"] = (
        quantized_activation
    )
    sys.modules["rtp_llm.ops"] = ops_mod


def _load_module(calls):
    _install_stubs(calls)
    path = Path(__file__).resolve().parents[1] / "fp8_deepgemm_linear.py"
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

    def test_skip_head_mid_reuses_quantized_input_and_preserves_gap(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        values = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        scale_wire = torch.empty((1, 4), dtype=torch.int32)
        activation = mod.QuantizedActivation(values, scale_wire)
        output = torch.full((2, 4), 11.0, dtype=torch.bfloat16)
        layer.supports_skip_head_mid = lambda input, splits: True

        got = layer.forward_skip_head_mid(activation, (1, 1, 2), output=output).view(
            2, 1, 4
        )

        self.assertEqual(
            got.untyped_storage().data_ptr(), output.untyped_storage().data_ptr()
        )
        self.assertEqual([call[0] for call in calls], ["fp8_gemm_nt_skip_head_mid"])
        self.assertIs(calls[0][1][0], values)
        self.assertTrue(torch.all(got[..., :1] == 5.0))
        self.assertTrue(torch.all(got[..., 1:2] == 11.0))
        self.assertTrue(torch.all(got[..., 2:] == 7.0))

    def test_skip_head_mid_does_not_advertise_bias_support(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod, bias=True)
        input = types.SimpleNamespace(
            ndim=2,
            shape=(2, layer.K),
            dtype=torch.bfloat16,
            is_cuda=True,
            device=layer.weight.device,
        )

        self.assertFalse(layer.supports_skip_head_mid(input, (1, 1, 2)))


if __name__ == "__main__":
    unittest.main()

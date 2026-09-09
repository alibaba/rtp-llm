"""CPU boundary/ABI tests; these do not qualify PPU kernel numerics or speed."""

import importlib.util
import os
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.platforms.ppu import runtime as RUNTIME
from rtp_llm.platforms.ppu.models.dsv4 import ppu_wo_a as WO_A
from rtp_llm.platforms.ppu.modules.linear import fp8_linear as MODULE


class Fp8CpuContractTest(unittest.TestCase):
    def scale(self, shape=(1, 1)):
        return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_checkpoint_scale_preserves_encoded_power_of_two(self):
        raw = torch.tensor([[0, 120, 127, 130, 254]], dtype=torch.uint8)
        with mock.patch.object(MODULE, "_require_m890p"):
            actual = MODULE.checkpoint_ue8m0_scale_to_fp32(
                raw.view(torch.float8_e8m0fnu), (128, 640)
            )
        expected = torch.pow(torch.tensor(2.0), raw.to(torch.int32) - 127)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (1, 5))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_scale_shape_dtype_and_weight_alignment_rejected(self):
        for scale, shape, error in [
            (self.scale(), (127, 128), ValueError),
            (self.scale((1, 2)), (128, 128), ValueError),
            (torch.ones((1, 1)), (128, 128), TypeError),
        ]:
            with self.subTest(shape=shape, dtype=scale.dtype):
                with self.assertRaises(error):
                    MODULE.checkpoint_ue8m0_scale_to_fp32(scale, shape)

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_cpu_tensors_are_rejected_before_native_resolution(self):
        with mock.patch.object(MODULE, "_resolve_deep_gemm_symbol") as native:
            with self.assertRaisesRegex(ValueError, "CUDA/PPU"):
                MODULE.PpuFp8Linear(
                    torch.zeros((128, 128), dtype=torch.float8_e4m3fn), self.scale()
                )
        native.assert_not_called()

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_dense_receives_fp32_scales_and_reuses_them(self):
        weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
        activation = torch.zeros((2, 128), dtype=torch.bfloat16)
        quantized = torch.zeros((2, 128), dtype=torch.float8_e4m3fn)
        scales = torch.ones((2, 1), dtype=torch.float32)
        seen = []

        def gemm(lhs, rhs, out):
            self.assertIs(lhs[0], quantized)
            self.assertIs(lhs[1], scales)
            self.assertIs(rhs[0], weight)
            self.assertEqual(rhs[1].dtype, torch.float32)
            self.assertEqual(tuple(rhs[1].shape), (1, 1))
            seen.append(rhs[1])
            out.fill_(2)

        with mock.patch.object(MODULE, "_require_m890p"), mock.patch.object(
            MODULE, "_resolve_deep_gemm_symbol", return_value=gemm
        ), mock.patch.object(
            MODULE, "quantize_ppu_fp8_activation", return_value=(quantized, scales)
        ):
            layer = MODULE.PpuFp8Linear(weight, self.scale())
            explicit = torch.empty((2, 128), dtype=torch.bfloat16)
            self.assertIs(layer(activation, out=explicit), explicit)
            self.assertTrue(torch.equal(explicit, torch.full_like(explicit, 2)))
            layer(activation)
            self.assertIs(seen[0], seen[1])
            with self.assertRaisesRegex(ValueError, "alias"):
                layer(activation, out=activation)

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_empty_batch_does_not_quantize_or_launch(self):
        with mock.patch.object(MODULE, "_require_m890p"), mock.patch.object(
            MODULE, "_resolve_deep_gemm_symbol"
        ) as resolver, mock.patch.object(
            MODULE, "quantize_ppu_fp8_activation"
        ) as quant:
            layer = MODULE.PpuFp8Linear(
                torch.zeros((128, 128), dtype=torch.float8_e4m3fn), self.scale()
            )
            output = layer(torch.empty((0, 128), dtype=torch.bfloat16))
            self.assertEqual(tuple(output.shape), (0, 128))
            resolver.return_value.assert_not_called()
            quant.assert_not_called()

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "Torch lacks UE8M0 dtype")
    def test_wo_a_tp4_group_geometry_and_einsum_abi(self):
        calls = []

        def einsum(equation, lhs, rhs, out, *, recipe):
            calls.append(equation)
            self.assertEqual(equation, "bhr,hdr->bhd")
            self.assertEqual(recipe, (1, 1, 128))
            self.assertEqual(tuple(lhs[0].shape), (3, 2, 128))
            self.assertEqual(tuple(rhs[0].shape), (2, 128, 128))
            self.assertEqual(rhs[1].dtype, torch.float32)
            out.zero_()

        quant = (torch.zeros((6, 128), dtype=torch.float8_e4m3fn), torch.ones((6, 1)))
        with mock.patch.object(MODULE, "_require_m890p"), mock.patch.object(
            WO_A, "_require_m890p"
        ), mock.patch.object(
            WO_A, "_resolve_deep_gemm_symbol", return_value=einsum
        ), mock.patch.object(
            WO_A, "quantize_ppu_fp8_activation", return_value=quant
        ):
            layer = WO_A.PpuWoAFp8Linear(
                torch.zeros((256, 128), dtype=torch.float8_e4m3fn),
                self.scale((2, 1)),
                groups=2,
                k_local=128,
            )
            output = layer(torch.zeros((3, 2, 128), dtype=torch.bfloat16))
            self.assertEqual(tuple(output.shape), (3, 2, 128))
            self.assertEqual(len(calls), 1)

    def test_missing_deepgemm_symbol_fails_without_fallback(self):
        with mock.patch.object(
            MODULE.importlib, "import_module", return_value=types.SimpleNamespace()
        ):
            with self.assertRaisesRegex(RuntimeError, "callable deep_gemm.fp8_gemm_nt"):
                MODULE._resolve_deep_gemm_symbol("fp8_gemm_nt")

    def test_jit_lock_serializes_cache_misses_and_is_idempotent(self):
        active = maximum = 0
        mutex = threading.Lock()

        def build(value):
            nonlocal active, maximum
            with mutex:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with mutex:
                active -= 1
            return value

        with tempfile.TemporaryDirectory() as cache:
            compiler = types.SimpleNamespace(build=build, get_cache_dir=lambda: cache)
            RUNTIME.install_deep_gemm_build_lock(compiler)
            wrapped = compiler.build
            RUNTIME.install_deep_gemm_build_lock(compiler)
            self.assertIs(compiler.build, wrapped)
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(
                    list(pool.map(compiler.build, range(8))), list(range(8))
                )
        self.assertEqual(maximum, 1)


if __name__ == "__main__":
    unittest.main()

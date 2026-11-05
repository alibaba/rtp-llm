"""CPU tests for the R6-2 FP8 wo_a prefill helpers (convention-level).

Verifies the UE8M0 scale convention and int32 packing match the decode
kernel's documented layout without needing a GPU (kernel launch itself is
exercised by the GPU bench + full-stack receipts).
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[6]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
    ".models_py.modules.dsv4.fp8",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

woa = importlib.import_module("rtp_llm.models_py.modules.dsv4.fp8._v41_woa_fp8_prefill")


class WoaFlagTest(unittest.TestCase):
    def test_default_on(self):
        # R45 adoption: default ON; =0 is the BF16 escape hatch.
        env = dict(os.environ)
        env.pop("DSV41_WOA_FP8_EINSUM", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(woa.woa_fp8_enabled())
        with mock.patch.dict(os.environ, {"DSV41_WOA_FP8_EINSUM": "0"}):
            self.assertFalse(woa.woa_fp8_enabled())

    def test_supported_rejects_cpu(self):
        w = torch.zeros(8, 1024, 4096, dtype=torch.bfloat16)
        self.assertFalse(woa.woa_fp8_supported(w))


class WoaWeightQuantTest(unittest.TestCase):
    def test_scale_bytes_and_packing(self):
        # H=1, N=4, K=512 -> 16 blocks of 32 -> P=4 int32 words per (h,n).
        w = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
        w[:, :, 0:128] = 1.0  # amax=1      -> 2^-8  -> 0x77
        w[:, :, 128:256] = 0.0  # clamped eps -> 2^-42 -> 0x55
        w[:, :, 256:384] = 448.0  # amax=448    -> 2^0   -> 0x7F
        w[:, :, 384:512] = 0.001  #             -> 2^-18 -> 0x6D
        wq, scale = woa.quantize_woa_weight(w)
        self.assertEqual(wq.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(wq.shape), (1, 4, 512))
        self.assertEqual(tuple(scale.shape), (1, 4, 4))
        # tma_n = 4 (N aligned); storage [H, P, tma_n] m-fastest
        self.assertEqual(scale.stride(), (16, 1, 4))
        expect = (0x77777777, 0x55555555, 0x7F7F7F7F, 0x6D6D6D6D)
        for n in range(4):
            for p in range(4):
                got = int(scale[0, n, p]) & 0xFFFFFFFF
                self.assertEqual(
                    got, expect[p], f"n={n} p={p}: {got:#x} != {expect[p]:#x}"
                )
        # dequantized values: block0 keeps 1.0 exactly (scale 2^-8, 256 exact)
        deq0 = wq[0, 0, 0:128].float() * float(2**-8)
        self.assertTrue(torch.allclose(deq0, torch.full_like(deq0, 1.0), atol=1e-3))
        # zero block stays zero
        self.assertEqual(wq[0, 0, 128:256].float().abs().max().item(), 0.0)
        # 448 block at scale 1.0
        self.assertAlmostEqual(wq[0, 0, 256].float().item(), 448.0, places=1)

    def test_k_multiple_requirement(self):
        w = torch.zeros(8, 1024, 4096 - 1, dtype=torch.bfloat16)
        self.assertFalse(woa.woa_fp8_supported(w))


if __name__ == "__main__":
    unittest.main()

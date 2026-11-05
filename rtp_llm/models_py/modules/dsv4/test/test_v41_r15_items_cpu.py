"""CPU tests for the Batch-4b part 2 (R15) item flags and layout math:
E5a strided wo_a quantizer, E5b Triton inverse RoPE, E6 shared QKV quant,
E7 scale swizzle, E1-alt fused SWA roundtrip. Kernel numerics are proven
bitwise on GPU (r15_microbench gates) and by the parity sha gate; these
tests cover flag parsing, guards, fallbacks and the swizzle index math."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


swz = load("_r15_swz", "_v41_scale_swizzle.py")
woa = load("_r15_woa", "fp8/_v41_woa_strided_quant.py")
e6m = load("_r15_e6", "_v41_shared_qkv_quant.py")
e5b = load("_r15_e5b", "_v41_inv_rope_triton.py")
swa = load("_r15_swa", "fp8/_v41_swa_triton.py")


class FlagParsingTest(unittest.TestCase):
    # Adopted defaults (R16): byte-exact + median better + >=2/3 pairs.
    DEFAULT_ON = (
        (woa, "woa_strided_quant_enabled", "DSV41_E5A_WOA_STRIDED_QUANT"),
        (swz, "e7_scale_swizzle_enabled", "DSV41_E7_SCALE_SWIZZLE"),
        (e6m, "e6_shared_qkv_quant_enabled", "DSV41_E6_SHARED_QKV_QUANT"),
        (swa, "e1_fused_roundtrip_enabled", "DSV41_E1_FUSED_ROUNDTRIP_SWA"),
    )
    # Dropped/neutral items keep default 0 with the =1 opt-in.
    DEFAULT_OFF = ((e5b, "e5b_inv_rope_enabled", "DSV41_E5B_INV_ROPE_TRITON"),)

    def test_defaults(self):
        clean = dict(os.environ)
        for module, fn, env in self.DEFAULT_ON + self.DEFAULT_OFF:
            clean.pop(env, None)
        for module, fn, env in self.DEFAULT_ON:
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, clean, clear=True):
                    self.assertTrue(getattr(module, fn)())
        for module, fn, env in self.DEFAULT_OFF:
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, clean, clear=True):
                    self.assertFalse(getattr(module, fn)())

    def test_explicit_values(self):
        for module, fn, env in self.DEFAULT_ON + self.DEFAULT_OFF:
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, {env: "1"}):
                    self.assertTrue(getattr(module, fn)())
                for value in ("0", "true", ""):
                    with mock.patch.dict(os.environ, {env: value}):
                        self.assertFalse(getattr(module, fn)())


class SwizzleReferenceTest(unittest.TestCase):
    def test_size(self):
        self.assertEqual(swz.swizzled_scale_size(1024, 16), 1024 * 16)
        self.assertEqual(swz.swizzled_scale_size(127, 22), 128 * 24)
        self.assertEqual(swz.swizzled_scale_size(129, 16), 256 * 16)

    def test_reference_matches_explicit_formula(self):
        g = torch.Generator().manual_seed(7)
        for m, k in ((5, 6), (128, 8), (130, 3), (1, 1)):
            src = torch.randint(0, 256, (m, k), generator=g, dtype=torch.uint8)
            ref = swz.swizzle_reference(src)
            pm = (m + 127) // 128 * 128
            pk = (k + 3) // 4 * 4
            self.assertEqual(ref.numel(), pm * pk)
            out = torch.zeros(pm * pk, dtype=torch.uint8)
            for row in range(m):
                for col in range(k):
                    off = (
                        ((row // 128) * (pk // 4) + col // 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) // 32) * 4
                        + col % 4
                    )
                    out[off] = src[row, col]
            self.assertTrue(torch.equal(ref, out), f"m={m} k={k}")

    def test_linear_swizzle_validation(self):
        with self.assertRaises(ValueError):
            swz.swizzle_linear(torch.zeros((4, 8), dtype=torch.int32))
        with self.assertRaises(ValueError):
            swz.swizzle_linear(torch.zeros((4, 8), dtype=torch.uint8).t())
        with self.assertRaises(ValueError):
            swz.swizzle_from_packed_int32(torch.zeros((4, 2), dtype=torch.int32), 4, 8)


class E5aWrapperGuardTest(unittest.TestCase):
    def test_rejects_cpu_and_bad_layout(self):
        # CPU tensors never take the fused path (is_cuda gate).
        base = torch.zeros((2, 4, 256), dtype=torch.bfloat16)
        self.assertIsNone(woa.quantize_strided_gtr(base.permute(1, 0, 2)))
        self.assertIsNone(woa.quantize_strided_gtr(base[:, :2]))


class E1RoundtripContractTest(unittest.TestCase):
    def test_cpu_contract_unchanged(self):
        kv = torch.zeros((3, 512), dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"DSV41_E1_FUSED_ROUNDTRIP_SWA": "1"}):
            with self.assertRaises(ValueError):
                swa.roundtrip_swa(kv)

    def test_cpu_zero_rows_still_validated(self):
        kv = torch.zeros((0, 512), dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            swa.roundtrip_swa(kv)


if __name__ == "__main__":
    unittest.main(verbosity=2)

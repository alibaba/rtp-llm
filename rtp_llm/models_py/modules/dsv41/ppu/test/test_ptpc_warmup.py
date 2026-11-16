"""Scratch isolation and failure cleanup for the pre-readiness enumeration."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import ptpc_warmup as warmup


class PTPCWarmupTest(unittest.TestCase):
    def test_disabled_does_not_inspect_model_or_import_vendor(self):
        with patch.dict(os.environ, {warmup.MAX_ROWS_ENV: "0"}):
            warmup.warmup_resident_ptpc(None)

    def test_negative_bound_rejected(self):
        with patch.dict(os.environ, {warmup.MAX_ROWS_ENV: "-1"}):
            with self.assertRaises(ValueError):
                warmup.warmup_resident_ptpc(None)

    def test_grouped_uses_exact_provider_selector_and_deduplicates(self):
        calls, modes = [], []
        weight = torch.ones((16, 32)).to(torch.float8_e4m3fn)
        scale = torch.ones((16, 1))

        def group(lhs, scales, rhs, rhs_scales, out):
            self.assertEqual(tuple(out.shape), (lhs.shape[1], 2, 16))
            self.assertTrue(torch.equal(lhs.float(), torch.zeros_like(lhs.float())))
            self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
            self.assertIs(rhs, grouped_weight)
            calls.append(lhs.shape[1])
            out.fill_(7)

        grouped_weight = torch.ones((2, 16, 32)).to(torch.float8_e4m3fn)
        op = SimpleNamespace(weight=weight, scale=scale, _gemm=lambda *args: None,
                             _group_weight=grouped_weight, _group_scale=torch.ones((2, 16, 1)),
                             _group_direct_gemm=group)
        provider = SimpleNamespace(
            GemmType=SimpleNamespace(Batched="batched"), get_num_sms=lambda: 78,
            get_best_int8_configs=lambda m, *args, **kw: (78, 16 if m < 3 else 32, 16, 32, 16, 16, 2, (0, 0, 0)),
        )
        vendor = SimpleNamespace(get_compile_mode=lambda: 0, set_compile_mode=modes.append)
        with tempfile.TemporaryDirectory() as cache, patch.dict(
            os.environ, {warmup.MAX_ROWS_ENV: "4", "DG_JIT_CACHE_DIR": cache}
        ), patch.dict(sys.modules, {"deep_gemm": vendor, "deep_gemm.jit_kernels.einsum": provider}), patch.object(
            warmup, "resident_families", return_value=({(16, 32): op}, {(2, 16, 32): op})
        ), patch.object(torch.cuda, "synchronize"):
            warmup.warmup_resident_ptpc(None)
        self.assertEqual(calls, [1, 3])
        self.assertEqual(modes, [1, 0])

    def test_dense_scratch_enumeration_and_mode_restoration(self):
        for fail in (False, True):
            calls, modes = [], []
            weight = torch.ones((16, 32)).to(torch.float8_e4m3fn)
            scale = torch.ones((16, 1))

            def gemm(a, b, out):
                lhs, lhs_scale = a
                self.assertTrue(torch.equal(lhs.float(), torch.zeros_like(lhs.float())))
                self.assertTrue(torch.equal(lhs_scale, torch.ones_like(lhs_scale)))
                self.assertIs(b[0], weight)
                self.assertIs(b[1], scale)
                self.assertNotEqual(out.untyped_storage()._cdata, weight.untyped_storage()._cdata)
                calls.append(lhs.shape[0])
                out.fill_(9)
                if fail:
                    raise RuntimeError("injected compiler failure")

            op = SimpleNamespace(weight=weight, scale=scale, _gemm=gemm)
            vendor = SimpleNamespace(get_compile_mode=lambda: 7, set_compile_mode=modes.append)
            with tempfile.TemporaryDirectory() as cache, patch.dict(
                os.environ, {warmup.MAX_ROWS_ENV: "3", "DG_JIT_CACHE_DIR": cache}
            ), patch.dict(sys.modules, {"deep_gemm": vendor}), patch.object(
                warmup, "resident_families", return_value=({(16, 32): op}, {})
            ), patch.object(torch.cuda, "synchronize"):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        warmup.warmup_resident_ptpc(None)
                else:
                    warmup.warmup_resident_ptpc(None)
                self.assertEqual(calls, [1] if fail else [1, 2, 3])
                self.assertEqual(modes, [1, 7])
                self.assertTrue(torch.equal(weight.float(), torch.ones_like(weight.float())))
                self.assertTrue(torch.equal(scale, torch.ones_like(scale)))


if __name__ == "__main__":
    unittest.main()

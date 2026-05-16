"""Correctness and dispatch tests for the TileLang mHC post kernel."""

from __future__ import annotations

import importlib
import importlib.util
import os
import unittest
from contextlib import contextmanager

import torch


def _prepare_tilelang_env() -> None:
    kernel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "tilelang_kernels.py",
    )
    spec = importlib.util.spec_from_file_location("_dsv4_tilelang_kernels", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "_ensure_libz3_loadable"):
        mod._ensure_libz3_loadable()
    if hasattr(mod, "_ensure_tvm_tmpdir_writable"):
        mod._ensure_tvm_tmpdir_writable()


_prepare_tilelang_env()

_POST = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel"
)


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _make_inputs(m: int, hidden: int = 4096, mhc: int = 4):
    torch.manual_seed(20260516 + m)
    x = (torch.randn(1, m, hidden, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    residual = (
        torch.randn(1, m, mhc, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
    ).contiguous()
    post = torch.rand(1, m, mhc, 1, device="cuda", dtype=torch.float32).contiguous()
    comb = (torch.randn(1, m, mhc, mhc, device="cuda", dtype=torch.float32) * 0.1).contiguous()
    return x, residual, post, comb


def _reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    out = post * x.unsqueeze(-2).float()
    out = out + torch.matmul(comb.transpose(-1, -2), residual.float())
    return out.to(torch.bfloat16)


class MHCPostDispatchTest(unittest.TestCase):
    def test_auto_selector_thresholds(self) -> None:
        select = _POST._select_mhc_post_fwd_variant_name
        self.assertEqual(
            select(1, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(128, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(257, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(512, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(1024, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(3079, 4, 4096, optimized_supported=True, requested="auto"),
            "mid",
        )
        self.assertEqual(
            select(4096, 4, 4096, optimized_supported=True, requested="auto"),
            "baseline",
        )
        self.assertEqual(
            select(1024, 8, 4096, optimized_supported=True, requested="auto"),
            "baseline",
        )
        self.assertEqual(
            select(1024, 4, 2048, optimized_supported=True, requested="auto"),
            "baseline",
        )
        self.assertEqual(
            select(1024, 4, 4096, optimized_supported=False, requested="large"),
            "baseline",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_auto_matches_baseline_and_reference(self) -> None:
        for m in (1, 2, 3, 7, 31, 128, 257, 512, 1024, 4096):
            with self.subTest(m=m), torch.inference_mode():
                x, residual, post, comb = _make_inputs(m)
                ref = _reference(x, residual, post, comb)
                with _env("DSV4_MHC_POST_VARIANT", "baseline"):
                    baseline = _POST.mhc_post_fwd(x, residual, post, comb)
                with _env("DSV4_MHC_POST_VARIANT", "auto"):
                    optimized = _POST.mhc_post_fwd(x, residual, post, comb)
                    selected = _POST.mhc_post_last_selected_variant()

                if m <= 3079:
                    expected = "mid"
                else:
                    expected = "baseline"
                self.assertEqual(selected, expected)
                self.assertEqual(optimized.dtype, torch.bfloat16)
                self.assertEqual(tuple(optimized.shape), (1, m, 4, 4096))
                self.assertTrue(optimized.is_contiguous())
                torch.testing.assert_close(baseline, ref, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(optimized, baseline, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()

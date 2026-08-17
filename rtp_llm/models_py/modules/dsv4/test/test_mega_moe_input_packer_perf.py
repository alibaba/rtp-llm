"""Performance guard for the DSV4 MegaMoE input packer."""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace

import torch

_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "moe",
    "_mega_input_pack_triton.py",
)
_SPEC = importlib.util.spec_from_file_location("_mega_input_pack_triton", _KERNEL_PATH)
_KERNEL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_KERNEL)
fused_pack_mega_moe_inputs_legacy = _KERNEL.fused_pack_mega_moe_inputs_legacy
fused_pack_mega_moe_inputs_optimized = _KERNEL.fused_pack_mega_moe_inputs_optimized


def _make_buf(tokens: int, dim: int, topk: int, device: str):
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


def _bench(fn, warmup: int = 30, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return sorted(times)[len(times) // 2]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MegaMoeInputPackerPerfTest(unittest.TestCase):
    def _case(self, tokens: int, dim: int = 4096, topk: int = 6):
        torch.manual_seed(tokens)
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.3
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(0, 256, (tokens, topk), device="cuda", dtype=torch.int64)
        ref = _make_buf(tokens, dim, topk, "cuda")
        got = _make_buf(tokens, dim, topk, "cuda")

        def run_legacy():
            fused_pack_mega_moe_inputs_legacy(
                x, weights, indices, ref.x, ref.x_sf, ref.topk_idx, ref.topk_weights
            )

        def run_optimized():
            fused_pack_mega_moe_inputs_optimized(
                x, weights, indices, got.x, got.x_sf, got.topk_idx, got.topk_weights
            )

        run_legacy()
        run_optimized()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu()))
        self.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
        self.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
        self.assertTrue(torch.equal(ref.topk_weights.cpu(), got.topk_weights.cpu()))

        legacy_ms = _bench(run_legacy)
        optimized_ms = _bench(run_optimized)
        print(
            f"[MegaMoE pack] T={tokens:5d} D={dim} topk={topk}: "
            f"legacy={legacy_ms * 1000:.2f}us optimized={optimized_ms * 1000:.2f}us "
            f"speedup={legacy_ms / optimized_ms:.2f}x"
        )
        return legacy_ms, optimized_ms

    def test_perf_token_sweep(self):
        gated = {}
        for tokens in (1, 4, 16, 64, 256, 1024, 8192):
            legacy_ms, optimized_ms = self._case(tokens)
            if tokens in (16, 64, 256, 1024):
                gated[tokens] = (legacy_ms, optimized_ms)

        failures = []
        for tokens, (legacy_ms, optimized_ms) in gated.items():
            if not (optimized_ms <= legacy_ms * 0.82):
                failures.append(
                    f"T={tokens}: optimized={optimized_ms * 1000:.2f}us, "
                    f"legacy={legacy_ms * 1000:.2f}us"
                )
        self.assertFalse(failures, "MegaMoE packer perf gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    unittest.main()

"""Performance guard for DSV4 fused inverse-RoPE + FP8 quantization."""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch

_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "_fused_inv_rope_fp8_quant_triton.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "_fused_inv_rope_fp8_quant_triton", _KERNEL_PATH
)
_KERNEL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_KERNEL)
fused_inv_rope_fp8_quant_legacy = _KERNEL.fused_inv_rope_fp8_quant_legacy
fused_inv_rope_fp8_quant_optimized = _KERNEL.fused_inv_rope_fp8_quant_optimized


N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
N_GROUPS = 8
HEADS_PER_GROUP = N_HEADS // N_GROUPS


def _make_freqs(rows: int) -> torch.Tensor:
    ang = torch.rand(rows, ROPE_DIM // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).contiguous()


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


def _run_legacy(o: torch.Tensor, freqs: torch.Tensor):
    return fused_inv_rope_fp8_quant_legacy(
        o,
        freqs,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
    )


def _run_optimized(o: torch.Tensor, freqs: torch.Tensor):
    return fused_inv_rope_fp8_quant_optimized(
        o,
        freqs,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedInvRopeFp8QuantPerfTest(unittest.TestCase):
    def _check_outputs(self, legacy, optimized, label: str):
        fp8_ref, scale_ref = legacy
        fp8_opt, scale_opt = optimized
        fp8_diff = (
            fp8_ref.contiguous().view(torch.uint8).to(torch.int16)
            - fp8_opt.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        scale_diff = (
            scale_ref.contiguous().view(torch.uint8).to(torch.int16)
            - scale_opt.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        fp8_exact = (fp8_diff == 0).float().mean().item()
        scale_exact = (scale_diff == 0).float().mean().item()
        print(
            f"[{label}] fp8_exact={fp8_exact * 100:.2f}% max_ulp={fp8_diff.max().item()} "
            f"scale_exact={scale_exact * 100:.2f}% max_scale_byte={scale_diff.max().item()}"
        )
        self.assertGreaterEqual(fp8_exact, 0.95)
        self.assertGreaterEqual(scale_exact, 0.99)
        self.assertLessEqual(scale_diff.max().item(), 1)

    def _case(self, label: str, o: torch.Tensor, freqs: torch.Tensor, iters: int = 200):
        legacy = _run_legacy(o, freqs)
        optimized = _run_optimized(o, freqs)
        torch.cuda.synchronize()
        self._check_outputs(legacy, optimized, label)

        legacy_ms = _bench(lambda: _run_legacy(o, freqs), iters=iters)
        optimized_ms = _bench(lambda: _run_optimized(o, freqs), iters=iters)
        print(
            f"[InvRoPE quant] {label}: "
            f"legacy={legacy_ms * 1000:.2f}us optimized={optimized_ms * 1000:.2f}us "
            f"speedup={legacy_ms / optimized_ms:.2f}x"
        )
        return legacy_ms, optimized_ms

    def test_decode_perf_gate(self):
        gated = {}
        for batch in (1, 4, 16, 64, 256):
            torch.manual_seed(batch)
            o = (
                torch.randn(batch, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
                * 0.3
            ).contiguous()
            freqs = _make_freqs(batch)
            legacy_ms, optimized_ms = self._case(f"decode_B={batch}", o, freqs)
            if batch in (1, 16, 64):
                gated[batch] = (legacy_ms, optimized_ms)

        failures = []
        for batch, (legacy_ms, optimized_ms) in gated.items():
            if not (optimized_ms <= legacy_ms * 0.75):
                failures.append(
                    f"B={batch}: optimized={optimized_ms * 1000:.2f}us, "
                    f"legacy={legacy_ms * 1000:.2f}us"
                )
        self.assertFalse(failures, "decode perf gate failed: " + "; ".join(failures))

    def test_prefill_perf_gate(self):
        gated = {}
        for seqlen, iters in ((256, 100), (4096, 30)):
            torch.manual_seed(seqlen)
            o = (
                torch.randn(1, seqlen, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
                * 0.3
            ).contiguous()
            freqs = _make_freqs(seqlen)
            legacy_ms, optimized_ms = self._case(
                f"prefill_S={seqlen}", o.view(seqlen, N_HEADS, HEAD_DIM), freqs, iters=iters
            )
            gated[seqlen] = (legacy_ms, optimized_ms)

        failures = []
        for seqlen, (legacy_ms, optimized_ms) in gated.items():
            if not (optimized_ms <= legacy_ms * 0.90):
                failures.append(
                    f"S={seqlen}: optimized={optimized_ms * 1000:.2f}us, "
                    f"legacy={legacy_ms * 1000:.2f}us"
                )
        self.assertFalse(failures, "prefill perf gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    unittest.main()

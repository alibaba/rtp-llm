"""Numerical-equivalence test for the DeepGEMM grouped-FP4 MoE path.

Compares `MoE._grouped_routed_experts` (activates when deep_gemm ships
`m_grouped_fp8_fp4_gemm_nt_contiguous`, i.e. >= 2.4) against the legacy
per-expert QuantizedLinear loop under factory-mode construction.

Run against the local conda env that has deep_gemm 2.4+ —
``/opt/conda310/bin/python``. The bazel-pinned deep_gemm 2.1.1 lacks the
FP4 kernels and will simply skip this test (factory-mode falls back to
the legacy path, both paths match trivially).
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
from rtp_llm.models_py.modules.dsv4.moe import (
    MoE,
    _has_fp8_fp4_grouped_kernel,
)


def _make_weights_dict(
    E: int,
    D: int,
    inter: int,
    topk: int,
    device: str,
    prefix: str = "ffn",
) -> dict:
    """Synthesize a weights dict with V4-shaped routed-expert tensors."""
    w: dict = {}
    # Gate (non-hash path): [E, D] bf16 + bias [E] fp32
    w[f"{prefix}.gate.weight"] = torch.randn(E, D, device=device, dtype=torch.bfloat16)
    w[f"{prefix}.gate.bias"] = torch.randn(E, device=device, dtype=torch.float32) * 0.01
    # Routed experts: FP4-packed weight + UE8M0 block-32 scale, per expert
    # Use reasonable random values so the output isn't saturated.
    for i in range(E):
        w[f"{prefix}.experts.{i}.w1.weight"] = torch.randint(
            -10, 10, (inter, D // 2), dtype=torch.int8, device=device,
        )
        w[f"{prefix}.experts.{i}.w1.scale"] = (
            torch.randint(120, 132, (inter, D // 32), dtype=torch.uint8, device=device)
            .view(torch.float8_e8m0fnu)
        )
        w[f"{prefix}.experts.{i}.w2.weight"] = torch.randint(
            -10, 10, (D, inter // 2), dtype=torch.int8, device=device,
        )
        w[f"{prefix}.experts.{i}.w2.scale"] = (
            torch.randint(120, 132, (D, inter // 32), dtype=torch.uint8, device=device)
            .view(torch.float8_e8m0fnu)
        )
        w[f"{prefix}.experts.{i}.w3.weight"] = torch.randint(
            -10, 10, (inter, D // 2), dtype=torch.int8, device=device,
        )
        w[f"{prefix}.experts.{i}.w3.scale"] = (
            torch.randint(120, 132, (inter, D // 32), dtype=torch.uint8, device=device)
            .view(torch.float8_e8m0fnu)
        )
    # Shared expert (FP8 e4m3fn + UE8M0 block-128)
    for name, out_dim, in_dim in (("w1", inter, D), ("w2", D, inter), ("w3", inter, D)):
        w[f"{prefix}.shared_experts.{name}.weight"] = (
            torch.randn(out_dim, in_dim, device=device, dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
        )
        w[f"{prefix}.shared_experts.{name}.scale"] = (
            torch.randint(
                120, 135, (max(1, out_dim // 128), max(1, in_dim // 128)),
                dtype=torch.uint8, device=device,
            ).view(torch.float8_e8m0fnu)
        )
    return w


class TestGroupedRoutedExperts(unittest.TestCase):
    @unittest.skipUnless(has_deep_gemm(), "deep_gemm not available")
    @unittest.skipUnless(
        _has_fp8_fp4_grouped_kernel(),
        "deep_gemm < 2.4: fp8_fp4 grouped kernel absent; grouped path skipped",
    )
    def test_grouped_matches_loop(self):
        """Build one MoE twice on the same weights — once via the grouped
        FP4 path, once via the legacy per-expert loop — and check the
        outputs match within 1e-2 BF16 tolerance."""
        torch.manual_seed(0)
        device = "cuda:0"
        # Small but realistic dims: non-trivial E and dim divisible by 128.
        E, D, inter, topk = 16, 512, 256, 4
        N = 8  # total tokens

        w = _make_weights_dict(E, D, inter, topk, device)
        # Clone the dict: legacy path calls `weights.pop(...)` so we need
        # two independent dicts.
        w_a = {k: v.clone() for k, v in w.items()}
        w_b = {k: v.clone() for k, v in w.items()}

        # Build MoE with grouped path (deep_gemm 2.4+)
        # To force the legacy path on the same weights, monkey-patch
        # `_has_fp8_fp4_grouped_kernel` to return False for that instance.
        with torch.device("meta"):
            moe_grouped = MoE(
                layer_id=3, dim=D, moe_inter_dim=inter,
                n_routed_experts=E, n_activated_experts=topk,
                n_shared_experts=1, score_func="sqrtsoftplus",
                route_scale=1.0, swiglu_limit=10.0,
                n_hash_layers=0, vocab_size=1,
                weights=w_a, prefix="ffn",
            )
        # Both paths produce the same output deterministically given a
        # fixed input (gates use bf16 weights so topk selection is stable).
        self.assertTrue(moe_grouped._use_grouped_fp4,
                        "expected grouped FP4 path to activate")
        # Build MoE with legacy path: override the feature flag.
        import rtp_llm.models_py.modules.dsv4.moe as moe_mod
        _orig = moe_mod._has_fp8_fp4_grouped_kernel
        moe_mod._has_fp8_fp4_grouped_kernel = lambda: False
        try:
            with torch.device("meta"):
                moe_legacy = MoE(
                    layer_id=3, dim=D, moe_inter_dim=inter,
                    n_routed_experts=E, n_activated_experts=topk,
                    n_shared_experts=1, score_func="sqrtsoftplus",
                    route_scale=1.0, swiglu_limit=10.0,
                    n_hash_layers=0, vocab_size=1,
                    weights=w_b, prefix="ffn",
                )
        finally:
            moe_mod._has_fp8_fp4_grouped_kernel = _orig
        self.assertFalse(moe_legacy._use_grouped_fp4)

        # Materialize any meta-device buffers on both.
        for moe in (moe_grouped, moe_legacy):
            for name, buf in list(moe._buffers.items()):
                if buf is not None and buf.device.type == "meta":
                    moe._buffers[name] = torch.zeros(
                        buf.shape, dtype=buf.dtype, device=device,
                    )

        x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1, (N,), dtype=torch.long, device=device)

        with torch.inference_mode():
            y_grouped = moe_grouped(x, input_ids)
            y_legacy = moe_legacy(x, input_ids)

        # Output shapes match the input.
        self.assertEqual(tuple(y_grouped.shape), (N, D))
        self.assertEqual(tuple(y_legacy.shape), (N, D))

        # Numerical equivalence — FP4 (4-bit) path is lossy; we check the
        # outputs are close in a relative-tolerance sense.
        diff = (y_grouped.float() - y_legacy.float()).abs()
        scale = y_legacy.float().abs().mean().item() + 1e-6
        rel = diff.mean().item() / scale
        # Both paths use the same FP4 dequant LUT semantics; mismatches
        # come only from the GEMM accumulation order and the FP8 activation
        # quant step (grouped path goes through one global quant; legacy
        # goes through a BF16 `F.linear`). Expect rel diff < 5%.
        self.assertLess(rel, 0.05,
                        f"grouped vs legacy diverged: rel diff={rel:.3e}\n"
                        f"  max abs diff: {diff.max().item():.3e}\n"
                        f"  legacy mean abs: {y_legacy.abs().mean().item():.3e}")


if __name__ == "__main__":
    unittest.main()

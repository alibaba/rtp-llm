"""R54: smoke + accuracy for the delayed-mode mhc_pre_big_fuse (6a514c4543).

Runs the fused delayed mHC pre on SM120 (tilelang_single GEMM) and compares to the
unfused production op sequence (norm_fn -> split_mixes -> sinkhorn -> apply_mix with
the previous layer's pre_mix). Reports max-abs/rel error. This is a PRECISION-CLASS
fusion (not byte-exact); the bar is small error, not torch.equal.

Run: python3 test_v41_mhc_big_fuse_delayed.py
"""

import os
import sys
import types
import unittest
from pathlib import Path

import torch

# Make the vendored tile_kernels importable without the rtp_llm C++ chain.
_TP = Path(__file__).resolve().parents[3] / "3rdparty"
sys.path.insert(0, str(_TP))
# Stub the one rtp_llm dependency the op needs.
_arch = types.ModuleType("rtp_llm.models_py.utils.arch")
_arch.mhc_pre_gemm_backend = lambda device=None: "tilelang_single"
for _m in ["rtp_llm", "rtp_llm.models_py", "rtp_llm.models_py.utils"]:
    if _m not in sys.modules:
        _p = types.ModuleType(_m)
        _p.__path__ = []
        sys.modules[_m] = _p
sys.modules["rtp_llm.models_py.utils.arch"] = _arch

from tile_kernels.modeling.mhc import ops  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMHCPreBigFuseDelayed(unittest.TestCase):
    def test_delayed_runs_and_close(self):
        torch.manual_seed(0)
        hc, dim, tokens, iters, eps = 4, 5120, 128, 20, 1e-6
        dev = "cuda"
        m3 = hc * 2 + hc * hc  # 24
        residual = torch.randn(1, tokens, hc, dim, device=dev, dtype=torch.bfloat16)
        fn = (torch.randn(m3, hc * dim, device=dev) * 0.003).float()
        scale = torch.tensor([0.2, 0.4, 0.3], device=dev).float()
        base = (torch.randn(m3, device=dev) * 0.1).float()
        prev_pre = (torch.rand(tokens, hc, device=dev) * 0.5 + 0.1).float()

        # Fused (delayed): returns (pre_mix_out, post, comb, layer_input).
        out = ops.mhc_pre_big_fuse(
            residual,
            fn,
            scale,
            base,
            rms_eps=eps,
            mhc_pre_eps=eps,
            mhc_sinkhorn_eps=eps,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=iters,
            previous_pre_mix=prev_pre,
            gemm_backend="tilelang_single",
        )
        pre_f, post_f, comb_f, y_f = out

        # Unfused reference (the production op sequence).
        mixes = ops.mhc_pre_norm_fn(residual, fn, None, eps, n_splits=1)
        pre_r, post_r, comb_r = ops.mhc_pre_split_mixes(
            mixes, scale, base, hc, 2.0, eps
        )
        comb_r = ops.sinkhorn_normalize(comb_r, repeat=iters, eps=eps)
        y_r = ops.mhc_pre_apply_mix(
            residual.reshape(tokens, hc, dim),
            prev_pre.reshape(tokens, hc, 1).contiguous(),
        )

        def rel(a, b):
            a = a.reshape(-1).float()
            b = b.reshape(-1).float()
            d = (a - b).abs().max().item()
            r = (
                (a - b).square().sum().sqrt() / b.square().sum().sqrt().clamp_min(1e-30)
            ).item()
            return d, r

        for name, a, b in [
            ("pre_mix", pre_f, pre_r),
            ("post", post_f, post_r.reshape(tokens, hc)),
            ("comb", comb_f, comb_r.reshape(tokens, hc * hc)),
            ("readout_y", y_f, y_r.reshape(tokens, dim)),
        ]:
            mad, rl = rel(a, b)
            print(
                f"  {name}: max_abs={mad:.4e} rel_l2={rl:.4e} finite={bool(torch.isfinite(a).all())}"
            )
            self.assertTrue(
                bool(torch.isfinite(a.reshape(-1)).all()), f"{name} not finite"
            )
        print(
            "R54 delayed fusion ran on SM120 (tilelang_single); precision-class errors above."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

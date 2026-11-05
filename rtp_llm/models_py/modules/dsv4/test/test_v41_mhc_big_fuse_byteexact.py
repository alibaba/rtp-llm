"""R52: byte-exactness evidence for the delayed mHC pre big-fuse (6a514c4543).

Compares the fused mhc_pre_big_fuse(delayed=True) against the production unfused
op sequence (norm_fn -> split_mixes -> sinkhorn -> apply_mix with the previous
layer's pre_mix). Reports torch.equal (the byte-exact gate) and max-abs diff.

Run: DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single python3 test_v41_mhc_big_fuse_byteexact.py
"""

import importlib.util
import unittest
from pathlib import Path

import torch

_OPS = (
    Path(__file__).resolve().parents[2]
    / "3rdparty"
    / "tile_kernels"
    / "modeling"
    / "mhc"
    / "ops"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ops():
    # Load the ops package directly (avoids the rtp_llm package import chain),
    # and stub the one rtp_llm.utils.arch dependency the op needs.
    import importlib
    import sys
    import types

    pkg_dir = _OPS.parent  # .../mhc
    modeling_dir = pkg_dir.parent  # .../modeling
    tk_dir = modeling_dir.parent  # .../tile_kernels

    arch = types.ModuleType("rtp_llm.models_py.utils.arch")
    arch.mhc_pre_gemm_backend = lambda device: "tilelang_single"
    for mod in ["rtp_llm", "rtp_llm.models_py", "rtp_llm.models_py.utils"]:
        if mod not in sys.modules:
            pkg = types.ModuleType(mod)
            pkg.__path__ = []
            sys.modules[mod] = pkg
    sys.modules["rtp_llm.models_py.utils.arch"] = arch

    for mod, p in [
        ("tile_kernels", tk_dir),
        ("tile_kernels.modeling", modeling_dir),
        ("tile_kernels.modeling.mhc", pkg_dir),
    ]:
        if mod not in sys.modules:
            pkg = types.ModuleType(mod)
            pkg.__path__ = [str(p)]
            sys.modules[mod] = pkg
    ops = importlib.import_module("tile_kernels.modeling.mhc.ops")
    return ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMHCBigFuseByteExact(unittest.TestCase):
    def test_fused_vs_unfused(self):
        ops = _load_ops()
        torch.manual_seed(0)
        hc, dim, tokens, iters, eps = 4, 5120, 64, 20, 1e-6
        dev = "cuda"
        residual = torch.randn(tokens, hc, dim, device=dev, dtype=torch.bfloat16)
        m3 = hc * 2 + hc * hc  # 24
        fn = (torch.randn(m3, hc * dim, device=dev) * 0.003).float()
        scale = torch.tensor([0.2, 0.4, 0.3], device=dev).float()
        base = (torch.randn(m3, device=dev) * 0.1).float()
        prev_pre = torch.rand(tokens, hc, device=dev).float() + 0.1

        # Unfused reference (the production op sequence).
        mixes = ops.mhc_pre_norm_fn(residual, fn, None, eps, n_splits=1)
        pre, post, comb = ops.mhc_pre_split_mixes(mixes, scale, base, hc, 2.0, eps)
        comb = ops.sinkhorn_normalize(comb, repeat=iters, eps=eps)
        layer_ref = ops.mhc_pre_apply_mix(
            residual, prev_pre.reshape(tokens, hc, 1).contiguous()
        )

        # Fused (delayed).
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
        )
        pre_f, post_f, comb_f, layer_f = out

        def cmp(name, a, b):
            a = a.reshape(-1).float()
            b = b.reshape(-1).float()
            eq = torch.equal(a, b)
            md = (a - b).abs().max().item()
            print(f"  {name}: torch.equal={eq} max_abs_diff={md:.3e}")
            return eq

        print("R52 fusion byte-exactness (fused delayed vs unfused):")
        r = [
            cmp("pre_mix_out", pre_f, pre.reshape(tokens, hc)),
            cmp("post", post_f, post.reshape(tokens, hc)),
            cmp("comb", comb_f, comb.reshape(tokens, hc * hc)),
            cmp("layer_input(readout)", layer_f, layer_ref.reshape(tokens, dim)),
        ]
        print("BYTE_EXACT:", all(r))


if __name__ == "__main__":
    unittest.main(verbosity=2)

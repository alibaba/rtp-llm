import unittest

import torch

from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (
    TileLangHCHead,
    TileLangHCUnit,
)


def _weights(hc: int, dim: int, device: str = "cpu"):
    torch.manual_seed(0)
    mix_hc = (2 + hc) * hc
    fn = torch.randn(mix_hc, hc * dim, device=device, dtype=torch.float32) * 0.02
    base = torch.zeros(mix_hc, device=device, dtype=torch.float32)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    return fn, base, scale


class TestHCImpl(unittest.TestCase):
    def test_factory_default_tilelang_fails_fast_on_cpu(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = build_hc_unit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        self.assertIsInstance(unit, TileLangHCUnit)
        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_tilelang_none_result_is_not_fallback(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = lambda *args, **kwargs: None
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_tilelang_wrap_is_view_and_requires_contiguous(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        seen = {}

        def fake_pre(residual, *args, **kwargs):
            seen["shape"] = tuple(residual.shape)
            seen["stride"] = tuple(residual.stride())
            seen["data_ptr"] = residual.data_ptr()
            seen["is_contiguous"] = residual.is_contiguous()
            return (
                torch.zeros(1, 5, dim, dtype=torch.bfloat16),
                torch.zeros(1, 5, hc, 1, dtype=torch.float32),
                torch.zeros(1, 5, hc, hc, dtype=torch.float32),
            )

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = fake_pre
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y, post, comb = unit.pre(x)
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertEqual(tuple(post.shape), (5, hc, 1))
        self.assertEqual(tuple(comb.shape), (5, hc, hc))
        self.assertEqual(seen["shape"], (1, 5, hc, dim))
        self.assertTrue(seen["is_contiguous"])
        self.assertEqual(seen["data_ptr"], x.data_ptr())

        x_noncontig = torch.randn(hc, 5, dim, dtype=torch.bfloat16).transpose(0, 1)
        self.assertEqual(tuple(x_noncontig.shape), (5, hc, dim))
        self.assertFalse(x_noncontig.is_contiguous())
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            unit.pre(x_noncontig)

    def test_tilelang_head_always_dispatches_fused_kernel(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        head = TileLangHCHead(
            fn[:hc],
            base[:hc],
            scale[:1],
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        calls: list[str] = []

        def fake_fused(residual, *args, **kwargs):
            calls.append("fused")
            return torch.ones(1, 5, dim, dtype=torch.bfloat16)

        old_fused = tilelang_impl.tk_mhc_head_fused
        tilelang_impl.tk_mhc_head_fused = fake_fused
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_head_fused", old_fused))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y = head.head(x)
        self.assertEqual(calls, ["fused"])
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertTrue(torch.all(y == 1))

        calls.clear()
        tilelang_impl.tk_mhc_head_fused = (
            lambda *args, **kwargs: calls.append("fused") or None
        )
        with self.assertRaisesRegex(RuntimeError, "fused head must succeed"):
            head.head(x)
        self.assertEqual(calls, ["fused"])

    def test_shape_contract_is_checked_before_impl(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = build_hc_unit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        with self.assertRaises(ValueError):
            unit.pre(torch.randn(2, hc * dim))
        residual = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        x = torch.randn(2, 5, dim, dtype=torch.bfloat16)
        bad_post = torch.randn(2, 5, hc)
        comb = torch.randn(2, 5, hc, hc)
        with self.assertRaises(ValueError):
            unit.post(x, residual, bad_post, comb)

    def test_tilelang_post_reuses_residual_buffer_in_place(self) -> None:
        # Pins the memory-saving wiring (no CUDA needed): _post_impl must pass
        # out=residual so the kernel writes in place instead of allocating a
        # fresh empty_like(residual). A future refactor that drops the alias
        # would silently re-introduce the per-call allocation; this catches it.
        hc, dim, T = 4, 16, 5
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        captured: dict = {}

        def fake_post(x, residual, post, comb, hc_mult=4, out=None):
            captured["out_is_residual"] = out is residual
            captured["out_ptr"] = None if out is None else out.data_ptr()
            return residual if out is None else out

        old_post = tilelang_impl.tk_mhc_post
        tilelang_impl.tk_mhc_post = fake_post
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_post", old_post))

        residual = torch.randn(T, hc, dim, dtype=torch.bfloat16)
        x = torch.randn(T, dim, dtype=torch.bfloat16)
        post = torch.randn(T, hc, 1, dtype=torch.float32)
        comb = torch.randn(T, hc, hc, dtype=torch.float32)
        out = unit.post(x, residual, post, comb)

        self.assertTrue(captured["out_is_residual"])
        # the aliased out points at the caller's residual storage
        self.assertEqual(captured["out_ptr"], residual.data_ptr())
        self.assertEqual(out.data_ptr(), residual.data_ptr())
        self.assertEqual(tuple(out.shape), (T, hc, dim))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_nonbf16_x_raises_dtype_error(self) -> None:
        # A non-bf16 sublayer output x is an upstream dtype bug, not a TileLang
        # availability miss: it must raise loudly (with the offending dtype),
        # not return None and get disguised as "TileLang unavailable". Reaches
        # the x.dtype check before any kernel import, so tilelang is not needed.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 8
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)
        x_fp32 = torch.randn(1, T, dim, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "bfloat16 sublayer output"):
                tk_mhc_post(x_fp32, residual, post, comb, hc_mult=hc)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_in_place_matches_fresh_buffer_cuda(self) -> None:
        # Pins the out=residual aliasing invariant: writing the post output back
        # into the residual buffer must be bit-identical to writing into a fresh
        # buffer. The vendored kernel's safety relies on reading residual[pid_n]
        # into shared memory before overwriting out[pid_n]; if a future kernel
        # change breaks that read-before-write ordering, this fails loudly.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 64
        torch.manual_seed(0)
        x = torch.randn(1, T, dim, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)

        residual_fresh = residual.clone()
        residual_alias = residual.clone()
        with torch.inference_mode():
            out_fresh = tk_mhc_post(x, residual_fresh, post, comb, hc_mult=hc)
            out_alias = tk_mhc_post(
                x, residual_alias, post, comb, hc_mult=hc, out=residual_alias
            )

        assert out_fresh is not None and out_alias is not None
        # fresh path allocated a new buffer; aliased path wrote into residual
        self.assertNotEqual(out_fresh.data_ptr(), residual_fresh.data_ptr())
        self.assertEqual(out_alias.data_ptr(), residual_alias.data_ptr())
        # in-place and fresh-buffer results must be bit-identical
        torch.testing.assert_close(out_alias, out_fresh, atol=0, rtol=0)

    def test_fused_mhc_pre_norm_gate_rejects_cpu_and_overrides(self) -> None:
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
        from rtp_llm.models_py.modules.dsv4.hc import can_use_fused_mhc_pre_norm

        hc, dim = 4, 256
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        norm = RMSNorm(torch.ones(dim, dtype=torch.bfloat16), eps=1e-6)
        x = torch.randn(4, hc, dim, dtype=torch.bfloat16)
        self.assertFalse(can_use_fused_mhc_pre_norm(x, unit, norm))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_mhc_pre_norm_gate_dispatches_small_batches_to_unfused(self) -> None:
        """Decode-scale batches must fall back to the unfused chain.

        The fused kernel is one CTA per token row; at decode M_local=3..4 it
        runs 18.3us/call vs ~10us unfused on SM103 (46 calls/step), so the
        default gate rejects small batches and only prefill-scale token
        counts keep the fused path.
        """
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
        from rtp_llm.models_py.modules.dsv4.hc import can_use_fused_mhc_pre_norm

        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            self.skipTest("requires SM100/SM103 for tf32_hc_prenorm_gemm")

        dim = 4096
        fn, base, scale = _weights(4, dim, device="cuda")
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=4,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        norm = RMSNorm(torch.ones(dim, device="cuda", dtype=torch.bfloat16), eps=1e-6)
        with torch.inference_mode():
            x_small = torch.randn(4, 4, dim, device="cuda", dtype=torch.bfloat16)
            self.assertFalse(can_use_fused_mhc_pre_norm(x_small, unit, norm))
            self.assertFalse(
                can_use_fused_mhc_pre_norm(x_small, unit, norm, quant_ue8m0=True)
            )
            # min_tokens=0 restores the raw capability gate for tiny inputs.
            if can_use_fused_mhc_pre_norm(x_small, unit, norm, min_tokens=0):
                x_large = torch.randn(1024, 4, dim, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(can_use_fused_mhc_pre_norm(x_large, unit, norm))
                # A 4D [B, S, hc, dim] tensor counts B*S rows, not B.
                x_4d = torch.randn(2, 600, 4, dim, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(can_use_fused_mhc_pre_norm(x_4d, unit, norm))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_mhc_pre_norm_matches_separate_tilelang_and_rmsnorm(self) -> None:
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
            rmsnorm_fp8_quant_ue8m0,
        )
        from rtp_llm.models_py.modules.dsv4.hc import (
            can_use_fused_mhc_pre_norm,
            fused_mhc_pre_norm,
            fused_mhc_pre_norm_quant,
        )

        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            self.skipTest("requires SM100/SM103 for tf32_hc_prenorm_gemm")

        for shape in ((17, 4, 4096), (8, 1, 4, 4096), (5, 4, 3072)):
            with self.subTest(shape=shape):
                dim = shape[-1]
                fn, base, scale = _weights(4, dim, device="cuda")
                unit = TileLangHCUnit(
                    fn,
                    base,
                    scale,
                    dim=dim,
                    hc_mult=4,
                    hc_sinkhorn_iters=10,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
                w = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.5 + 1.0
                norm = RMSNorm(w, eps=1e-6)
                x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    if not can_use_fused_mhc_pre_norm(x, unit, norm, min_tokens=0):
                        self.skipTest("tf32_hc_prenorm_gemm unavailable in DeepGEMM")
                    x_pre_ref, post_ref, comb_ref = unit.pre(x)
                    if x_pre_ref.dim() > 2:
                        b, s, d = x_pre_ref.shape
                        x_norm_ref = norm(x_pre_ref.reshape(b * s, d)).view(b, s, d)
                    else:
                        x_norm_ref = norm(x_pre_ref)
                    x_norm_got, post_got, comb_got = fused_mhc_pre_norm(x, unit, norm)

                torch.testing.assert_close(post_got, post_ref, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(comb_got, comb_ref, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(x_norm_got, x_norm_ref, atol=2e-2, rtol=2e-2)

                if len(shape) == 3 and dim % 128 == 0:
                    with torch.inference_mode():
                        nq_ref, q_ref, s_ref = rmsnorm_fp8_quant_ue8m0(
                            x_pre_ref.clone(), w, eps=1e-6
                        )
                        nq_got, q_post, q_comb, (q_got, s_got) = (
                            fused_mhc_pre_norm_quant(x, unit, norm)
                        )
                    torch.testing.assert_close(q_post, post_ref, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(q_comb, comb_ref, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(nq_got, nq_ref, atol=2e-2, rtol=2e-2)
                    torch.testing.assert_close(
                        q_got.float(), q_ref.float(), atol=1.0, rtol=5e-2
                    )
                    self.assertEqual(tuple(s_got.shape), tuple(s_ref.shape))
                    self.assertEqual(tuple(s_got.stride()), tuple(s_ref.stride()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mhc_pre_and_post_decode_prefill_and_dim7168(self) -> None:
        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            self.skipTest("requires SM100/SM103 for tf32_hc_prenorm_gemm")

        def _ref_mhc_pre(
            x: torch.Tensor,
            fn: torch.Tensor,
            base: torch.Tensor,
            scale: torch.Tensor,
            *,
            hc: int,
            dim: int,
            sinkhorn_iters: int = 10,
            eps: float = 1e-6,
        ):
            x_f = x.float()
            flat = x_f.reshape(x_f.shape[0], hc * dim)
            rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + eps)
            mixes = (flat @ fn.t()) * rms
            pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
            post = (
                torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
                * 2.0
            )
            cm = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(
                x_f.shape[0], hc, hc
            )
            cm = torch.softmax(cm, dim=-1) + eps
            cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
            for _ in range(sinkhorn_iters - 1):
                cm = cm / (cm.sum(dim=-1, keepdim=True) + eps)
                cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
            y = (pre.unsqueeze(-1) * x_f).sum(dim=1).to(torch.bfloat16)
            return y, post.unsqueeze(-1), cm

        def _ref_mhc_post(
            layer_out: torch.Tensor,
            residual: torch.Tensor,
            post: torch.Tensor,
            comb: torch.Tensor,
        ):
            res = post * layer_out.float().unsqueeze(1) + torch.einsum(
                "tio,tid->tod", comb, residual.float()
            )
            return res.to(torch.bfloat16)

        # Cover both decode-split (T=18 <= 256) and prefill (T=512 > 256)
        # dispatch regimes plus large hidden dimension (dim=7168) for register
        # tiling safety in pre_big_fuse_kernel / post_kernel.
        for tokens, dim in ((18, 4096), (512, 4096), (18, 7168)):
            with self.subTest(tokens=tokens, dim=dim):
                hc = 4
                fn, base, scale = _weights(hc, dim, device="cuda")
                unit = TileLangHCUnit(
                    fn,
                    base,
                    scale,
                    dim=dim,
                    hc_mult=hc,
                    hc_sinkhorn_iters=10,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
                x = torch.randn(tokens, hc, dim, device="cuda", dtype=torch.bfloat16)
                layer_out = torch.randn(
                    tokens, dim, device="cuda", dtype=torch.bfloat16
                )
                with torch.inference_mode():
                    pre_got, post_got, comb_got = unit.pre(x)
                    pre_ref, post_ref, comb_ref = _ref_mhc_pre(
                        x, fn, base, scale, hc=hc, dim=dim
                    )
                    # Pre-norm linear uses DeepGEMM TF32 (~5e-4 vs FP32 matmul).
                    torch.testing.assert_close(post_got, post_ref, atol=5e-3, rtol=5e-3)
                    torch.testing.assert_close(comb_got, comb_ref, atol=5e-3, rtol=5e-3)
                    torch.testing.assert_close(pre_got, pre_ref, atol=2e-2, rtol=2e-2)

                    out_got = unit.post(layer_out, x.clone(), post_got, comb_got)
                    out_ref = _ref_mhc_post(layer_out, x.clone(), post_ref, comb_ref)
                    torch.testing.assert_close(out_got, out_ref, atol=2e-2, rtol=2e-2)

        # Decode h_split=4/8 (T <= 256) must match the 1-CTA (h_split=1) kernel
        # outputs bit-for-bit, including dim=7168 (hidden_block=256 tiling).
        import importlib

        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        _mhc_post_fwd = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel"
        )._mhc_post_fwd
        _mhc_pre_big_fuse = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.mhc.pre_big_fuse_kernel"
        )._mhc_pre_big_fuse
        pre_ops = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )
        _compute_num_split = pre_ops._compute_num_split
        _run_deepgemm_splitk_gemm = pre_ops._run_deepgemm_splitk_gemm

        for tokens, dim in ((18, 4096), (18, 7168)):
            with self.subTest(bit_exact_tokens=tokens, dim=dim):
                hc = 4
                fn, base, scale = _weights(hc, dim, device="cuda")
                unit = TileLangHCUnit(
                    fn,
                    base,
                    scale,
                    dim=dim,
                    hc_mult=hc,
                    hc_sinkhorn_iters=10,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
                x = torch.randn(tokens, hc, dim, device="cuda", dtype=torch.bfloat16)
                layer_out = torch.randn(
                    tokens, dim, device="cuda", dtype=torch.bfloat16
                )
                with torch.inference_mode():
                    pre_got, post_got, comb_got = unit.pre(x)
                    out_got = unit.post(layer_out, x.clone(), post_got, comb_got)

                    n_splits = _compute_num_split(64, hc * dim, (tokens + 63) // 64)
                    g_mul = torch.empty(
                        n_splits, tokens, 24, dtype=torch.float32, device="cuda"
                    )
                    g_sq = torch.empty(
                        n_splits, tokens, dtype=torch.float32, device="cuda"
                    )
                    _run_deepgemm_splitk_gemm(
                        x.view(tokens, hc * dim), fn, g_mul, g_sq, n_splits
                    )
                    post_1 = torch.empty(tokens, hc, dtype=torch.float32, device="cuda")
                    comb_1 = torch.empty(
                        tokens, hc * hc, dtype=torch.float32, device="cuda"
                    )
                    pre_1 = torch.empty(
                        tokens, dim, dtype=torch.bfloat16, device="cuda"
                    )
                    _mhc_pre_big_fuse(
                        dim,
                        1e-6,
                        1e-6,
                        1e-6,
                        2.0,
                        10,
                        n_splits=n_splits,
                        mhc_mult=hc,
                        h_split=1,
                    )(g_mul, g_sq, scale, base, x, post_1, comb_1, pre_1)
                    out_1 = x.clone()
                    _mhc_post_fwd(hc, dim, h_blk=1024, h_split=1)(
                        comb_got.view(tokens, hc, hc),
                        out_1,
                        post_got.view(tokens, hc),
                        layer_out,
                        out_1,
                    )

                    self.assertTrue(torch.equal(post_got, post_1.view(tokens, hc, 1)))
                    self.assertTrue(torch.equal(comb_got, comb_1.view(tokens, hc, hc)))
                    self.assertTrue(torch.equal(pre_got, pre_1))
                    self.assertTrue(torch.equal(out_got, out_1))


if __name__ == "__main__":
    unittest.main()

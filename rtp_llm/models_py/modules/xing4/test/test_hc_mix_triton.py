"""Equivalence UT for the Xing4 HC mix/sinkhorn Triton kernel (K2).

Reference (numerical authority, handwritten torch mirror of
``rtp_llm/models_py/model_desc/xing4_0_model.py`` ``Xing4HCUnit._pre_impl``
L130-158, strictly downstream of the ``F.linear`` GEMM)::

    pre_w, post_w, comb_w = mixes.split([4, 4, 16], dim=-1)
    pre_b, post_b, comb_b = base.split([4, 4, 16])
    pre_s, post_s, comb_s = scale.unbind(0)
    pre  = torch.sigmoid(pre_w * pre_s + pre_b)              # no +eps
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.view(T, 4, 4) * comb_s + comb_b.view(4, 4)
    comb_logits = comb_logits.clamp(min=-30.0, max=30.0)     # mhc_h_res_clamp
    comb = torch.exp(comb_logits - comb_logits.amax(-1, keepdim=True))
    for _ in range(iters):                                   # 20 in config
        comb = comb / (comb.sum(-1, keepdim=True) + eps)     # row (dim=-1) first
        comb = comb / (comb.sum(-2, keepdim=True) + eps)     # then col (dim=-2)
    return pre.to(dt), post.to(dt), comb.to(dt)

Tolerance policy: the fp32 output mode isolates logic errors from bf16
rounding (atol 1e-6, rtol 0; measured max diff <= 3.3e-7); the bf16 mode
gates the production cast point with rtol=atol=2**-7 (2 bf16 ulp).  Every
case reports the measured max diff on both output dtypes.

Adversarial domains: T in {1, 2, 8, 512, 4096}; clamp saturation at +-30
(with the resulting exact amax ties at the clamp ceiling); exact amax ties
via row-constant logits; eps-only denominators -- a row of ``exp`` can never
be all-zero (the row max maps to exp(0)=1), so the degenerate
"denominator = eps only" case is realized as all-zero *columns* after exp
underflow to exact 0 with the max concentrated in column 0 (wide clamp
bounds so nothing clips); iters in {1, 20}; eps in {1e-6, 1e-2}; saturating
sigmoid gates (+-40 inputs).

Launch counting: CUPTI is unavailable in the CUDA13 containers, so both
paths are captured into a CUDA graph instead -- every launch becomes
exactly one graph node, counted via ``cudaGraphGetNodes`` (driver API).
Measured: reference torch chain 136 nodes vs fused Triton 1 node.
"""

import ctypes
import unittest
from collections import Counter

import torch

from rtp_llm.models_py.modules.xing4.hc.mix_triton import hc_mix_triton

HC = 4
_TS = (1, 2, 8, 512, 4096)
_BF16_RTOL = 2**-7
_BF16_ATOL = 2**-7
_FP32_ATOL = 1e-6


def _reference_mix(mixes, base, scale, sinkhorn_iters, eps, clamp_min, clamp_max, out_dtype):
    """Torch step-by-step mirror of Xing4HCUnit._pre_impl (xing4_0_model.py:130-158)."""
    T = mixes.shape[0]
    pre_w, post_w, comb_w = mixes.split([HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = base.split([HC, HC, HC * HC])
    pre_s, post_s, comb_s = scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b)
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)

    comb_logits = comb_w.view(T, HC, HC) * comb_s + comb_b.view(HC, HC)
    comb_logits = comb_logits.clamp(min=clamp_min, max=clamp_max)
    comb = torch.exp(comb_logits - comb_logits.amax(dim=-1, keepdim=True))
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre.to(out_dtype), post.to(out_dtype), comb.to(out_dtype)


def _gen_case(case, T):
    """Deterministic fixture -> (mixes, base, scale, kwargs) on CPU."""
    g = torch.Generator().manual_seed(20260928)
    kw = dict(sinkhorn_iters=20, eps=1e-6, clamp_min=-30.0, clamp_max=30.0)
    base = torch.randn(24, generator=g) * 0.1
    scale = torch.ones(3)

    if case == "default":
        return torch.randn(T, 24, generator=g) * 0.5, base, scale, kw

    if case == "clamp_active":
        # comb_scale=8 with |comb_w| ~ 4.5 -> logits ~ +-36 -> clamped to +-30;
        # first half of tokens saturates, second half stays unclamped.
        scale = torch.tensor([0.7, 1.3, 8.0])
        prepost = torch.randn(T, 8, generator=g) * 0.5
        sign = torch.where(torch.rand(T, 16, generator=g) < 0.5, -1.0, 1.0)
        comb = torch.randn(T, 16, generator=g) * 0.3 + sign * 4.5
        half = max(T // 2, 1)
        comb[half:] = torch.randn(T - half, 16, generator=g) * 0.5
        return torch.cat([prepost, comb], dim=-1), base, scale, kw

    if case == "amax_ties":
        # comb_s=1 + row-constant comb_b and comb_w -> every row's four logits
        # are bit-identical -> exact 4-way amax ties, exp(0)=1 across the row.
        row_bias = torch.randn(4, generator=g) * 0.5
        comb_b = row_bias[:, None].expand(4, 4).reshape(16)
        base = torch.cat([base[:8], comb_b])
        scale = torch.tensor([0.7, 1.3, 1.0])
        w = torch.randn(T, 4, generator=g) * 0.8
        comb_w = w[:, :, None].expand(T, 4, 4).reshape(T, 16)
        prepost = torch.randn(T, 8, generator=g) * 0.5
        return torch.cat([prepost, comb_w], dim=-1), base, scale, kw

    if case == "zero_cols":
        # Wide clamp so nothing clips; per-row max concentrated in column 0,
        # other columns' logits <= -110 -> exp underflows to EXACTLY 0 ->
        # after row normalization the zero columns' sums are 0, so their
        # normalization denominator is eps only.
        scale = torch.tensor([0.7, 1.3, 1.0])
        kw["clamp_min"], kw["clamp_max"] = -200.0, 200.0
        prepost = torch.randn(T, 8, generator=g) * 0.5
        comb = torch.randn(T, 16, generator=g) * 0.05
        comb[:, 0::4] = 2.0
        comb[:, 1::4] = -115.0
        comb[:, 2::4] = -120.0
        comb[:, 3::4] = -125.0
        return torch.cat([prepost, comb], dim=-1), base, scale, kw

    if case == "eps_1e2":
        kw["eps"] = 1e-2
        return torch.randn(T, 24, generator=g) * 0.5, base, scale, kw

    if case == "iters_1":
        kw["sinkhorn_iters"] = 1
        return torch.randn(T, 24, generator=g) * 0.5, base, scale, kw

    if case == "saturated_gates":
        # +-40 pre/post inputs: sigmoid saturates to exactly 1.0 / 4e-18.
        prepost = torch.randn(T, 8, generator=g) * 0.5
        prepost[0:T:2] = 40.0
        prepost[1:T:2] = -40.0
        comb = torch.randn(T, 16, generator=g) * 0.5
        return torch.cat([prepost, comb], dim=-1), base, scale, kw

    raise ValueError(f"unknown case {case}")


class TestXing4HCMixTriton(unittest.TestCase):
    """K2 lane: mixes[T, 24] fp32 -> (pre, post, comb) in one Triton launch."""

    def _check_case(self, case):
        for T in _TS:
            with self.subTest(case=case, T=T):
                mixes, base, scale, kw = _gen_case(case, T)
                mixes, base, scale = (
                    mixes.cuda(),
                    base.cuda(),
                    scale.cuda(),
                )
                for out_dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(dtype=out_dtype):
                        ref = _reference_mix(
                            mixes,
                            base,
                            scale,
                            kw["sinkhorn_iters"],
                            kw["eps"],
                            kw["clamp_min"],
                            kw["clamp_max"],
                            out_dtype,
                        )
                        got = hc_mix_triton(
                            mixes,
                            base,
                            scale,
                            out_dtype=out_dtype,
                            **kw,
                        )
                        for name, want, out in zip(("pre", "post", "comb"), ref, got):
                            self.assertEqual(out.dtype, out_dtype)
                            self.assertEqual(tuple(out.shape), tuple(want.shape))
                            max_diff = (
                                (out.float() - want.float()).abs().max().item()
                            )
                            if out_dtype == torch.float32:
                                # Logic-error isolation: fp32 path must match
                                # the torch chain closely (measured <= 3.3e-7).
                                torch.testing.assert_close(
                                    out, want, rtol=0.0, atol=_FP32_ATOL
                                )
                            else:
                                # Production cast point: 2 bf16 ulp.
                                torch.testing.assert_close(
                                    out.float(),
                                    want.float(),
                                    rtol=_BF16_RTOL,
                                    atol=_BF16_ATOL,
                                )
                            self.assertFalse(
                                torch.isnan(out.float()).any().item(),
                                f"NaN in {name}",
                            )
                            print(
                                f"[{case}] T={T} dtype={out_dtype} {name}: "
                                f"max|diff|={max_diff:.3e}"
                            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_default(self):
        self._check_case("default")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_clamp_active(self):
        mixes, base, scale, kw = _gen_case("clamp_active", 64)
        raw = (
            mixes[:, 8:].view(64, HC, HC).cuda() * scale[2].cuda()
            + base[8:].view(HC, HC).cuda()
        )
        n_beyond = int(((raw > 30.0) | (raw < -30.0)).sum())
        self.assertGreater(
            n_beyond,
            0,
            f"clamp_active fixture failed to engage the +-30 clamp ({n_beyond} logits beyond)",
        )
        print(f"[clamp_active] {n_beyond}/1024 logits beyond +-30 (clamp engaged)")
        self._check_case("clamp_active")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_amax_ties(self):
        mixes, _, scale, _ = _gen_case("amax_ties", 16)
        logits = mixes[:, 8:].view(16, HC, HC)  # comb_s == 1.0
        self.assertTrue(
            torch.all(logits[:, :, 0] == logits[:, :, 3]),
            "amax_ties fixture must produce bit-identical per-row logits",
        )
        self._check_case("amax_ties")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_zero_cols(self):
        mixes, _, _, kw = _gen_case("zero_cols", 16)
        comb_w = mixes[:, 8:].view(16, HC, HC)
        self.assertTrue(
            (comb_w[:, :, 0] > 0).all() and (comb_w[:, :, 1:] < -110).all(),
            "zero_cols fixture must put the row max in column 0 and "
            "underflow the other columns' exp to exactly 0",
        )
        self._check_case("zero_cols")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_eps_1e2(self):
        self._check_case("eps_1e2")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_iters_1(self):
        self._check_case("iters_1")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_equivalence_saturated_gates(self):
        self._check_case("saturated_gates")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_launch_count_vs_reference(self):
        """One Triton launch replaces the whole ~130-launch torch chain.

        CUPTI activity tracing is unavailable in the CUDA13 dev containers,
        so launches are counted by CUDA-graph capture: every launch becomes
        exactly one graph node, enumerated with cuGraphGetNodes.
        """

        def graph_node_types(fn):
            try:
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    fn()
                torch.cuda.current_stream().wait_stream(side)
                with torch.cuda.graph(graph):
                    fn()
                handle = graph.raw_cuda_graph()
                cuda = ctypes.CDLL("libcuda.so.1")
                count = ctypes.c_size_t(0)
                res = cuda.cuGraphGetNodes(
                    ctypes.c_void_p(handle), None, ctypes.byref(count)
                )
                if res != 0 or count.value == 0:
                    return None
                nodes = (ctypes.c_void_p * count.value)()
                res = cuda.cuGraphGetNodes(
                    ctypes.c_void_p(handle), nodes, ctypes.byref(count)
                )
                if res != 0:
                    return None
                types = Counter()
                for i in range(count.value):
                    node_type = ctypes.c_int(0)
                    if cuda.cuGraphNodeGetType(
                        ctypes.c_void_p(nodes[i]), ctypes.byref(node_type)
                    ) != 0:
                        return None
                    types[node_type.value] += 1
                return types
            except Exception as exc:  # pragma: no cover - infra variance
                print(f"[launch-count] graph capture unavailable: {exc!r}")
                return None

        T = 512
        mixes, base, scale, kw = _gen_case("default", T)
        mixes, base, scale = (
            mixes.cuda(),
            base.cuda(),
            scale.cuda(),
        )

        def ref_call():
            _reference_mix(
                mixes,
                base,
                scale,
                kw["sinkhorn_iters"],
                kw["eps"],
                kw["clamp_min"],
                kw["clamp_max"],
                torch.bfloat16,
            )

        def fused_call():
            hc_mix_triton(mixes, base, scale, **kw)

        ref_types = graph_node_types(ref_call)
        fused_types = graph_node_types(fused_call)
        if ref_types is None or fused_types is None:
            self.skipTest("CUDA graph launch counting unavailable in this environment")
        ref_nodes = sum(ref_types.values())
        fused_nodes = sum(fused_types.values())
        # Node type 0 = kernel (CUgraphNodeTypeKernel).
        print(
            f"[launch-count] T={T} iters=20: reference torch chain = {ref_nodes} "
            f"nodes {dict(ref_types)}, fused Triton = {fused_nodes} nodes "
            f"{dict(fused_types)}"
        )
        self.assertGreaterEqual(
            ref_nodes, 50, "reference chain should be ~130 launches, got {ref_nodes}"
        )
        self.assertLessEqual(fused_nodes, 2, "fused path must be a single launch")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_zero_tokens(self):
        pre, post, comb = hc_mix_triton(
            torch.zeros(0, 24, device="cuda"),
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
        )
        self.assertEqual(tuple(pre.shape), (0, 4))
        self.assertEqual(tuple(post.shape), (0, 4))
        self.assertEqual(tuple(comb.shape), (0, 4, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_noncontiguous_mixes(self):
        T = 37
        g = torch.Generator().manual_seed(11)
        mixes = torch.randn(T * 2, 24, generator=g).cuda()[::2]
        base = (torch.randn(24, generator=g) * 0.1).cuda()
        scale = torch.tensor([0.7, 1.3, 1.0]).cuda()
        self.assertFalse(mixes.is_contiguous())
        ref = _reference_mix(mixes.contiguous(), base, scale, 20, 1e-6, -30.0, 30.0, torch.float32)
        got = hc_mix_triton(mixes, base, scale, out_dtype=torch.float32)
        for want, out in zip(ref, got):
            torch.testing.assert_close(out, want, rtol=0.0, atol=_FP32_ATOL)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_bf16_mixes_bitwise_matches_widened_fp32(self):
        """The bf16 hc_fn GEMM output path must match fp32(widened) bitwise.

        The kernel widens bf16 loads losslessly; the removed eager widen
        launch (aten::_to_copy) changes no values.
        """
        g = torch.Generator(device="cuda").manual_seed(7)
        T = 257
        mixes = (torch.randn(T, 24, generator=g, device="cuda") * 6.0)
        base = torch.randn(24, generator=g, device="cuda") * 0.5
        scale = torch.randn(3, generator=g, device="cuda").abs() + 0.5
        kw = dict(sinkhorn_iters=20, eps=1e-6, out_dtype=torch.bfloat16)
        mixes_bf16 = mixes.to(torch.bfloat16)
        # Same input VALUES through both paths: the eager widen (bf16->fp32
        # aten copy) vs the in-kernel widen. (Comparing against the original
        # fp32 tensor would differ by the bf16 rounding of the input itself.)
        b = hc_mix_triton(mixes_bf16.float(), base, scale, **kw)
        c = hc_mix_triton(mixes_bf16, base, scale, **kw)
        for x, y in zip(b, c):
            self.assertTrue(torch.equal(x, y))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_contract_validation(self):
        mixes = torch.randn(8, 24, device="cuda")
        base = torch.zeros(24, device="cuda")
        scale = torch.ones(3, device="cuda")
        with self.assertRaises(AssertionError):
            hc_mix_triton(torch.randn(8, 16, device="cuda"), base, scale)
        with self.assertRaises(AssertionError):
            hc_mix_triton(mixes, torch.zeros(16, device="cuda"), scale)
        with self.assertRaises(AssertionError):
            hc_mix_triton(mixes, base, torch.ones(2, device="cuda"))
        # bf16 mixes (the hc_fn GEMM output dtype) is accepted and widened
        # losslessly in-kernel; only non-fp32 base/scale still reject.
        hc_mix_triton(mixes.to(torch.bfloat16), base, scale)
        with self.assertRaises(AssertionError):
            hc_mix_triton(mixes, base.to(torch.bfloat16), scale)
        with self.assertRaises(AssertionError):
            hc_mix_triton(mixes, base, scale, sinkhorn_iters=-1)
        with self.assertRaises(ValueError):
            hc_mix_triton(mixes, base, scale, out_dtype=torch.float64)
        with self.assertRaises(AssertionError):
            hc_mix_triton(mixes.cpu(), base.cpu(), scale.cpu())


if __name__ == "__main__":
    unittest.main()

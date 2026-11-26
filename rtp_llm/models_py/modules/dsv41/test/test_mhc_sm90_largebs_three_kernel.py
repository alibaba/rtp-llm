"""SM90 ``64 < M <= 512`` three-kernel delayed-mHC boundary.

The reference is the production unfused chain under ``inference_mode`` (post +
Phase-1 fused pre + RMSNorm).  The DeepGEMM projection is bit-identical to the
Phase-1 chain, so the fused seam must reproduce the residual, the delayed
readout and every coefficient byte-for-byte.  The TileLang split-K fallback is
checked against the same chain and an independent FP64 oracle.
"""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.hc import mhc_sm90_largebs_three_kernel as lb
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.moe.shared_expert import PendingCombine

DIM = 5120
HC = 4
MIX = 24
EPS = 1e-6
NORM_EPS = 1e-20
SINKHORN = 20


def _unit(fn, base, scale):
    return DelayedHCUnit(
        fn,
        base,
        scale,
        dim=DIM,
        hc_mult=HC,
        hc_sinkhorn_iters=SINKHORN,
        norm_eps=NORM_EPS,
        hc_eps=EPS,
    )


def make_case(tokens, seed, device="cuda"):
    torch.manual_seed(seed)
    previous = _unit(
        torch.zeros((MIX, HC * DIM), device=device),
        torch.zeros(MIX, device=device),
        torch.ones(3, device=device),
    )
    following = _unit(
        torch.randn((MIX, HC * DIM), device=device) * 0.003,
        torch.randn(MIX, device=device) * 0.2,
        torch.tensor([0.2, 0.4, 0.3], device=device),
    )
    following.set_previous(previous)
    previous.pre_mix_out = torch.rand((1, tokens, HC), device=device)
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

    return dict(
        attn_out=torch.randn((1, tokens, DIM), device=device).bfloat16(),
        residual=torch.randn((1, tokens, HC, DIM), device=device).bfloat16(),
        post=torch.rand((1, tokens, HC, 1), device=device) * 0.5,
        comb=(
            torch.eye(HC, device=device)
            .expand(1, tokens, HC, HC)
            .contiguous()
            .add_(torch.randn((1, tokens, HC, HC), device=device) * 0.01)
        ),
        previous=previous,
        next_hc=following,
        norm=RMSNorm(torch.ones(DIM, device=device, dtype=torch.bfloat16), EPS),
    )


def golden(case, x=None):
    """Production Phase-1 chain under inference mode."""
    x = case["attn_out"] if x is None else x
    residual = case["previous"].post(
        x, case["residual"].clone(), case["post"], case["comb"]
    )
    layer_input, post, comb = case["next_hc"].pre(residual)
    normalized = case["norm"](layer_input.reshape(-1, DIM)).view_as(layer_input)
    return residual, normalized, post, comb, case["next_hc"].pre_mix_out


def oracle(case, x=None):
    """Independent FP64 reference for the delayed boundary."""
    tokens = case["attn_out"].shape[1]
    x = case["attn_out"] if x is None else x
    fn = case["next_hc"].fn.double()
    residual = case["residual"].double()
    post = case["post"].double()
    comb = case["comb"].double()
    scale = case["next_hc"].scale.double()
    base = case["next_hc"].base.double()
    previous_pre = case["previous"].pre_mix_out.double()
    residual_out = post * x.double().unsqueeze(-2) + torch.einsum(
        "...sd,...sh->...dh", comb, residual
    )
    flat = residual_out.reshape(1, tokens, HC * DIM)
    mixes = (
        flat @ fn.T * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    )
    pre = torch.sigmoid(mixes[..., :HC] * scale[0] + base[:HC]) + EPS
    post_mix = 2.0 * torch.sigmoid(
        mixes[..., HC : 2 * HC] * scale[1] + base[HC : 2 * HC]
    )
    cm = (mixes[..., 2 * HC :] * scale[2] + base[2 * HC :]).view(1, tokens, HC, HC)
    cm = cm.softmax(dim=-1) + EPS
    cm = cm / (cm.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(SINKHORN - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + EPS)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + EPS)
    y = (residual_out * previous_pre.unsqueeze(-1)).sum(dim=-2).to(torch.bfloat16)
    normalized = (
        y.double()
        * torch.rsqrt(y.double().square().mean(dim=-1, keepdim=True) + EPS)
        * case["norm"].weight.double()
    ).to(torch.bfloat16)
    return (
        residual_out.to(torch.bfloat16),
        normalized,
        pre,
        post_mix,
        cm,
    )


def _rel_l2(actual, expected):
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    return ((actual - expected).norm() / expected.norm()).item()


def _assert_bitwise(test, actual, expected, name):
    test.assertTrue(
        torch.equal(actual.reshape(-1), expected.reshape(-1)),
        f"{name} is not bitwise identical",
    )


def _assert_normalized_close(test, actual, expected):
    """The fused RMSNorm uses a different reduction order than flashinfer.

    Phase-1's fixtures measured ``rel_l2 <= 1.73e-5``.  The collapse also uses
    a 1024-wide block instead of 512, so a few BF16 roundings can flip on other
    seeds; the observed worst across these fixtures is ~7e-5, still far below
    one BF16 ULP (~7.8e-3).
    """
    rel = _rel_l2(actual, expected)
    test.assertLessEqual(rel, 2e-4, f"normalized rel_l2={rel:.3e}")
    test.assertLessEqual((actual.float() - expected.float()).abs().max().item(), 0.05)


def _assert_coeffs_close(test, actual, expected, atol=1e-6):
    for name, got, ref in zip(("post", "comb", "pre"), actual[2:], expected[2:]):
        test.assertTrue(torch.isfinite(got).all(), name)
        test.assertLessEqual((got.float() - ref.float()).abs().max().item(), atol, name)


class LargebsHostTest(unittest.TestCase):
    """Pure dispatch contracts; no GPU is touched."""

    def setUp(self):
        lb._WARMED_STREAMS.clear()
        self.addCleanup(lb._WARMED_STREAMS.clear)

    def test_window_and_backend_selection(self):
        self.assertEqual(lb.MIN_TOKENS, 64)
        self.assertEqual(lb.MAX_TOKENS, 512)
        with mock.patch.object(lb, "_has_deepgemm", return_value=True):
            with mock.patch.dict(os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": ""}):
                self.assertEqual(lb._select_gemm_backend(), "deepgemm")
            with mock.patch.dict(
                os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_splitk"}
            ):
                self.assertEqual(lb._select_gemm_backend(), "tilelang_splitk")
        with mock.patch.object(lb, "_has_deepgemm", return_value=False):
            with mock.patch.dict(os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": ""}):
                self.assertEqual(lb._select_gemm_backend(), "tilelang_splitk")

    def test_projection_splits_match_runtime_planners(self):
        with mock.patch.object(
            lb._pre_big_fuse, "_compute_num_split", return_value=26
        ) as compute:
            self.assertEqual(lb.projection_splits(264, "deepgemm"), 26)
            compute.assert_called_once_with(64, HC * DIM, 5)
        with mock.patch.object(
            lb._norm_fn, "_tilelang_split_plan", return_value=10
        ) as plan, mock.patch.object(
            lb._norm_fn, "_tilelang_device_num_sms", return_value=132
        ), mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ):
            self.assertEqual(lb.projection_splits(264, "tilelang_splitk"), 10)
            plan.assert_called_once_with(264, HC * DIM, 132)

    def test_stream_key_distinguishes_backend_splits_and_combine(self):
        fake_stream = mock.Mock(device=torch.device("cuda", 0), cuda_stream=0x1234)
        with mock.patch.object(
            torch.cuda, "current_stream", return_value=fake_stream
        ), mock.patch.object(
            lb, "projection_splits", side_effect=lambda t, b, d=None: 52 if t < 128 else 26
        ):
            dev = torch.device("cuda", 0)
            k_96_plain = lb._stream_key(dev, 96, "deepgemm", False)
            k_96_comb = lb._stream_key(dev, 96, "deepgemm", True)
            k_264_plain = lb._stream_key(dev, 264, "deepgemm", False)
            k_96_tl = lb._stream_key(dev, 96, "tilelang_splitk", False)
            self.assertEqual(len({k_96_plain, k_96_comb, k_264_plain, k_96_tl}), 4)

    def test_combine_dispatch_uses_largebs_above_64(self):
        from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc as mega

        sentinel = object()
        backend = mock.Mock()
        backend.try_sm90_largebs_post_pre.return_value = sentinel
        args = dict(
            attn_out=torch.zeros((1, 65, DIM), dtype=torch.bfloat16),
            residual=torch.zeros((1, 65, HC, DIM), dtype=torch.bfloat16),
            post=torch.zeros((1, 65, HC, 1), dtype=torch.float32),
            comb=torch.zeros((1, 65, HC, HC), dtype=torch.float32),
            previous=object(),
            next_hc=object(),
            norm=object(),
        )
        routed = torch.zeros((1, 65, DIM), dtype=torch.bfloat16)
        shared = torch.zeros((1, 65, DIM), dtype=torch.bfloat16)
        with mock.patch.object(
            mega, "_can_try_sm90_mega_mhc", return_value=False
        ), mock.patch.object(
            mega, "_can_try_sm90_largebs_mega_mhc", return_value=True
        ), mock.patch.object(
            mega, "_get_sm90_largebs_mega_mhc", return_value=backend
        ):
            self.assertIs(
                mega.try_fused_post_pre(**args, combine=(routed, shared)), sentinel
            )
        backend.try_sm90_largebs_post_pre.assert_called_once()

    def test_extended_modes_skip_largebs(self):
        from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc as mega

        args = dict(
            attn_out=torch.zeros((1, 65, DIM), dtype=torch.bfloat16),
            residual=torch.zeros((1, 65, HC, DIM), dtype=torch.bfloat16),
            post=torch.zeros((1, 65, HC, 1), dtype=torch.float32),
            comb=torch.zeros((1, 65, HC, HC), dtype=torch.float32),
            previous=object(),
            next_hc=object(),
            norm=object(),
        )
        for kwargs in (
            {"original_tokens": 65},
            {"reuse_residual": True},
            {"output_fp8": True},
        ):
            with self.subTest(kwargs=kwargs), mock.patch.object(
                mega, "_get_sm90_largebs_mega_mhc"
            ) as backend:
                self.assertIsNone(mega.try_fused_post_pre(**args, **kwargs))
            backend.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LargebsCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or tuple(
            torch.cuda.get_device_capability()
        ) != (9, 0):
            raise unittest.SkipTest("exact SM90 required")

    def setUp(self):
        lb._WARMED_STREAMS.clear()
        self.addCleanup(lb._WARMED_STREAMS.clear)

    @torch.inference_mode()
    def test_sm90_dispatch_window(self):
        from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc as mega

        with mock.patch.object(mega, "_get_sm90_largebs_mega_mhc", return_value=lb):
            for tokens in (1, 64, 513):
                with self.subTest(tokens=tokens):
                    tensor = torch.zeros(
                        (1, tokens, DIM), dtype=torch.bfloat16, device="cuda"
                    )
                    self.assertFalse(mega._can_try_sm90_largebs_mega_mhc(tensor))
            for tokens in (65, 128, 264, 512):
                with self.subTest(tokens=tokens):
                    tensor = torch.zeros(
                        (1, tokens, DIM), dtype=torch.bfloat16, device="cuda"
                    )
                    self.assertTrue(mega._can_try_sm90_largebs_mega_mhc(tensor))

    @torch.inference_mode()
    def test_attn_seam_deepgemm_is_bitwise_identical(self):
        for tokens in (65, 80, 92, 96, 128, 160, 256, 264, 320, 352, 512):
            with self.subTest(tokens=tokens):
                reference = make_case(tokens, 3100 + tokens)
                candidate = make_case(tokens, 3100 + tokens)
                expected = tuple(value.clone() for value in golden(reference))
                actual = lb.try_sm90_largebs_post_pre(**candidate)
                self.assertIsNotNone(actual)
                torch.cuda.synchronize()
                _assert_bitwise(self, actual[0], expected[0], "residual")
                _assert_normalized_close(self, actual[1], expected[1])
                _assert_coeffs_close(
                    self, (*actual, candidate["next_hc"].pre_mix_out), expected
                )

    @torch.inference_mode()
    def test_ffn_seam_combine_is_bitwise_identical(self):
        for tokens in (65, 92, 96, 128, 264, 352, 512):
            with self.subTest(tokens=tokens):
                reference = make_case(tokens, 3200 + tokens)
                candidate = make_case(tokens, 3200 + tokens)
                routed = torch.randn_like(reference["attn_out"].float()).bfloat16()
                shared = (torch.randn_like(routed.float()) * 0.5).bfloat16()
                pair = PendingCombine(routed, shared, torch.bfloat16)
                reference["attn_out"] = pair.materialize()
                expected = tuple(value.clone() for value in golden(reference))
                candidate["attn_out"] = pair.routed
                actual = lb.try_sm90_largebs_post_pre(
                    **candidate, combine=(pair.routed, pair.shared)
                )
                self.assertIsNotNone(actual)
                torch.cuda.synchronize()
                _assert_bitwise(self, actual[0], expected[0], "residual")
                _assert_normalized_close(self, actual[1], expected[1])
                _assert_coeffs_close(
                    self, (*actual, candidate["next_hc"].pre_mix_out), expected
                )

    @torch.inference_mode()
    def test_post_combine_is_bitwise_add_cast_plus_post(self):
        from rtp_llm.models_py.modules.dsv41.moe.shared_expert import (
            combine_routed_and_shared,
        )

        for tokens in (65, 92, 96, 128, 264, 352, 512):
            with self.subTest(tokens=tokens):
                case = make_case(tokens, 3300 + tokens)
                routed = torch.randn_like(case["attn_out"].float()).bfloat16()
                shared = (torch.randn_like(routed.float()) * 0.5).bfloat16()
                materialized = combine_routed_and_shared(
                    routed.view(tokens, DIM),
                    shared.view(tokens, DIM),
                    torch.bfloat16,
                ).view(1, tokens, DIM)
                ops = lb._functional
                expected = ops.mhc_post(
                    materialized,
                    case["residual"],
                    case["post"],
                    case["comb"],
                    out=torch.empty_like(case["residual"]),
                )
                actual = lb.post_combine_fwd(
                    routed.view(tokens, DIM),
                    shared.view(tokens, DIM),
                    case["residual"].view(tokens, HC, DIM),
                    case["post"].view(tokens, HC),
                    case["comb"].view(tokens, HC, HC),
                    out=torch.empty_like(case["residual"]).view(tokens, HC, DIM),
                ).view_as(expected)
                torch.cuda.synchronize()
                _assert_bitwise(self, actual, expected, "combine residual")

    @torch.inference_mode()
    def test_tilelang_backend_matches_fp64_oracle(self):
        for tokens in (65, 96, 264, 352, 512):
            with self.subTest(tokens=tokens):
                case = make_case(tokens, 3400 + tokens)
                expected = oracle(case)
                result = lb.run(
                    case["attn_out"].view(tokens, DIM),
                    case["residual"].view(tokens, HC, DIM),
                    case["post"].view(tokens, HC),
                    case["comb"].view(tokens, HC, HC),
                    case["next_hc"].fn,
                    case["previous"].pre_mix_out.view(tokens, HC),
                    case["next_hc"].scale,
                    case["next_hc"].base,
                    case["norm"].weight,
                    hc_norm_eps=NORM_EPS,
                    pre_eps=EPS,
                    sinkhorn_eps=EPS,
                    sinkhorn_repeat=SINKHORN,
                    rmsnorm_eps=EPS,
                    backend="tilelang_splitk",
                )
                torch.cuda.synchronize()
                residual_out, normalized, pre, post, comb = result
                # The post mapping is the same kernel as the baseline.
                reference = make_case(tokens, 3400 + tokens)
                golden_residual = reference["previous"].post(
                    reference["attn_out"],
                    reference["residual"].clone(),
                    reference["post"],
                    reference["comb"],
                )
                _assert_bitwise(self, residual_out, golden_residual, "residual")
                self.assertLessEqual(_rel_l2(normalized, expected[1]), 1e-2)
                for name, got, ref in zip(
                    ("pre", "post", "comb"),
                    (pre, post, comb),
                    (expected[2], expected[3], expected[4]),
                ):
                    self.assertLessEqual(_rel_l2(got, ref), 2e-4, name)

    @torch.inference_mode()
    def test_cuda_graph_capture_and_replay(self):
        for tokens in (96, 264):
            with self.subTest(tokens=tokens):
                case = make_case(tokens, 3500 + tokens)
                routed = torch.randn_like(case["attn_out"].float()).bfloat16()
                shared = (torch.randn_like(routed.float()) * 0.5).bfloat16()
                case["attn_out"] = routed
                originals = {
                    "routed": routed.clone(),
                    "shared": shared.clone(),
                    "residual": case["residual"].clone(),
                    "post": case["post"].clone(),
                    "comb": case["comb"].clone(),
                    "pre": case["previous"].pre_mix_out.clone(),
                }

                def run():
                    return lb.try_sm90_largebs_post_pre(
                        **case, combine=(routed, shared)
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                self.assertIsNotNone(captured)
                keys = tuple(range(4))
                for scale in (1.0, -0.75):
                    routed.copy_(originals["routed"] * scale)
                    shared.copy_(originals["shared"] * scale)
                    case["residual"].copy_(originals["residual"] * scale)
                    case["post"].copy_(originals["post"] * scale)
                    case["comb"].copy_(originals["comb"] * scale)
                    case["previous"].pre_mix_out.copy_(originals["pre"] * scale)
                    expected = tuple(value.clone() for value in run())
                    stream.wait_stream(torch.cuda.current_stream())
                    graph.replay()
                    torch.cuda.synchronize()
                    for index in keys:
                        _assert_bitwise(
                            self,
                            captured[index],
                            expected[index],
                            f"replay field {index}",
                        )

    @torch.inference_mode()
    def test_unwarmed_capture_stream_declines(self):
        tokens = 96
        case = make_case(tokens, 3600 + tokens)
        lb._WARMED_STREAMS.clear()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = lb.try_sm90_largebs_post_pre(**case)
                torch.zeros(1, device="cuda").fill_(0)
        self.assertIsNone(result)

    @torch.inference_mode()
    def test_two_kernel_owns_64_rows(self):
        from rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_two_kernel import (
            try_vllm_two_kernel_post_pre,
        )

        case = make_case(64, 3700)
        self.assertIsNone(lb.try_sm90_largebs_post_pre(**case))
        self.assertIsNotNone(try_vllm_two_kernel_post_pre(**case))


if __name__ == "__main__":
    unittest.main()

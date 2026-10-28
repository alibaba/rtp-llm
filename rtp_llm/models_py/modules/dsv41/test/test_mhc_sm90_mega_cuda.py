"""SM90 exactness and CUDA-graph gate for the guarded MegaMHC seam."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.hc import mhc_sm90_mega
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.hc.delayed import _tile_ops
from rtp_llm.models_py.modules.dsv41.hc.v41_mega_mhc import (
    _SM90_WARMED_STREAMS,
    is_sm90_supported,
    try_fused_post_pre,
)

DIM = 5120
HC = 4
EPS = 1e-6


def _unit(fn, base, scale):
    return DelayedHCUnit(
        fn,
        base,
        scale,
        dim=DIM,
        hc_mult=HC,
        hc_sinkhorn_iters=20,
        norm_eps=EPS,
        hc_eps=EPS,
    )


def _case(tokens, seed, mode="random"):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    previous = _unit(
        torch.zeros((24, HC * DIM), device=device),
        torch.zeros(24, device=device),
        torch.ones(3, device=device),
    )
    base = torch.randn(24, device=device) * 0.2
    scale = torch.tensor([0.2, 0.4, 0.3], device=device)
    if mode == "saturated":
        base.copy_(torch.linspace(-12, 12, 24, device=device))
        scale.copy_(torch.tensor([2.0, -2.0, 4.0], device=device))
    following = _unit(
        torch.randn((24, HC * DIM), device=device) * 0.003,
        base,
        scale,
    )
    following.set_previous(previous)
    previous.pre_mix_out = torch.rand((1, tokens, HC), device=device)

    amplitude = 0.0 if mode == "zero" else 8.0 if mode == "saturated" else 1.0
    residual = (
        torch.randn((1, tokens, HC, DIM), device=device) * amplitude
    ).bfloat16()
    attn_out = (torch.randn((1, tokens, DIM), device=device) * amplitude).bfloat16()
    post = torch.rand((1, tokens, HC, 1), device=device) * 0.5
    comb = (
        torch.eye(HC, device=device)
        .expand(1, tokens, HC, HC)
        .contiguous()
        .add_(torch.randn((1, tokens, HC, HC), device=device) * 0.01)
    )
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

    norm = RMSNorm(
        (1.0 + torch.randn(DIM, device=device) * 0.05).bfloat16(), EPS
    )
    return dict(
        attn_out=attn_out,
        residual=residual,
        post=post,
        comb=comb,
        previous=previous,
        next_hc=following,
        norm=norm,
    )


def _unfused(case):
    residual = case["residual"].clone()
    residual = case["previous"].post(
        case["attn_out"], residual, case["post"], case["comb"]
    )
    layer_input, post, comb = case["next_hc"].pre(residual)
    normalized = case["norm"](layer_input.reshape(-1, DIM)).view_as(layer_input)
    return residual, normalized, post, comb, case["next_hc"].pre_mix_out


def _fused(case):
    result = try_fused_post_pre(**case)
    if result is None:
        raise AssertionError("guarded SM90 MegaMHC unexpectedly fell back")
    return (*result, case["next_hc"].pre_mix_out)


def _assert_exact(test, actual, expected):
    for name, got, ref in zip(
        ("residual", "normalized", "post", "comb", "next_pre"), actual, expected
    ):
        test.assertEqual((got.shape, got.dtype), (ref.shape, ref.dtype), name)
        test.assertTrue(torch.equal(got, ref), name)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MhcSm90MegaCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capability = tuple(torch.cuda.get_device_capability())
        if torch.version.hip is not None or capability != (9, 0):
            raise unittest.SkipTest("exact SM90 required")

    def setUp(self):
        _SM90_WARMED_STREAMS.clear()
        self.addCleanup(_SM90_WARMED_STREAMS.clear)

    @torch.inference_mode()
    def test_materialized_mixes_epilogue_is_bitwise_standalone(self):
        tokens = 4
        mixes = torch.linspace(
            -12, 12, tokens * 24, dtype=torch.float32, device="cuda"
        ).view(tokens, 24)
        scale = torch.tensor([0.2, -0.4, 0.3], device="cuda")
        base = torch.linspace(-1, 1, 24, dtype=torch.float32, device="cuda")
        hidden = torch.arange(DIM, dtype=torch.float32, device="cuda")
        residual = torch.stack(
            [
                ((hidden.remainder(257) - 128) / 32 + lane * 0.125).bfloat16()
                for lane in range(HC)
            ]
        ).expand(tokens, HC, DIM).contiguous()
        previous_pre = torch.tensor(
            [
                [0.0, 0.25, 0.5, 1.0],
                [-0.5, 0.75, 1.25, -1.0],
                [2.0, -2.0, 0.125, 0.0625],
                [1e-6, 1.0, -0.25, 0.5],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        ops = _tile_ops()
        ref_pre, ref_post, ref_comb = ops.mhc_pre_split_mixes(
            mixes.view(1, tokens, 24), scale, base, HC, 2.0, EPS
        )
        ref_comb = ops.sinkhorn_normalize(ref_comb, repeat=20, eps=EPS)
        ref_y = ops.mhc_pre_apply_mix(
            residual.view(1, tokens, HC, DIM),
            previous_pre.view(1, tokens, HC, 1),
        )
        actual = mhc_sm90_mega.pre_mixes_fwd(
            mixes,
            residual,
            previous_pre,
            scale,
            base,
            pre_eps=EPS,
            sinkhorn_eps=EPS,
            post_mult=2.0,
            sinkhorn_repeat=20,
        )
        expected = (
            ref_pre.view(tokens, HC),
            ref_post.view(tokens, HC),
            ref_comb.view(tokens, HC * HC),
            ref_y.view(tokens, DIM),
        )
        torch.cuda.synchronize()
        for name, got, ref in zip(("pre", "post", "comb", "y"), actual, expected):
            self.assertTrue(torch.equal(got, ref), name)

    @torch.inference_mode()
    def test_exact_random_and_boundary_outputs(self):
        for tokens in (1, 4, 12):
            for mode in ("random", "zero", "saturated"):
                with self.subTest(tokens=tokens, mode=mode):
                    case = _case(tokens, 4100 + tokens, mode)
                    expected = tuple(value.clone() for value in _unfused(case))
                    residual_pointer = case["residual"].data_ptr()
                    actual = _fused(case)
                    torch.cuda.synchronize()
                    self.assertEqual(actual[0].data_ptr(), residual_pointer)
                    _assert_exact(self, actual, expected)

    @torch.inference_mode()
    def test_gate_is_decode_bounded_and_extended_modes_fall_back(self):
        supported = _case(4, 4204)
        self.assertTrue(is_sm90_supported(**supported))
        too_large = _case(17, 4217)
        self.assertFalse(is_sm90_supported(**too_large))
        self.assertIsNone(try_fused_post_pre(**supported, original_tokens=16))
        self.assertIsNone(try_fused_post_pre(**supported, reuse_residual=True))
        self.assertIsNone(try_fused_post_pre(**supported, output_fp8=True))

    @torch.inference_mode()
    def test_cuda_graph_replays_mutated_inputs_and_predecessor_mix(self):
        case = _case(4, 4304)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _fused(case)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_outputs = _fused(case)
        self.assertEqual(graph_outputs[0].data_ptr(), case["residual"].data_ptr())

        # Replay must not grow live allocations or the caching allocator's
        # reservation after the graph pool has reached steady state.
        graph.replay()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), allocated)
        self.assertEqual(torch.cuda.memory_reserved(), reserved)

        for step, amplitude in enumerate((0.0, -0.75, 2.0)):
            torch.manual_seed(4400 + step)
            case["attn_out"].copy_(
                (torch.randn_like(case["attn_out"].float()) * amplitude).bfloat16()
            )
            case["residual"].copy_(
                (torch.randn_like(case["residual"].float()) * amplitude).bfloat16()
            )
            case["post"].copy_(torch.rand_like(case["post"]) * (step + 1))
            case["comb"].copy_(torch.randn_like(case["comb"]) * 0.2)
            case["previous"].pre_mix_out.copy_(
                torch.rand_like(case["previous"].pre_mix_out) * (step + 1)
            )
            # Build the reference from the pristine input before graph replay
            # overwrites the residual in place.
            expected = tuple(value.clone() for value in _unfused(case))
            stream.wait_stream(torch.cuda.current_stream())
            graph.replay()
            torch.cuda.synchronize()
            actual = tuple(value.clone() for value in graph_outputs)
            _assert_exact(self, actual, expected)


if __name__ == "__main__":
    unittest.main()

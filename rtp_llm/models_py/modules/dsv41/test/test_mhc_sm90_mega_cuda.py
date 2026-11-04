"""SM90 numerical and graph gates for the vLLM two-kernel MegaMHC path."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_two_kernel import (
    _WARMED_STREAMS,
)
from rtp_llm.models_py.modules.dsv41.hc.v41_mega_mhc import try_fused_post_pre
from rtp_llm.models_py.modules.dsv41.moe.shared_expert import PendingCombine

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


def _case(tokens, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    previous = _unit(
        torch.zeros((24, HC * DIM), device=device),
        torch.zeros(24, device=device),
        torch.ones(3, device=device),
    )
    following = _unit(
        torch.randn((24, HC * DIM), device=device) * 0.003,
        torch.randn(24, device=device) * 0.2,
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


def _golden(case):
    residual = case["previous"].post(
        case["attn_out"],
        case["residual"].clone(),
        case["post"],
        case["comb"],
    )
    layer_input, post, comb = case["next_hc"].pre(residual)
    normalized = case["norm"](layer_input.reshape(-1, DIM)).view_as(layer_input)
    return residual, normalized, post, comb, case["next_hc"].pre_mix_out


def _candidate(case, combine=None):
    result = try_fused_post_pre(**case, combine=combine)
    if result is None:
        raise AssertionError("SM90 two-kernel MegaMHC unexpectedly fell back")
    return (*result, case["next_hc"].pre_mix_out)


def _candidate_pair(case):
    """Drive the deferred-combine seam with ``case["pair"]``."""
    pair = case["pair"]
    return _candidate(
        {key: value for key, value in case.items() if key != "pair"},
        combine=(pair.routed, pair.shared),
    )


def _assert_outputs(test, actual, expected):
    test.assertTrue(torch.equal(actual[0], expected[0]), "residual")
    test.assertTrue(torch.equal(actual[1], expected[1]), "normalized")
    for name, got, ref in zip(
        ("post", "comb", "pre"), actual[2:], expected[2:]
    ):
        test.assertTrue(torch.isfinite(got).all(), name)
        test.assertLessEqual((got.float() - ref.float()).abs().max().item(), 1e-6, name)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MhcSm90MegaCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or tuple(torch.cuda.get_device_capability()) != (
            9,
            0,
        ):
            raise unittest.SkipTest("exact SM90 required")

    def setUp(self):
        _WARMED_STREAMS.clear()
        self.addCleanup(_WARMED_STREAMS.clear)

    @torch.inference_mode()
    def test_low_token_outputs(self):
        for tokens in (1, 4, 8, 16, 24, 32):
            with self.subTest(tokens=tokens):
                reference = _case(tokens, 20261029 + tokens)
                candidate = _case(tokens, 20261029 + tokens)
                expected = tuple(value.clone() for value in _golden(reference))
                input_pointer = candidate["residual"].data_ptr()
                actual = _candidate(candidate)
                torch.cuda.synchronize()
                self.assertNotEqual(actual[0].data_ptr(), input_pointer)
                _assert_outputs(self, actual, expected)

    @torch.inference_mode()
    def test_add_cast_fold(self):
        """The deferred routed/shared pair must replace add+cast bit-exactly.

        ``_add_cast_kernel`` stores ``bf16(fp32(routed) + fp32(shared))``. The
        fused post kernel rebuilds that value, so it may read the two BF16
        operands directly. A fold that skips the intermediate BF16 rounding is
        not equivalent, which the last assertion pins down.
        """
        for tokens in (1, 3, 4, 16, 32):
            with self.subTest(tokens=tokens):
                reference = _case(tokens, 20261105 + tokens)
                candidate = _case(tokens, 20261105 + tokens)
                routed = torch.randn_like(reference["attn_out"].float()).bfloat16()
                shared = (torch.randn_like(routed.float()) * 0.5).bfloat16()
                pair = PendingCombine(routed, shared, torch.bfloat16)

                reference["attn_out"] = pair.materialize()
                expected = tuple(value.clone() for value in _golden(reference))
                candidate["attn_out"] = pair.routed
                actual = _candidate(
                    candidate, combine=(pair.routed, pair.shared)
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual[0], expected[0]), "residual")
                self.assertTrue(torch.equal(actual[1], expected[1]), "normalized")

                # The arithmetic reason the round trip is load-bearing.
                naive = (
                    candidate["post"].float()
                    * (routed.float() + shared.float()).unsqueeze(-2)
                    + torch.matmul(
                        candidate["comb"].float().transpose(-1, -2),
                        candidate["residual"].float(),
                    )
                ).to(torch.bfloat16)
                self.assertFalse(
                    torch.equal(naive, expected[0]), "unrounded fold must differ"
                )

    @torch.inference_mode()
    def test_alternating_input_graph_replay(self):
        for tokens in (1, 8, 32):
            with self.subTest(tokens=tokens):
                eager = _case(tokens, 20300000 + tokens)
                captured_case = _case(tokens, 20300000 + tokens)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _candidate(captured_case)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = _candidate(captured_case)

                for amplitude in (1.0, -0.75, 2.0, 1.0):
                    values = dict(
                        attn_out=(
                            torch.randn_like(eager["attn_out"].float()) * amplitude
                        ).bfloat16(),
                        residual=(
                            torch.randn_like(eager["residual"].float()) * amplitude
                        ).bfloat16(),
                        post=torch.rand_like(eager["post"]),
                        comb=torch.randn_like(eager["comb"]).softmax(-1),
                    )
                    previous_pre = torch.rand_like(eager["previous"].pre_mix_out)
                    for case in (eager, captured_case):
                        for key, value in values.items():
                            case[key].copy_(value)
                        case["previous"].pre_mix_out.copy_(previous_pre)
                    expected = tuple(value.clone() for value in _candidate(eager))
                    stream.wait_stream(torch.cuda.current_stream())
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = tuple(value.clone() for value in outputs)
                    for name, got, ref in zip(
                        ("residual", "normalized", "post", "comb"), actual, expected
                    ):
                        self.assertTrue(torch.equal(got, ref), name)


    @torch.inference_mode()
    def test_deferred_combine_graph_replay(self):
        """The deferred pair must replay identically under a captured graph."""
        for tokens in (1, 8, 32):
            with self.subTest(tokens=tokens):
                eager = _case(tokens, 20400000 + tokens)
                captured_case = _case(tokens, 20400000 + tokens)
                for case in (eager, captured_case):
                    case["pair"] = PendingCombine(
                        torch.randn_like(case["attn_out"].float()).bfloat16(),
                        (torch.randn_like(case["attn_out"].float()) * 0.5).bfloat16(),
                        torch.bfloat16,
                    )
                    case["attn_out"] = case["pair"].routed
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _candidate_pair(captured_case)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = _candidate_pair(captured_case)

                for amplitude in (1.0, -0.75, 2.0):
                    # One set of inputs, replayed into both cases in place.
                    values = dict(
                        routed=(
                            torch.randn_like(eager["pair"].routed.float()) * amplitude
                        ).bfloat16(),
                        shared=(
                            torch.randn_like(eager["pair"].shared.float()) * amplitude
                        ).bfloat16(),
                        residual=(
                            torch.randn_like(eager["residual"].float()) * amplitude
                        ).bfloat16(),
                        post=torch.rand_like(eager["post"]),
                        comb=torch.randn_like(eager["comb"]).softmax(-1),
                        previous_pre=torch.rand_like(eager["previous"].pre_mix_out),
                    )
                    for case in (eager, captured_case):
                        for name in ("routed", "shared", "residual", "post", "comb"):
                            target = (
                                getattr(case["pair"], name)
                                if name in ("routed", "shared")
                                else case[name]
                            )
                            target.copy_(values[name])
                        case["previous"].pre_mix_out.copy_(values["previous_pre"])
                    expected = tuple(
                        value.clone() for value in _candidate_pair(eager)
                    )
                    stream.wait_stream(torch.cuda.current_stream())
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = tuple(value.clone() for value in outputs)
                    for name, got, ref in zip(
                        ("residual", "normalized", "post", "comb"), actual, expected
                    ):
                        self.assertTrue(torch.equal(got, ref), name)


if __name__ == "__main__":
    unittest.main()

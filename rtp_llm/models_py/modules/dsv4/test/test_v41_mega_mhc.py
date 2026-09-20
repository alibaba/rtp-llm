"""The fused attention-post -> delayed FFN-pre -> RMSNorm transition.

The reference is the production unfused chain, with independent residual
storage because its post operation writes in place. CUDA graph tests update
the input tensors between replays, including the delayed predecessor mix.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc import (
    is_supported,
    try_fused_post_pre,
)

_MODULE = "rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc"
_DIM = 5120
_HC = 4
_EPS = 1e-6
_DECODE_LAYOUTS = tuple((batch, seq) for batch in (1, 2, 4) for seq in (1, 6))


def make_unit(fn, base, scale):
    return DelayedHCUnit(
        fn,
        base,
        scale,
        dim=_DIM,
        hc_mult=_HC,
        hc_sinkhorn_iters=20,
        norm_eps=_EPS,
        hc_eps=_EPS,
    )


def make_case(leading, device="cuda", weights=None):
    """Build independent previous/next modules and explicit runtime tensors."""
    residual = torch.randn(*leading, _HC, _DIM, device=device, dtype=torch.bfloat16)
    attn_out = torch.randn(*leading, _DIM, device=device, dtype=torch.bfloat16)
    if weights is None:
        weights = [
            (
                torch.randn(24, _HC * _DIM, device=device) * 0.003,
                torch.randn(24, device=device) * 0.1,
                torch.tensor([0.2, 0.4, 0.3], device=device),
            )
            for _ in range(2)
        ]
    previous, next_hc = [make_unit(*group) for group in weights]
    next_hc.set_previous(previous)
    if device == "cpu":
        # CPU gate checks must not initialize CUDA or import its backend.
        norm = SimpleNamespace(
            weight=torch.ones(_DIM, dtype=torch.bfloat16), variance_epsilon=_EPS
        )
    else:
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

        norm = RMSNorm((1.0 + 0.1 * torch.randn(_DIM, device=device)).bfloat16(), _EPS)
    _, post, comb = previous.pre(residual)
    return dict(
        attn_out=attn_out,
        residual=residual,
        post=post,
        comb=comb,
        previous=previous,
        next_hc=next_hc,
        norm=norm,
    )


def unfused(case, residual=None):
    """Run the existing chain, using a private residual unless supplied."""
    residual = case["residual"].clone() if residual is None else residual
    previous, next_hc = case["previous"], case["next_hc"]
    residual = previous.post(case["attn_out"], residual, case["post"], case["comb"])
    y, post, comb = next_hc.pre(residual)
    y = case["norm"](y.reshape(-1, _DIM)).reshape(*residual.shape[:-2], _DIM)
    return residual, y, post, comb, next_hc.pre_mix_out


def fused(case):
    out = try_fused_post_pre(**case)
    if out is None:
        raise AssertionError("mega_mhc unexpectedly fell back for a supported case")
    return (*out, case["next_hc"].pre_mix_out)


def errors(actual, expected):
    """Also used by the standalone benchmark to retain numeric evidence."""
    result = {}
    for name, got, ref in zip(
        ("residual", "y", "post", "comb", "next_pre"), actual, expected
    ):
        diff = got.float() - ref.float()
        result[name] = {
            "max_abs": diff.abs().max().item(),
            "rms": diff.square().mean().sqrt().item(),
            "reference_rms": ref.float().square().mean().sqrt().item(),
        }
    return result


def assert_outputs_close(actual, expected):
    for index, (got, ref) in enumerate(zip(actual, expected)):
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError((index, got.shape, ref.shape, got.dtype, ref.dtype))
        if not torch.isfinite(got).all().item():
            raise AssertionError(f"nonfinite output {index}")
        if index < 2:
            # Preserve the established BF16 full delayed-HC tolerance. A
            # separate RMS bound prevents a systematic bias from passing it.
            torch.testing.assert_close(got, ref, rtol=0.02, atol=0.03125)
            diff_rms = (got.float() - ref.float()).square().mean().sqrt().item()
            ref_rms = ref.float().square().mean().sqrt().item()
            if diff_rms > 1e-5 + 0.005 * ref_rms:
                raise AssertionError((index, "RMS error", diff_rms, ref_rms))
        else:
            torch.testing.assert_close(got, ref, rtol=2e-4, atol=1e-5)


class V41MegaMHCCPUTest(unittest.TestCase):
    def test_cpu_fallback_does_not_load_backend_or_change_state(self):
        torch.manual_seed(41)
        case = make_case((1, 6), "cpu")
        tensors = [case[name] for name in ("attn_out", "residual", "post", "comb")]
        before = [tensor.clone() for tensor in tensors]
        previous_pre = case["previous"].pre_mix_out
        with patch(f"{_MODULE}._get_mega_mhc") as backend:
            self.assertFalse(is_supported(**case))
            self.assertIsNone(try_fused_post_pre(**case))
            backend.assert_not_called()
        self.assertIs(case["previous"].pre_mix_out, previous_pre)
        self.assertIsNone(case["next_hc"].pre_mix_out)
        for actual, expected in zip(tensors, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41MegaMHCCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("mega_mhc requires Blackwell")

    def setUp(self):
        env = patch.dict(os.environ, {"DSV41_MEGA_MHC": "1"})
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(41)

    @torch.no_grad()
    def check_case(self, case):
        expected = tuple(t.clone() for t in unfused(case))
        previous_pre = case["previous"].pre_mix_out
        protected = [
            case["attn_out"],
            case["residual"],
            case["post"],
            case["comb"],
            previous_pre,
            case["next_hc"].fn,
            case["next_hc"].base,
            case["next_hc"].scale,
            case["norm"].weight,
        ]
        before = [value.clone() for value in protected]
        self.assertTrue(is_supported(**case))
        actual = fused(case)
        self.assertNotEqual(actual[0].data_ptr(), case["residual"].data_ptr())
        assert_outputs_close(actual, expected)
        self.assertIs(case["previous"].pre_mix_out, previous_pre)
        for got, ref in zip(protected, before):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        return actual

    @torch.no_grad()
    def test_decode_layouts_and_prefill(self):
        for leading in (*_DECODE_LAYOUTS, (64,), (128,), (1024,)):
            with self.subTest(leading=leading):
                self.check_case(make_case(leading))

    @torch.no_grad()
    def test_zero_and_rescaled_inputs(self):
        for amplitude in (0.0, 1e-5, 1.0, 32.0):
            with self.subTest(amplitude=amplitude):
                case = make_case((1, 6))
                case["residual"].mul_(amplitude)
                case["attn_out"].mul_(amplitude)
                _, case["post"], case["comb"] = case["previous"].pre(case["residual"])
                self.check_case(case)

    @torch.no_grad()
    def test_gate_and_predecessor_contract(self):
        case = make_case((1, 6))
        with patch.dict(os.environ, {"DSV41_MEGA_MHC": "0"}):
            self.assertFalse(is_supported(**case))
            self.assertIsNone(try_fused_post_pre(**case))
        for key in ("attn_out", "residual", "post", "comb"):
            with self.subTest(cpu_input=key):
                changed = {**case, key: case[key].cpu()}
                self.assertFalse(is_supported(**changed))
                self.assertIsNone(try_fused_post_pre(**changed))
        for changed in (
            {**case, "residual": case["residual"].float()},
            {**case, "attn_out": case["attn_out"].float()},
            {**case, "post": case["post"].bfloat16()},
            {**case, "comb": case["comb"].bfloat16()},
            {**case, "residual": case["residual"].transpose(-1, -2)},
        ):
            with self.subTest(
                dtypes=[changed[key].dtype for key in ("residual", "post")]
            ):
                self.assertFalse(is_supported(**changed))
                self.assertIsNone(try_fused_post_pre(**changed))
        case["next_hc"].set_previous(None)
        self.assertFalse(is_supported(**case))
        self.assertIsNone(try_fused_post_pre(**case))
        case["next_hc"].set_previous(case["previous"])
        case["previous"].pre_mix_out = None
        self.assertFalse(is_supported(**case))
        self.assertIsNone(try_fused_post_pre(**case))

    @torch.no_grad()
    def test_backend_execution_failure_is_not_hidden(self):
        case = make_case((1, 1))

        def failure(**kwargs):
            raise RuntimeError("injected mega_mhc execution failure")

        with patch(f"{_MODULE}._get_mega_mhc", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "injected mega_mhc"):
                try_fused_post_pre(**case)

    @torch.no_grad()
    def test_cuda_graph_replays_updated_inputs_and_delayed_mix(self):
        for leading in (*_DECODE_LAYOUTS, (64,), (128,), (1024,)):
            with self.subTest(leading=leading):
                case = make_case(leading)
                graph_case = {**case, "residual": case["residual"].clone()}
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        graph_case["residual"].copy_(case["residual"])
                        fused(graph_case)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph_case["residual"].copy_(case["residual"])
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=stream):
                    actual = fused(graph_case)
                initial = None
                for amplitude in (1.0, -0.75, 2.0):
                    case["residual"].copy_(
                        torch.randn_like(case["residual"]) * amplitude
                    )
                    case["attn_out"].copy_(
                        torch.randn_like(case["attn_out"]) * amplitude
                    )
                    old_pre = case["previous"].pre_mix_out
                    _, post, comb = case["previous"].pre(case["residual"])
                    # Keep graph pointers stable while updating all mixer
                    # values; assigning new Python tensors would test nothing.
                    old_pre.copy_(case["previous"].pre_mix_out)
                    case["previous"].pre_mix_out = old_pre
                    case["post"].copy_(post)
                    case["comb"].copy_(comb)
                    expected = tuple(t.clone() for t in unfused(case))
                    graph_case["residual"].copy_(case["residual"])
                    graph.replay()
                    assert_outputs_close(actual, expected)
                    if initial is None:
                        initial = actual[1].clone()
                    else:
                        self.assertFalse(torch.equal(initial, actual[1]))

    @torch.no_grad()
    def test_graph_publishes_mix_to_next_layer_and_final_head(self):
        from rtp_llm.models_py.modules.dsv4.hc.delayed import (
            DelayedHCHead,
            collapse_delayed,
        )

        for leading in ((4, 1), (4, 6), (128,)):
            with self.subTest(leading=leading):
                source = make_case(leading)
                variants = {}
                for kind in ("baseline", "candidate"):
                    case = make_case(leading)
                    for name in ("previous", "next_hc"):
                        for weight in ("fn", "base", "scale"):
                            getattr(case[name], weight).copy_(
                                getattr(source[name], weight)
                            )
                    case["norm"].weight.copy_(source["norm"].weight)
                    for name in ("attn_out", "residual", "post", "comb"):
                        case[name].copy_(source[name])
                    case["previous"].pre_mix_out.copy_(source["previous"].pre_mix_out)
                    # Make the successor's OWN mix very different. Its input
                    # must use the preceding FFN's published delayed mix.
                    base = torch.zeros(24, device="cuda")
                    base[:4] = 9.0
                    follower = make_unit(
                        torch.zeros(24, _HC * _DIM, device="cuda"),
                        base,
                        torch.ones(3, device="cuda"),
                    )
                    follower.set_previous(case["next_hc"])
                    head = DelayedHCHead(case["next_hc"])
                    ffn_out = torch.empty_like(case["attn_out"])
                    ffn_out.copy_(source["attn_out"])

                    def run_transition():
                        result = (
                            unfused(case, residual=case["residual"])
                            if kind == "baseline"
                            else fused(case)
                        )
                        residual, normalized, post, comb, pre = result
                        final_residual = case["next_hc"].post(
                            ffn_out, residual, post, comb
                        )
                        next_input, _, _ = follower.pre(final_residual)
                        head_input = head.head(final_residual)
                        return normalized, final_residual, pre, next_input, head_input

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            case["residual"].copy_(source["residual"])
                            run_transition()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    case["residual"].copy_(source["residual"])
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = run_transition()
                    variants[kind] = (case, follower, ffn_out, graph, result)

                pre_pointers = {
                    kind: case["next_hc"].pre_mix_out.data_ptr()
                    for kind, (case, _, _, _, _) in variants.items()
                }
                initial = None
                for amplitude in (1.0, -0.75, 2.0):
                    updated = make_case(leading)
                    updated["residual"].mul_(amplitude)
                    updated["attn_out"].mul_(amplitude)
                    # Only update tensor contents after capture. Do not call
                    # pre/fused again on the captured modules or replace the
                    # Python references that name graph-owned coefficient data.
                    for kind, (
                        case,
                        follower,
                        ffn_out,
                        graph,
                        result,
                    ) in variants.items():
                        for name in ("attn_out", "residual", "post", "comb"):
                            case[name].copy_(updated[name])
                        case["previous"].pre_mix_out.copy_(
                            updated["previous"].pre_mix_out
                        )
                        ffn_out.copy_(updated["attn_out"])
                        graph.replay()
                        self.assertEqual(
                            case["next_hc"].pre_mix_out.data_ptr(), pre_pointers[kind]
                        )
                        self.assertEqual(result[2].data_ptr(), pre_pointers[kind])
                        torch.testing.assert_close(result[3], result[4], rtol=0, atol=0)
                        own_readout = collapse_delayed(result[1], follower.pre_mix_out)
                        self.assertFalse(torch.equal(result[3], own_readout))
                    expected = variants["baseline"][-1]
                    actual = variants["candidate"][-1]
                    for index, (got, ref) in enumerate(zip(actual, expected)):
                        torch.testing.assert_close(
                            got,
                            ref,
                            rtol=2e-4 if index == 2 else 0.02,
                            atol=1e-5 if index == 2 else 0.03125,
                        )
                    if initial is None:
                        initial = actual[4].clone()
                    else:
                        self.assertFalse(torch.equal(initial, actual[4]))

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set DSV41_TEST_CHECKPOINT for real weights",
    )
    @torch.no_grad()
    def test_checkpoint_attention_to_ffn_transition(self):
        from safetensors import safe_open

        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in (0, 39):
            weights = []
            for suffix in ("attn", "ffn"):
                group = []
                for key in ("fn", "base", "scale"):
                    name = f"layers.{layer}.hc_{suffix}_{key}"
                    with safe_open(
                        str(checkpoint / index[name]), framework="pt", device="cpu"
                    ) as source:
                        group.append(source.get_tensor(name).float().cuda())
                weights.append(group)
            for leading in ((4, 1), (4, 6), (128,)):
                with self.subTest(layer=layer, leading=leading):
                    case = make_case(leading, weights=weights)
                    name = f"layers.{layer}.ffn_norm.weight"
                    with safe_open(
                        str(checkpoint / index[name]), framework="pt", device="cpu"
                    ) as source:
                        case["norm"].weight.copy_(source.get_tensor(name).cuda())
                    self.check_case(case)


if __name__ == "__main__":
    unittest.main()

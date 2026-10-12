"""Released output GEMMs: independent oracles, native scope and component graphs."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    _dense_down,
    _routed_down,
    expert_down_reference,
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import (
    V41MXFP8Linear,
    quantize_group32,
)
from rtp_llm.models_py.modules.dsv4.v41_routed import (
    V41RoutedExperts,
    _fp8_fp4_eager,
)


class ExpertPrecisionSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Expert output diagnostic requires SM120")
        cls.kernel = load_reference_kernel()

    def setUp(self):
        torch.manual_seed(41431)
        original = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", original)

    @staticmethod
    def shared():
        return W13SharedExpert(
            128,
            128,
            {
                "w13_w": torch.randn(256, 128, device="cuda").to(torch.float8_e4m3fn),
                "w13_s": torch.full((8, 4), 123, device="cuda", dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
                "w2_w": torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn),
                "w2_s": torch.full((4, 4), 124, device="cuda", dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            },
            swiglu_limit=10,
        )

    @staticmethod
    def routed_weights():
        return {
            "w13_w": torch.randint(
                0, 256, (3, 256, 64), device="cuda", dtype=torch.uint8
            ),
            "w13_s": torch.full((3, 256, 4), 122, device="cuda", dtype=torch.uint8),
            "w2_w": torch.randint(
                0, 256, (3, 128, 64), device="cuda", dtype=torch.uint8
            ),
            "w2_s": torch.full((3, 128, 4), 123, device="cuda", dtype=torch.uint8),
        }

    def assert_bytes(self, before, after):
        for a, b in zip(before, after):
            torch.testing.assert_close(
                a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
            )

    def test_selection_nesting_exception_and_arch_admission(self):
        original = (
            W13SharedExpert.forward,
            V41RoutedExperts._run,
            V41MXFP8Linear.forward_quantized,
        )
        with expert_down_reference(True, kernel=self.kernel):
            outer = W13SharedExpert.forward
            self.assertIsNot(outer, original[0])
            self.assertIs(V41RoutedExperts._run, original[1])
            with expert_down_reference(False):
                self.assertIs(W13SharedExpert.forward, outer)
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with expert_down_reference(True, routed=True, kernel=self.kernel):
                    self.assertIsNot(V41RoutedExperts._run, original[1])
                    raise RuntimeError("restore")
            self.assertIs(W13SharedExpert.forward, outer)
            self.assertIs(V41RoutedExperts._run, original[1])
        self.assertEqual(
            original,
            (
                W13SharedExpert.forward,
                V41RoutedExperts._run,
                V41MXFP8Linear.forward_quantized,
            ),
        )
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "SM120"):
                with expert_down_reference(True, kernel=self.kernel):
                    self.fail("Admitted another architecture")
            with expert_down_reference(False):
                pass

    @torch.inference_mode()
    def test_shared_owner_oracle_scope_and_source_bytes(self):
        owner = self.shared()
        x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
        sources = (
            x,
            owner.w13.weight,
            owner.w13._sm120_linear.scales,
            owner.w2.weight,
            owner.w2._sm120_linear.scales,
        )
        before = [value.clone() for value in sources]
        original_projection = owner.w13(x.reshape(-1, 128))
        captured = []
        handle = owner.w2.register_forward_pre_hook(
            lambda _m, args: captured.append(args[0].clone())
        )
        try:
            with expert_down_reference(True, kernel=self.kernel):
                actual = owner(x)
                torch.testing.assert_close(
                    owner.w13(x.reshape(-1, 128)), original_projection, rtol=0, atol=0
                )
        finally:
            handle.remove()
        reference = V41MXFP8Linear(
            owner.w2.weight.cpu(), owner.w2._sm120_linear.scales.cpu(), backend="eager"
        )
        expected = reference(captured[0].cpu()).reshape_as(actual)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        self.assert_bytes(before, sources)

    @torch.inference_mode()
    def test_dense_empty_output_and_alias_rejection(self):
        owner = self.shared().w2._sm120_linear
        values = torch.zeros(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.full((64, 4), 127, device="cuda", dtype=torch.uint8)
        out = torch.full((64, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
        self.assertIs(_dense_down(owner, values, scales, self.kernel, out=out), out)
        self.assertEqual(torch.count_nonzero(out).item(), 0)
        empty = _dense_down(owner, values[:0], scales[:0], self.kernel)
        self.assertEqual(empty.shape, (0, 128))
        alias = owner.weight.view(torch.bfloat16).reshape(64, 128)
        with self.assertRaisesRegex(ValueError, "share"):
            _dense_down(owner, values, scales, self.kernel, out=alias)
        with self.assertRaisesRegex(ValueError, "scales"):
            _dense_down(owner, values, scales[:, :3].contiguous(), self.kernel)

    @torch.inference_mode()
    def test_routed_layouts_match_independent_cpu_oracle(self):
        weights = self.routed_weights()
        owner = V41RoutedExperts(weights)
        x = torch.randn(3, 128, 128, device="cuda", dtype=torch.bfloat16)
        values, scales = quantize_group32(x)
        scales = scales.reshape(3, 128, 4)
        protected = (values, scales, *weights.values())
        before = [value.clone() for value in protected]
        expected = torch.stack(
            [
                _fp8_fp4_eager(
                    values[e].cpu(),
                    scales[e].cpu(),
                    weights["w2_w"][e].cpu(),
                    weights["w2_s"][e].cpu(),
                )
                for e in range(3)
            ]
        )
        masked = _routed_down(owner, values, scales, ("masked", 128, 128), self.kernel)
        contiguous = _routed_down(
            owner,
            values.reshape(-1, 128),
            scales.reshape(-1, 4),
            ("contiguous", (128, 128, 128)),
            self.kernel,
        )
        torch.testing.assert_close(masked.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            contiguous.cpu(), expected.reshape(-1, 128), rtol=0, atol=0
        )
        # Zero-sized segments retain expert index mapping.
        partial = _routed_down(
            owner,
            values[1:].reshape(-1, 128),
            scales[1:].reshape(-1, 4),
            ("contiguous", (0, 128, 128)),
            self.kernel,
        )
        torch.testing.assert_close(
            partial.cpu(), expected[1:].reshape(-1, 128), rtol=0, atol=0
        )
        self.assert_bytes(before, protected)

    @torch.inference_mode()
    def test_shared_changed_input_graph_parity(self):
        owner = self.shared()
        x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        with expert_down_reference(True, kernel=self.kernel):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    out.copy_(owner(x))
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(owner(x))
            for factor in (0, 0.37, -1.5):
                x.copy_(torch.randn_like(x) * factor)
                graph.replay()
                torch.testing.assert_close(out, owner(x), rtol=0, atol=0)

    @torch.inference_mode()
    def test_local_masking_continuation_and_changed_input_graph(self):
        weights = self.routed_weights()
        owner = V41LocalMoE(V41RoutedExperts(weights), max_tokens=128, max_topk=2)
        reference = V41LocalMoE(
            V41RoutedExperts(
                {name: value.cpu() for name, value in weights.items()}, backend="eager"
            ),
            max_tokens=128,
            max_topk=2,
        )
        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor(
            [[0, 1], [1, 2], [-1, -1], [2, 0], [0, 2], [1, 0], [2, 1]], device="cuda"
        )
        route = torch.full((7, 2), 0.37, device="cuda", dtype=torch.float32)
        x[2].fill_(float("nan"))
        out = torch.empty_like(x, dtype=torch.float32)
        with expert_down_reference(True, shared=False, routed=True, kernel=self.kernel):
            for layout in ("masked", "contiguous"):
                actual = owner(x, route, ids, layout=layout)
                expected = reference(x.cpu(), route.cpu(), ids.cpu(), layout=layout)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)
            self.assertEqual(
                owner(x[:0], route[:0], ids[:0], layout="masked").shape, (0, 128)
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    out.copy_(owner(x, route, ids, layout="masked"))
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(owner(x, route, ids, layout="masked"))
            for factor in (0, 0.2, 1.5):
                x.copy_(torch.randn_like(x) * factor)
                x[2].fill_(float("nan"))
                route.copy_(torch.full_like(route, factor))
                graph.replay()
                torch.testing.assert_close(
                    out, owner(x, route, ids, layout="masked"), rtol=0, atol=0
                )
                self.assertEqual(torch.count_nonzero(out[2]).item(), 0)


if __name__ == "__main__":
    unittest.main()

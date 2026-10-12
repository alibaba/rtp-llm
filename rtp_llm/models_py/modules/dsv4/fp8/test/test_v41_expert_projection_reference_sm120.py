"""Validate explicit projection scope, released oracles and changing CUDA graphs."""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    v41_expert_projection_reference as scope,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    expert_down_reference,
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, quantize_group32
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


class ExpertProjectionReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Expert projection diagnostic requires SM120")
        cls.kernel = load_reference_kernel()
        path = Path(os.environ["DSV41_TEST_REFERENCE_KERNEL"]).resolve()
        sys.path.insert(0, str(path.parent))
        cls.official = importlib.import_module("model")
        assert Path(cls.official.__file__).resolve() == path.parent / "model.py"

    def setUp(self):
        torch.manual_seed(41561)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def exact(self, left, right):
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.dtype, right.dtype)
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def shared(self, width=128):
        return V41MXFP8Linear(
            torch.randn(256, width, device="cuda").to(torch.float8_e4m3fn),
            torch.full((8, width // 32), 123, device="cuda", dtype=torch.uint8),
        )

    def routed(self):
        return V41RoutedExperts(
            {
                "w13_w": torch.randint(
                    0, 256, (3, 256, 64), device="cuda", dtype=torch.uint8
                ),
                "w13_s": torch.full((3, 256, 4), 122, device="cuda", dtype=torch.uint8),
                "w2_w": torch.randint(
                    0, 256, (3, 128, 64), device="cuda", dtype=torch.uint8
                ),
                "w2_s": torch.full((3, 128, 4), 123, device="cuda", dtype=torch.uint8),
            }
        )

    def shared_oracle(self, owner, x):
        outputs = []
        for offset in (0, owner.N // 2):
            with torch.device("cuda"):
                linear = self.official.Linear(
                    owner.K, owner.N // 2, dtype=torch.float8_e4m3fn
                )
            linear.weight.view(torch.uint8).copy_(
                owner.weight[offset : offset + owner.N // 2].view(torch.uint8)
            )
            linear.scale.view(torch.uint8).copy_(
                owner.scales[offset // 32 : (offset + owner.N // 2) // 32]
            )
            outputs.append(linear(x))
        return torch.cat(outputs, dim=-1)

    def routed_oracle(self, owner, x, counts, route):
        result = torch.zeros_like(x)
        for e, rows in enumerate(counts.cpu().tolist()):
            if not rows:
                continue
            with torch.device("cuda"):
                expert = self.official.Expert(
                    owner.dim,
                    owner.inter_dim,
                    dtype=torch.float4_e2m1fn_x2,
                    swiglu_limit=owner.swiglu_limit,
                )
            for part, offset in (("w1", 0), ("w3", owner.inter_dim)):
                getattr(expert, part).weight.view(torch.uint8).copy_(
                    owner.w13[e, offset : offset + owner.inter_dim]
                )
                getattr(expert, part).scale.view(torch.uint8).copy_(
                    owner.s13[e, offset : offset + owner.inter_dim]
                )
            expert.w2.weight.view(torch.uint8).copy_(owner.w2[e])
            expert.w2.scale.view(torch.uint8).copy_(owner.s2[e])
            result[e, :rows] = expert(
                x[e, :rows].contiguous(), route[e, :rows, None].contiguous()
            )
        return result

    def originals(self):
        return V41MXFP8Linear.forward_quantized, V41RoutedExperts._run

    @torch.inference_mode()
    def test_boolean_owner_and_kernel_admission(self):
        shared, routed = self.shared(), self.routed()
        original = self.originals()
        for enabled in (1, "yes", None):
            with self.assertRaisesRegex(ValueError, "bool"):
                with scope.expert_projection_reference(enabled):
                    pass
        for args in (
            {},
            {"shared": [object()]},
            {"routed": [object()]},
            {"shared": [shared, shared]},
            {"routed": [routed, routed]},
        ):
            with self.assertRaises(ValueError):
                with scope.expert_projection_reference(
                    True, kernel=self.kernel, **args
                ):
                    pass
        with self.assertRaisesRegex(ValueError, "GEMM provider"):
            with scope.expert_projection_reference(
                True, shared=[shared], kernel=object()
            ):
                pass
        self.assertEqual(self.originals(), original)

    @torch.inference_mode()
    def test_metadata_backend_and_architecture_admission(self):
        shared, routed = self.shared(), self.routed()
        original = self.originals()
        for owner, field, value, argument in (
            (shared, "backend", "eager", "shared"),
            (shared, "N", 128, "shared"),
            (
                shared,
                "scales",
                torch.zeros(7, 4, dtype=torch.uint8, device="cuda"),
                "shared",
            ),
            (routed, "backend", "eager", "routed"),
            (routed, "inter_dim", 64, "routed"),
            (
                routed,
                "s13",
                torch.zeros(3, 255, 4, device="cuda", dtype=torch.uint8),
                "routed",
            ),
        ):
            with patch.object(owner, field, value), self.assertRaises(ValueError):
                with scope.expert_projection_reference(
                    True, kernel=self.kernel, **{argument: [owner]}
                ):
                    pass
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ), self.assertRaisesRegex(ValueError, "SM120"):
            with scope.expert_projection_reference(
                True, shared=[shared], kernel=self.kernel
            ):
                pass
        self.assertEqual(self.originals(), original)

    @torch.inference_mode()
    def test_disabled_nested_and_exception_restoration(self):
        shared, other, routed = self.shared(), self.shared(), self.routed()
        original = self.originals()
        with scope.expert_projection_reference(False, shared=[object()]):
            self.assertEqual(self.originals(), original)
        with scope.expert_projection_reference(
            True, shared=[shared], kernel=self.kernel
        ):
            outer = self.originals()
            self.assertIs(outer[1], original[1])
            with scope.expert_projection_reference(False):
                self.assertEqual(self.originals(), outer)
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with scope.expert_projection_reference(
                    True, shared=[other], routed=[routed], kernel=self.kernel
                ):
                    raise RuntimeError("restore")
            self.assertEqual(self.originals(), outer)
        self.assertEqual(self.originals(), original)

    @torch.inference_mode()
    def test_shared_entry_paths_buffers_and_unselected(self):
        for width in (128, 5120):
            owner, other = self.shared(width), self.shared(width)
            x = torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
            expected, unselected = self.shared_oracle(owner, x), other(x).clone()
            protected = (x.clone(), owner.weight.clone(), owner.scales.clone())
            values, scales = quantize_group32(x)
            out = torch.empty_like(expected)
            with scope.expert_projection_reference(
                True, shared=[owner], kernel=self.kernel
            ):
                self.exact(expected, owner(x))
                self.exact(expected, owner.forward_quantized(values, scales))
                self.assertIs(owner.forward_quantized(values, scales, out=out), out)
                self.exact(expected, out)
                self.exact(unselected, other(x))
            for before, after in zip(protected, (x, owner.weight, owner.scales)):
                self.exact(before, after)

    @torch.inference_mode()
    def test_shared_bad_inputs_and_output_aliases(self):
        owner = self.shared()
        values, scales = quantize_group32(
            torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        )
        with scope.expert_projection_reference(
            True, shared=[owner], kernel=self.kernel
        ):
            with self.assertRaisesRegex(ValueError, "E4M3"):
                owner.forward_quantized(values.float(), scales)
            with self.assertRaisesRegex(ValueError, "scales"):
                owner.forward_quantized(values, scales[:, :3].contiguous())
            with self.assertRaisesRegex(ValueError, "output"):
                owner.forward_quantized(
                    values,
                    scales,
                    out=torch.empty(3, 256, device="cuda", dtype=torch.float32),
                )
            alias = (
                owner.weight.view(torch.uint8)
                .flatten()[: 3 * 256 * 2]
                .view(torch.bfloat16)
                .reshape(3, 256)
            )
            with self.assertRaisesRegex(ValueError, "share"):
                owner.forward_quantized(values, scales, out=alias)

    @torch.inference_mode()
    def test_shared_changed_input_graph_and_empty_rows(self):
        owner = self.shared()
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(3, 256, device="cuda", dtype=torch.bfloat16)
        protected = (owner.weight.clone(), owner.scales.clone())
        with scope.expert_projection_reference(
            True, shared=[owner], kernel=self.kernel
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    owner(x, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                owner(x, out=out)
            for scale in (1.0, 0.0, -0.5):
                x.normal_().mul_(scale)
                expected = self.shared_oracle(owner, x)
                graph.replay()
                self.exact(expected, out)
            self.assertEqual(owner(x[:0]).shape, (0, 256))
        for before, after in zip(protected, (owner.weight, owner.scales)):
            self.exact(before, after)

    @torch.inference_mode()
    def test_routed_masked_received_contiguous_released_oracle(self):
        owner = self.routed()
        x = torch.randn(3, 128, 128, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([1, 3, 0], device="cuda", dtype=torch.int32)
        valid = torch.arange(128, device="cuda")[None] < counts[:, None]
        route = torch.full((3, 128), 0.37, device="cuda", dtype=torch.float32)
        x[~valid], route[~valid] = float("nan"), float("nan")
        expected = self.routed_oracle(owner, x, counts, route)
        protected = tuple(
            t.clone() for t in (x, route, owner.w13, owner.s13, owner.w2, owner.s2)
        )
        with expert_down_reference(
            True, routed=True, kernel=self.kernel
        ), scope.expert_projection_reference(True, routed=[owner], kernel=self.kernel):
            actual = owner.forward_masked(x, counts, route)
            self.exact(expected, actual)
            values, scales = quantize_group32(torch.where(valid[..., None], x, 0))
            received = owner.forward_masked(
                values, counts, route, input_scales=scales.reshape(3, 128, 4)
            )
            self.exact(expected, received)
            contiguous = owner.forward_contiguous(
                torch.cat([x[0], x[1]]),
                (128, 128, 0),
                torch.cat([valid[0], valid[1]]),
                torch.cat([route[0], route[1]]),
            )
            self.exact(torch.cat([expected[0], expected[1]]), contiguous)
        for before, after in zip(
            protected, (x, route, owner.w13, owner.s13, owner.w2, owner.s2)
        ):
            self.exact(before, after)

    @torch.inference_mode()
    def test_routed_unselected_and_scope_restoration(self):
        owner, other = self.routed(), self.routed()
        x = torch.randn(3, 128, 128, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([1, 2, 1], device="cuda", dtype=torch.int32)
        route = torch.full((3, 128), 0.5, device="cuda", dtype=torch.float32)
        baseline, unselected = (
            owner.forward_masked(x, counts, route).clone(),
            other.forward_masked(x, counts, route).clone(),
        )
        original = self.originals()
        with expert_down_reference(
            True, routed=True, kernel=self.kernel
        ), scope.expert_projection_reference(True, routed=[owner], kernel=self.kernel):
            actual = owner.forward_masked(x, counts, route)
            self.exact(self.routed_oracle(owner, x, counts, route), actual)
            with patch.object(
                scope, "_routed_projection", wraps=scope._routed_projection
            ) as selected:
                other.forward_masked(x, counts, route)
                self.assertEqual(selected.call_count, 0)
        self.exact(baseline, owner.forward_masked(x, counts, route))
        self.exact(unselected, other.forward_masked(x, counts, route))
        self.assertEqual(self.originals(), original)

    @torch.inference_mode()
    def test_routed_graph_changed_counts_inputs_routes(self):
        owner = self.routed()
        x = torch.randn(3, 128, 128, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([1, 3, 0], device="cuda", dtype=torch.int32)
        route = torch.full((3, 128), 0.37, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        protected = tuple(t.clone() for t in (owner.w13, owner.s13, owner.w2, owner.s2))
        with expert_down_reference(
            True, routed=True, kernel=self.kernel
        ), scope.expert_projection_reference(True, routed=[owner], kernel=self.kernel):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    owner.forward_masked(x, counts, route, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                owner.forward_masked(x, counts, route, out=out)
            for changed, scale in (
                ((3, 0, 2), 0.5),
                ((0, 0, 0), 0.0),
                ((1, 2, 3), 1.5),
            ):
                counts.copy_(torch.tensor(changed, device="cuda", dtype=torch.int32))
                x.normal_()
                route.fill_(scale)
                expected = self.routed_oracle(owner, x, counts, route)
                graph.replay()
                self.exact(expected, out)
        for before, after in zip(protected, (owner.w13, owner.s13, owner.w2, owner.s2)):
            self.exact(before, after)


if __name__ == "__main__":
    unittest.main()

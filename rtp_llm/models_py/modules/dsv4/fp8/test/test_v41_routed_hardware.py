"""Actual SM120 grouped FP4 provider acceptance, including both EP shard sizes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


class V41RoutedHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Routed expert gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41)

    @staticmethod
    def weights(experts):
        rng = torch.Generator(device="cuda").manual_seed(41340 + experts)
        shapes = {
            "w13_w": (experts, 4608, 2560),
            "w2_w": (experts, 5120, 1152),
            "w13_s": (experts, 4608, 160),
            "w2_s": (experts, 5120, 72),
        }
        return {
            name: torch.randint(
                0 if name.endswith("_w") else 119,
                256 if name.endswith("_w") else 124,
                shape,
                device="cuda",
                dtype=torch.uint8,
                generator=rng,
            )
            for name, shape in shapes.items()
        }

    def assert_output(self, op, tensors, x, counts, route, actual):
        valid = torch.arange(x.shape[1], device="cuda")[None] < counts[:, None]
        active = valid & (route != 0)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool((actual[~active] == 0).all()))
        # Bound CPU diagnostics to occupied experts/rows, not the full EP shard.
        for e, n in enumerate(counts.cpu().tolist()):
            if not n:
                continue
            reference = V41RoutedExperts(
                {name: tensor[e : e + 1].cpu() for name, tensor in tensors.items()},
                backend="eager",
            )
            expected = reference.forward_masked(
                x[e : e + 1, :n].cpu().contiguous(),
                torch.tensor([n], dtype=torch.int32),
                route[e : e + 1, :n].cpu().contiguous(),
            )[0]
            torch.testing.assert_close(
                actual[e, :n].cpu(), expected, rtol=2.5e-2, atol=1.5e-2
            )

    def test_ep48_ep96_masked_contiguous_and_updated_graph_inputs(self):
        for experts in (48, 96):
            tensors = self.weights(experts)
            op = V41RoutedExperts(tensors)
            x = torch.randn(experts, 128, 5120, device="cuda", dtype=torch.bfloat16)
            counts = torch.zeros(experts, dtype=torch.int32, device="cuda")
            counts[0], counts[2], counts[-1] = 1, 5, 3
            valid = torch.arange(128, device="cuda")[None] < counts[:, None]
            route = torch.full((experts, 128), 0.37, dtype=torch.float32, device="cuda")
            x[~valid], route[~valid] = float("nan"), float("nan")
            masked = op.forward_masked(x, counts, route)
            self.assert_output(op, tensors, x, counts, route, masked)

            owners = (0, 2, experts - 1)
            rows = tuple(128 if e in owners else 0 for e in range(experts))
            contiguous = op.forward_contiguous(
                torch.cat([x[e] for e in owners]),
                rows,
                torch.cat([valid[e] for e in owners]),
                torch.cat([route[e] for e in owners]),
            )
            torch.testing.assert_close(
                contiguous,
                torch.cat([masked[e] for e in owners]),
                rtol=2.5e-2,
                atol=1.5e-2,
            )

            out = torch.empty_like(x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    op.forward_masked(x, counts, route, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.forward_masked(x, counts, route, out=out)
            counts.zero_()
            route.zero_()
            graph.replay()
            self.assertTrue(bool((out == 0).all()))
            counts[1] = 2
            route[1, :2] = 1.5
            x[1, :2].normal_()
            graph.replay()
            self.assert_output(op, tensors, x, counts, route, out)
            del graph, op, tensors, x, out, contiguous, masked

    def test_local_routing_fp32_combine_and_graph_route_updates(self):
        tensors = self.weights(48)
        local = V41LocalMoE(V41RoutedExperts(tensors), expert_start_id=48)
        x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([[95, 48, 52]], device="cuda", dtype=torch.int64).repeat(
            4, 1
        )
        weights = torch.tensor(
            [[0.37, 0.25, 0.5]], device="cuda", dtype=torch.float32
        ).repeat(4, 1)
        valid = torch.tensor([True, False, True, True], device="cuda")
        x[~valid] = float("nan")
        references = {
            e: V41RoutedExperts(
                {name: tensor[e : e + 1].cpu() for name, tensor in tensors.items()},
                backend="eager",
            )
            for e in (0, 4, 47)
        }

        def check(actual, layout):
            self.assertEqual(actual.dtype, torch.float32)
            expected = torch.zeros((4, 5120), dtype=torch.float32)
            error_bound = torch.zeros_like(expected)
            gpu_expected = torch.zeros_like(actual)
            expert_x = torch.zeros((48, 128, 5120), dtype=x.dtype, device=x.device)
            expert_weights = torch.zeros((48, 128), device=x.device)
            counts = torch.zeros(48, dtype=torch.int32, device=x.device)
            row_indices = {}
            reference_outputs = {}
            for e in (0, 4, 47):
                route = torch.where(ids == 48 + e, weights, 0).sum(1)
                route = torch.where(valid, route, 0)
                selected = (route != 0).nonzero().flatten()
                row_indices[e] = selected
                n = selected.numel()
                counts[e] = n
                expert_x[e, :n] = x.index_select(0, selected)
                expert_weights[e, :n] = route.index_select(0, selected)
                ref = (
                    references[e]
                    .forward_masked(
                        x.cpu()[None],
                        torch.tensor([4], dtype=torch.int32),
                        route.cpu()[None],
                    )[0]
                    .float()
                )
                reference_outputs[e] = ref
                expected.add_(ref)
                # Summed per-expert tolerances remain valid under cancellation.
                error_bound.add_(1.5e-2 + 2.5e-2 * ref.abs())
            if layout == "masked":
                expert_out = local.experts.forward_masked(
                    expert_x, counts, expert_weights
                )
            else:
                segments = tuple(128 if n else 0 for n in counts.cpu().tolist())
                owners = [e for e, n in enumerate(segments) if n]
                expert_out = torch.zeros_like(expert_x)
                if owners:
                    compact_x = torch.cat([expert_x[e] for e in owners])
                    compact_weights = torch.cat([expert_weights[e] for e in owners])
                    compact_valid = torch.cat(
                        [torch.arange(128, device=x.device) < counts[e] for e in owners]
                    )
                    compact_out = local.experts.forward_contiguous(
                        compact_x, segments, compact_valid, compact_weights
                    )
                    for i, e in enumerate(owners):
                        expert_out[e] = compact_out[i * 128 : (i + 1) * 128]
            for e in (0, 4, 47):
                selected = row_indices[e]
                contribution = torch.zeros_like(actual)
                contribution[selected] = expert_out[e, : selected.numel()].float()
                torch.testing.assert_close(
                    contribution.cpu(), reference_outputs[e], rtol=2.5e-2, atol=1.5e-2
                )
                gpu_expected.add_(contribution)
            torch.testing.assert_close(actual, gpu_expected, rtol=0, atol=0)
            self.assertTrue(
                bool(((actual.cpu() - expected).abs() <= error_bound).all())
            )
            active = valid & (weights != 0).any(1)
            self.assertTrue(bool((actual[~active] == 0).all()))

        check(
            local(x, weights, ids, valid_rows=valid, layout="contiguous"), "contiguous"
        )
        check(local(x, weights, ids, valid_rows=valid, layout="masked"), "masked")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                local(x, weights, ids, valid_rows=valid, layout="masked")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = local(x, weights, ids, valid_rows=valid, layout="masked")
        ids.copy_(ids.flip(1))
        weights.copy_(weights.flip(1))
        graph.replay()
        check(output, "masked")
        weights.zero_()
        graph.replay()
        check(output, "masked")

    def test_received_fp8_ep48_ep96_and_dispatch_graph_updates(self):
        wire = V41DispatchLayout(5120, 6, 384)
        for experts in (48, 96):
            local = V41LocalMoE(
                V41RoutedExperts(self.weights(experts)),
                expert_start_id=experts,
                max_tokens=8,
                masked_token_limit=8,
            )
            x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
            weights = torch.rand(4, 6, device="cuda", dtype=torch.float32)
            ids = torch.tensor(
                [experts, 2 * experts - 1, experts + 4, 0, 1, 383],
                device="cuda",
                dtype=torch.int32,
            ).repeat(4, 1)
            valid = torch.tensor([True, False, True, True], device="cuda")
            x[~valid] = float("nan")
            payload = wire.allocate(8, device="cuda")
            pack_dispatch(x, weights, ids, valid, wire, out=payload)

            def check(actual, layout):
                expected = local(x, weights, ids, valid_rows=valid, layout=layout)
                torch.testing.assert_close(actual[:4], expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(bool((actual[4:] == 0).all()))

            for layout in ("contiguous", "masked"):
                check(local.forward_received(payload, wire, layout=layout), layout)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    pack_dispatch(x, weights, ids, valid, wire, out=payload)
                    local.forward_received(payload, wire, layout="masked")
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pack_dispatch(x, weights, ids, valid, wire, out=payload)
                actual = local.forward_received(payload, wire, layout="masked")
            for route_weight in (0, 0.37, 1.5):
                x.normal_()
                valid.logical_not_()
                weights.fill_(route_weight)
                ids.copy_(ids.flip(1))
                graph.replay()
                check(actual, "masked")
            del graph, local, actual


if __name__ == "__main__":
    unittest.main()

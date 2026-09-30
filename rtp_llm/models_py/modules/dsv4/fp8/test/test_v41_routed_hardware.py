"""Actual SM120 grouped FP4 provider acceptance, including both EP shard sizes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


class V41RoutedHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Routed expert gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

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


if __name__ == "__main__":
    unittest.main()

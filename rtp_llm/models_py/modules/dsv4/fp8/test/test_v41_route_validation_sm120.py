"""Exact route predicates, received views and live updates on SM120."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import v41_dispatch


class V41RouteValidationSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Route validation requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    def inputs(self, rows, topk, dtype=torch.int64, *, strided=False):
        weights = torch.full((rows, topk * 2), 0.37, device="cuda")[:, ::2]
        indices = torch.arange(topk * 2, device="cuda", dtype=dtype)[None, :]
        indices = indices.expand(rows, -1).clone()[:, ::2]
        valid = torch.ones(rows * 2, dtype=torch.bool, device="cuda")[::2]
        if not strided:
            weights, indices, valid = (
                tensor.contiguous() for tensor in (weights, indices, valid)
            )
        return weights, indices, valid

    @staticmethod
    def snapshot(tensor):
        # Empty and singleton views can be contiguous with a nonunit stride.
        copy = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        copy.copy_(tensor.cpu())
        return copy.view(torch.uint8)

    def check(self, weights, indices, valid, experts=384):
        inputs = weights, indices, valid
        cpu = tuple(tensor.cpu() for tensor in inputs)
        expected_mask = cpu[2][:, None] & (cpu[0] != 0)
        before = [self.snapshot(tensor) for tensor in cpu]
        try:
            expected = v41_dispatch._validate_routes(*cpu, experts)
            legal = True
            self.assertTrue(torch.equal(expected, expected_mask))
        except ValueError as error:
            self.assertEqual(
                str(error),
                "V4.1 dispatch requires finite nonnegative weights and unique in-range active IDs",
            )
            legal = False
        assertions = []

        def capture(predicate, message):
            assertions.append((bool(predicate.cpu()), message))

        with patch.object(torch, "_assert_async", capture):
            actual = v41_dispatch._validate_routes(*inputs, experts)
        self.assertEqual(assertions, [(legal, "Invalid V4.1 dispatch route values")])
        self.assertTrue(torch.equal(actual.cpu(), expected_mask))
        for tensor, original in zip(inputs, before):
            self.assertTrue(torch.equal(self.snapshot(tensor), original))
        return legal

    def test_mask_and_predicate_match_cpu_for_strides_and_boundaries(self):
        for rows in (0, 1, 8, 17, 32, 33):
            for topk in (1, 6, 16, 32, 33):
                for dtype in (torch.int32, torch.int64):
                    for strided in (False, True):
                        with self.subTest(
                            rows=rows, topk=topk, dtype=dtype, strided=strided
                        ):
                            inputs = self.inputs(rows, topk, dtype, strided=strided)
                            if rows:
                                inputs[0][-1, -1] = 0
                                inputs[1][-1, -1] = -1
                                inputs[2][0] = False
                            self.assertTrue(self.check(*inputs))

    def test_invalid_weights_ids_and_duplicates_respect_active_rows(self):
        for value in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(weight=value):
                weights, indices, valid = self.inputs(2, 6, strided=True)
                weights[-1, -1] = value
                self.assertFalse(self.check(weights, indices, valid))
                valid[-1] = False
                self.assertTrue(self.check(weights, indices, valid))
        for value in (-1, 384, 2**31, 2**63 - 1, -(2**63)):
            with self.subTest(index=value):
                weights, indices, valid = self.inputs(2, 6, strided=True)
                indices[-1, -1] = value
                self.assertFalse(self.check(weights, indices, valid))
                weights[-1, -1] = 0
                self.assertTrue(self.check(weights, indices, valid))
        weights, indices, valid = self.inputs(2, 6, strided=True)
        indices[-1, -1] = indices[-1, 0]
        self.assertFalse(self.check(weights, indices, valid))
        weights[-1, -1] = -0.0
        self.assertTrue(self.check(weights, indices, valid))
        weights[-1, -1] = 1.0e-45
        self.assertFalse(self.check(weights, indices, valid))

    def test_received_payload_views_revalidate_same_storage(self):
        layout = v41_dispatch.V41DispatchLayout(5120, 6, 384)
        payload = layout.allocate(8, device="cuda")
        _, _, weights, indices = layout.views(payload)
        weights.fill_(0.37)
        indices.copy_(torch.arange(48, device="cuda").view(8, 6))
        valid = torch.ones(8, dtype=torch.bool, device="cuda")
        self.assertTrue(self.check(weights, indices, valid))
        indices[-1, -1] = indices[-1, 0]
        self.assertFalse(self.check(weights, indices, valid))
        indices[-1, -1] = 47
        self.assertTrue(self.check(weights, indices, valid))
        weights[-1, -1] = float("nan")
        self.assertFalse(self.check(weights, indices, valid))
        valid[-1] = False
        self.assertTrue(self.check(weights, indices, valid))

    def test_graph_replay_uses_updated_weights_and_valid_rows(self):
        weights, indices, valid = self.inputs(8, 6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                v41_dispatch._validate_routes(weights, indices, valid, 384)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            active = v41_dispatch._validate_routes(weights, indices, valid, 384)
        for scale in (0, 0.37, 1.5):
            weights.fill_(scale)
            indices.add_(1).remainder_(384)
            valid.logical_not_()
            weights[~valid] = float("nan")
            graph.replay()
            expected = v41_dispatch._validate_routes(
                weights.cpu(), indices.cpu(), valid.cpu(), 384
            )
            self.assertTrue(torch.equal(active.cpu(), expected))


if __name__ == "__main__":
    unittest.main()

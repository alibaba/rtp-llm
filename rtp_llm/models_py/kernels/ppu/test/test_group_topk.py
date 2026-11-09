"""Native routing parity, Graph replay, and the specialization boundary."""

from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.group_topk import try_fused_group_topk
from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK


@skipUnless(is_ppu(), "PPU routing qualification")
class GroupTopKTest(TestCase):
    def setUp(self):
        torch.manual_seed(25683)
        self.router = GroupTopK()

    def reference(self, scores, bias, dtype=torch.int64):
        weights = torch.empty(scores.shape[0], 8, device="cuda")
        ids = torch.empty_like(weights, dtype=dtype)
        probability = scores.float().sigmoid()
        self.router.group_topk_op.forward(
            weights,
            ids,
            probability,
            probability + bias.unsqueeze(0),
            1,
            1,
            8,
            True,
            2.5,
        )
        return weights, ids

    def check(self, scores, bias, dtype=torch.int64):
        expected, expected_ids = self.reference(scores, bias, dtype)
        weights, ids = torch.empty_like(expected), torch.empty_like(expected_ids)
        self.router(weights, ids, scores, bias, 1, 1, 8, True, 2.5)
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(
            torch.equal(weights.view(torch.int32), expected.view(torch.int32))
        )

    def test_exact_random_routes_and_weights(self):
        for rows in (1, 7, 17, 128, 256, 1024):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(rows=rows, dtype=dtype):
                    self.check(
                        torch.randn(rows, 256, device="cuda") * 5,
                        torch.randn(256, device="cuda"),
                        dtype,
                    )

    def test_ieee_values_ties_and_safe_fallback(self):
        bits = torch.randint(0, 2**32, (1024, 256), device="cuda", dtype=torch.int64)
        self.check(
            bits.to(torch.int32).view(torch.float32), torch.zeros(256, device="cuda")
        )
        for value in (0.0, -0.0, float("nan"), float("inf"), -float("inf")):
            self.check(
                torch.full((17, 256), value, device="cuda"),
                torch.zeros(256, device="cuda"),
            )
        scores = torch.zeros(1, 256, device="cuda")
        for value in (
            float("nan"),
            float("inf"),
            -float("inf"),
            torch.finfo(torch.float32).max,
            -torch.finfo(torch.float32).max / 2,
        ):
            self.check(scores, torch.full((256,), value, device="cuda"))
        bias = torch.full((256,), float("nan"), device="cuda")
        for finite_count in (1, 2, 7, 8):
            bias[:finite_count] = 0
            self.check(scores, bias)

    def test_graph_reads_new_logits_and_bias_on_nondefault_stream(self):
        for rows in (1, 17, 128, 1024):
            x, bias = torch.randn(rows, 256, device="cuda"), torch.randn(
                256, device="cuda"
            )
            weights, ids = self.reference(x, bias)
            for _ in range(3):
                self.router(weights, ids, x, bias, 1, 1, 8, True, 2.5)
            torch.cuda.synchronize()
            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=stream):
                self.router(weights, ids, x, bias, 1, 1, 8, True, 2.5)
            for _ in range(3):
                x.normal_()
                bias.normal_()
                torch.cuda.synchronize()
                graph.replay()
                expected, expected_ids = self.reference(x, bias)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(ids, expected_ids))
                self.assertTrue(
                    torch.equal(weights.view(torch.int32), expected.view(torch.int32))
                )

    def test_unsupported_shapes_dtypes_and_configs_fall_back(self):
        x, bias = torch.randn(17, 256, device="cuda"), torch.randn(256, device="cuda")
        weights, ids = self.reference(x, bias)
        args = [weights, ids, x, bias, 1, 1, 8, True, 2.5]
        replacements = [
            (0, weights.to(torch.bfloat16)),
            (1, ids.to(torch.int16)),
            (2, x.to(torch.bfloat16)),
            (3, bias.to(torch.bfloat16)),
            (2, torch.randn(17, 512, device="cuda")[:, ::2]),
            (2, torch.randn(1025, 256, device="cuda")),
            (2, x.cpu()),
            (4, 2),
            (5, 2),
            (6, 4),
            (7, False),
            (8, 1.0),
        ]
        for index, value in replacements:
            changed = list(args)
            changed[index] = value
            self.assertFalse(try_fused_group_topk(*changed))
        # Exercise the real native fallback, rather than only the guard predicate.
        self.check(torch.randn(1025, 256, device="cuda"), bias)
        self.check(x.to(torch.bfloat16), bias)
        self.check(x, bias.to(torch.bfloat16))

    def test_cuda_dispatch_keeps_native_router(self):
        with patch(
            "rtp_llm.models_py.modules.base.cuda.select_topk.is_ppu", return_value=False
        ):
            module = GroupTopK()
        x, bias = torch.randn(17, 256, device="cuda"), torch.randn(256, device="cuda")
        expected, expected_ids = self.reference(x, bias)
        weights, ids = torch.empty_like(expected), torch.empty_like(expected_ids)
        with patch(
            "rtp_llm.models_py.kernels.ppu.group_topk.try_fused_group_topk",
            side_effect=AssertionError("CUDA must retain native routing"),
        ):
            module(weights, ids, x, bias, 1, 1, 8, True, 2.5)
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(
            torch.equal(weights.view(torch.int32), expected.view(torch.int32))
        )

    def test_empty_input_does_not_launch(self):
        self.assertTrue(
            try_fused_group_topk(
                torch.empty(0, 8, device="cuda"),
                torch.empty(0, 8, dtype=torch.int32, device="cuda"),
                torch.empty(0, 256, device="cuda"),
                torch.zeros(256, device="cuda"),
                1,
                1,
                8,
                True,
                2.5,
            )
        )


if __name__ == "__main__":
    main()

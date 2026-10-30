"""Bitwise regression for opt-in register Hadamard against the scratch kernel."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
    fused_q_rope_quant,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RegisterHadamardTest(unittest.TestCase):
    def compare(self, args):
        with patch.dict("os.environ", {"RTP_LLM_CP_REGISTER_Q_HADAMARD": "0"}):
            ref, ref_scale = fused_q_rope_quant(*args)
        with patch.dict("os.environ", {"RTP_LLM_CP_REGISTER_Q_HADAMARD": "1"}):
            actual, scale = fused_q_rope_quant(*args)
        self.assertTrue(torch.equal(ref.view(torch.uint8), actual.view(torch.uint8)))
        self.assertTrue(
            torch.equal(ref_scale.view(torch.int32), scale.view(torch.int32))
        )

    def test_shapes_styles_strides_and_fallbacks(self):
        torch.manual_seed(321)
        for n in (0, 1, 511, 512, 544, 3114):
            for heads in (4, 5, 32):
                for rope in (64, 128):
                    for neox in (False, True):
                        with self.subTest(n=n, heads=heads, rope=rope, neox=neox):
                            q = torch.randn(
                                n, heads * 2, 128, device="cuda", dtype=torch.bfloat16
                            )[:, ::2]
                            pos = torch.arange(n, device="cuda", dtype=torch.int64)
                            cs = torch.randn(max(n, 1) * 2, rope, device="cuda")[::2]
                            self.compare((q, pos, cs, heads, 128, rope, neox))
        # Unsupported register dimensions and dtype retain the original path.
        for dtype in (torch.bfloat16, torch.float32):
            q = torch.randn(544, 32, 64, device="cuda", dtype=dtype)
            self.compare(
                (
                    q,
                    torch.arange(544, device="cuda"),
                    torch.randn(544, 64, device="cuda"),
                    32,
                    64,
                    64,
                    True,
                )
            )

    def test_special_values(self):
        for value in (
            0.0,
            -0.0,
            1e-5,
            100.0,
            1e30,
            float("inf"),
            -float("inf"),
            float("nan"),
        ):
            with self.subTest(value=value):
                q = torch.full(
                    (544, 32, 128), value, device="cuda", dtype=torch.bfloat16
                )
                self.compare(
                    (
                        q,
                        torch.arange(544, device="cuda"),
                        torch.randn(544, 64, device="cuda"),
                        32,
                        128,
                        64,
                        True,
                    )
                )

    def test_graph_replay_and_input_changes(self):
        q = torch.randn(544, 32, 128, device="cuda", dtype=torch.bfloat16)
        args = (
            q,
            torch.arange(544, device="cuda"),
            torch.randn(544, 64, device="cuda"),
            32,
            128,
            64,
            True,
        )
        with patch.dict("os.environ", {"RTP_LLM_CP_REGISTER_Q_HADAMARD": "1"}):
            fused_q_rope_quant(*args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, scale = fused_q_rope_quant(*args)
        for _ in range(5):
            q.normal_()
            graph.replay()
            with patch.dict("os.environ", {"RTP_LLM_CP_REGISTER_Q_HADAMARD": "0"}):
                ref, ref_scale = fused_q_rope_quant(*args)
            self.assertTrue(
                torch.equal(ref.view(torch.uint8), actual.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(ref_scale.view(torch.int32), scale.view(torch.int32))
            )

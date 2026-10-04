# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.modules.qwen4_exp.small_moe_gemv import (
    is_supported,
    small_moe_gemv,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallMoeGemvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("SM120 required")
        torch.manual_seed(1847)
        cls.module = TritonFusedMoeExecutor.__new__(TritonFusedMoeExecutor)
        for name, value in dict(
            ep_size=1,
            E=512,
            N=160,
            K=2560,
            inter_size=80,
            _qwen4_decode_alignment=True,
            _qwen4_decode_tiles=True,
            _qwen4_small_moe_gemv=False,
        ).items():
            setattr(cls.module, name, value)
        cls.module.w1 = (
            torch.randn(512, 160, 2560, device="cuda", dtype=torch.bfloat16) * 0.05
        )
        cls.module.w2 = (
            torch.randn(512, 2560, 80, device="cuda", dtype=torch.bfloat16) * 0.05
        )

    def payload(self, m, spread=512, scale=0.125):
        return ExpertForwardPayload(
            expert_x=torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16) * scale,
            expert_topk_ids=torch.randint(
                spread, (m, 10), device="cuda", dtype=torch.int32
            ),
            expert_topk_weights=torch.rand(m, 10, device="cuda") * 0.1,
        )

    def ref(self, p):
        return self.module.execute(
            p, "silu", None, None, False, None
        ).fused_expert_output

    def test_eager_and_live_graph_routes(self):
        for m in (1, 2, 3, 4):
            p = self.payload(m)
            for _ in range(3):
                small_moe_gemv(
                    p.expert_x,
                    self.module.w1,
                    self.module.w2,
                    p.expert_topk_ids,
                    p.expert_topk_weights,
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = small_moe_gemv(
                    p.expert_x,
                    self.module.w1,
                    self.module.w2,
                    p.expert_topk_ids,
                    p.expert_topk_weights,
                )
            for spread, scale in ((512, 0.125), (1, 0.125), (3, 1.0), (512, 0.0)):
                with self.subTest(m=m, spread=spread, scale=scale):
                    p.expert_x.copy_(torch.randn_like(p.expert_x) * scale)
                    p.expert_topk_ids.copy_(
                        torch.randint(spread, (m, 10), device="cuda", dtype=torch.int32)
                    )
                    p.expert_topk_weights.copy_(
                        torch.rand_like(p.expert_topk_weights) * 0.1
                    )
                    expected = self.ref(p)
                    graph.replay()
                    torch.testing.assert_close(
                        actual, expected, atol=0.0078125, rtol=0.008
                    )

    def test_unsupported_falls_back_before_launch(self):
        p = self.payload(1)
        args = [
            p.expert_x,
            self.module.w1,
            self.module.w2,
            p.expert_topk_ids,
            p.expert_topk_weights,
        ]
        for index, tensor in (
            (0, p.expert_x.float()),
            (0, p.expert_x[:, ::2]),
            (3, p.expert_topk_ids.long()),
            (4, p.expert_topk_weights.half()),
        ):
            with self.subTest(index=index, dtype=tensor.dtype):
                values = list(args)
                values[index] = tensor
                self.assertFalse(is_supported(*values))
                self.assertIsNone(small_moe_gemv(*values))
        p = self.payload(5)
        self.assertIsNone(
            small_moe_gemv(
                p.expert_x,
                self.module.w1,
                self.module.w2,
                p.expert_topk_ids,
                p.expert_topk_weights,
            )
        )

    def test_executor_dispatch_and_nondefault_semantics(self):
        p = self.payload(1)
        expected = self.ref(p)
        with patch.object(self.module, "_qwen4_small_moe_gemv", True):
            from rtp_llm.models_py.modules.qwen4_exp.small_moe_gemv import (
                small_moe_gemv as candidate,
            )

            for activation in ("silu", "SiGLU"):
                with patch(
                    "rtp_llm.models_py.modules.qwen4_exp.small_moe_gemv.small_moe_gemv",
                    wraps=candidate,
                ) as mock:
                    self.module.execute(p, activation, None, None, False, None)
                    mock.assert_called_once()
            actual = self.ref(p)
            torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
            for activation, expert_map, scale, apply, extra in (
                ("gelu", None, None, False, None),
                ("silu", p.expert_topk_ids, None, False, None),
                ("silu", None, p.expert_topk_weights, False, None),
                ("silu", None, None, True, None),
                ("silu", None, None, False, {"custom": True}),
            ):
                with patch(
                    "rtp_llm.models_py.modules.qwen4_exp.small_moe_gemv.small_moe_gemv"
                ) as mock:
                    self.module.execute(p, activation, expert_map, scale, apply, extra)
                    mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

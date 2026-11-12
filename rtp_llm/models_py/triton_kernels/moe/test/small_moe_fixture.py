import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)


class SmallMoeGemvTestFixture:
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
            _decode_alignment=True,
            _decode_tiles=True,
            _small_moe_gemv=False,
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

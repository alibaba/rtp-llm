"""Numerical contract for the DeepGEMM-independent FP8 MoE executor."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fp8_per_block_executor import (
    TritonFp8PerBlockExecutor,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class TritonFp8PerBlockExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("FP8 tensor cores require SM89 or newer")
        torch.manual_seed(20261002)
        model = ModelConfig()
        model.quant_config = Fp8BlockWiseQuantConfig()
        model.data_type = "bf16"
        model.expert_num = 4
        model.moe_k = 2
        model.hidden_size = 256
        model.moe_inter_size = 128
        model.activation_type = "SiGLU"
        parallel = ParallelismConfig()
        parallel.world_size = 1
        parallel.local_world_size = 1
        parallel.tp_size = 1
        parallel.ep_size = 1
        moe = MoeConfig()
        moe.use_all_gather = True
        self.config = MoEConfigAdapter(model, parallel, moe)

        w1 = torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.04
        w2 = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.04
        # Different N and K blocks expose mistakes in scale indexing.
        w1[:, 128:, :] *= 3
        w1[:, :, 128:] *= 2
        w2[:, 128:, :] *= 2
        q1, s1, q2, s2 = [], [], [], []
        for expert in range(4):
            q, s = per_block_cast_to_fp8(w1[expert], use_ue8m0=False)
            q1.append(q)
            s1.append(s)
            q, s = per_block_cast_to_fp8(w2[expert], use_ue8m0=False)
            q2.append(q)
            s2.append(s)
        self.weights = {
            W.moe_w1: torch.stack(q1),
            W.moe_s1: torch.stack(s1),
            W.moe_w2: torch.stack(q2),
            W.moe_s2: torch.stack(s2),
        }
        self.executor = TritonFp8PerBlockExecutor(
            self.config,
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
            ),
            self.weights,
        )

    @staticmethod
    def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight.float() * scale.repeat_interleave(128, -2).repeat_interleave(
            128, -1
        )

    def test_matches_dequantized_reference_and_ignores_remote_routes(self) -> None:
        hidden = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16) * 0.3
        hidden[:, 128:] *= 4
        x, x_scale = sgl_per_token_group_quant_fp8(hidden, group_size=128)
        ids = torch.tensor([[0, 3], [2, -1], [1, 0]], device="cuda", dtype=torch.int32)
        weights = torch.tensor(
            [[0.25, 0.75], [0.4, 0.6], [0.7, 0.3]],
            device="cuda",
            dtype=torch.float32,
        )
        payload = ExpertForwardPayload(
            expert_x=x,
            expert_x_scale=x_scale,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=ids,
            expert_topk_weights=weights,
        )
        actual = self.executor.execute(
            payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        ).fused_expert_output

        x_dequant = x.float() * x_scale.repeat_interleave(128, -1)
        w1 = self._dequant_weight(self.weights[W.moe_w1], self.weights[W.moe_s1])
        w2 = self._dequant_weight(self.weights[W.moe_w2], self.weights[W.moe_s2])
        first_stage = []
        for token in range(ids.shape[0]):
            for route in range(ids.shape[1]):
                expert = max(0, int(ids[token, route]))
                gate_up = (x_dequant[token] @ w1[expert].T).to(torch.bfloat16)
                activated = (F.silu(gate_up[128:].float()) * gate_up[:128].float()).to(
                    torch.bfloat16
                )
                first_stage.append(activated)
        down_input = torch.stack(first_stage)
        down_fp8, down_scale = sgl_per_token_group_quant_fp8(down_input, group_size=128)
        down_dequant = down_fp8.float() * down_scale.repeat_interleave(128, -1)
        route_output = []
        for token in range(ids.shape[0]):
            for route in range(ids.shape[1]):
                expert = int(ids[token, route])
                if expert < 0:
                    route_output.append(
                        torch.zeros(256, device="cuda", dtype=torch.bfloat16)
                    )
                    continue
                out = (
                    (down_dequant[token * ids.shape[1] + route] @ w2[expert].T)
                    * weights[token, route]
                ).to(torch.bfloat16)
                route_output.append(out)
        expected = torch.stack(route_output).view(3, 2, 256).sum(dim=1)
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.035)


if __name__ == "__main__":
    unittest.main()

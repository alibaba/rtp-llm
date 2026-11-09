"""GPU-only expert counts must preserve ragged MoE output and padding."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.ppu.fp8_quant import fp8_quantize_128
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.executors.deepgemm_hybrid_executor import (
    DeepGemmFp8HybridExecutor,
)
from rtp_llm.utils.model_weight import W


class Fp8NormalLayoutTest(TestCase):
    def test_device_counts_static_capacity_and_local_ids_match_exact_layout(self):
        torch.manual_seed(1792)
        torch.set_num_threads(1)
        hidden, intermediate, experts = 512, 512, 4
        weights = {
            W.moe_w1: torch.randn(experts, intermediate, hidden, device="cuda").to(
                torch.float8_e4m3fn
            ),
            W.moe_w2: torch.randn(experts, hidden, intermediate // 2, device="cuda").to(
                torch.float8_e4m3fn
            ),
            W.moe_s1: torch.rand(
                experts, intermediate // 128, hidden // 128, device="cuda"
            )
            * 0.01,
            W.moe_s2: torch.rand(
                experts, hidden // 128, intermediate // 256, device="cuda"
            )
            * 0.01,
        }
        config = SimpleNamespace(
            ep_size=2,
            ep_rank=1,
            expert_num=8,
            moe_k=2,
            activation_type="silu",
            masked_max_token_num=0,
        )
        with patch.object(DeepGemmFp8HybridExecutor, "_warmup_fp8_kernels"):
            executor = DeepGemmFp8HybridExecutor(
                config,
                FusedMoEQuantConfig(
                    quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
                ),
                weights,
            )
        self.assertTrue(executor.use_nopad)
        for rows in [0, 1, 17, 129]:
            x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(0, experts - 1, (rows, 2), device="cuda")
            ids[:, 1] = (ids[:, 0] + 1) % (experts - 1)
            # Leave the last expert empty and route some rows nowhere.
            ids[::3] = -1
            weights_per_token = torch.rand(rows, 2, device="cuda")
            q, scales = fp8_quantize_128(x)
            counts = torch.stack(
                [(ids == expert).sum() for expert in range(experts)]
            ).int()
            cpu_counts = counts.cpu().tolist()

            def forward(static):
                capacity = rows * 2 if static else rows
                data = torch.zeros(
                    capacity, hidden, device="cuda", dtype=torch.float8_e4m3fn
                )
                data[:rows].copy_(q)
                data_scales = torch.ones(capacity, hidden // 128, device="cuda")
                data_scales[:rows].copy_(scales)
                selected = torch.full(
                    (capacity, 2), -1, device="cuda", dtype=torch.int64
                )
                selected[:rows].copy_(
                    torch.where(ids >= 0, ids if static else ids + experts, -1)
                )
                routing_weights = torch.zeros(capacity, 2, device="cuda")
                routing_weights[:rows].copy_(weights_per_token)
                payload = ExpertForwardPayload(
                    expert_x=data,
                    expert_x_scale=data_scales,
                    expert_x_origin_dtype=torch.bfloat16,
                    expert_topk_ids=selected,
                    expert_topk_weights=routing_weights,
                    expert_ids_are_local=static,
                    expert_tokens_meta=ExpertTokensMetadata(
                        expert_num_tokens=counts,
                        expert_num_tokens_cpu=None if static else cpu_counts,
                    ),
                )
                return executor.execute(
                    payload, "silu", None, None, False, None
                ).fused_expert_output

            expected, actual = forward(False), forward(True)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertTrue(torch.equal(expected, actual[:rows]))
            self.assertEqual(torch.count_nonzero(actual[rows:]).item(), 0)


if __name__ == "__main__":
    main()

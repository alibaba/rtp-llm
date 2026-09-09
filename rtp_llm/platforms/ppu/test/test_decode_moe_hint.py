"""Launch geometry must not change engine dispatch capacity or route width."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.platforms.ppu.models.dsv4 import pluggable_builders
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import PpuDeepEPFP4Strategy


class DecodeMoeHintTest(unittest.TestCase):
    def test_instance_policy_preserves_dispatch_contract(self):
        cfg = MoeCfg(
            layer_id=0,
            dim=4096,
            moe_inter_dim=2048,
            n_routed_experts=256,
            n_activated_experts=6,
            swiglu_limit=10.0,
            ep_size=8,
            ep_rank=0,
            n_local_experts=32,
            local_expert_start=0,
            local_expert_end=32,
            max_tokens_per_rank=128,
        )
        options = {
            "DSV4_PPU_DECODE_MOE_HINT": "capacity",
            "DSV4_PPU_DECODE_MOE_OUTPUT": "bf16",
        }
        provider = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_MOE_HINT"] = "batch"
        options["DSV4_PPU_DECODE_MOE_OUTPUT"] = "fp32"

        def factory(**kwargs):
            return kwargs["strategy_type"](cfg, **kwargs["strategy_kwargs"])

        context = SimpleNamespace(
            selection=SimpleNamespace(
                model_metadata={"execution_options": provider.execution_options}
            )
        )
        with patch.object(
            pluggable_builders.baseline, "build_moe", side_effect=factory
        ):
            strategy = pluggable_builders.build_decode_moe(
                build_ctx=context,
                request=object(),
                platform_provider=provider,
                tp_size=1,
                ep_size=8,
                is_decode_role=True,
            )
        capacity = PpuDeepEPFP4Strategy(cfg)
        self.assertEqual(capacity.output_dtype, torch.float32)
        self.assertEqual(strategy.output_dtype, torch.bfloat16)
        buffer = object()
        strategy._wrapper = SimpleNamespace(
            buffer=buffer, ll_num_max_token_per_rank=256
        )
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        target = "rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency.low_latency_mxfp4_moe"
        for batch in (0, 1, 8, 32, 128):
            x = torch.empty((batch, cfg.dim), dtype=torch.bfloat16)
            weights = torch.empty((batch, 6))
            indices = torch.empty((batch, 6), dtype=torch.int64)
            with patch(target) as launch:
                strategy(x, weights, indices)
            self.assertIs(launch.call_args.args[0], buffer)
            self.assertIs(launch.call_args.args[2], weights)
            self.assertIs(launch.call_args.args[3], indices)
            self.assertEqual(launch.call_args.kwargs["max_dispatch_tokens"], 256)
            self.assertEqual(launch.call_args.kwargs["expected_m"], 24)
            self.assertEqual(launch.call_args.kwargs["output_dtype"], torch.bfloat16)
            self.assertEqual(capacity.expected_rows(batch), 24)
        with self.assertRaisesRegex(ValueError, "MoE hint"):
            PpuDecodeProvider({"DSV4_PPU_DECODE_MOE_HINT": "unknown"})
        with self.assertRaisesRegex(ValueError, "rows policy"):
            PpuDeepEPFP4Strategy(cfg, expected_m_policy="unknown")
        with self.assertRaisesRegex(ValueError, "MoE output"):
            PpuDecodeProvider({"DSV4_PPU_DECODE_MOE_OUTPUT": "fp16"})
        with self.assertRaisesRegex(ValueError, "routed output"):
            PpuDeepEPFP4Strategy(cfg, output_dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()

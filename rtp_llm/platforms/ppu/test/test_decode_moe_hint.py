"""Launch geometry must not change engine dispatch capacity or route width."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import PpuMoeConfig as MoeCfg
from rtp_llm.platforms.ppu.models.dsv4 import pluggable_builders
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import PpuDeepEPFP4Strategy
from rtp_llm.platforms.ppu.kernels.ppu_mxfp4_masked import _decode_tile_configs


class DecodeMoeHintTest(unittest.TestCase):
    def test_tile_is_opt_in_and_model_shape_bounded(self):
        self.assertIsNone(_decode_tile_configs("auto", 32, 4096, 2048, 8))
        self.assertIsNone(_decode_tile_configs("n128", 32, 4096, 2048, 17))
        self.assertIsNone(_decode_tile_configs("n128", 64, 4096, 2048, 8))
        self.assertIsNone(_decode_tile_configs("n128", 32, 8192, 2048, 8))
        self.assertIsNone(_decode_tile_configs("n128", 32, 4096, 4096, 8))
        with self.assertRaisesRegex(ValueError, "GEMM tile"):
            _decode_tile_configs("unknown", 32, 4096, 2048, 8)
        with patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=64), patch(
            "deep_gemm.jit_kernels.gemm_fp4.get_smem_config_fp4", return_value=(1, 2, 3)
        ) as smem:
            for hint in (1, 8, 16):
                self.assertEqual(_decode_tile_configs("n128", 32, 4096, 2048, hint),
                                 (64, 32, 128, 128, 32, 64, 2, (1, 2, 3)))
            smem.assert_called_with(2, 32, 128, 32, 64, 128)

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
        options = dict(DECODE_EXECUTION_OPTIONS)
        provider = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_MOE_HINT"] = "batch"
        options["DSV4_PPU_DECODE_MOE_OUTPUT"] = "fp32"

        def factory(**kwargs):
            return kwargs["strategy_type"](cfg, **kwargs["strategy_kwargs"])

        context = SimpleNamespace(
            selection=SimpleNamespace(
                model_metadata={
                    "execution_options": provider.execution_options,
                    "hidden_size": cfg.dim,
                    "tp_size": 1,
                }
            )
        )
        request = SimpleNamespace(
            module_id="rtp.dsv4.moe",
            path="layers.0.ffn",
            metadata={"layer_id": cfg.layer_id},
        )
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe.PpuEPMoE",
            side_effect=factory,
        ):
            strategy = pluggable_builders.build_decode_moe(
                build_ctx=context,
                request=request,
                platform_provider=provider,
                layer_id=cfg.layer_id,
                dim=cfg.dim,
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
            self.assertEqual(launch.call_args.kwargs["gemm_tile"], "auto")
            self.assertEqual(capacity.expected_rows(batch), 24)
        batch_provider = PpuDecodeProvider(
            {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_MTP_MOE_HINT": "batch",
             "DSV4_PPU_MTP_MOE_TILE": "n128"}
        )
        context.selection.model_metadata["execution_options"] = batch_provider.execution_options
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe.PpuEPMoE",
            side_effect=factory,
        ):
            batch_strategy = pluggable_builders.build_decode_moe(
                build_ctx=context, request=request, platform_provider=batch_provider,
                layer_id=cfg.layer_id, dim=cfg.dim, tp_size=1, ep_size=8,
                is_decode_role=True,
            )
        batch_strategy._wrapper = strategy._wrapper
        batch_strategy._w13 = batch_strategy._s13 = object()
        batch_strategy._w2 = batch_strategy._s2 = object()
        for local_rows, expected_hint in ((0, 1), (1, 1), (8, 2), (32, 6), (128, 24)):
            x = torch.empty((local_rows, cfg.dim), dtype=torch.bfloat16)
            weights = torch.empty((local_rows, 6))
            indices = torch.empty((local_rows, 6), dtype=torch.int64)
            with patch(target) as launch:
                batch_strategy(x, weights, indices)
            self.assertEqual(launch.call_args.kwargs["expected_m"], expected_hint)
            self.assertEqual(launch.call_args.kwargs["gemm_tile"], "n128")
            self.assertEqual(launch.call_args.kwargs["max_dispatch_tokens"], 256)
            self.assertIs(launch.call_args.args[0], buffer)
            self.assertIs(launch.call_args.args[2], weights)
            self.assertIs(launch.call_args.args[3], indices)
            self.assertEqual(capacity.expected_rows(local_rows), 24)
        with self.assertRaisesRegex(ValueError, "DSV4_PPU_MTP_MOE_HINT"):
            PpuDecodeProvider(
                {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_MTP_MOE_HINT": "unknown"}
            )
        with self.assertRaisesRegex(ValueError, "DSV4_PPU_DECODE_MOE_HINT"):
            PpuDecodeProvider(
                {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_HINT": "unknown"}
            )
        with self.assertRaisesRegex(ValueError, "rows policy"):
            PpuDeepEPFP4Strategy(cfg, expected_m_policy="unknown")
        with self.assertRaisesRegex(ValueError, "GEMM tile"):
            PpuDeepEPFP4Strategy(cfg, gemm_tile="unknown")
        with self.assertRaisesRegex(ValueError, "DSV4_PPU_MTP_MOE_TILE"):
            PpuDecodeProvider({**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_MTP_MOE_TILE": "unknown"})
        with self.assertRaisesRegex(ValueError, "DSV4_PPU_DECODE_MOE_OUTPUT"):
            PpuDecodeProvider(
                {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_OUTPUT": "fp16"}
            )
        with self.assertRaisesRegex(ValueError, "routed output"):
            PpuDeepEPFP4Strategy(cfg, output_dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()

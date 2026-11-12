import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import Nvfp4MarlinMoE
from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.test.marlin_fixture import (
    MarlinNvfp4TestFixture,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.marlin_nvfp4_executor import (
    Qwen4DecodeMarlinExecutor,
    _linear_scales,
)
from rtp_llm.models_py.modules.qwen4_exp.weight_views import PreparedWeightDict
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(
    Nvfp4MarlinMoE.is_supported(),
    "requires a supported NVFP4 Marlin runtime and GPU",
)
class MarlinNvfp4Test(MarlinNvfp4TestFixture, unittest.TestCase):

    @unittest.skipUnless(
        os.environ.get("QWEN4_NVFP4_FIXTURE"), "real NVFP4 fixture not supplied"
    )
    def test_checkpoint_experts_against_dequantized_reference(self):
        from safetensors import safe_open

        with safe_open(os.environ["QWEN4_NVFP4_FIXTURE"], framework="pt") as tensors:
            for prefix in ("model.language_model", "mtp"):

                def load(projection, suffix):
                    return torch.stack(
                        [
                            tensors.get_tensor(
                                f"{prefix}.layers.0.mlp.experts.{expert}.{projection}_proj.{suffix}"
                            )
                            for expert in (0, 117, 0, 117)
                        ]
                    ).cuda()

                self.q1 = torch.cat((load("up", "weight"), load("gate", "weight")), 1)
                self.q2 = load("down", "weight")
                self.s1 = torch.cat(
                    (load("up", "weight_scale"), load("gate", "weight_scale")), 1
                )
                self.s2 = load("down", "weight_scale")
                self.g1, self.g2 = load("up", "weight_scale_2"), load(
                    "down", "weight_scale_2"
                )
                torch.testing.assert_close(
                    self.g1, load("gate", "weight_scale_2"), rtol=0, atol=0
                )
                lut = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                    device="cuda",
                )

                def dequant(q, scale, outer):
                    codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                    return (
                        lut[codes]
                        * scale.float().repeat_interleave(16, -1)
                        * outer[:, None, None]
                    )

                self.w1 = dequant(self.q1, self.s1, self.g1)
                self.w2 = dequant(self.q2, self.s2, self.g2)
                for rank in (0, 1):
                    model = Nvfp4MarlinMoE(
                        self.q1, self.s1, self.g1, self.q2, self.s2, self.g2, 2, rank
                    )
                    for rows in (1, 7, 33):
                        with self.subTest(prefix=prefix, rank=rank, rows=rows):
                            x, ids, weights = self.inputs(rows, rank)
                            actual = model(x, ids, weights)
                            expected = self.reference(x, ids, weights, rank)
                            self.assert_reference(actual, expected)
                            rmse = (
                                (
                                    (actual.float() - expected).square().mean()
                                    / expected.square().mean().clamp_min(1e-30)
                                )
                                .sqrt()
                                .item()
                            )
                            print(
                                "REAL_NVFP4_RELATIVE_RMSE",
                                prefix,
                                rank,
                                rows,
                                rmse,
                                flush=True,
                            )

    def test_repack_reuses_original_storage_and_executor_contract(self):
        def swizzle(sf):
            e, n, k = sf.shape
            return (
                sf.view(torch.uint8)
                .reshape(e, n // 128, 4, 32, k // 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .contiguous()
                .reshape(e, n, k)
                .view(torch.int32)
            )

        q1, q2 = self.q1.clone(), self.q2.clone()
        sf1, sf2 = swizzle(self.s1), swizzle(self.s2)
        torch.testing.assert_close(
            _linear_scales(sf1, 4, 1280, 160).float(), self.s1.float(), rtol=0, atol=0
        )
        native = SimpleNamespace(
            config=None,
            quant_config=None,
            weights=PreparedWeightDict(),
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            w1=q1.view(torch.int64),
            w2=q2.view(torch.int64),
            block_scales=(sf1, sf2),
        )
        native.weights.update(
            {
                W.moe_w1: q1,
                W.moe_w2: q2,
                W.moe_s1: sf1,
                W.moe_s2: sf2,
                W.moe_w1_s2: self.g1,
                W.moe_w2_s2: self.g2,
            }
        )
        ptrs = [v.data_ptr() for v in (q1, q2, sf1, sf2)]
        executor = Qwen4DecodeMarlinExecutor(native)
        self.assertEqual(
            ptrs,
            [
                v.data_ptr()
                for v in (
                    executor.moe.w1,
                    executor.moe.w2,
                    executor.moe.s1,
                    executor.moe.s2,
                )
            ],
        )
        args = self.inputs(4, 1)
        payload = ExpertForwardPayload(
            expert_x=args[0], expert_topk_ids=args[1], expert_topk_weights=args[2]
        )
        actual = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output
        self.assert_reference(actual, self.reference(*args, 1))
        self.assertIs(executor.weights, native.weights)
        self.assertIsNotNone(native.weights[W.moe_w1]._qwen4_marlin_layout)
        del executor
        # A later model can reuse the packed tensors after the first model dies.
        # It must rebuild views and workspaces without repacking their bytes.
        for _ in range(2):
            executor = Qwen4DecodeMarlinExecutor(native)
            repeated = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
            torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
            one_row = self.inputs(1, 1)
            one_payload = ExpertForwardPayload(
                expert_x=one_row[0],
                expert_topk_ids=one_row[1],
                expert_topk_weights=one_row[2],
            )
            one_output = executor.execute(
                one_payload, "silu", None, None, False, None
            ).fused_expert_output
            self.assert_reference(one_output, self.reference(*one_row, 1))
        for activation, expert_map, apply_input in [
            ("relu", None, False),
            ("silu", torch.zeros(8, device="cuda"), False),
            ("silu", None, True),
        ]:
            with self.assertRaises(ValueError):
                executor.execute(
                    payload, activation, expert_map, None, apply_input, None
                )

    def test_explicit_strategy_uses_prepared_weights(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.quant_config import (
            QWEN4_FP8_NVFP4_RECIPE,
            ModelOptFp4Config,
        )
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaNvfp4MarlinStrategy,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        model = ModelConfig()
        model.model_type = "qwen4_exp"
        model.data_type = "bf16"
        model.expert_num = 8
        model.hidden_size, model.moe_inter_size, model.moe_k = self.k, self.n, 3
        model.quant_config = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=False,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
            source_fp8_block_size=128,
            require_contiguous_experts=True,
        )
        pc = ParallelismConfig()
        pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = 2
        set_parallelism_config(pc)
        from rtp_llm.config.qwen4_exp_config import resolve_qwen4_parallelism

        pc = resolve_qwen4_parallelism(model, pc)
        moe = MoeConfig()
        moe.moe_strategy, moe.fp4_moe_op = "nvfp4_marlin", "marlin"
        moe.use_all_gather, moe.use_deepep_moe = True, False
        config = MoEConfigAdapter(model, pc, moe)
        strategy = CudaNvfp4MarlinStrategy()
        config.moe_strategy = "auto"
        with patch.object(
            strategy,
            "get_attributes",
            side_effect=AssertionError("auto imported Marlin"),
        ):
            self.assertFalse(strategy.can_handle(config))
        config.moe_strategy = "nvfp4_marlin"
        self.assertTrue(strategy.can_handle(config))
        factory = FusedMoeFactory()
        self.assertIsInstance(
            factory.registry.get_strategy(config), CudaNvfp4MarlinStrategy
        )
        weights = PreparedWeightDict(
            {
                W.moe_w1: self.q1.clone(),
                W.moe_w2: self.q2.clone(),
                W.moe_s1: CudaImpl.swizzle_blockscale(self.s1.cpu()),
                W.moe_s2: CudaImpl.swizzle_blockscale(self.s2.cpu()),
                W.moe_w1_s2: self.g1,
                W.moe_w2_s2: self.g2,
                W.moe_w1_i_s: torch.ones(self.experts, device="cuda"),
                W.moe_w2_i_s: torch.ones(self.experts, device="cuda"),
            }
        )
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "cutlass_nvfp4_executor.CutlassNvfp4Executor.__init__",
            side_effect=AssertionError("Marlin must not initialize a CUTLASS executor"),
        ):
            executor = factory.create_fused_moe(config, weights).fused_experts
        self.assertIs(executor.weights, weights)
        for rows in (1, 64, 256):
            args = self.inputs(rows)
            payload = ExpertForwardPayload(
                expert_x=args[0], expert_topk_ids=args[1], expert_topk_weights=args[2]
            )
            actual = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
            self.assert_reference(actual, self.reference(*args, 0))
        model.model_type = "other"
        self.assertFalse(strategy.can_handle(config))
        model.model_type = "qwen4_exp"
        moe.fp4_moe_op = "cutlass"
        self.assertFalse(strategy.can_handle(config))


if __name__ == "__main__":
    unittest.main()

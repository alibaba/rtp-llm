"""CPU contract tests for registry admission and physical-M dispatch."""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.config.qwen4_exp_config import build_qwen4_quant_config
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor import (
    Nvfp4HybridExecutor,
    decode_tuning,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
    CudaNvfp4CutlassStrategy,
    CudaNvfp4HybridStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)

_FORCED_BACKEND = "dynamic"

MODULE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor"


def config_for(
    role="DECODE", ep=2, hidden=2560, intermediate=640, experts=512, top_k=10
):
    quant = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
    pc = SimpleNamespace(
        role_type=role,
        tp_size=ep,
        ep_size=ep,
        dp_size=1,
        pp_size=1,
        get_ffn_tp_size=lambda: 1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            quant_config=quant,
            data_type=torch.bfloat16,
            eplb_config=SimpleNamespace(enable_eplb=lambda: False),
        ),
        parallelism_config=pc,
        ep_size=ep,
        tp_size=ep,
        moe_quant_method="modelopt_fp4",
        moe_strategy="nvfp4_cutlass",
        has_redundant_experts=False,
        moe_config=SimpleNamespace(
            fp4_moe_op="cutlass",
            moe_strategy="nvfp4_cutlass",
            use_all_gather=True,
            use_deepep_moe=False,
        ),
        hidden_size=hidden,
        moe_inter_dim=intermediate,
        n_local_experts=experts // ep,
        expert_num=experts,
        moe_k=top_k,
        moe_w1_layout="up_gate",
    )


class Nvfp4HybridExecutorTest(unittest.TestCase):
    def test_registry_selects_by_stage_and_conditions_without_model_name(self):
        registry = StrategyRegistry()
        baseline, hybrid = CudaNvfp4CutlassStrategy(), CudaNvfp4HybridStrategy()
        registry.register(baseline)
        registry.register(hybrid)
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ):
            for enabled, role, expected in [
                ("1", "DECODE", hybrid),
                ("1", "PREFILL", baseline),
                ("1", "PDFUSION", baseline),
                ("0", "DECODE", baseline),
            ]:
                with self.subTest(enabled=enabled, role=role), patch.dict(
                    "os.environ", {"RTP_NVFP4_HYBRID_MOE": enabled}
                ):
                    self.assertIs(
                        registry.get_strategy(config_for(role=role)), expected
                    )
            with patch.dict("os.environ", {"RTP_NVFP4_HYBRID_MOE": "1"}):
                for ep, k, n, e, top_k in [
                    (1, 2048, 1024, 64, 4),
                    (4, 4096, 1536, 128, 8),
                ]:
                    self.assertIs(
                        registry.get_strategy(
                            config_for(
                                ep=ep, hidden=k, intermediate=n, experts=e, top_k=top_k
                            )
                        ),
                        hybrid,
                    )
                self.assertIs(
                    registry.get_strategy(config_for(intermediate=192)), baseline
                )
                config = config_for()
                config.moe_w1_layout = "gate_up"
                self.assertIs(registry.get_strategy(config), baseline)
                with patch("torch.cuda.get_device_capability", return_value=(12, 1)):
                    self.assertIs(registry.get_strategy(config_for()), baseline)

    def test_dispatch_uses_actual_rows_and_configured_boundary(self):
        executor = Nvfp4HybridExecutor.__new__(Nvfp4HybridExecutor)
        executor.hidden_size = 2560
        executor.top_k = 10
        executor.max_b12x_rows = 16
        executor.cutlass_profiles = [13, 37]
        executor.ep_size = 2
        executor.ep_rank = 0
        executor.tp_size = 1
        executor.tp_rank = 0
        executor.w1 = object()
        executor.w2 = object()
        executor.quant_scales = []
        executor.b12x = Mock(side_effect=lambda x, ids, weights: x + 1)
        with patch(
            MODULE + ".cutlass_fused_moe",
            side_effect=lambda **kwargs: [kwargs["input"] - 1],
        ) as cutlass:
            # Includes the threshold and its immediate neighbor, not request concurrency.
            for rows in (1, 2, 8, 16, 17, 32, 64):
                x = torch.zeros(rows, 2560, dtype=torch.bfloat16)
                payload = ExpertForwardPayload(
                    expert_x=x,
                    expert_topk_ids=torch.zeros(rows, 10, dtype=torch.int64),
                    expert_topk_weights=torch.ones(rows, 10),
                )
                output = executor.execute(
                    payload, "SiGLU", None, None, False, None
                ).fused_expert_output
                self.assertTrue(torch.equal(output, x + (1 if rows <= 16 else -1)))
            self.assertEqual(executor.b12x.call_count, 4)
            self.assertEqual(cutlass.call_count, 3)
            self.assertEqual(cutlass.call_args.kwargs["profile_ids"], [13, 37])
            executor.max_b12x_rows = 8
            executor.cutlass_profiles = None
            payload.expert_x = torch.zeros(16, 2560, dtype=torch.bfloat16)
            payload.expert_topk_ids = torch.zeros(16, 10, dtype=torch.int64)
            payload.expert_topk_weights = torch.ones(16, 10)
            executor.execute(payload, "SiGLU", None, None, False, None)
            self.assertNotIn("profile_ids", cutlass.call_args.kwargs)
            payload.expert_ids_are_local = True
            with self.assertRaisesRegex(ValueError, "global expert IDs"):
                executor.execute(payload, "SiGLU", None, None, False, None)

    def test_b12x_adapter_derives_geometry_and_isolates_workspace_shapes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            B12xNvfp4MoE,
        )

        def launch(*, quant_mode):
            return _FORCED_BACKEND

        def weight_views(*, branches, **kwargs):
            return dict(branches=branches, **kwargs)

        convert = Mock(side_effect=lambda value, **kwargs: value)
        dispatch = SimpleNamespace(
            launch_sm120_static_moe=launch, _get_weight_views=weight_views
        )
        modules = {
            "flashinfer.cute_dsl.utils": SimpleNamespace(
                convert_sf_to_mma_layout=convert
            ),
            "flashinfer.fused_moe.cute_dsl.blackwell_sm12x": SimpleNamespace(
                moe_dispatch=dispatch
            ),
        }
        model = object()
        pools = []
        with patch.dict(sys.modules, modules):
            for e, k, n, topk in [(4, 256, 256, 2), (8, 512, 384, 4)]:
                convert.reset_mock()
                w1 = torch.zeros(e, 2 * n, k // 16, dtype=torch.int64)
                w2 = torch.zeros(e, k, n // 16, dtype=torch.int64)
                scales = [
                    torch.ones(1),
                    torch.zeros(16, dtype=torch.int32),
                    torch.ones(e),
                    torch.ones(1),
                    torch.zeros(16, dtype=torch.int32),
                    torch.ones(e),
                ]
                adapter = B12xNvfp4MoE(w1, w2, scales, 1, model, topk)
                self.assertEqual(
                    (
                        adapter.num_experts,
                        adapter.hidden_size,
                        adapter.intermediate_size,
                        adapter.top_k,
                    ),
                    (e, k, n, topk),
                )
                self.assertEqual(adapter.weights["w1_fp4"].shape, (e, 2 * n, k // 2))
                self.assertEqual(adapter.weights["w2_fp4"].shape, (e, k, n // 2))
                self.assertEqual(
                    convert.call_args_list[0].kwargs,
                    dict(m=2 * n, k=k, num_groups=e, sf_vec_size=16),
                )
                self.assertEqual(
                    convert.call_args_list[1].kwargs,
                    dict(m=k, k=n, num_groups=e, sf_vec_size=16),
                )
                self.assertEqual(adapter.launch(quant_mode="nvfp4"), "static")
                self.assertEqual(launch(quant_mode="nvfp4"), "dynamic")
                same_shape = B12xNvfp4MoE(w1, w2, scales, 1, model, topk)
                self.assertIs(adapter.states, same_shape.states)
                pools.append(adapter.workspace_pool)
        self.assertIsNot(pools[0], pools[1])

    def test_tuning_defaults_do_not_apply_qwen_tactics_to_other_models(self):
        config = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        self.assertEqual(decode_tuning(config), (16, None))
        config.decode_cutlass_profile_ids = [13, 37]
        self.assertEqual(decode_tuning(config), (16, [13, 37]))
        requested = json.dumps(
            dict(
                method="modelopt_fp4",
                bits=4,
                group_size=16,
                is_quanted=False,
                model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                decode_b12x_max_rows=8,
                decode_cutlass_profile_ids=[13, 37],
            )
        )
        resolved = build_qwen4_quant_config(requested, source_fp8=True)
        self.assertEqual(decode_tuning(resolved), (8, [13, 37]))
        for bad in ([13], [13, 37, 1], [13, True], [-2, 37]):
            config.decode_cutlass_profile_ids = bad
            with self.assertRaises(ValueError):
                decode_tuning(config)
        config.decode_cutlass_profile_ids = None
        config.decode_b12x_max_rows = -1
        with self.assertRaises(ValueError):
            decode_tuning(config)


if __name__ == "__main__":
    unittest.main()

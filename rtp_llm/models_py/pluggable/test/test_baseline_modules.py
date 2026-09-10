"""Construct baseline modules through builders with mocked submodules and packing.

These checks cover builder compatibility; they do not execute model kernels.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires Torch and native model imports")
class BaselineModuleConstructionTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.platform_provider import (
            DefaultDsv4PlatformProvider,
        )

        self.provider = DefaultDsv4PlatformProvider()
        self.context = SimpleNamespace(
            selection=SimpleNamespace(model_metadata={"hidden_size": 128, "tp_size": 1})
        )

    def request(self, kind):
        return SimpleNamespace(
            module_id="rtp.dsv4." + kind,
            metadata={"layer_id": 0},
            path="v4.layers.0." + kind,
        )

    def test_decode_runtime_buffer_capacity_uses_current_moe_interface(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

        model = DeepSeekV4Model.__new__(DeepSeekV4Model)
        torch.nn.Module.__init__(model)
        model._is_decode_role = True
        model._v4_args = SimpleNamespace(max_seq_len=8192, max_tokens_per_rank=128)
        model._max_generate_batch_size = 3
        model._is_speculative = False
        model._gen_num_per_cycle = 4
        self.assertEqual(model._resolve_prefill_q_token_capacity(), 3)
        self.assertEqual(model._resolve_mtp_hidden_token_capacity(), 3)
        model._is_speculative = True
        self.assertEqual(model._resolve_prefill_q_token_capacity(), 15)
        self.assertEqual(model._resolve_mtp_hidden_token_capacity(), 15)

    def test_attention_constructor_does_not_require_provider_builder(self):
        from rtp_llm.models.dsv4.builders import build_attention
        from rtp_llm.models_py.modules.dsv4.fp8 import attention
        from rtp_llm.utils.model_weight import W

        weights = {}
        for prefix in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            weights[getattr(W, "v4_attn_" + prefix + "_w")] = torch.zeros(
                128, 128, dtype=torch.bfloat16
            )
            weights[getattr(W, "v4_attn_" + prefix + "_s")] = torch.ones(
                128, 1, dtype=torch.int32
            )
        weights.update(
            {
                W.v4_attn_q_norm: torch.ones(128, dtype=torch.bfloat16),
                W.v4_attn_kv_norm: torch.ones(128, dtype=torch.bfloat16),
                W.v4_attn_sink: torch.zeros(1),
            }
        )
        self.assertFalse(hasattr(self.provider, "build_attention"))
        with patch.object(
            attention, "_v4_fp8_linear", side_effect=lambda *a, **k: torch.nn.Identity()
        ), patch.object(
            attention,
            "_prepare_wo_a_stacked",
            return_value=(torch.zeros(1, 128, 128), torch.ones(1, 128, 1)),
        ):
            module = build_attention(
                build_ctx=self.context,
                request=self.request("attention"),
                platform_provider=self.provider,
                layer_id=0,
                dim=128,
                n_heads=1,
                q_lora_rank=128,
                head_dim=128,
                rope_head_dim=64,
                o_lora_rank=128,
                o_groups=1,
                window_size=128,
                compress_ratio=0,
                compress_rope_theta=10000,
                rope_theta=10000,
                rope_factor=1,
                beta_fast=32,
                beta_slow=1,
                original_seq_len=0,
                max_batch_size=1,
                max_seq_len=128,
                index_n_heads=1,
                index_head_dim=128,
                index_topk=512,
                layer_weights=weights,
                tp_size=1,
            )
        self.assertIsInstance(module, attention.AttentionFP8)
        self.assertIsNone(module.compressor)
        self.assertIsInstance(module.wq_a, torch.nn.Module)

    def test_moe_constructor_does_not_require_provider_builder(self):
        from rtp_llm.models.dsv4.builders import build_moe
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import layer
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            ChunkedFp8Fp4MoeLayer,
        )
        from rtp_llm.utils.model_weight import W

        weights = {
            W.v4_router_w: torch.zeros(2, 128, dtype=torch.bfloat16),
            W.v4_router_bias: torch.zeros(2),
            W.v4_shared_w13_w: torch.zeros(256, 128),
            W.v4_shared_w13_s: torch.ones(2, 1),
            W.v4_shared_w2_w: torch.zeros(128, 128),
            W.v4_shared_w2_s: torch.ones(1, 1),
        }
        for suffix in ("1_w", "2_w", "3_w"):
            weights[getattr(W, "v4_routed_w" + suffix)] = torch.zeros(
                2, 128, 64, dtype=torch.int8
            )
        for suffix in ("1_s", "2_s", "3_s"):
            weights[getattr(W, "v4_routed_w" + suffix)] = torch.ones(2, 128, 4)
        router = weights[W.v4_router_w]
        fused = torch.nn.Identity()
        fused.strategy_name = "local_loop"
        fused.includes_shared_expert = False
        executor = SimpleNamespace(prepare=lambda shared: None)
        self.assertFalse(hasattr(self.provider, "build_moe"))
        with patch.object(
            layer, "W13SharedExpert", return_value=torch.nn.Identity()
        ), patch.object(
            layer, "get_shared_expert_executor", return_value=executor
        ), patch.object(
            layer.FusedMoeFactory, "create_fused_moe", return_value=fused
        ) as factory:
            module = build_moe(
                build_ctx=self.context,
                request=self.request("moe"),
                platform_provider=self.provider,
                layer_id=0,
                dim=128,
                moe_inter_dim=128,
                n_routed_experts=2,
                n_activated_experts=1,
                n_shared_experts=1,
                score_func="sqrtsoftplus",
                route_scale=1,
                swiglu_limit=10,
                n_hash_layers=0,
                vocab_size=0,
                layer_weights=weights,
                tp_size=1,
                max_tokens_per_rank=1,
            )
        self.assertIsInstance(module, ChunkedFp8Fp4MoeLayer)
        self.assertIs(module.fused_moe, fused)
        self.assertIs(module.gate.weight, router)
        self.assertIs(factory.call_args.args[1], weights)
        self.assertEqual(weights[W.moe_w1].shape, (2, 256, 64))
        self.assertNotIn(W.v4_routed_w1_w, weights)
        self.assertNotIn(W.v4_router_w, weights)


if __name__ == "__main__":
    unittest.main()

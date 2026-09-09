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
        from rtp_llm.models_py.modules.dsv4.moe import moe_layer
        from rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop import (
            LocalLoopStrategy,
        )
        from rtp_llm.utils.model_weight import W

        weights = {
            key: torch.empty(0)
            for key in (
                W.v4_shared_w13_w,
                W.v4_shared_w13_s,
                W.v4_shared_w2_w,
                W.v4_shared_w2_s,
            )
        }
        weights.update(
            {
                W.v4_router_w: torch.zeros(2, 128, dtype=torch.bfloat16),
                W.v4_router_bias: torch.zeros(2),
            }
        )
        executor = SimpleNamespace(prepare=lambda shared: None)
        self.assertFalse(hasattr(self.provider, "build_moe"))
        with patch.object(
            moe_layer,
            "W13SharedExpert",
            side_effect=lambda *a, **k: torch.nn.Identity(),
        ), patch.object(
            moe_layer, "get_shared_expert_executor", return_value=executor
        ), patch.object(
            LocalLoopStrategy, "setup_weights"
        ):
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
                strategy_type=LocalLoopStrategy,
            )
        self.assertIsInstance(module, moe_layer.MoE)
        self.assertIsInstance(module._strategy, LocalLoopStrategy)
        self.assertIs(module._platform_provider, self.provider)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
from rtp_llm.models.kimi_k3.kimi_k3_weight import _merge_mla_input_projections
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3MLA
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class _CountingProjection(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.mm(hidden_states, self.weight)


class KimiK3MLAProjectionFusionUnitTest(unittest.TestCase):
    @staticmethod
    def _projection_module() -> tuple[KimiK3MLA, _CountingProjection]:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module.q_lora_rank = 3
        module.kv_lora_rank = 2
        module.suffix_dim = 1
        module.local_heads = 2
        module.value_dim = 4
        module.attn_tp_size = 1
        module.attn_tp_rank = 0
        module.use_output_gate = True
        module._mla_backend = "kernel"
        module._accuracy_full_weight_cache = {}
        projection = _CountingProjection(torch.randn(5, 14))
        module.fused_qkv_a_proj = projection
        module._packed_qkv_gate_w = projection.weight
        return module, projection

    def test_loader_packs_replicated_latents_and_local_gate(self) -> None:
        q_a = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        kv_a = torch.arange(20, dtype=torch.float32).reshape(4, 5) + 100
        output_gate = torch.arange(40, dtype=torch.float32).reshape(8, 5) + 200

        for tp_size in (1, 2, 4, 8):
            for tp_rank in range(tp_size):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    actual = _merge_mla_input_projections(
                        [q_a, kv_a, output_gate],
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                    )
                    expected = torch.cat(
                        (q_a, kv_a, output_gate.chunk(tp_size, dim=0)[tp_rank]),
                        dim=0,
                    ).T.contiguous()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(actual.is_contiguous())

    def test_k3_projects_q_kv_and_gate_with_one_gemm(self) -> None:
        module, projection = self._projection_module()
        hidden_states = torch.randn(7, 5)

        with patch.dict(
            os.environ,
            {
                "KIMI_K3_PERF_FUSIONS": "1",
                "KIMI_K3_ACCURACY_CANONICAL_TP": "0",
                "KIMI_K3_ACCURACY_CANONICAL_MLA": "0",
                "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA": "0",
                "KIMI_K3_ACCURACY_TRACE_DIR": "",
            },
            clear=False,
        ):
            qkv_a, output_gate = module._project_qkv_a_input(hidden_states)
        expected = torch.mm(hidden_states, projection.weight)

        self.assertEqual(projection.calls, 1)
        torch.testing.assert_close(qkv_a, expected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(output_gate, expected[:, 6:], rtol=0, atol=0)

    def test_accuracy_projects_q_kv_and_gate_with_source_gemm_boundaries(self) -> None:
        module, projection = self._projection_module()
        hidden_states = torch.randn(7, 5)

        with (
            patch.dict(
                os.environ,
                {
                    "KIMI_K3_PERF_FUSIONS": "0",
                    "KIMI_K3_ACCURACY_CANONICAL_TP": "0",
                    "KIMI_K3_ACCURACY_CANONICAL_MLA": "0",
                    "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA": "0",
                    "KIMI_K3_ACCURACY_TRACE_DIR": "",
                },
                clear=False,
            ),
            patch.object(kimi_k3, "_linear", wraps=kimi_k3._linear) as linear,
        ):
            qkv_a, output_gate = module._project_qkv_a_input(hidden_states)

        expected_q = torch.mm(hidden_states, projection.weight[:, :3])
        expected_kv = torch.mm(hidden_states, projection.weight[:, 3:6])
        expected_gate = torch.mm(hidden_states, projection.weight[:, 6:])
        self.assertEqual(projection.calls, 0)
        self.assertEqual(linear.call_count, 3)
        torch.testing.assert_close(
            qkv_a, torch.cat((expected_q, expected_kv), dim=-1), rtol=0, atol=0
        )
        torch.testing.assert_close(output_gate, expected_gate, rtol=0, atol=0)

    def test_reference_backend_restores_source_gemm_boundaries(self) -> None:
        module, _ = self._projection_module()
        module._mla_backend = "reference"
        with patch.dict(
            os.environ,
            {
                "KIMI_K3_PERF_FUSIONS": "1",
                "KIMI_K3_ACCURACY_CANONICAL_TP": "0",
                "KIMI_K3_ACCURACY_CANONICAL_MLA": "0",
                "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA": "0",
                "KIMI_K3_ACCURACY_TRACE_DIR": "",
            },
            clear=False,
        ):
            self.assertTrue(module._use_source_projection_boundaries())

    def test_canonical_tp_reconstructs_full_width_gate_projection(self) -> None:
        module, projection = self._projection_module()
        module.attn_tp_size = 2
        module.attn_tp_rank = 1
        hidden_states = torch.randn(7, 5)
        expected_gate = torch.randn(7, 8)

        with (
            patch.dict(
                os.environ,
                {
                    "KIMI_K3_PERF_FUSIONS": "0",
                    "KIMI_K3_ACCURACY_CANONICAL_TP": "1",
                },
                clear=False,
            ),
            patch.object(
                kimi_k3,
                "_column_parallel_linear",
                return_value=expected_gate,
            ) as column_linear,
        ):
            _, output_gate = module._project_qkv_a_input(hidden_states)

        args = column_linear.call_args.args
        self.assertIs(args[0], hidden_states)
        torch.testing.assert_close(args[1], projection.weight[:, 6:])
        self.assertEqual(args[2:4], (2, 1))
        self.assertIs(args[4], module._accuracy_full_weight_cache)
        self.assertEqual(args[5], "mla_output_gate")
        self.assertIs(output_gate, expected_gate)

    def test_model_config_accepts_only_bf16_without_runtime_quantization(
        self,
    ) -> None:
        config = KimiK3ModelConfig()
        with patch.object(ModelConfig, "init_precision_config", return_value=None):
            config.data_type = "bf16"
            config.quant_config = None
            config.init_precision_config(None, None)

            config.data_type = "fp16"
            with self.assertRaisesRegex(ValueError, "only BF16 compute"):
                config.init_precision_config(None, None)

            config.data_type = "bf16"
            config.quant_config = object()
            with self.assertRaisesRegex(ValueError, "runtime weight quantization"):
                config.init_precision_config(None, None)

    def test_base_mla_keeps_gate_hook_optional(self) -> None:
        module = MlaAttention.__new__(MlaAttention)
        nn.Module.__init__(module)
        module.fused_qkv_a_proj = nn.Linear(5, 7, bias=False)
        hidden_states = torch.randn(3, 5)

        projected, output_gate = module._project_qkv_a_input(hidden_states)

        torch.testing.assert_close(
            projected,
            module.fused_qkv_a_proj(hidden_states),
            rtol=0,
            atol=0,
        )
        self.assertIsNone(output_gate)
        self.assertIs(module._apply_output_gate(projected, output_gate), projected)


if __name__ == "__main__":
    unittest.main()

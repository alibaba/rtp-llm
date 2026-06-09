import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
        _flydsl_fused_moe_enabled,
        _flydsl_fused_moe_unsupported_reason,
        _should_use_flydsl_fused_moe,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.flydsl.tuning import (
        get_qwen_ptpc_fp8_tuning,
    )
    from rtp_llm.models_py.model_desc.qwen3_next import (
        _use_qwen35_flydsl_moe_phase_hint,
    )

    _IMPORT_ERROR = None
except ImportError as exc:
    _IMPORT_ERROR = exc


class RocmMoeFlydslGateTest(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"ROCm fused_moe deps unavailable: {_IMPORT_ERROR}")

    def _reason(
        self,
        m: int,
        *,
        is_gfx942_device: bool = True,
        activation: str = "silu",
        experts: int = 512,
        topk: int = 10,
        model_dim: int = 4096,
        inter_dim: int = 256,
        effective_expert_mask=None,
        expert_ids_are_local: bool = False,
        ep_size: int = 1,
        is_prefill: bool = False,
    ):
        hidden_states = torch.empty((m, model_dim), dtype=torch.bfloat16, device="meta")
        w1 = torch.empty(
            (experts, 2 * inter_dim, model_dim),
            dtype=torch.float8_e4m3fnuz,
            device="meta",
        )
        w2 = torch.empty(
            (experts, model_dim, inter_dim), dtype=torch.float8_e4m3fnuz, device="meta"
        )
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device="meta")
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe.is_gfx942",
            return_value=is_gfx942_device,
        ):
            return _flydsl_fused_moe_unsupported_reason(
                hidden_states,
                w1,
                w2,
                topk_ids,
                activation=activation,
                is_prefill=is_prefill,
                effective_expert_mask=effective_expert_mask,
                expert_ids_are_local=expert_ids_are_local,
                num_experts=experts,
                local_num_experts=experts,
                ep_size=ep_size,
            )

    def test_tp4_m512_is_supported(self):
        self.assertIsNone(self._reason(512))

    def test_prefill_falls_back_to_aiter(self):
        reason = self._reason(1, is_prefill=True)
        self.assertIn("prefill", reason)

    def test_non_mi308_device_falls_back(self):
        reason = self._reason(1, is_gfx942_device=False)
        self.assertIn("MI308X/gfx942", reason)

    def test_qwen35_next_shape_passes_phase_hint(self):
        qwen35_cfg = SimpleNamespace(
            moe_style=2,
            expert_num=512,
            moe_k=10,
            hidden_size=4096,
        )
        other_cfg = SimpleNamespace(
            moe_style=2,
            expert_num=512,
            moe_k=10,
            hidden_size=8192,
        )
        self.assertTrue(_use_qwen35_flydsl_moe_phase_hint(qwen35_cfg))
        self.assertFalse(_use_qwen35_flydsl_moe_phase_hint(other_cfg))

    def test_tp4_m513_falls_back_by_default(self):
        reason = self._reason(513)
        self.assertIn("M=513", reason)

    def test_tp8_m512_is_supported(self):
        self.assertIsNone(self._reason(512, inter_dim=128))

    def test_tp8_m513_falls_back_by_default(self):
        reason = self._reason(513, inter_dim=128)
        self.assertIn("M=513", reason)

    def test_flydsl_moe_env_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_flydsl_fused_moe_enabled())

    def test_flydsl_moe_env_requires_global_flydsl(self):
        with patch.dict(os.environ, {"USE_FLYDSL_MOE": "1"}):
            self.assertFalse(_flydsl_fused_moe_enabled())

    def test_global_flydsl_env_does_not_enable_moe_by_itself(self):
        with patch.dict(os.environ, {"USE_FLYDSL": "1"}):
            self.assertFalse(_flydsl_fused_moe_enabled())

    def test_flydsl_moe_env_enables_experiment_with_global_flydsl(self):
        with patch.dict(os.environ, {"USE_FLYDSL": "1", "USE_FLYDSL_MOE": "1"}):
            self.assertTrue(_flydsl_fused_moe_enabled())

    def test_should_use_flydsl_only_checks_gate(self):
        hidden_states = torch.empty((1, 4096), dtype=torch.bfloat16, device="meta")
        w1 = torch.empty(
            (512, 512, 4096), dtype=torch.float8_e4m3fnuz, device="meta"
        )
        w2 = torch.empty(
            (512, 4096, 256), dtype=torch.float8_e4m3fnuz, device="meta"
        )
        topk_ids = torch.empty((1, 10), dtype=torch.int32, device="meta")
        with patch.dict(os.environ, {"USE_FLYDSL": "1", "USE_FLYDSL_MOE": "1"}):
            with patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe.is_gfx942",
                return_value=True,
            ):
                with patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe._load_flydsl_fused_moe",
                    side_effect=AssertionError("load should happen only on execution"),
                ) as load_flydsl:
                    self.assertTrue(
                        _should_use_flydsl_fused_moe(
                            hidden_states,
                            w1,
                            w2,
                            topk_ids,
                            activation="silu",
                            is_prefill=False,
                            effective_expert_mask=None,
                            expert_ids_are_local=False,
                            num_experts=512,
                            local_num_experts=512,
                            ep_size=1,
                        )
                    )
                    load_flydsl.assert_not_called()

    def test_grouped_tile_m_comes_from_table(self):
        tp4_tuning = get_qwen_ptpc_fp8_tuning(256)
        tp8_tuning = get_qwen_ptpc_fp8_tuning(128)
        self.assertIsNotNone(tp4_tuning)
        self.assertIsNotNone(tp8_tuning)
        self.assertEqual(16, tp4_tuning.grouped_tile_m(512, default=32))
        self.assertEqual(64, tp4_tuning.grouped_tile_m(2048, default=32))
        self.assertEqual(16, tp8_tuning.grouped_tile_m(512, default=32))

    def test_m1_aiter_quant_comes_from_table(self):
        tp4_tuning = get_qwen_ptpc_fp8_tuning(256)
        tp8_tuning = get_qwen_ptpc_fp8_tuning(128)
        self.assertIsNotNone(tp4_tuning)
        self.assertIsNotNone(tp8_tuning)
        self.assertTrue(tp4_tuning.use_aiter_quant(1))
        self.assertFalse(tp4_tuning.use_aiter_quant(2))
        self.assertFalse(tp8_tuning.use_aiter_quant(1))

    def test_small_m_route_free_comes_from_table(self):
        tp4_tuning = get_qwen_ptpc_fp8_tuning(256)
        tp8_tuning = get_qwen_ptpc_fp8_tuning(128)
        self.assertIsNotNone(tp4_tuning)
        self.assertIsNotNone(tp8_tuning)
        self.assertTrue(tp4_tuning.use_route_free(1))
        self.assertTrue(tp4_tuning.use_route_free(15))
        self.assertFalse(tp4_tuning.use_route_free(16))
        self.assertTrue(tp8_tuning.use_route_free(1))
        self.assertTrue(tp8_tuning.use_route_free(15))
        self.assertFalse(tp8_tuning.use_route_free(16))

    def test_unsupported_activation_falls_back(self):
        reason = self._reason(16, activation="gelu")
        self.assertIn("activation", reason)

    def test_expert_mask_falls_back(self):
        mask = torch.empty((512,), dtype=torch.int32, device="meta")
        reason = self._reason(16, effective_expert_mask=mask)
        self.assertIn("expert_mask", reason)

    def test_non_qwen_shape_falls_back(self):
        reason = self._reason(16, experts=4, topk=2, model_dim=256, inter_dim=128)
        self.assertIn("shape", reason)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.glm5_mega_moe import fused_moe_wrapper
from rtp_llm.utils.model_weight import W


class _FakeMegaMoE:
    instance = None

    @classmethod
    def from_params(cls, **kwargs):
        cls.instance = cls()
        cls.instance.params = kwargs
        return cls.instance

    def setup_weights_from_fp4(self, **kwargs):
        self.fp4_kwargs = kwargs


def _config(hidden_size=8, inter=4, max_seq_len=16, gen_num_per_cycle=0):
    return SimpleNamespace(
        hidden_size=hidden_size,
        expert_num=2,
        moe_k=1,
        moe_inter_size=inter,
        max_seq_len=max_seq_len,
        gen_num_per_cycle=gen_num_per_cycle,
    )


def _parallelism(role_type=None):
    return SimpleNamespace(ep_size=1, ep_rank=0, role_type=role_type)


class MegaMoeFusedWrapperLayoutTest(unittest.TestCase):
    def test_bf16_stacked_moe_w1_is_rejected(self):
        config = _config(hidden_size=8, inter=4)
        up = torch.full((2, 4, 8), 3, dtype=torch.bfloat16)
        gate = torch.full((2, 4, 8), 7, dtype=torch.bfloat16)
        weights = {
            W.moe_w1: torch.cat([up, gate], dim=1),
            W.moe_w2: torch.ones((2, 8, 4), dtype=torch.bfloat16),
        }

        with patch.object(fused_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            with self.assertRaisesRegex(ValueError, "load-time FP4"):
                fused_moe_wrapper.MegaMoeFusedWrapper(
                    config, _parallelism(), weights, moe_config=None, layer_idx=0
                )

    def test_fp8_stacked_moe_w1_is_rejected(self):
        config = _config(hidden_size=8, inter=4)
        w1 = torch.zeros((2, 8, 8), dtype=torch.float32).to(torch.float8_e4m3fn)
        w2 = torch.zeros((2, 8, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        weights = {
            W.moe_w1: w1,
            W.moe_s1: torch.ones((2, 8, 1), dtype=torch.float32),
            W.moe_w2: w2,
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
        }

        with patch.object(fused_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            with self.assertRaisesRegex(ValueError, "load-time FP4 int8"):
                fused_moe_wrapper.MegaMoeFusedWrapper(
                    config, _parallelism(), weights, moe_config=None, layer_idx=0
                )

    def test_fp4_stacked_moe_w1_reorders_up_gate_for_deepgemm(self):
        config = _config(hidden_size=8, inter=4)
        up_w = torch.full((2, 4, 4), 3, dtype=torch.int8)
        gate_w = torch.full((2, 4, 4), 7, dtype=torch.int8)
        up_s = torch.full((2, 4, 2), 5, dtype=torch.float32)
        gate_s = torch.full((2, 4, 2), 11, dtype=torch.float32)
        weights = {
            W.moe_w1: torch.cat([up_w, gate_w], dim=1),
            W.moe_s1: torch.cat([up_s, gate_s], dim=1),
            W.moe_w2: torch.ones((2, 8, 2), dtype=torch.int8),
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
        }

        with patch.object(fused_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            fused_moe_wrapper.MegaMoeFusedWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        captured = _FakeMegaMoE.instance.fp4_kwargs
        torch.testing.assert_close(captured["w1_w"][:, :4], gate_w)
        torch.testing.assert_close(captured["w1_w"][:, 4:], up_w)
        torch.testing.assert_close(captured["w1_s"][:, :4], gate_s)
        torch.testing.assert_close(captured["w1_s"][:, 4:], up_s)

    def test_decode_mtp_budget_includes_verify_width(self):
        from rtp_llm.ops import RoleType

        config = _config(
            hidden_size=8,
            inter=4,
            max_seq_len=4096,
            gen_num_per_cycle=3,
        )
        weights = {
            W.moe_w1: torch.zeros((2, 8, 4), dtype=torch.int8),
            W.moe_s1: torch.ones((2, 8, 2), dtype=torch.float32),
            W.moe_w2: torch.ones((2, 8, 2), dtype=torch.int8),
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
        }

        with patch.object(fused_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            fused_moe_wrapper.MegaMoeFusedWrapper(
                config,
                _parallelism(role_type=RoleType.DECODE),
                weights,
                moe_config=None,
                layer_idx=0,
                max_generate_batch_size=8,
            )

        self.assertEqual(_FakeMegaMoE.instance.params["max_tokens_per_rank"], 32)


if __name__ == "__main__":
    unittest.main()

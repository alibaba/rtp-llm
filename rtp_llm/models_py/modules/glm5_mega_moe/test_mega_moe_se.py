"""Focused unit tests for the FP8xFP4 MegaMoE shared-expert strategy."""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# The source-tree-only unit test does not load the compiled compute_ops
# extension.  Stub only the pure token-budget helper imported by the wrapper;
# Bazel/service execution uses the real module.
_fake_moe_package = types.ModuleType("rtp_llm.models_py.modules.dsv4.moe")
_fake_moe_package.__path__ = []
_fake_moe_layer = types.ModuleType("rtp_llm.models_py.modules.dsv4.moe.moe_layer")
_fake_moe_layer.resolve_moe_max_tokens_per_rank = (
    lambda *, current_max_tokens_per_rank, **_: current_max_tokens_per_rank
)
sys.modules.setdefault("rtp_llm.models_py.modules.dsv4.moe", _fake_moe_package)
sys.modules.setdefault("rtp_llm.models_py.modules.dsv4.moe.moe_layer", _fake_moe_layer)

from rtp_llm.models_py.modules.glm5_mega_moe import mega_moe_se, mega_moe_se_wrapper
from rtp_llm.utils.model_weight import W


class _FakeMegaMoESE:
    instance = None

    @classmethod
    def from_params(cls, **kwargs):
        cls.instance = cls()
        cls.instance.params = kwargs
        return cls.instance

    def setup_weights_from_fp4(self, **kwargs):
        self.routed = kwargs

    def setup_weights_from_fp8(self, **kwargs):
        raise AssertionError("offline FP4 test must not use routed FP8 setup")

    def setup_shared_expert_from_fp8(self, **kwargs):
        self.shared = kwargs

    def maybe_warmup_fused_shared_jit_once(self):
        self.warmed = True


def _config():
    return SimpleNamespace(
        hidden_size=8,
        expert_num=2,
        moe_k=1,
        moe_inter_size=4,
        max_seq_len=16,
        gen_num_per_cycle=0,
        swiglu_limit=10.0,
    )


def _parallelism():
    return SimpleNamespace(
        ep_size=1,
        ep_rank=0,
        role_type=None,
        get_ffn_tp_size=lambda: 1,
    )


class MegaMoeSEWrapperTest(unittest.TestCase):
    def test_fp4_routed_and_fp8_shared_weights_are_consumed(self):
        routed_up = torch.full((2, 4, 4), 3, dtype=torch.int8)
        routed_gate = torch.full((2, 4, 4), 7, dtype=torch.int8)
        routed_up_sf = torch.full((2, 4, 2), 5, dtype=torch.float32)
        routed_gate_sf = torch.full((2, 4, 2), 11, dtype=torch.float32)
        shared_w13 = torch.full((8, 8), 2.0).to(torch.float8_e4m3fn)
        shared_w2 = torch.full((8, 4), 4.0).to(torch.float8_e4m3fn)
        weights = {
            W.moe_w1: torch.cat([routed_up, routed_gate], dim=1),
            W.moe_s1: torch.cat([routed_up_sf, routed_gate_sf], dim=1),
            W.moe_w2: torch.ones((2, 8, 2), dtype=torch.int8),
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
            W.ffn_w13: shared_w13,
            W.ffn_s13: torch.ones((1, 1), dtype=torch.int32),
            W.ffn_w2: shared_w2,
            W.ffn_s2: torch.ones((1, 1), dtype=torch.int32),
        }

        with patch.object(mega_moe_se_wrapper, "GLM5MegaMoESE", _FakeMegaMoESE):
            mega_moe_se_wrapper.MegaMoeSEWrapper(
                _config(), _parallelism(), weights, layer_idx=3
            )

        captured = _FakeMegaMoESE.instance
        torch.testing.assert_close(captured.routed["w1_w"][:, :4], routed_up)
        torch.testing.assert_close(captured.routed["w1_w"][:, 4:], routed_gate)
        torch.testing.assert_close(captured.routed["w1_s"][:, :4], routed_up_sf)
        torch.testing.assert_close(captured.routed["w1_s"][:, 4:], routed_gate_sf)
        self.assertEqual(captured.routed["w1_layout"], "up_gate")
        torch.testing.assert_close(captured.shared["w1_w"], shared_w13)
        torch.testing.assert_close(captured.shared["w2_w"], shared_w2)
        self.assertTrue(captured.warmed)
        self.assertEqual(captured.params["layer_id"], 3)
        for key in (
            W.moe_w1,
            W.moe_s1,
            W.moe_w2,
            W.moe_s2,
            W.ffn_w13,
            W.ffn_s13,
            W.ffn_w2,
            W.ffn_s2,
        ):
            self.assertNotIn(key, weights)

    def test_shared_ffn_tp_is_rejected(self):
        parallelism = _parallelism()
        parallelism.get_ffn_tp_size = lambda: 2
        with self.assertRaisesRegex(ValueError, "ffn_tp_size == 1"):
            mega_moe_se_wrapper.MegaMoeSEWrapper(
                _config(), parallelism, {}, layer_idx=0
            )


class MegaMoeSEBufferCompatibilityTest(unittest.TestCase):
    def test_deep_gemm_261_buffer_need_not_expose_shared_count(self):
        cfg = SimpleNamespace(
            layer_id=3,
            n_activated_experts=1,
            n_routed_experts=2,
            max_tokens_per_rank=16,
            dim=8,
            moe_inter_dim=4,
        )
        module = mega_moe_se.GLM5MegaMoESE(cfg)
        module._num_shared_experts = 1
        module._mega_l1_w = torch.empty(1)
        fake_buffer = SimpleNamespace(num_max_tokens_per_rank=16)
        fake_output = torch.empty((16, 8), dtype=torch.bfloat16)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(
                mega_moe_se,
                "get_or_create_mega_moe_se_buf",
                return_value=fake_buffer,
            ),
            patch.object(
                mega_moe_se,
                "get_or_create_mega_moe_se_output",
                return_value=fake_output,
            ),
            patch.object(
                mega_moe_se,
                "get_mega_moe_se_input_packer",
                return_value=object(),
            ),
        ):
            module._setup_buffer_and_warmup()

        self.assertIs(module._mega_buf, fake_buffer)
        self.assertIs(module._mega_y, fake_output)


if __name__ == "__main__":
    unittest.main()

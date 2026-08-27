import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
    fused_pack_mega_moe_inputs,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper
from rtp_llm.utils.model_weight import W


class _RecordingPacker:
    def __init__(self):
        self.calls = []

    def pack(self, x, weights, indices, buf, tokens):
        self.calls.append((x, weights, indices, buf, tokens))


class MegaMoeConfigTest(unittest.TestCase):
    def test_from_params_builds_exact_ep_partition(self):
        moe = GLM5MegaMoE.from_params(
            layer_id=3,
            dim=128,
            moe_inter_dim=32,
            n_routed_experts=12,
            n_activated_experts=2,
            ep_size=3,
            ep_rank=1,
            max_tokens_per_rank=16,
        )
        self.assertEqual(moe.cfg.n_local_experts, 4)
        self.assertEqual(moe.cfg.local_expert_start, 4)
        self.assertEqual(moe.cfg.local_expert_end, 8)

    def test_from_params_rejects_invalid_distributed_geometry(self):
        cases = (
            ({"n_routed_experts": 10, "ep_size": 3}, "divisible"),
            ({"n_routed_experts": 8, "ep_size": 2, "ep_rank": 2}, "ep_rank"),
            ({"n_routed_experts": 8, "n_activated_experts": 9}, r"in \[1, 8\]"),
            ({"dim": 96}, "divisible by 128"),
            ({"moe_inter_dim": 31}, "divisible by 32"),
            ({"max_tokens_per_rank": 0}, "must be positive"),
        )
        defaults = {
            "layer_id": 0,
            "dim": 128,
            "moe_inter_dim": 32,
            "n_routed_experts": 8,
            "n_activated_experts": 2,
            "ep_size": 1,
            "ep_rank": 0,
            "max_tokens_per_rank": 8,
        }
        for overrides, pattern in cases:
            with self.subTest(overrides=overrides):
                params = {**defaults, **overrides}
                with self.assertRaisesRegex(ValueError, pattern):
                    GLM5MegaMoE.from_params(**params)


class MegaMoeForwardContractTest(unittest.TestCase):
    @staticmethod
    def _module(swiglu_limit: float):
        moe = GLM5MegaMoE.from_params(
            layer_id=0,
            dim=128,
            moe_inter_dim=32,
            n_routed_experts=2,
            n_activated_experts=1,
            swiglu_limit=swiglu_limit,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=4,
        )
        moe._mega_l1_w = torch.empty(0)
        moe._mega_l1_sf = torch.empty(0)
        moe._mega_l2_w = torch.empty(0)
        moe._mega_l2_sf = torch.empty(0)
        moe._mega_buf = SimpleNamespace(num_max_tokens_per_rank=4)
        moe._mega_y = torch.empty(4, 128, dtype=torch.bfloat16)
        moe._input_packer = _RecordingPacker()
        return moe

    @staticmethod
    def _fake_deep_gemm(record):
        module = types.ModuleType("deep_gemm")

        def fp8_fp4_mega_moe(output, *_args, **kwargs):
            record.append(kwargs)
            output.zero_()

        module.fp8_fp4_mega_moe = fp8_fp4_mega_moe
        return module

    def test_forward_propagates_required_swiglu_clamp(self):
        calls = []
        moe = self._module(10.0)
        with patch.dict(sys.modules, {"deep_gemm": self._fake_deep_gemm(calls)}):
            output = moe(
                torch.zeros(2, 128, dtype=torch.bfloat16),
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.int64),
            )

        self.assertEqual(tuple(output.shape), (2, 128))
        self.assertEqual(calls[0]["activation_clamp"], 10.0)
        self.assertEqual(len(moe._input_packer.calls), 1)

    def test_zero_limit_disables_clamp_but_still_launches_for_empty_rank(self):
        calls = []
        moe = self._module(0.0)
        with patch.dict(sys.modules, {"deep_gemm": self._fake_deep_gemm(calls)}):
            output = moe(
                torch.zeros(0, 128, dtype=torch.bfloat16),
                torch.zeros(0, 1),
                torch.zeros(0, 1, dtype=torch.int64),
            )

        self.assertEqual(tuple(output.shape), (0, 128))
        self.assertIsNone(calls[0]["activation_clamp"])
        self.assertEqual(moe._input_packer.calls[0][-1], 0)

    def test_fp4_setup_rejects_mismatched_weight_shape_before_deep_gemm(self):
        moe = self._module(10.0)
        with self.assertRaisesRegex(ValueError, "w1_w shape"):
            moe.setup_weights_from_fp4(
                torch.zeros(2, 63, 64, dtype=torch.int8),
                torch.ones(2, 64, 4),
                torch.zeros(2, 128, 16, dtype=torch.int8),
                torch.ones(2, 128, 1),
            )

    def test_fused_packer_rejects_router_shape_before_cuda_launch(self):
        with self.assertRaisesRegex(ValueError, "match x rows"):
            fused_pack_mega_moe_inputs(
                torch.zeros(2, 128, dtype=torch.bfloat16),
                torch.zeros(1, 1),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.empty(2, 128, dtype=torch.float8_e4m3fn),
                torch.empty(2, 1, dtype=torch.int32),
                torch.empty(2, 1, dtype=torch.int64),
                torch.empty(2, 1),
            )


class MegaMoeWrapperConfigTest(unittest.TestCase):
    def test_explicit_zero_swiglu_limit_is_not_replaced_by_default(self):
        captured = {}

        class FakeMegaMoe:
            @classmethod
            def from_params(cls, **kwargs):
                captured.update(kwargs)
                return cls()

            def setup_weights_from_fp4(self, **_kwargs):
                return None

        class TestWrapper(MegaMoeWrapper):
            def _get_mega_moe_cls(self):
                return FakeMegaMoe

        config = SimpleNamespace(
            hidden_size=128,
            expert_num=2,
            moe_k=1,
            swiglu_limit=0.0,
            moe_inter_size=32,
            max_seq_len=128,
            gen_num_per_cycle=0,
        )
        parallelism = SimpleNamespace(
            ep_size=1,
            ep_rank=0,
            role_type=None,
            prefill_cp_config=None,
        )
        weights = {
            W.moe_w1: torch.zeros(2, 64, 64, dtype=torch.int8),
            W.moe_w2: torch.zeros(2, 128, 16, dtype=torch.int8),
            W.moe_s1: torch.ones(2, 64, 4),
            W.moe_s2: torch.ones(2, 128, 1),
        }

        TestWrapper(config, parallelism, weights, max_generate_batch_size=1)

        self.assertEqual(captured["swiglu_limit"], 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.glm5_mega_moe import (
    mega_moe,
    mega_moe_fp8,
    mega_moe_fp8_se_wrapper,
    mega_moe_fp8_wrapper,
    mega_moe_fused_wrapper,
    mega_moe_wrapper,
    quant_layouts,
)
from rtp_llm.models_py.model_desc.generic_moe import (
    _validate_hy4_mxfp8_moe_strategy,
)
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

    def setup_weights_from_fp8(self, **kwargs):
        self.fp8_kwargs = kwargs

    def setup_shared_expert_from_fp8(self, **kwargs):
        self.shared_fp8_kwargs = kwargs

    def maybe_warmup_fused_shared_jit_once(self):
        self.fused_shared_jit_warmed = True


class _FakeForwardMegaMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, topk_weights, topk_ids, **kwargs):
        self.calls.append(kwargs)
        return hidden_states

    def forward_prepacked(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states


def _config(
    hidden_size=8,
    inter=4,
    max_seq_len=16,
    gen_num_per_cycle=0,
    swiglu_limit=0.0,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        expert_num=2,
        moe_k=1,
        moe_inter_size=inter,
        max_seq_len=max_seq_len,
        gen_num_per_cycle=gen_num_per_cycle,
        swiglu_limit=swiglu_limit,
    )


def _parallelism(role_type=None):
    return SimpleNamespace(ep_size=1, ep_rank=0, role_type=role_type)


class MegaMoeWrapperLayoutTest(unittest.TestCase):
    def test_hy4_mxfp8_rejects_backend_that_drops_routed_clamp(self):
        config = _config(swiglu_limit=10.0)
        config.model_type = "hy_v4"
        config.quant_config = SimpleNamespace(get_method=lambda: "MXFP8")
        with self.assertRaisesRegex(ValueError, "online FP8-to-FP4"):
            _validate_hy4_mxfp8_moe_strategy(
                config, SimpleNamespace(moe_strategy="auto")
            )

        _validate_hy4_mxfp8_moe_strategy(
            config, SimpleNamespace(moe_strategy="mega_moe_fp8")
        )
        _validate_hy4_mxfp8_moe_strategy(
            config, SimpleNamespace(moe_strategy="mega_moe")
        )
        with self.assertRaisesRegex(ValueError, "clamps routed experts only"):
            _validate_hy4_mxfp8_moe_strategy(
                config, SimpleNamespace(moe_strategy="mega_moe_se")
            )

    def test_fp4_wrapper_forwards_routed_clamp(self):
        wrapper = object.__new__(mega_moe_wrapper.MegaMoeWrapper)
        torch.nn.Module.__init__(wrapper)
        fake_mega_moe = _FakeForwardMegaMoE()
        wrapper.mega_moe = fake_mega_moe
        wrapper._activation_clamp = 10.0

        hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        wrapper(
            hidden,
            topk_weights,
            topk_ids,
            extra_expert_args={"swiglu_limit": 10.0},
        )

        self.assertEqual(fake_mega_moe.calls[0]["activation_clamp"], 10.0)

    def test_fp4_prepacked_path_forwards_routed_clamp(self):
        wrapper = object.__new__(mega_moe_wrapper.MegaMoeWrapper)
        torch.nn.Module.__init__(wrapper)
        fake_mega_moe = _FakeForwardMegaMoE()
        wrapper.mega_moe = fake_mega_moe
        wrapper._activation_clamp = 10.0

        hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
        wrapper.forward_prepacked(hidden)

        self.assertEqual(fake_mega_moe.calls[0]["activation_clamp"], 10.0)

    def test_fp8_scale_recipe_infers_mxfp8_and_legacy_block_fp8(self):
        mxfp8_scale = torch.empty((2, 64, 4), dtype=torch.float32)
        self.assertEqual(
            mega_moe_fp8._infer_fp8_scale_recipe(mxfp8_scale, mn=64, k=128),
            (1, 32),
        )

        block_fp8_scale = torch.empty((2, 2, 3), dtype=torch.float32)
        self.assertEqual(
            mega_moe_fp8._infer_fp8_scale_recipe(
                block_fp8_scale, mn=256, k=384
            ),
            (128, 128),
        )

    def test_fp8_scale_layout_transform_receives_mxfp8_recipe(self):
        scale = torch.ones((2, 64, 4), dtype=torch.float32)
        packed = torch.ones((2, 64, 1), dtype=torch.int32)
        transform = Mock(return_value=packed)
        fake_deep_gemm = SimpleNamespace(
            transform_sf_into_required_layout=transform,
        )

        with patch.dict("sys.modules", {"deep_gemm": fake_deep_gemm}):
            actual = quant_layouts.prepare_fp8_weight_scale_for_deepgemm(
                scale,
                mn=64,
                k=128,
                num_groups=2,
                recipe=(1, 32),
            )

        self.assertIs(actual, packed)
        transform.assert_called_once_with(scale, 64, 128, (1, 32), 2)

    def test_fp8_wrapper_only_enables_clamp_when_explicitly_requested(self):
        wrapper = object.__new__(mega_moe_fp8_wrapper.MegaMoeFp8Wrapper)
        torch.nn.Module.__init__(wrapper)
        fake_mega_moe = _FakeForwardMegaMoE()
        wrapper.mega_moe = fake_mega_moe

        hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        wrapper(hidden, topk_weights, topk_ids, extra_expert_args={})
        wrapper(
            hidden,
            topk_weights,
            topk_ids,
            extra_expert_args={"swiglu_limit": 10.0},
        )

        self.assertIsNone(fake_mega_moe.calls[0]["activation_clamp"])
        self.assertEqual(fake_mega_moe.calls[1]["activation_clamp"], 10.0)

    def test_fp8_mega_forward_passes_inferred_weight_recipe(self):
        moe = mega_moe_fp8.GLM5MegaMoEFP8.from_params(
            layer_id=0,
            dim=8,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=1,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=4,
        )
        moe._fp8_weight_recipe = (1, 32)
        moe._mega_l1_w = torch.empty((2, 8, 8))
        moe._mega_l1_sf = torch.empty((2, 8, 1), dtype=torch.int32)
        moe._mega_l2_w = torch.empty((2, 8, 4))
        moe._mega_l2_sf = torch.empty((2, 8, 1), dtype=torch.int32)
        moe._mega_buf = SimpleNamespace(num_max_tokens_per_rank=4)
        moe._mega_y = torch.empty((4, 8), dtype=torch.bfloat16)
        moe._input_packer = SimpleNamespace(pack=Mock())
        moe._maybe_pre_kernel_barrier = Mock()

        run_mega = Mock()
        fake_deep_gemm = SimpleNamespace(fp8_fp8_mega_moe=run_mega)
        x = torch.zeros((2, 8), dtype=torch.bfloat16)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        with patch.dict("sys.modules", {"deep_gemm": fake_deep_gemm}), patch.object(
            mega_moe_fp8, "_sync_cuda_graph_warmup_ranks"
        ):
            moe(x, topk_weights, topk_ids, activation_clamp=10.0)

        self.assertEqual(run_mega.call_args.kwargs["weight_recipe"], (1, 32))
        self.assertEqual(run_mega.call_args.kwargs["activation_clamp"], 10.0)

    def test_fp4_mega_forward_passes_activation_clamp(self):
        moe = mega_moe.GLM5MegaMoE.from_params(
            layer_id=0,
            dim=8,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=1,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=4,
        )
        moe._mega_l1_w = torch.empty((2, 8, 4), dtype=torch.int8)
        moe._mega_l1_sf = torch.empty((2, 8, 1), dtype=torch.int32)
        moe._mega_l2_w = torch.empty((2, 8, 2), dtype=torch.int8)
        moe._mega_l2_sf = torch.empty((2, 8, 1), dtype=torch.int32)
        moe._mega_buf = SimpleNamespace(num_max_tokens_per_rank=4)
        moe._mega_y = torch.empty((4, 8), dtype=torch.bfloat16)
        moe._input_packer = SimpleNamespace(pack=Mock())
        moe._maybe_pre_kernel_barrier = Mock()

        run_mega = Mock()
        fake_deep_gemm = SimpleNamespace(fp8_fp4_mega_moe=run_mega)
        x = torch.zeros((2, 8), dtype=torch.bfloat16)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        with patch.dict("sys.modules", {"deep_gemm": fake_deep_gemm}), patch.object(
            mega_moe, "_sync_cuda_graph_warmup_ranks"
        ):
            moe(x, topk_weights, topk_ids, activation_clamp=10.0)

        self.assertEqual(run_mega.call_args.kwargs["activation_clamp"], 10.0)

    def test_bf16_stacked_moe_w1_is_rejected(self):
        config = _config(hidden_size=8, inter=4)
        up = torch.full((2, 4, 8), 3, dtype=torch.bfloat16)
        gate = torch.full((2, 4, 8), 7, dtype=torch.bfloat16)
        weights = {
            W.moe_w1: torch.cat([up, gate], dim=1),
            W.moe_w2: torch.ones((2, 8, 4), dtype=torch.bfloat16),
        }

        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            with self.assertRaisesRegex(ValueError, "load-time FP4"):
                mega_moe_wrapper.MegaMoeWrapper(
                    config, _parallelism(), weights, moe_config=None, layer_idx=0
                )

    def test_fp8_stacked_moe_w1_reorders_up_gate_for_deepgemm(self):
        config = _config(hidden_size=8, inter=4)
        up_w = torch.full((2, 4, 8), 3, dtype=torch.float32).to(torch.float8_e4m3fn)
        gate_w = torch.full((2, 4, 8), 7, dtype=torch.float32).to(torch.float8_e4m3fn)
        up_s = torch.full((2, 4, 1), 5, dtype=torch.float32)
        gate_s = torch.full((2, 4, 1), 11, dtype=torch.float32)
        w2 = torch.zeros((2, 8, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        s2 = torch.ones((2, 8, 1), dtype=torch.float32)
        weights = {
            W.moe_w1: torch.cat([up_w, gate_w], dim=1),
            W.moe_s1: torch.cat([up_s, gate_s], dim=1),
            W.moe_w2: w2,
            W.moe_s2: s2,
        }

        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            mega_moe_wrapper.MegaMoeWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        captured = _FakeMegaMoE.instance.fp8_kwargs
        torch.testing.assert_close(captured["w1_fp8"], gate_w)
        torch.testing.assert_close(captured["w1_scale"], gate_s)
        torch.testing.assert_close(captured["w2_fp8"], w2)
        torch.testing.assert_close(captured["w2_scale"], s2)
        torch.testing.assert_close(captured["w3_fp8"], up_w)
        torch.testing.assert_close(captured["w3_scale"], up_s)

    def test_fp8_wrapper_uses_fp8_mega_moe_class_and_reorders_layout(self):
        config = _config(hidden_size=8, inter=4, swiglu_limit=10.0)
        up_w = torch.full((2, 4, 8), 3, dtype=torch.float32).to(torch.float8_e4m3fn)
        gate_w = torch.full((2, 4, 8), 7, dtype=torch.float32).to(torch.float8_e4m3fn)
        up_s = torch.full((2, 4, 1), 5, dtype=torch.int32)
        gate_s = torch.full((2, 4, 1), 11, dtype=torch.int32)
        w2 = torch.zeros((2, 8, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        s2 = torch.ones((2, 8, 1), dtype=torch.int32)
        weights = {
            W.moe_w1: torch.cat([up_w, gate_w], dim=1),
            W.moe_s1: torch.cat([up_s, gate_s], dim=1),
            W.moe_w2: w2,
            W.moe_s2: s2,
        }

        with patch.object(mega_moe_fp8_wrapper, "GLM5MegaMoEFP8", _FakeMegaMoE):
            mega_moe_fp8_wrapper.MegaMoeFp8Wrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        self.assertEqual(_FakeMegaMoE.instance.params["swiglu_limit"], 10.0)
        captured = _FakeMegaMoE.instance.fp8_kwargs
        torch.testing.assert_close(captured["w1_fp8"], gate_w)
        torch.testing.assert_close(captured["w1_scale"], gate_s)
        torch.testing.assert_close(captured["w2_fp8"], w2)
        torch.testing.assert_close(captured["w2_scale"], s2)
        torch.testing.assert_close(captured["w3_fp8"], up_w)
        torch.testing.assert_close(captured["w3_scale"], up_s)

    def test_missing_config_swiglu_limit_defaults_to_ten(self):
        config = _config(hidden_size=8, inter=4)
        delattr(config, "swiglu_limit")
        weights = {
            W.moe_w1: torch.zeros((2, 8, 4), dtype=torch.int8),
            W.moe_s1: torch.ones((2, 8, 2), dtype=torch.float32),
            W.moe_w2: torch.ones((2, 8, 2), dtype=torch.int8),
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
        }

        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            mega_moe_wrapper.MegaMoeWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        self.assertEqual(_FakeMegaMoE.instance.params["swiglu_limit"], 10.0)

    def test_fp8_se_wrapper_loads_routed_and_shared_fp8_weights(self):
        config = _config(hidden_size=8, inter=4, swiglu_limit=10.0)
        up_w = torch.full((2, 4, 8), 3, dtype=torch.float32).to(torch.float8_e4m3fn)
        gate_w = torch.full((2, 4, 8), 7, dtype=torch.float32).to(torch.float8_e4m3fn)
        routed_w2 = torch.zeros((2, 8, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        shared_w13 = torch.full((8, 8), 5, dtype=torch.float32).to(torch.float8_e4m3fn)
        shared_w2 = torch.full((8, 4), 11, dtype=torch.float32).to(torch.float8_e4m3fn)
        weights = {
            W.moe_w1: torch.cat([up_w, gate_w], dim=1),
            W.moe_s1: torch.ones((2, 8, 1), dtype=torch.int32),
            W.moe_w2: routed_w2,
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.int32),
            W.ffn_w13: shared_w13,
            W.ffn_s13: torch.ones((8, 2), dtype=torch.int32),
            W.ffn_w2: shared_w2,
            W.ffn_s2: torch.ones((8, 1), dtype=torch.int32),
        }

        with patch.object(mega_moe_fp8_se_wrapper, "GLM5MegaMoEFP8SE", _FakeMegaMoE):
            mega_moe_fp8_se_wrapper.MegaMoeFp8SEWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        captured_routed = _FakeMegaMoE.instance.fp8_kwargs
        torch.testing.assert_close(captured_routed["w1_fp8"], gate_w)
        torch.testing.assert_close(captured_routed["w3_fp8"], up_w)
        torch.testing.assert_close(captured_routed["w2_fp8"], routed_w2)
        captured_shared = _FakeMegaMoE.instance.shared_fp8_kwargs
        torch.testing.assert_close(captured_shared["w1_w"], shared_w13)
        torch.testing.assert_close(captured_shared["w2_w"], shared_w2)
        for key in (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2):
            self.assertNotIn(key, weights)
        self.assertTrue(_FakeMegaMoE.instance.fused_shared_jit_warmed)

    def test_from_params_swiglu_limit_defaults_to_ten(self):
        moe = mega_moe.GLM5MegaMoE.from_params(
            layer_id=0,
            dim=8,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=1,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=16,
        )

        self.assertEqual(moe.cfg.swiglu_limit, 10.0)

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

        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            mega_moe_wrapper.MegaMoeWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        captured = _FakeMegaMoE.instance.fp4_kwargs
        self.assertEqual(captured["w1_layout"], "up_gate")
        torch.testing.assert_close(captured["w1_w"][:, :4], up_w)
        torch.testing.assert_close(captured["w1_w"][:, 4:], gate_w)
        torch.testing.assert_close(captured["w1_s"][:, :4], up_s)
        torch.testing.assert_close(captured["w1_s"][:, 4:], gate_s)

    def test_fp4_up_gate_interleave_matches_deepgemm_layout(self):
        up = torch.arange(2 * 16 * 4, dtype=torch.int32).reshape(2, 16, 4)
        gate = up + 10000
        stacked = torch.cat([up, gate], dim=1)

        actual = mega_moe._interleave_stacked_up_gate(stacked)
        expected = torch.stack(
            [gate.reshape(2, 2, 8, 4), up.reshape(2, 2, 8, 4)], dim=2
        ).reshape(2, 32, 4)

        torch.testing.assert_close(actual, expected)
        self.assertNotEqual(actual.data_ptr(), stacked.data_ptr())

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

        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            mega_moe_wrapper.MegaMoeWrapper(
                config,
                _parallelism(role_type=RoleType.DECODE),
                weights,
                moe_config=None,
                layer_idx=0,
                max_generate_batch_size=8,
            )

        self.assertEqual(_FakeMegaMoE.instance.params["max_tokens_per_rank"], 32)

    def test_fused_wrapper_loads_shared_expert_fp8(self):
        config = _config(hidden_size=8, inter=4)
        w1_w = torch.full((8, 8), 3, dtype=torch.float32).to(torch.float8_e4m3fn)
        w2_w = torch.full((8, 4), 7, dtype=torch.float32).to(torch.float8_e4m3fn)
        weights = {
            W.moe_w1: torch.zeros((2, 8, 4), dtype=torch.int8),
            W.moe_s1: torch.ones((2, 8, 2), dtype=torch.float32),
            W.moe_w2: torch.ones((2, 8, 2), dtype=torch.int8),
            W.moe_s2: torch.ones((2, 8, 1), dtype=torch.float32),
            W.ffn_w13: w1_w,
            W.ffn_s13: torch.full((8, 2), 5, dtype=torch.float32),
            W.ffn_w2: w2_w,
            W.ffn_s2: torch.full((8, 1), 11, dtype=torch.float32),
        }

        with patch.object(mega_moe_fused_wrapper, "GLM5MegaMoEFused", _FakeMegaMoE):
            mega_moe_fused_wrapper.MegaMoeFusedWrapper(
                config, _parallelism(), weights, moe_config=None, layer_idx=0
            )

        captured = _FakeMegaMoE.instance.shared_fp8_kwargs
        self.assertTrue(_FakeMegaMoE.instance.fused_shared_jit_warmed)
        torch.testing.assert_close(captured["w1_w"], w1_w)
        torch.testing.assert_close(
            captured["w1_s"], torch.full((8, 2), 5, dtype=torch.float32)
        )
        torch.testing.assert_close(captured["w2_w"], w2_w)
        torch.testing.assert_close(
            captured["w2_s"], torch.full((8, 1), 11, dtype=torch.float32)
        )


if __name__ == "__main__":
    unittest.main()

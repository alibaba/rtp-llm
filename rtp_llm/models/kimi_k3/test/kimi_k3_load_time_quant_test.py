import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    split_kda_qkvg,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
from rtp_llm.models.kimi_k3.quantization import (
    KimiK3LoadTimeFp8Weight,
    get_kimi_k3_load_time_fp8_config,
    quantize_rank_local_fp8,
    wrap_kimi_k3_load_time_fp8_weight,
)
from rtp_llm.models_py.model_desc.kimi_k3_eagle3 import (
    _GatedEagle3MLA,
    _SplitFusedSiluAndMul,
    _split_checkpoint_native_fp8_ffn_weights,
)
from rtp_llm.models_py.modules.kimi_k3.kda.module import KimiK3KDA
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class _TensorSource(TensorSource):
    def __init__(self, tensors):
        self._tensors = tensors

    def load_tensor(self, name, data_type=torch.float16):
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name):
        return name in self._tensors

    def get_database(self):
        return None


class KimiK3LoadTimeQuantTest(unittest.TestCase):
    def test_eagle3_split_fp8_ffn_activation_accepts_gate_and_up(self):
        activation = _SplitFusedSiluAndMul()
        gate = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
        up = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)

        actual = activation(gate, up)

        torch.testing.assert_close(actual, torch.nn.functional.silu(gate) * up)

    def test_eagle3_keeps_checkpoint_native_fp8_split_ffn_projections(self):
        weights = {
            W.ffn_w1: torch.zeros((6, 4), dtype=torch.float8_e4m3fn),
            W.ffn_s1: torch.zeros((6, 1), dtype=torch.int32),
            W.ffn_w3: torch.zeros((6, 4), dtype=torch.float8_e4m3fn),
            W.ffn_s3: torch.zeros((6, 1), dtype=torch.int32),
        }

        split_weights, use_split_gate_up = (
            _split_checkpoint_native_fp8_ffn_weights(weights, hidden_size=4)
        )

        self.assertTrue(use_split_gate_up)
        self.assertIs(split_weights, weights)

    def test_eagle3_splits_checkpoint_native_fp8_merged_ffn_input_axis(self):
        weights = {
            W.ffn_w13: torch.zeros((6, 8), dtype=torch.float8_e4m3fn),
            W.ffn_s13: torch.zeros((6, 2), dtype=torch.int32),
        }

        split_weights, use_split_gate_up = (
            _split_checkpoint_native_fp8_ffn_weights(weights, hidden_size=4)
        )

        self.assertTrue(use_split_gate_up)
        self.assertNotIn(W.ffn_w13, split_weights)
        self.assertNotIn(W.ffn_s13, split_weights)
        self.assertEqual(split_weights[W.ffn_w1].shape, (6, 4))
        self.assertEqual(split_weights[W.ffn_w3].shape, (6, 4))
        self.assertEqual(split_weights[W.ffn_s1].shape, (6, 1))
        self.assertEqual(split_weights[W.ffn_s3].shape, (6, 1))

    def test_eagle3_uses_separate_mla_output_gate_when_present(self):
        attention = _GatedEagle3MLA.__new__(_GatedEagle3MLA)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 2
        attention.kv_lora_rank = 3
        attention.qk_rope_head_dim = 1
        attention._uses_separate_gate_layout = True
        attention.fused_qkv_a_proj = nn.Linear(2, 6, bias=False)
        attention._gate_proj = nn.Linear(2, 4, bias=False)

        hidden_states = torch.tensor([[1.0, 2.0]])
        fused_qkv, output_gate = attention._project_qkv_a_input(hidden_states)

        self.assertEqual(fused_qkv.shape, (1, 6))
        self.assertEqual(output_gate.shape, (1, 4))

    def test_kda_and_mla_switches_are_independent_and_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_kimi_k3_load_time_fp8_config(is_kda=True))
            self.assertIsNone(get_kimi_k3_load_time_fp8_config(is_kda=False))

        with mock.patch.dict(
            os.environ,
            {"KIMI_K3_W8A8_KDA": "1", "KIMI_K3_W8A8_MLA": "0"},
            clear=True,
        ):
            kda_config = get_kimi_k3_load_time_fp8_config(is_kda=True)
            self.assertIsNotNone(kda_config)
            self.assertFalse(kda_config.is_quanted())
            self.assertEqual(kda_config.group_size(), 128)
            self.assertIsNone(get_kimi_k3_load_time_fp8_config(is_kda=False))

    def test_rejects_invalid_switch_value(self):
        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_MLA": "yes"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "KIMI_K3_W8A8_MLA"):
                get_kimi_k3_load_time_fp8_config(is_kda=False)

    def test_k3_fmha_uses_mla_load_time_quant_config(self):
        config = KimiK3ModelConfig()
        config.quant_config = None

        with mock.patch.dict(
            os.environ,
            {"KIMI_K3_W8A8_KDA": "0", "KIMI_K3_W8A8_MLA": "1"},
            clear=True,
        ):
            fmha_quant_config = config.get_fmha_quant_config()

        self.assertIsNotNone(fmha_quant_config)
        self.assertEqual(fmha_quant_config.get_method(), "FP8_PER_BLOCK")
        self.assertEqual(fmha_quant_config.group_size(), 128)
        self.assertFalse(fmha_quant_config.is_quanted())

    def test_eagle3_accepts_checkpoint_native_fp8_quantization(self):
        config = KimiK3ModelConfig()
        config.ckpt_path = ""
        config.model_type = "kimi_k3_mla_swa_eagle3"
        config.config_dtype = "bfloat16"
        config.quantization = "FP8_PER_BLOCK"

        config.init_precision_config(None, None)

        self.assertIsNotNone(config.quant_config)
        self.assertEqual(config.quant_config.get_method(), "FP8_PER_BLOCK")

    def test_main_k3_still_rejects_framework_quantization(self):
        config = KimiK3ModelConfig()
        config.ckpt_path = ""
        config.model_type = "kimi_k3"
        config.config_dtype = "bfloat16"
        config.quantization = "FP8_PER_BLOCK"

        with self.assertRaisesRegex(ValueError, "runtime weight quantization"):
            config.init_precision_config(None, None)

    def test_quantizes_rank_local_runtime_layout(self):
        weight = torch.empty((130, 257), dtype=torch.bfloat16)
        weight[:128, :128] = 0.5
        weight[:128, 128:] = 1.0
        weight[128:, :128] = 1.5
        weight[128:, 128:] = 2.0

        quantized, scale = quantize_rank_local_fp8(weight, group_size=128)

        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(quantized.shape, (257, 130))
        self.assertEqual(scale.shape, (3, 2))
        expanded_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        reconstructed = (
            quantized.float() * expanded_scale[:257, :130]
        ).T.contiguous()
        torch.testing.assert_close(
            reconstructed, weight.float(), rtol=0.03, atol=0.03
        )

    def test_quantizes_and_pads_tp16_beta_output(self):
        weight = torch.empty((256, 6), dtype=torch.bfloat16)
        weight[:128, :3] = 0.5
        weight[:128, 3:] = 1.0
        weight[128:, :3] = 1.5
        weight[128:, 3:] = 2.0

        quantized, scale = quantize_rank_local_fp8(
            weight, group_size=128, pad_output_to=128
        )

        self.assertEqual(quantized.shape, (128, 256))
        self.assertEqual(scale.shape, (1, 2))
        expanded_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        reconstructed = quantized.float() * expanded_scale[:128, :256]
        torch.testing.assert_close(
            reconstructed[:6].T,
            weight.float(),
            rtol=0.03,
            atol=0.03,
        )
        torch.testing.assert_close(
            reconstructed[6:], torch.zeros_like(reconstructed[6:])
        )

    def test_kda_qkvg_split_keeps_each_projection_section(self):
        weight = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        load_config = SimpleNamespace(tp_size=2, tp_rank=1)
        linear_config = SimpleNamespace(
            linear_num_value_heads=2,
            linear_value_head_dim=2,
        )

        rank_local = split_kda_qkvg(weight, load_config, linear_config)

        expected_columns = [2, 3, 6, 7, 10, 11, 14, 15]
        torch.testing.assert_close(rank_local, weight[:, expected_columns])

    def test_kda_wraps_sglang_projection_scope(self):
        projection_names = (
            W.linear_attn_qkvg_w,
            W.linear_attn_f_a_w,
            W.linear_attn_f_b_w,
            W.linear_attn_b_w,
            W.linear_attn_out_w,
        )
        projections = [
            AtomicWeight(
                name,
                [CkptWeightInfo(f"self_attn.{index}.weight", identity)],
            )
            for index, name in enumerate(projection_names)
        ]

        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_KDA": "1"}, clear=True
        ):
            wrapped = [
                wrap_kimi_k3_load_time_fp8_weight(weight, is_kda=True)
                for weight in projections
            ]

        self.assertTrue(
            all(
                isinstance(weight, KimiK3LoadTimeFp8Weight)
                for weight in wrapped
            )
        )

    def test_beta_wrapper_pads_after_tp16_split(self):
        source_weight = LinearAttnAtomicWeight(
            W.linear_attn_b_w,
            [CkptWeightInfo("self_attn.b_proj.weight", identity)],
            transpose,
            SimpleNamespace(),
        )
        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_KDA": "1"}, clear=True
        ):
            wrapped = wrap_kimi_k3_load_time_fp8_weight(
                source_weight, is_kda=True
            )

        checkpoint_weight = torch.empty((96, 256), dtype=torch.bfloat16)
        checkpoint_weight[:48] = 0.5
        checkpoint_weight[48:] = 1.0
        loaded = wrapped._load_raw_tensor(
            _TensorSource({"self_attn.b_proj.weight": checkpoint_weight}),
            None,
            "cpu",
            SimpleNamespace(
                compute_dtype=torch.bfloat16,
                tp_size=16,
                tp_rank=15,
            ),
        )

        self.assertEqual(
            loaded[W.linear_attn_b_w].shape,
            (128, 256),
        )
        self.assertEqual(
            loaded[W.linear_attn_b_s].shape,
            (1, 2),
        )

    def test_kda_fp8_projection_uses_quantized_gate_linears_and_trims_beta(self):
        layer = KimiK3KDA.__new__(KimiK3KDA)
        nn.Module.__init__(layer)
        layer._uses_separate_projection_layout = True
        layer.projection_size = 2
        layer.local_heads = 2
        layer.kda_qkvg_proj = nn.Linear(2, 8, bias=False)
        layer.kda_f_a_proj = nn.Linear(2, 2, bias=False)
        layer.kda_f_b_proj = nn.Linear(2, 2, bias=False)
        layer.kda_beta_proj = nn.Linear(2, 128, bias=False)
        with torch.no_grad():
            layer.kda_qkvg_proj.weight.copy_(
                torch.arange(16, dtype=torch.float32).reshape(8, 2)
            )
            layer.kda_f_a_proj.weight.copy_(torch.eye(2))
            layer.kda_f_b_proj.weight.copy_(2 * torch.eye(2))
            layer.kda_beta_proj.weight.zero_()
            layer.kda_beta_proj.weight[:2].copy_(3 * torch.eye(2))

        hidden_states = torch.tensor([[1.0, 2.0]])
        (
            projected_qkv,
            q_projected,
            k_projected,
            v_projected,
            raw_gate,
            raw_beta,
            output_gate,
        ) = layer._project_fused_kda_inputs(
            hidden_states, prefill_sp_layout=None
        )

        self.assertEqual(projected_qkv.shape, (1, 6))
        torch.testing.assert_close(q_projected, torch.tensor([[2.0, 8.0]]))
        torch.testing.assert_close(k_projected, torch.tensor([[14.0, 20.0]]))
        torch.testing.assert_close(v_projected, torch.tensor([[26.0, 32.0]]))
        torch.testing.assert_close(output_gate, torch.tensor([[38.0, 44.0]]))
        torch.testing.assert_close(raw_gate, torch.tensor([[2.0, 4.0]]))
        self.assertEqual(raw_beta.shape, (1, 2))
        torch.testing.assert_close(raw_beta, torch.tensor([[3.0, 6.0]]))

    def test_wraps_only_selected_attention_weights_and_never_moe(self):
        attention = AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("self_attn.g_proj.weight", identity)],
        )
        moe = AtomicWeight(
            W.moe_w1,
            [CkptWeightInfo("block_sparse_moe.experts.w1", identity)],
        )
        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_MLA": "1"}, clear=True
        ):
            wrapped_attention = wrap_kimi_k3_load_time_fp8_weight(
                attention, is_kda=False
            )
            wrapped_moe = wrap_kimi_k3_load_time_fp8_weight(moe, is_kda=False)

        self.assertIsInstance(wrapped_attention, KimiK3LoadTimeFp8Weight)
        self.assertIs(wrapped_moe, moe)

    def test_wrapper_splits_before_quantizing(self):
        source_weight = AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("self_attn.g_proj.weight", identity)],
            process_fun=transpose,
        )
        config = get_kimi_k3_load_time_fp8_config
        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_MLA": "1"}, clear=True
        ):
            wrapped = KimiK3LoadTimeFp8Weight(
                source_weight, config(is_kda=False)
            )

        load_config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            tp_size=2,
            tp_rank=1,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            hidden_size=4,
            head_num=2,
            head_num_kv=2,
            size_per_head=2,
            moe_pure_tp_mode=False,
            bit=16,
        )
        checkpoint_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        loaded = wrapped._load_raw_tensor(
            _TensorSource({"self_attn.g_proj.weight": checkpoint_weight}),
            None,
            "cpu",
            load_config,
        )

        self.assertEqual(loaded[W.attn_gate_w].shape, (2, 4))
        self.assertEqual(loaded[W.attn_gate_s].shape, (1, 1))

    def test_mla_gate_split_uses_value_head_dim(self):
        source_weight = MlaAttnAtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("self_attn.g_proj.weight", identity)],
            process_fun=transpose,
            config=MlaConfig(head_num=4, v_head_dim=2),
        )
        with mock.patch.dict(
            os.environ, {"KIMI_K3_W8A8_MLA": "1"}, clear=True
        ):
            wrapped = KimiK3LoadTimeFp8Weight(
                source_weight,
                get_kimi_k3_load_time_fp8_config(is_kda=False),
            )

        load_config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            tp_size=2,
            tp_rank=1,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            hidden_size=4,
            head_num=4,
            head_num_kv=4,
            size_per_head=3,
            moe_pure_tp_mode=False,
            bit=16,
        )
        checkpoint_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        loaded = wrapped._load_raw_tensor(
            _TensorSource({"self_attn.g_proj.weight": checkpoint_weight}),
            None,
            "cpu",
            load_config,
        )

        self.assertEqual(loaded[W.attn_gate_w].shape, (4, 4))
        self.assertEqual(loaded[W.attn_gate_s].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()

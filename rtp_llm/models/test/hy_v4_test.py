import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.config.quant_config import (
    Fp8MxBlockWiseQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
)
from rtp_llm.model_loader.mxfp8_quant_weight import Mxfp8Weight
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.models.hy_v4 import (
    Hy4,
    Hy4MtpWeight,
    Hy4Weight,
    _move_indexer_rope_to_front,
    _transpose_stacked_gate_up,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, stack_


class Hy4ConfigTest(unittest.TestCase):
    def test_parse_explicit_layer_schedules(self):
        raw = {
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 16,
            "indexer_types": ["full", "shared", "full"],
            "intermediate_size": 48,
            "moe_intermediate_size": 12,
            "n_shared_experts": 1,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
            "routed_scaling_factor": 2.827,
            "norm_topk_prob": True,
            "enable_ihc": True,
            "hc_mult": 4,
            "gated_mla": True,
            "gating_type": "elementwise",
            "learnable_sink": True,
            "swiglu_limit": 10.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(raw))
            config = Hy4._create_config(tmp)
        self.assertEqual(config.model_type, "hy_v4")
        self.assertEqual(config.moe_layer_index, [1, 2])
        self.assertEqual(config.dense_inter_size, 48)
        self.assertEqual(config.inter_size, 12)
        self.assertEqual(config.scoring_func, 1)
        self.assertTrue(config.force_sparse_mla)
        self.assertFalse(config.attn_config.rope_config.is_neox_style)
        self.assertFalse(config.attn_config.rope_config.indexer_is_neox_style)
        self.assertEqual(config.indexer_layernorm_eps, 1e-6)
        self.assertEqual(config.indexer_scale_fmt, "ue8m0")
        self.assertFalse(config.indexer_use_hadamard)

    def test_indexer_layernorm_epsilon_is_independent_from_model_rmsnorm(self):
        raw = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 16,
            "indexer_types": ["full"],
            "mlp_layer_types": ["dense"],
            "intermediate_size": 48,
            "moe_intermediate_size": 12,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 1e-5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(raw))
            config = Hy4._create_config(tmp)
        self.assertEqual(config.layernorm_eps, 1e-5)
        self.assertEqual(config.indexer_layernorm_eps, 1e-6)
        self.assertEqual(config.indexer_scale_fmt, "ue8m0")
        self.assertFalse(config.indexer_use_hadamard)

    def test_rejects_unknown_indexer_type(self):
        raw = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 16,
            "indexer_types": ["silently_dense"],
            "intermediate_size": 48,
            "moe_intermediate_size": 12,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "indexer types"):
                Hy4._create_config(tmp)

    def test_requires_explicit_layer_schedules(self):
        raw = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 16,
            "intermediate_size": 48,
            "moe_intermediate_size": 12,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(KeyError, "indexer_types"):
                Hy4._create_config(tmp)


class Hy4WeightTest(unittest.TestCase):
    def test_indexer_checkpoint_layout_moves_rope_to_front_per_head(self):
        tensor = torch.arange(2 * 128 * 3).reshape(2 * 128, 3)
        actual = _move_indexer_rope_to_front(
            [tensor], output_dim=2 * 128, head_dim=128, rope_dim=64
        )
        expected = torch.cat(
            (
                tensor.reshape(2, 128, 3)[:, 64:],
                tensor.reshape(2, 128, 3)[:, :64],
            ),
            dim=1,
        ).reshape_as(tensor)
        torch.testing.assert_close(actual, expected)

    def test_indexer_coarse_scale_layout_is_unchanged(self):
        scale = torch.arange(2 * 3).reshape(2, 3)
        actual = _move_indexer_rope_to_front(
            [scale], output_dim=2 * 128, head_dim=128, rope_dim=64
        )
        torch.testing.assert_close(actual, scale)

    def test_indexer_layout_adapter_matches_vllm_rope_norm_chain(self):
        def interleaved_rope(tensor, offset):
            output = tensor.clone()
            angles = torch.arange(32, dtype=tensor.dtype) / 17
            even = tensor[offset : offset + 64 : 2]
            odd = tensor[offset + 1 : offset + 64 : 2]
            output[offset : offset + 64 : 2] = (
                even * angles.cos() - odd * angles.sin()
            )
            output[offset + 1 : offset + 64 : 2] = (
                odd * angles.cos() + even * angles.sin()
            )
            return output

        def layer_norm(tensor, weight, bias):
            normalized = (tensor - tensor.mean()) * torch.rsqrt(
                tensor.var(unbiased=False) + 1e-6
            )
            return normalized * weight + bias

        def swap(tensor):
            return _move_indexer_rope_to_front(
                [tensor], output_dim=128, head_dim=128, rope_dim=64
            )

        q = torch.arange(128, dtype=torch.float64) - 37
        wk_output = torch.arange(128, dtype=torch.float64).flip(0) + 11
        norm_weight = torch.arange(128, dtype=torch.float64) / 127 + 0.5
        norm_bias = torch.arange(128, dtype=torch.float64) / 251

        # vLLM/PTM path: K norm in checkpoint order, then RoPE on the tail.
        k = layer_norm(wk_output, norm_weight, norm_bias)
        q_roped = interleaved_rope(q, offset=64)
        k_roped = interleaved_rope(k, offset=64)

        # RTP path after load-time adaptation: projection rows and LayerNorm
        # parameters are all swapped, then the existing first-slice RoPE runs.
        k_swapped = layer_norm(
            swap(wk_output), swap(norm_weight), swap(norm_bias)
        )
        torch.testing.assert_close(k_swapped, swap(k))
        q_swapped_roped = interleaved_rope(swap(q), offset=0)
        k_swapped_roped = interleaved_rope(k_swapped, offset=0)

        # RTP stores the two half-heads in the opposite order, but after the
        # load-time permutation every value is exactly the corresponding vLLM
        # value. Per-vector absmax quantization and QK scores are invariant to
        # this fixed permutation.
        torch.testing.assert_close(q_swapped_roped, swap(q_roped))
        torch.testing.assert_close(k_swapped_roped, swap(k_roped))
        torch.testing.assert_close(
            q_swapped_roped * k_swapped_roped,
            swap(q_roped * k_roped),
        )

    def test_fused_gate_up_checkpoint_order_is_swapped_for_rtp(self):
        expert0 = torch.tensor([[1.0], [2.0], [11.0], [12.0]])
        expert1 = torch.tensor([[3.0], [4.0], [13.0], [14.0]])
        actual = _transpose_stacked_gate_up([expert0, expert1])
        expected = torch.tensor(
            [
                [[11.0], [12.0], [1.0], [2.0]],
                [[13.0], [14.0], [3.0], [4.0]],
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_fused_expert_mapping_keeps_stacked_ep_loader(self):
        loader = Hy4Weight.__new__(Hy4Weight)
        loader.moe_layer_index_ = [1]
        loader.has_fused_experts = True
        loader._align_size = 0
        loader._is_gated_activation = True
        loader.expert_num_ = 8
        loader.has_e_score_correction_bias = True
        weights = loader._get_hf_ffn_layer_weight_info(1)
        components = [c for weight in weights for c in weight.get_components()]
        by_name = {component.name: component for component in components}
        self.assertTrue(by_name[W.moe_w1].stacked_ckpt_keys)
        self.assertTrue(by_name[W.moe_w2].stacked_ckpt_keys)
        self.assertEqual(
            by_name[W.moe_w1].weights[0].name,
            "model.layers.{i}.mlp.experts.gate_up_proj",
        )
        self.assertIs(by_name[W.moe_w1].process_fun, _transpose_stacked_gate_up)
        self.assertEqual(by_name[W.moe_gate].data_type, torch.float32)

    def test_metadata_detection_accumulates_across_checkpoint_groups(self):
        loader = Hy4MtpWeight.__new__(Hy4MtpWeight)
        loader.q_use_lora = False
        loader.has_e_score_correction_bias = False
        loader.has_fused_experts = False
        loader.fused_gate_up_suffix = ""
        loader.fused_down_suffix = ""
        loader._process_meta(
            None,
            {
                "model.mtp_layers.0.self_attn.q_a_proj.weight",
                "model.mtp_layers.0.mlp.gate.e_score_correction_bias",
                "model.mtp_layers.0.mlp.experts.gate_up_proj",
                "model.mtp_layers.0.mlp.experts.down_proj",
            },
        )
        loader._process_meta(None, {"unrelated.finetune.weight"})
        self.assertTrue(loader.q_use_lora)
        self.assertTrue(loader.has_e_score_correction_bias)
        self.assertTrue(loader.has_fused_experts)

    def test_mtp_final_norm_uses_checkpoint_final_layernorm(self):
        loader = Hy4MtpWeight.__new__(Hy4MtpWeight)
        loader._num_layers = 1
        loader._hidden_size = 32
        loader._get_hf_layer_weight_info = lambda _: []
        info = loader._get_weight_info()
        by_name = {weight.name: weight for weight in info.weights}
        final_norm = by_name[W.multi_tokens_predict_final_ln_gamma]
        self.assertEqual(
            final_norm.weights[0].name,
            "model.mtp_layers.0.final_layernorm.weight",
        )


class Hy4Mxfp8WeightTest(unittest.TestCase):
    def setUp(self):
        self.quant_config = Fp8MxBlockWiseQuantConfig(
            is_quanted=True,
            checkpoint_scale_suffix=".weight_scale",
            packed_scale_suffix="_scale",
        )
        self.mla_config = MlaConfig(
            head_num=4,
            nope_head_dim=64,
            rope_head_dim=32,
            kv_lora_rank=128,
            v_head_dim=64,
            use_mla=True,
            q_use_lora=True,
        )

    @staticmethod
    def _ckpt_names(weight):
        return [item.name for item in weight.weights]

    def _assert_mapping(self, src, kernel_names, scale_names):
        self.assertTrue(Mxfp8Weight.support(self.quant_config, src))
        self.assertFalse(PerBlockFp8Weight.support(self.quant_config, src))
        wrapped = src.create(src, self.quant_config)
        self.assertIsInstance(wrapped, Mxfp8Weight)
        self.assertEqual(self._ckpt_names(wrapped.kernel), kernel_names)
        self.assertEqual(self._ckpt_names(wrapped.scale), scale_names)
        self.assertEqual(wrapped.kernel.data_type, torch.float8_e4m3fn)
        self.assertEqual(wrapped.scale.data_type, torch.float32)
        return wrapped

    def test_modelopt_config_uses_native_mxfp8_suffixes(self):
        raw = {
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "MXFP8",
                    "exclude_modules": [
                        "model.layers.0.self_attn.linear_gate"
                    ],
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(raw))
            config = QuantizationConfig.load_from_ckpt(tmp)
        self.assertIsInstance(config, Fp8MxBlockWiseQuantConfig)
        self.assertEqual(config.get_method(), "MXFP8")
        self.assertEqual(config.group_size(), 128)
        self.assertEqual(config.weight_block_size, [1, 32])
        self.assertEqual(config.checkpoint_scale_suffix, ".weight_scale")
        self.assertEqual(config.packed_scale_suffix, "_scale")

    def test_indexer_q_layout_merge_propagates_to_mxfp8_weight_and_scale(self):
        prefix = "model.layers.{i}.self_attn.indexer.wq_b"

        def merge(ts):
            return _move_indexer_rope_to_front(
                ts, output_dim=128, head_dim=128, rope_dim=64
            )

        src = MlaAttnAtomicWeight(
            W.mla_indexer_qb_w,
            [CkptWeightInfo(f"{prefix}.weight", merge)],
            identity,
            config=self.mla_config,
        )
        wrapped = self._assert_mapping(
            src,
            [f"{prefix}.weight"],
            [f"{prefix}.weight_scale"],
        )
        kernel = torch.arange(128 * 4).reshape(128, 4)
        scale = torch.arange(128 * 2).reshape(128, 2)
        expected_kernel = torch.cat((kernel[64:], kernel[:64]))
        expected_scale = torch.cat((scale[64:], scale[:64]))
        torch.testing.assert_close(
            wrapped.kernel.weights[0].merge_fun([kernel]), expected_kernel
        )
        torch.testing.assert_close(
            wrapped.scale.weights[0].merge_fun([scale]), expected_scale
        )

    def test_indexer_wk_is_dequantized_to_bf16(self):
        prefix = "model.layers.{i}.self_attn.indexer.wk"

        def merge(ts):
            return _move_indexer_rope_to_front(
                ts, output_dim=128, head_dim=128, rope_dim=64
            )

        src = MlaAttnAtomicWeight(
            W.mla_indexer_k_w,
            [CkptWeightInfo(f"{prefix}.weight", merge)],
            identity,
            config=self.mla_config,
        )
        self.assertTrue(Mxfp8Weight.support(self.quant_config, src))
        wrapped = src.create(src, self.quant_config)
        self.assertIsInstance(wrapped, Mxfp8Weight)
        self.assertEqual(wrapped.kernel.data_type, torch.bfloat16)
        self.assertIsNone(wrapped.scale)
        self.assertEqual(
            self._ckpt_names(wrapped.kernel),
            [f"{prefix}.weight", f"{prefix}.weight_scale"],
        )

        weight = torch.arange(128 * 32, dtype=torch.float32).reshape(128, 32)
        weight = (weight.remainder(31) - 15).to(torch.float8_e4m3fn)
        scale_exponents = torch.full((128, 1), 125.0)
        dequantized = (weight.float() * 0.25).bfloat16()
        expected = merge([dequantized])
        actual = wrapped.kernel.process_fun([weight, scale_exponents])
        torch.testing.assert_close(actual, expected)

    def test_mla_and_indexer_mappings(self):
        prefix = "model.layers.{i}.self_attn"
        cases = [
            (W.attn_o_w, [f"{prefix}.o_proj.weight"]),
            (W.mla_kv_b_w, [f"{prefix}.kv_b_proj.weight"]),
            (W.mla_q_b_w, [f"{prefix}.q_b_proj.weight"]),
            (W.mla_indexer_qb_w, [f"{prefix}.indexer.wq_b.weight"]),
        ]
        for internal_name, kernel_names in cases:
            with self.subTest(internal_name=internal_name):
                src = MlaAttnAtomicWeight(
                    internal_name,
                    [CkptWeightInfo(name, identity) for name in kernel_names],
                    identity,
                    config=self.mla_config,
                )
                scale_names = [
                    name[: -len(".weight")] + ".weight_scale"
                    for name in kernel_names
                ]
                self._assert_mapping(src, kernel_names, scale_names)

        fused_names = [
            f"{prefix}.q_a_proj.weight",
            f"{prefix}.kv_a_proj_with_mqa.weight",
        ]
        fused = MlaAttnAtomicWeight(
            W.mla_fusedqkrope_w,
            [CkptWeightInfo(name, identity) for name in fused_names],
            identity,
            config=self.mla_config,
        )
        self._assert_mapping(
            fused,
            fused_names,
            [name[: -len(".weight")] + ".weight_scale" for name in fused_names],
        )

    def test_absorbed_mla_bmm_weights_are_bf16(self):
        source_name = "model.layers.{i}.self_attn.kv_b_proj.weight"
        for internal_name in (W.mla_kc, W.mla_vc):
            with self.subTest(internal_name=internal_name):
                src = MlaAttnAtomicWeight(
                    internal_name,
                    [CkptWeightInfo(source_name, identity)],
                    identity,
                    config=self.mla_config,
                )
                wrapped = src.create(src, self.quant_config)
                self.assertIsInstance(wrapped, Mxfp8Weight)
                self.assertEqual(wrapped.kernel.data_type, torch.bfloat16)
                self.assertIsNone(wrapped.scale)
                self.assertEqual(
                    self._ckpt_names(wrapped.kernel),
                    [source_name, source_name.replace(".weight", ".weight_scale")],
                )

    def test_dense_and_shared_ffn_mappings(self):
        for module_prefix in (
            "model.layers.{i}.mlp",
            "model.layers.{i}.mlp.shared_experts",
        ):
            ffn_config = FfnConfig(
                align_size=0, is_gated_activation=True, is_moe=False
            )
            gate = f"{module_prefix}.gate_proj.weight"
            up = f"{module_prefix}.up_proj.weight"
            w13 = FfnAtomicWeight(
                W.ffn_w13,
                [CkptWeightInfo(gate), CkptWeightInfo(up)],
                identity,
                config=ffn_config,
            )
            self._assert_mapping(
                w13,
                [gate, up],
                [
                    gate[: -len(".weight")] + ".weight_scale",
                    up[: -len(".weight")] + ".weight_scale",
                ],
            )

            down = f"{module_prefix}.down_proj.weight"
            w2 = FfnAtomicWeight(
                W.ffn_w2,
                [CkptWeightInfo(down)],
                identity,
                config=ffn_config,
            )
            self._assert_mapping(
                w2,
                [down],
                [down[: -len(".weight")] + ".weight_scale"],
            )

    def test_packed_routed_expert_mappings(self):
        moe_config = MoeConfig(align_size=0, expert_num=256)
        gate_up = "model.layers.{i}.mlp.experts.gate_up_proj"
        w1 = MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo(gate_up)],
            _transpose_stacked_gate_up,
            config=moe_config,
            stacked_ckpt_keys=True,
        )
        wrapped_w1 = self._assert_mapping(
            w1, [gate_up], [gate_up + "_scale"]
        )
        self.assertTrue(wrapped_w1.kernel.stacked_ckpt_keys)
        self.assertTrue(wrapped_w1.scale.stacked_ckpt_keys)

        down = "model.layers.{i}.mlp.experts.down_proj"
        w2 = MoeAtomicWeight(
            W.moe_w2,
            [CkptWeightInfo(down)],
            stack_,
            config=moe_config,
            stacked_ckpt_keys=True,
        )
        self._assert_mapping(w2, [down], [down + "_scale"])

    def test_modelopt_excluded_linear_stays_unquantized(self):
        self.quant_config.exclude_modules = {
            "model.layers.0.self_attn.linear_gate"
        }
        src = MlaAttnAtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.linear_gate.weight")],
            identity,
            config=self.mla_config,
        )
        src.layer_id = 0
        self.assertFalse(Mxfp8Weight.support(self.quant_config, src))


if __name__ == "__main__":
    unittest.main()

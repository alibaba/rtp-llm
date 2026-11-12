"""CPU coverage of NVFP4 decoding, load-time conversion and mixed BF16 layers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    _apply_mega_moe_fp4_wrappers,
)
from rtp_llm.model_loader.offline_modelopt_fp4_quant_weight import (
    is_offline_mega_moe_fp4_ckpt,
)
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineMegaMoeFp4FromNvfp4Weight,
    OnlineMegaMoeFp4Weight,
    dequantize_nvfp4_expert,
    wrap_moe_for_mega_moe,
)
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, stack_, stack_moe_w1


def nvfp4_config():
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
            }
        },
        "ignore": [
            "model.layers.0*",
            "model.layers.1.*",
            "model.layers.2.*",
            "model.layers.*.self_attn*",
            "model.layers.*.mlp.shared_experts*",
            "model.layers.45*",
        ],
    }


class MemorySource(TensorSource):
    def __init__(self, tensors):
        self.tensors = tensors
        self.loaded = []

    def load_tensor(self, name, data_type=torch.float16):
        self.loaded.append(name)
        return [self.tensors[name].to(data_type)]

    def has_tensor(self, name):
        return name in self.tensors

    def get_database(self):
        return None


def moe_weight(name=W.moe_w1, prefix="model.layers.{i}.mlp.experts"):
    projections = ("up_proj", "gate_proj") if name == W.moe_w1 else ("down_proj",)
    return MoeAtomicWeight(
        name,
        [
            CkptWeightInfo(prefix + ".{expert_id}." + p + ".weight", identity)
            for p in projections
        ],
        stack_moe_w1 if name == W.moe_w1 else stack_,
        config=MoeConfig(expert_num=4),
    )


class Nvfp4ToMxfp4Test(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "config.json").write_text(
            json.dumps({"quantization_config": nvfp4_config()})
        )
        self.quant = QuantizationConfig.load_from_ckpt(self.directory.name)

    def test_all_codes_nibble_order_and_block_scales(self):
        codes = torch.arange(16, dtype=torch.uint8).repeat(2).reshape(1, 32)
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        scale = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        expected = torch.cat((values * 0.25, values * 0.5)).reshape(1, 32).bfloat16()
        actual = dequantize_nvfp4_expert(packed, scale, torch.tensor(0.25))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_scale_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale shape"):
            dequantize_nvfp4_expert(
                torch.zeros(2, 16, dtype=torch.uint8),
                torch.ones(2, 1, dtype=torch.float8_e4m3fn),
                torch.tensor(1.0),
            )

    def test_real_deepgemm_quantization_matches_projection_reference(self):
        from deep_gemm.utils import per_token_cast_to_fp4

        src = moe_weight()
        converter = OnlineMegaMoeFp4FromNvfp4Weight(src)
        tensors, references = {}, []
        values = [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        for j, ckpt in enumerate(src.weights):
            name = ckpt.name.format(i=3, expert_id=1)
            packed = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
            scales = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float8_e4m3fn
            )
            global_scale = torch.tensor(0.375 if j == 0 else 1.25)
            tensors[name] = packed
            tensors[name[:-7] + ".weight_scale"] = scales
            tensors[name[:-7] + ".weight_scale_2"] = global_scale
            # Scalar reference independently decodes nibble order and the 16-value blocks.
            dense = torch.empty(2, 64)
            for row in range(2):
                for col in range(64):
                    byte = int(packed[row, col // 2])
                    code = (byte >> (4 * (col % 2))) & 15
                    dense[row, col] = (
                        values[code]
                        * float(scales[row, col // 16])
                        * float(global_scale)
                    )
            references.append(
                per_token_cast_to_fp4(dense.bfloat16(), use_ue8m0=True, gran_k=32)
            )
        config = SimpleNamespace(get_selected_experts=lambda *_: [1])
        actual = converter._load_raw_tensor(MemorySource(tensors), 3, "cpu", config)
        expected_packed = torch.cat([p for p, _ in references]).unsqueeze(0)
        expected_scales = torch.cat([s for _, s in references]).unsqueeze(0)
        torch.testing.assert_close(actual[W.moe_w1], expected_packed, rtol=0, atol=0)
        torch.testing.assert_close(actual[W.moe_s1], expected_scales, rtol=0, atol=0)

    def test_config_retains_ignore_and_bf16_layers(self):
        self.assertIsInstance(self.quant, ModelOptFp4Config)
        self.assertEqual(self.quant.exclude_modules, set(nvfp4_config()["ignore"]))
        for layer in (0, 1, 2, 45):
            src = moe_weight()
            src.layer_id = layer
            self.assertIs(src.create(src, self.quant), src)
        for layer in (3, 44):
            src = moe_weight()
            src.layer_id = layer
            wrapped = src.create(src, self.quant)
            self.assertIsInstance(wrapped, PerGroupFp4Weight)
            self.assertIsInstance(
                wrap_moe_for_mega_moe(wrapped), OnlineMegaMoeFp4FromNvfp4Weight
            )
        shared = FfnAtomicWeight(
            W.ffn_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.shared_experts.down_proj.weight")],
            config=FfnConfig(),
        )
        shared.layer_id = 3
        self.assertIs(shared.create(shared, self.quant), shared)

    def test_mtp_concrete_offset_takes_precedence_over_runtime_layer(self):
        src = moe_weight(prefix="model.layers.45.mlp.experts")
        src.layer_id = 0
        self.assertIs(src.create(src, self.quant), src)
        self.assertIsInstance(wrap_moe_for_mega_moe(src), OnlineMegaMoeFp4Weight)

    def test_nvfp4_never_uses_offline_mxfp4_loader(self):
        database = MagicMock()
        database.path = self.directory.name
        database.get_pretrain_tensor_names.return_value = [
            "model.layers.3.mlp.experts.0.up_proj.weight_scale"
        ]
        self.assertFalse(is_offline_mega_moe_fp4_ckpt(database))
        database.path = None
        self.assertTrue(is_offline_mega_moe_fp4_ckpt(database))  # legacy MXFP4
        database.get_pretrain_tensor_names.return_value += [
            "model.layers.3.mlp.experts.0.up_proj.weight_scale_2"
        ]
        self.assertFalse(is_offline_mega_moe_fp4_ckpt(database))
        for has_database in (True, False):
            src = moe_weight()
            src.layer_id = 3
            weights = ModelWeightInfo([], [[src.create(src, self.quant)]])
            with patch.dict(os.environ, {"MOE_STRATEGY": "mega_moe"}):
                result = _apply_mega_moe_fp4_wrappers(
                    weights, database if has_database else None, self.quant
                )
            self.assertIsInstance(
                result.layer_weights[0][0], OnlineMegaMoeFp4FromNvfp4Weight
            )

    def test_separate_scale_shards_survive_file_pruning(self):
        wrapped = OnlineMegaMoeFp4FromNvfp4Weight(moe_weight())
        info = ModelWeightInfo([], [[wrapped]])
        patterns = ModelDeployWeightInfo._collect_ckpt_tensor_name_regexes(info)
        for suffix in (".weight", ".weight_scale", ".weight_scale_2"):
            for proj in ("gate_proj", "up_proj"):
                name = "model.layers.3.mlp.experts.2." + proj + suffix
                self.assertTrue(any(p.fullmatch(name) for p in patterns), name)

    def test_ordinary_nvfp4_strategy_is_unchanged(self):
        src = moe_weight()
        src.layer_id = 3
        wrapped = src.create(src, self.quant)
        info = ModelWeightInfo([], [[wrapped]])
        with patch.dict(os.environ, {"MOE_STRATEGY": ""}):
            self.assertIs(
                _apply_mega_moe_fp4_wrappers(info, quant_config=self.quant), info
            )
        self.assertIs(info.layer_weights[0][0], wrapped)

    def test_streaming_preserves_expert_order_and_independent_projection_scales(self):
        for name in (W.moe_w1, W.moe_w2):
            with self.subTest(name=name):
                src = moe_weight(name)
                wrapped = OnlineMegaMoeFp4FromNvfp4Weight(src)
                tensors, expected = {}, []
                # Deliberately non-contiguous and duplicated experts, as with EPLB.
                selected = [3, 1, 3]
                for expert in selected:
                    parts = []
                    for j, ckpt in enumerate(src.weights):
                        key = ckpt.name.format(i=3, expert_id=expert)
                        # 0x21 means [0.5, 1.0], not [1.0, 0.5].
                        tensors[key] = torch.full((2, 16), 0x21, dtype=torch.uint8)
                        tensors[key[:-7] + ".weight_scale"] = torch.tensor(
                            [[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e4m3fn
                        )
                        factor = (expert + 1) * (j + 1)
                        tensors[key[:-7] + ".weight_scale_2"] = torch.tensor(
                            float(factor)
                        )
                        base = (
                            torch.tensor([0.5, 1.0])
                            .repeat(16)
                            .reshape(1, 32)
                            .repeat(2, 1)
                        )
                        scales = torch.tensor(
                            [[1.0, 2.0], [4.0, 8.0]]
                        ).repeat_interleave(16, -1)
                        parts.append((base * scales * factor).bfloat16())
                    expected.append(torch.cat(parts, dim=0).unsqueeze(0))
                source = MemorySource(tensors)
                load_config = SimpleNamespace(get_selected_experts=lambda *_: selected)
                observed = []

                def quantizer(expert, block_size):
                    self.assertEqual(block_size, 32)
                    self.assertEqual(expert.shape[0], 1)
                    observed.append(expert.clone())
                    e, n, k = expert.shape
                    return torch.full(
                        (e, n, k // 2), len(observed), dtype=torch.int8
                    ), torch.ones(e, n, k // 32)

                with patch(
                    "rtp_llm.model_loader.online_modelopt_fp4_quant_weight.quantize_moe_weight_to_fp4_ue8m0",
                    side_effect=quantizer,
                ):
                    result = wrapped._load_raw_tensor(source, 3, "cpu", load_config)
                for actual, reference in zip(observed, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertEqual(len(observed), len(selected))
                self.assertEqual(
                    set(source.loaded), wrapped.get_tensor_names(3, load_config)
                )
                self.assertFalse(any(n.endswith(".input_scale") for n in source.loaded))
                self.assertEqual(result[name].dtype, torch.int8)
                self.assertEqual(result[wrapped.scale.name].dtype, torch.float32)
                self.assertEqual(result[name][:, 0, 0].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

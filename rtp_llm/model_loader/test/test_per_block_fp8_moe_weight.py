"""CPU coverage for loading per-expert FP8 checkpoint weights and scales."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.offline_modelopt_fp4_quant_weight import (
    is_offline_mega_moe_fp4_ckpt,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class InMemoryTensorSource(TensorSource):
    def __init__(self, tensors):
        self.tensors = tensors

    def load_tensor(self, name, data_type=torch.float16):
        return [self.tensors[name].to(data_type)]

    def has_tensor(self, name):
        return name in self.tensors

    def get_database(self):
        return None


class PerBlockFp8MoeWeightTest(unittest.TestCase):
    def test_fp4_checkpoint_detection_does_not_confuse_fp8_scales(self):
        prefix = "model.layers.0.mlp.experts.0.gate_proj"
        for config, suffix, expected in (
            ({"quantization_config": {"expert_dtype": "fp4"}}, ".scale", True),
            ({"quantization_config": {"expert_dtype": "fp8"}}, ".scale", False),
            ({}, ".weight_scale", True),
            ({}, ".weight_scale_inv", False),
            (None, ".weight_scale", True),
            (None, ".scale", False),
        ):
            with self.subTest(
                config=config, suffix=suffix
            ), tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(
                    json.dumps(config) if config is not None else "invalid json"
                )
                database = SimpleNamespace(
                    path=directory, get_pretrain_tensor_names=lambda: [prefix + suffix]
                )
                self.assertEqual(is_offline_mega_moe_fp4_ckpt(database), expected)

    def _check_checkpoint_load(self, weight_name, scale_fmt, suffix, scale_dtype):
        layer_id = 3
        # Exercise expert remapping as well as the gate/up projection ordering.
        selected_experts = [2, 0]
        load_config = LoadConfig.model_construct(
            compute_dtype=torch.bfloat16,
            ep_size=1,
            ep_rank=0,
            phy2log=[[], [], [], selected_experts],
        )
        projections = (
            ["gate_proj", "up_proj"] if weight_name == W.moe_w1 else ["down_proj"]
        )
        prefix = "model.layers.{i}.mlp.experts.{expert_id}."
        src = MoeAtomicWeight(
            weight_name,
            [CkptWeightInfo(prefix + proj + ".weight") for proj in projections],
            config=MoeConfig(expert_num=3),
        )
        quant_config = Fp8BlockWiseQuantConfig(is_quanted=True, scale_fmt=scale_fmt)
        wrapped = src.create(src, quant_config)
        self.assertIsInstance(wrapped, PerBlockFp8Weight)

        weight_shape = (128, 256) if weight_name == W.moe_w1 else (256, 128)
        scale_shape = tuple(size // 128 for size in weight_shape)
        tensors = {}
        expected_weights = []
        expected_scales = []
        for expert_id in selected_experts:
            expert_weights = []
            expert_scales = []
            for projection_id, projection in enumerate(projections):
                key = prefix.format(i=layer_id, expert_id=expert_id) + projection
                weight = torch.full(weight_shape, 1.0 + expert_id + projection_id)
                scale = torch.exp2(
                    torch.arange(2, dtype=torch.float32).reshape(scale_shape)
                    + expert_id
                    + 3 * projection_id
                )
                tensors[key + ".weight"] = weight.to(torch.float8_e4m3fn)
                tensors[key + suffix] = scale.to(scale_dtype)
                expert_weights.append(weight)
                expert_scales.append(scale)
            expected_weights.append(torch.cat(expert_weights, dim=0))
            expected_scales.append(torch.cat(expert_scales, dim=0))

        # Read the actual checkpoint keys and merge experts on CPU. Device layout
        # conversion is independent of checkpoint naming and is tested elsewhere.
        loaded = wrapped._load_raw_tensor(
            InMemoryTensorSource(tensors), layer_id, "cpu", load_config
        )
        self.assertEqual(wrapped.get_tensor_names(layer_id, load_config), set(tensors))
        self.assertEqual(loaded[wrapped.kernel.name].dtype, torch.float8_e4m3fn)
        self.assertEqual(loaded[wrapped.scale.name].dtype, scale_dtype)
        torch.testing.assert_close(
            loaded[wrapped.kernel.name].float(),
            torch.stack(expected_weights),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            loaded[wrapped.scale.name].float(),
            torch.stack(expected_scales),
            rtol=0,
            atol=0,
        )

    def test_native_ue8m0_gate_up_checkpoint(self):
        self._check_checkpoint_load(W.moe_w1, "ue8m0", ".scale", torch.float8_e8m0fnu)

    def test_native_ue8m0_down_checkpoint(self):
        self._check_checkpoint_load(W.moe_w2, "ue8m0", ".scale", torch.float8_e8m0fnu)

    def test_legacy_float32_checkpoints(self):
        for weight_name in (W.moe_w1, W.moe_w2):
            for scale_fmt in (None, "float32"):
                with self.subTest(weight_name=weight_name, scale_fmt=scale_fmt):
                    self._check_checkpoint_load(
                        weight_name, scale_fmt, ".weight_scale_inv", torch.float32
                    )


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextBaseWeight
from rtp_llm.models.qwen4_exp.moe_nvfp4 import (
    Qwen4Nvfp4MoeWeight,
    validate_nvfp4_manifest,
)
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_mtp import Qwen4ExpMTP
from rtp_llm.models.qwen4_exp.test.config_fixture import qwen4_config
from rtp_llm.utils.model_weight import W


def dequantize_nvfp4(weight, scale, scale_2, dtype):
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("Qwen4 NVFP4 requires packed uint8 expert matrices")
    if (
        scale.dtype != torch.float8_e4m3fn
        or scale.shape != (weight.shape[0], weight.shape[1] // 8)
        or weight.shape[1] % 8
        or scale_2.numel() != 1
        or not torch.isfinite(scale_2).all()
        or not (scale_2 > 0).all()
    ):
        raise ValueError(
            "Qwen4 NVFP4 requires E4M3 group-16 and positive scalar scales"
        )
    # E2M1 stores adjacent K values in the low then high nibble.
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    values = levels[codes].reshape(weight.shape[0], -1, 16)
    return (
        (values * scale.float().unsqueeze(-1) * scale_2.float())
        .reshape(weight.shape[0], -1)
        .to(dtype)
    )


def _config():
    config = qwen4_config()
    config["text_config"]["mtp"] = {
        "hybrid": True,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "rope_theta": 10000000,
    }
    config["text_config"]["mtp_num_hidden_layers"] = 1
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "with_input_scale": True,
    }
    return config


class Qwen4Nvfp4Test(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("QWEN4_NVFP4_FIXTURE"), "real NVFP4 fixture not supplied"
    )
    def test_real_experts_against_independent_decode(self):
        with safe_open(os.environ["QWEN4_NVFP4_FIXTURE"], framework="pt") as tensors:
            weights = [name for name in tensors.keys() if name.endswith(".weight")]
            self.assertGreater(len(weights), 0)
            for name in weights:
                with self.subTest(name=name):
                    weight = tensors.get_tensor(name)
                    stem = name.removesuffix(".weight")
                    scale = tensors.get_tensor(stem + ".weight_scale")
                    outer = tensors.get_tensor(stem + ".weight_scale_2")
                    actual = dequantize_nvfp4(weight, scale, outer, torch.bfloat16)
                    # Independent E2M1 sign/exponent/mantissa decode, without
                    # the production lookup table or tensor source adapter.
                    packed = weight.numpy()
                    codes = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
                    codes[:, 0::2], codes[:, 1::2] = packed % 16, packed // 16
                    exponent, mantissa = (codes // 2) % 4, codes % 2
                    magnitude = np.where(
                        exponent == 0,
                        mantissa * 0.5,
                        (1.0 + mantissa * 0.5)
                        * np.exp2(exponent.astype(np.float32) - 1),
                    )
                    values = np.where(codes >= 8, -magnitude, magnitude).astype(
                        np.float32
                    )
                    values *= np.repeat(scale.float().numpy(), 16, axis=1)
                    values *= outer.numpy().item()
                    expected = torch.from_numpy(values).bfloat16()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_factory_retains_bf16_ple_and_prequantized_experts(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps(_config()))
            for model in (Qwen4Exp, Qwen4ExpMTP):
                with self.subTest(model=model.__name__):
                    config = model.create_config(root)
                    config.quant_config = ModelOptFp4Config(4, 16, True)
                    config.quant_algo.setQuantAlgo("modelopt_fp4", 4, 16)
                    model._post_build_model_config(config)
                    self.assertTrue(config._qwen4_nvfp4_source)
                    self.assertFalse(config._qwen4_fp8_source)
                    self.assertFalse(config._qwen4_full_fp8_nvfp4)
                    self.assertTrue(config.quant_config.is_quanted())
                    self.assertTrue(config.quant_algo.isQuant())
                    self.assertEqual(config.expert_weight_dtype, "fp4")
                    if not config.is_mtp:
                        self.assertGreater(config._extra_weight_bytes, 100e9)

    def test_decode_all_nibbles_and_independent_group_scales(self):
        packed = torch.tensor(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8
        )
        scale = torch.tensor([[2.0, 0.5]], dtype=torch.float8_e4m3fn)
        actual = dequantize_nvfp4(packed, scale, torch.tensor([0.25]), torch.bfloat16)
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        expected = torch.cat((values * 0.5, values * 0.125)).reshape(1, -1).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_bad_scales(self):
        for outer in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(outer=outer), self.assertRaises(ValueError):
                dequantize_nvfp4(
                    torch.zeros(2, 8, dtype=torch.uint8),
                    torch.ones(2, 1).to(torch.float8_e4m3fn),
                    torch.tensor([outer]),
                    torch.bfloat16,
                )

    def test_manifest_rejects_missing_or_nonexpert_scales(self):
        config = _config()
        config["text_config"].update(num_hidden_layers=1, num_experts=1)
        keys = {
            f"{prefix}.layers.0.mlp.experts.0.{projection}_proj.{suffix}": "model.safetensors"
            for prefix in ("model.language_model", "mtp")
            for projection in ("gate", "up", "down")
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        }
        validate_nvfp4_manifest(config, keys)
        for bad in (
            {**keys, "lm_head.weight_scale": "model.safetensors"},
            {k: v for k, v in keys.items() if not k.endswith("up_proj.weight_scale_2")},
        ):
            with self.assertRaises(ValueError):
                validate_nvfp4_manifest(config, bad)

    def test_fused_outer_scale_mismatch_is_rejected(self):
        class Source(TensorSource):
            def load_tensor(self, name, data_type):
                return [torch.tensor(1.0 if ".up_proj." in name else 2.0)]

        source = Qwen3NextBaseWeight._create_moe_expert_weights(
            SimpleNamespace(prefix="model."), MoeConfig(expert_num=2)
        )[1]
        descriptor = Qwen4Nvfp4MoeWeight(source, ModelOptFp4Config(4, 16, True))
        load = SimpleNamespace(get_selected_experts=lambda *_: [0, 1])
        with self.assertRaisesRegex(ValueError, "equal up/gate"):
            descriptor._load_raw_tensor(Source(), 0, "cpu", load)

    def test_descriptor_preserves_packed_expert_merge_order(self):
        class Source(TensorSource):
            def load_tensor(self, name, data_type):
                parts = name.split(".")
                expert = int(parts[parts.index("experts") + 1])
                projection = {"up_proj": 0x11, "gate_proj": 0x22, "down_proj": 0x33}[
                    parts[-2]
                ]
                if name.endswith(".weight"):
                    return [
                        torch.full((2, 8), projection + 16 * expert, dtype=torch.uint8)
                    ]
                if name.endswith(".weight_scale"):
                    return [torch.ones(2, 1).to(torch.float8_e4m3fn)]
                return [torch.tensor([2.0])]

        originals = Qwen3NextBaseWeight._create_moe_expert_weights(
            SimpleNamespace(prefix="model."), MoeConfig(expert_num=2)
        )
        load = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            moe_pure_tp_preshard=False,
            get_selected_experts=lambda *_: [1, 0],
        )
        for source in originals:
            descriptor = Qwen4Nvfp4MoeWeight(source, ModelOptFp4Config(4, 16, True))
            raw = descriptor._load_raw_tensor(Source(), 0, "cpu", load)[source.name]
            self.assertEqual(raw.shape, (2, 4 if source.name == W.moe_w1 else 2, 8))
            rows = (
                [[0x21, 0x21, 0x32, 0x32], [0x11, 0x11, 0x22, 0x22]]
                if source.name == W.moe_w1
                else [[0x43, 0x43], [0x33, 0x33]]
            )
            expected = (
                torch.tensor(rows, dtype=torch.uint8).unsqueeze(-1).expand_as(raw)
            )
            torch.testing.assert_close(raw, expected, rtol=0, atol=0)
            self.assertEqual(raw.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()

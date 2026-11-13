import base64
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import torch

from rtp_llm.config.nvfp4_utils import reciprocal_global_scale
from rtp_llm.config.quant_config import (
    CompressedTensorsNvfp4Config,
    ModelOptFp4Config,
    QuantizationConfig,
)


class CompressedTensorsNvfp4Test(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).parent / "fixtures/nvfp4/config.json").read_text()
        )
        self.quant = self.config["quantization_config"]

    def load(self, config):
        def exists(path):
            return str(path).endswith("config.json")

        with patch(
            "rtp_llm.config.quant_config.os.path.exists", side_effect=exists
        ), patch("builtins.open", mock_open(read_data=json.dumps(config))):
            return QuantizationConfig.load_from_ckpt("unused")

    def test_real_checkpoint_dispatch(self):
        result = self.load(self.config)
        self.assertIsInstance(result, CompressedTensorsNvfp4Config)
        self.assertEqual(result.get_algo(), ModelOptFp4Config.get_algo())
        self.assertEqual(result.group_size(), 16)
        self.assertTrue(result.is_module_excluded("lm_head.weight"))
        self.assertFalse(result.is_module_excluded("model.lm_head_extra.weight"))
        self.assertIsInstance(
            QuantizationConfig.from_config(
                dict(method=result.get_method(), bits=4, group_size=16, is_quanted=True)
            ),
            CompressedTensorsNvfp4Config,
        )

    def test_unsupported_schemes_fail(self):
        for field, value in (
            ("group_size", 32),
            ("symmetric", False),
            ("dynamic", True),
            ("strategy", "group"),
        ):
            config = copy.deepcopy(self.config)
            config["quantization_config"]["config_groups"]["group_0"]["weights"][
                field
            ] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(config)

    def test_multiple_groups_fail(self):
        self.quant["config_groups"]["other"] = copy.deepcopy(
            self.quant["config_groups"]["group_0"]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.load(self.config)

    def test_transforms_and_quantized_head_fail(self):
        for key, value in (
            ("transform_config", {"rotation": True}),
            ("ignore", []),
            ("kv_cache_scheme", {"num_bits": 8}),
        ):
            config = copy.deepcopy(self.config)
            config["quantization_config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(config)

    def test_non_qwen3_fails(self):
        self.config["model_type"] = "qwen3_moe"
        with self.assertRaisesRegex(ValueError, "dense Qwen3"):
            self.load(self.config)

    def test_layer_exclusions_rejected_before_loading(self):
        for patterns in (
            ["lm_head", "model.layers.0.mlp.down_proj"],
            ["lm_head", r"re:^model\.layers\..*\.mlp\."],
            [r"re:^lm_head$"],
            [],
            "lm_head",
        ):
            config = copy.deepcopy(self.config)
            config["quantization_config"]["ignore"] = patterns
            with self.subTest(patterns=patterns), self.assertRaises(ValueError):
                self.load(config)

    def test_global_scale_rank_is_canonical(self):
        for shape in ((), (1,), (1, 1)):
            with self.subTest(shape=shape):
                value = reciprocal_global_scale([torch.tensor(2.0).reshape(shape)])
                self.assertEqual(value.shape, torch.Size([]))
                self.assertEqual(value.item(), 0.5)

    def test_reciprocal_and_gemm_scale_convention(self):
        weight = reciprocal_global_scale([torch.tensor([10560.0])] * 3)
        activation = reciprocal_global_scale([torch.tensor([[3968.0]])] * 3)
        self.assertAlmostEqual((1 / activation).item(), 3968.0, places=3)
        torch.testing.assert_close(
            weight * activation, torch.tensor(1 / (10560.0 * 3968.0))
        )

    def test_invalid_or_unequal_globals_fail(self):
        cases = [
            [],
            [torch.tensor([0.0])],
            [torch.tensor([-1.0])],
            [torch.tensor([float("nan")])],
            [torch.tensor([float("inf")])],
            [torch.ones(2)],
            [torch.tensor([2.0]), torch.tensor([3.0])],
        ]
        for tensors in cases:
            with self.subTest(tensors=tensors), self.assertRaises(ValueError):
                reciprocal_global_scale(tensors)

    def test_real_checkpoint_tile_payload(self):
        tile = json.loads(
            (Path(__file__).parent / "fixtures/nvfp4/q_proj_tile.json").read_text()
        )
        for key, count in (("weight_packed", 128 * 64), ("weight_scale", 128 * 8)):
            payload = base64.b64decode(tile[key])
            self.assertEqual(len(payload), count)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), tile[key + "_sha256"])
        scales = (
            torch.tensor(
                list(base64.b64decode(tile["weight_scale"])), dtype=torch.uint8
            )
            .view(torch.float8_e4m3fn)
            .float()
        )
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales >= 0).all())
        self.assertEqual(tile["weight_global_scale"], 10560.0)
        self.assertEqual(tile["input_global_scale"], 3968.0)

    def test_full_checkpoint_global_audit(self):
        fixtures = Path(__file__).parent / "fixtures/nvfp4"
        report = json.loads((fixtures / "checkpoint_audit.json").read_text())
        self.assertEqual(report["revision"], "e391349c110709b87bfc2ad2fde3f50dc5839fd8")
        self.assertEqual(
            hashlib.sha256((fixtures / "config.json").read_bytes()).hexdigest(),
            report["config_sha256"],
        )
        self.assertEqual(len(report["values"]), 504)
        self.assertEqual(
            report["global_storage"], dict(dtype="F32", shape=[1], count=504)
        )
        checks = 0
        for layer in range(36):
            for suffix in ("weight_global_scale", "input_global_scale"):
                for group in (
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                    ("mlp.gate_proj", "mlp.up_proj"),
                ):
                    values = [
                        report["values"][f"model.layers.{layer}.{name}.{suffix}"]
                        for name in group
                    ]
                    self.assertEqual(len(set(values)), 1)
                    actual = reciprocal_global_scale(
                        [torch.tensor([value]) for value in values]
                    )
                    self.assertEqual(actual.shape, torch.Size([]))
                    checks += 1
        self.assertEqual(checks, report["fused_global_checks"])
        for source in report["range_sources"]:
            payload = base64.b64decode(source["payload_base64"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), source["payload_sha256"]
            )

    def test_modelopt_dispatch_unchanged(self):
        # Legacy native configuration round trip remains distinct from CT.
        config = QuantizationConfig.from_config(
            dict(method="modelopt_fp4", bits=4, group_size=16, is_quanted=True)
        )
        self.assertIs(type(config), ModelOptFp4Config)


if __name__ == "__main__":
    unittest.main()

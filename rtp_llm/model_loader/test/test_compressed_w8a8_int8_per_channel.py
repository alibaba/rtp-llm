import inspect
import json
import os
import tempfile
import unittest

import torch

from rtp_llm.config.quant_config import (
    CompressedW8A8Int8PerChannelQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.compressed_w8a8_int8_per_channel_weight import (
    CompressedW8A8Int8PerChannelWeight,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_channel_fp8_quant_weight import PerChannelFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.ops import QuantAlgo
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _compressed_group(weight_type="int", symmetric=True):
    return {
        "weights": {
            "num_bits": 8,
            "type": weight_type,
            "strategy": "channel",
            "symmetric": symmetric,
            "dynamic": False,
        },
        "input_activations": {
            "num_bits": 8,
            "type": weight_type,
            "strategy": "token" if weight_type == "int" else "tensor",
            "symmetric": symmetric,
            "dynamic": True,
        },
        "targets": ["Linear"],
    }


class _RecordingDevice:
    """Stand-in for the exported device, the one external dependency here.

    Records how often the FP8 layout conversion runs and returns tensors that
    cannot be confused with the inputs, so the gate is asserted on behaviour
    rather than on the class attribute that drives it.
    """

    def __init__(self):
        self.convert_calls = 0
        self.sentinel_kernel = torch.full((1, 1), 7, dtype=torch.int32)
        self.sentinel_scale = torch.full((1, 1), 9, dtype=torch.int32)

    def maybe_rewrite_weight_by_key(self, key, tensor):
        return tensor

    def convert_fp8_weight_params(self, kernel, scale):
        self.convert_calls += 1
        return self.sentinel_kernel, self.sentinel_scale


def _load_config(exported_device):
    return LoadConfig(
        database=BaseDatabase(),
        num_layers=1,
        hidden_size=2,
        head_num=1,
        head_num_kv=1,
        size_per_head=2,
        moe_pure_tp_mode=False,
        align_size=1,
        moe_align_size=1,
        moe_layer_index=[],
        moe_n_group=1,
        expert_num=0,
        enable_eplb=False,
        phy_exp_num=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=1,
        lm_head_tp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        num_nodes=1,
        exported_device=exported_device,
    )


class CompressedW8A8ConfigTest(unittest.TestCase):
    def _load(self, quantization_config):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "config.json"), "w") as output:
                json.dump({"quantization_config": quantization_config}, output)
            return QuantizationConfig.load_from_ckpt(model_dir)

    def test_parses_checkpoint_defined_group_name_and_ignore(self):
        config = self._load(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"W8A8": _compressed_group()},
                "ignore": ["model.layers.0.mlp.gate"],
            }
        )
        self.assertIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)
        self.assertEqual(config.get_algo(), "w8a8_int8_per_channel")
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.group_size(), 0)
        self.assertEqual(config.exclude_modules, {"model.layers.0.mlp.gate"})

    def test_group_0_still_wins_so_existing_checkpoints_are_unchanged(self):
        config = self._load(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": _compressed_group("float"),
                    "W8A8": _compressed_group("int"),
                },
            }
        )
        self.assertIsInstance(config, Fp8PerChannelCompressedQuantConfig)

    def test_rejects_a_group_scoped_to_specific_targets(self):
        # Nothing reads targets, so a narrower scope would be applied as if it
        # covered the whole model. Fail at parse time instead.
        group = _compressed_group()
        group["targets"] = ["re:.*mlp.*"]
        with self.assertRaisesRegex(ValueError, "targets"):
            self._load(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"W8A8": group},
                }
            )

    def test_group_0_wins_and_says_the_siblings_are_dropped(self):
        with self.assertLogs(level="WARNING") as logs:
            config = self._load(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": _compressed_group(),
                        "group_1": _compressed_group(),
                    },
                }
            )
        self.assertIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)
        self.assertIn("group_1", "\n".join(logs.output))

    def test_misspelled_ignore_patterns_raises_instead_of_emptying_excludes(self):
        # Matched on the keyword name: an abstract-class TypeError would
        # otherwise let this pass while the protection it asserts is gone.
        with self.assertRaisesRegex(TypeError, "ignore_pattern"):
            CompressedW8A8Int8PerChannelQuantConfig(ignore_pattern=["lm_head"])

    def test_rejects_asymmetric_w8a8(self):
        with self.assertRaisesRegex(ValueError, "asymmetric INT8"):
            self._load(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"W8A8": _compressed_group(symmetric=False)},
                }
            )

    def test_unrecognised_scheme_names_itself_instead_of_failing_abstractly(self):
        # Weight-only INT8 (activations absent or null) and per-tensor INT8
        # activations must not be read as dynamic per-token W8A8, and the
        # fall-through must not surface an abstract-class TypeError.
        weight_only = _compressed_group()
        weight_only["input_activations"] = None
        missing_key = {
            k: v for k, v in _compressed_group().items() if k != "input_activations"
        }
        per_tensor = _compressed_group()
        per_tensor["input_activations"]["strategy"] = "tensor"
        for label, group in (
            ("null activations", weight_only),
            ("missing activations", missing_key),
            ("per-tensor activations", per_tensor),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(
                    ValueError, "unsupported compressed-tensors scheme"
                ):
                    self._load(
                        {
                            "quant_method": "compressed-tensors",
                            "config_groups": {"W8A8": group},
                        }
                    )

    def test_moe_activation_spec_requires_int8_per_token(self):
        self.assertEqual(
            CompressedW8A8Int8PerChannelQuantConfig().get_moe_activation_quant_spec(),
            (torch.int8, True),
        )
        self.assertIsNone(
            Fp8PerChannelCompressedQuantConfig(
                bits=8, is_quanted=True
            ).get_moe_activation_quant_spec()
        )


class CompressedW8A8WeightTest(unittest.TestCase):
    def _source(self):
        return AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.gate.weight", identity)],
        )

    def test_support_and_checkpoint_tensor_dtypes(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        source = self._source()
        self.assertTrue(CompressedW8A8Int8PerChannelWeight.support(config, source))

        weight = WeightModule.create(source, config)
        self.assertIsInstance(weight, CompressedW8A8Int8PerChannelWeight)
        self.assertEqual(weight.kernel.data_type, torch.int8)
        self.assertEqual(weight.scale.data_type, torch.float32)
        self.assertEqual(
            weight.kernel.weights[0].name, "model.layers.{i}.self_attn.gate.weight"
        )
        self.assertEqual(
            weight.scale.weights[0].name,
            "model.layers.{i}.self_attn.gate.weight_scale",
        )

    def test_ignore_matching_the_template_disables_the_quantized_loader(self):
        config = CompressedW8A8Int8PerChannelQuantConfig(
            ignore_patterns=["model.layers.7.self_attn.gate"]
        )
        self.assertFalse(
            CompressedW8A8Int8PerChannelWeight.support(config, self._source())
        )

    def test_non_matching_ignore_keeps_the_quantized_loader(self):
        config = CompressedW8A8Int8PerChannelQuantConfig(
            ignore_patterns=["model.layers.7.mlp.down_proj"]
        )
        self.assertTrue(
            CompressedW8A8Int8PerChannelWeight.support(config, self._source())
        )


class PerChannelFp8PostprocessTest(unittest.TestCase):
    """Assert the _postprocess behaviour, not the attribute that drives it."""

    def _run(self, quant_config, kernel_dtype):
        source = AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.gate.weight", identity)],
        )
        weight = WeightModule.create(source, quant_config)
        device = _RecordingDevice()
        # Non-square so the reshape inside _postprocess is observable.
        tensors = {
            weight.kernel.name: torch.arange(6, dtype=torch.int32)
            .reshape(2, 3)
            .to(kernel_dtype),
            weight.scale.name: torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        }
        processed = weight._postprocess(tensors, "cpu", _load_config(device))
        return weight, device, processed

    def test_int8_skips_the_fp8_conversion(self):
        # _RecordingDevice.maybe_rewrite_weight_by_key is an identity, so the
        # tensor comparisons below pin what this loader does to the weights, not
        # what a production device may still rewrite afterwards.
        weight, device, processed = self._run(
            CompressedW8A8Int8PerChannelQuantConfig(), torch.int8
        )
        self.assertEqual(device.convert_calls, 0)
        kernel = processed[weight.kernel.name]
        self.assertEqual(kernel.dtype, torch.int8)
        self.assertEqual(tuple(kernel.shape), (3, 2))
        torch.testing.assert_close(
            kernel.to(torch.int32),
            torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        scale = processed[weight.scale.name]
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (1, 3))

    def test_fp8_still_runs_the_device_conversion(self):
        weight, device, processed = self._run(
            Fp8PerChannelCompressedQuantConfig(bits=8, is_quanted=True),
            torch.float8_e4m3fn,
        )
        self.assertEqual(device.convert_calls, 1)
        torch.testing.assert_close(
            processed[weight.kernel.name], device.sentinel_kernel, rtol=0, atol=0
        )
        torch.testing.assert_close(
            processed[weight.scale.name], device.sentinel_scale, rtol=0, atol=0
        )

    def test_no_hardcoded_fp8_dtype_survives_in_the_template(self):
        # A missed call site would keep loading FP8 kernels for an INT8
        # checkpoint, and the dtype is only reachable through the class attribute.
        source = inspect.getsource(PerChannelFp8Weight)
        self.assertEqual(source.count("data_type=torch.float8_e4m3fn"), 0)


class QuantAlgoBindingTest(unittest.TestCase):
    """Pin the python -> C++ string contract the loader relies on."""

    def test_algo_string_round_trips_to_w8a8_int8_ptpc(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        algo = QuantAlgo()
        algo.setQuantAlgo(config.get_algo().lower(), config.bits, config.group_size())
        self.assertTrue(algo.isW8a8Int8PTPC())
        self.assertTrue(algo.isQuant())
        self.assertFalse(algo.isGroupwise())
        self.assertEqual(algo.getWeightBits(), 8)
        self.assertEqual(algo.getActivationBits(), 8)
        self.assertEqual(algo.getGroupSize(), 0)
        self.assertIn("W8A8INT8PTPC", str(algo.getQuantMethod()))


if __name__ == "__main__":
    unittest.main()

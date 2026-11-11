import os
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_desc.module_base import GptModelBase, required_config_value
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@register_model("gpt_model_base_contract_test")
class _ContractModel(GptModelBase):
    def __init__(self, model_config, load_config):
        super().__init__(model_config, types.SimpleNamespace())
        self.projection = torch.nn.Linear(2, 2, bias=False)
        self.postprocessed = False

    def process_weights_after_loading(self):
        self.postprocessed = True


class ModelBaseContractTest(unittest.TestCase):
    def test_required_config_value_accepts_mapping_and_attribute_aliases(self):
        for config in (
            {"num_layers": None, "num_hidden_layers": 3},
            types.SimpleNamespace(num_layers=None, num_hidden_layers=3),
        ):
            with self.subTest(config=config):
                self.assertEqual(
                    required_config_value(config, "num_layers", "num_hidden_layers"),
                    3,
                )

    def test_required_config_value_preserves_false_and_zero(self):
        for value in (False, 0):
            with self.subTest(value=value):
                self.assertIs(required_config_value({"value": value}, "value"), value)

    def test_missing_required_config_reports_all_aliases(self):
        for config in ({}, types.SimpleNamespace(num_layers=None)):
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "num_layers or num_hidden_layers"
            ):
                required_config_value(config, "num_layers", "num_hidden_layers")

    def test_base_accepts_service_and_hf_configs_without_runtime_weights(self):
        for config in (
            types.SimpleNamespace(num_layers=2, vocab_size=8),
            {"num_hidden_layers": 2, "vocab_size": 8},
        ):
            with self.subTest(config=config):
                model = GptModelBase(config, types.SimpleNamespace())
                self.assertIsInstance(model, RtpModule)
                self.assertEqual(model.layer_num, 2)
                self.assertEqual(model.vocab_size, 8)
                self.assertFalse(model.supports_hidden_state_capture)
                model.begin_aux_hidden_capture(torch.ones(2, 4), False)
                model.finish_aux_hidden_capture()
                self.assertIsNone(model.get_mtp_target_hidden_states(2))

    def test_public_loader_uses_inherited_dispatch_and_completeness(self):
        config = {
            "model_type": "gpt_model_base_contract_test",
            "num_hidden_layers": 1,
            "vocab_size": 8,
        }
        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"projection.weight": expected},
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                config,
                NewLoaderConfig(device="cpu", compute_dtype=torch.float32),
                model_path=model_path,
            ).load()
        self.assertIsInstance(model, RtpModule)
        self.assertTrue(model.postprocessed)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.projection.weight, expected)
        NewModelLoader._validate_loaded_weights(model)

    def test_base_does_not_bypass_missing_weight_validation(self):
        model = _ContractModel(types.SimpleNamespace(num_layers=1, vocab_size=8), None)
        with self.assertRaisesRegex(RuntimeError, "projection.weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_base_preserves_shared_buffer_aliases_during_dtype_migration(self):
        model = GptModelBase(
            types.SimpleNamespace(num_layers=1, vocab_size=8),
            types.SimpleNamespace(),
        )
        shared = torch.ones(2, dtype=torch.float32)
        model.register_buffer("first", shared)
        model.child = torch.nn.Module()
        model.child.register_buffer("second", shared)
        model.to(dtype=torch.float64)
        self.assertIs(model.first, model.child.second)
        self.assertEqual(model.first.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()

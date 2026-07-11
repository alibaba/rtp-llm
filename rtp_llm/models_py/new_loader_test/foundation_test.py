import os
import sys
import tempfile
import types
import unittest

import torch
import torch.nn as nn
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import LoadConfig, LoadMethod, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import (
    get_model_class,
    register_lazy_model,
    register_model,
)
from rtp_llm.models_py.weight_mapper import discover_ckpt_files, get_all_weights


class _Block(RtpModule):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.bias = nn.Parameter(torch.empty(2))


class _FoundationModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])
        self.final = nn.Parameter(torch.empty(2))
        self.post_device = None
        self.post_count = 0

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        self.post_device = self.final.device.type
        self.post_count += 1


register_model("foundation_test_model")(_FoundationModel)


def _weights():
    return {
        "layers.0.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "layers.0.bias": torch.tensor([5.0, 6.0]),
        "final": torch.tensor([7.0, 8.0]),
    }


class FoundationLoaderTest(unittest.TestCase):
    def _loader(self, model_path, **kwargs):
        config = types.SimpleNamespace(model_type="foundation_test_model")
        load_config = LoadConfig(device="cpu", compute_dtype=torch.float32, **kwargs)
        return NewModelLoader(config, load_config, model_path=model_path)

    def test_real_safetensors_stream_load_and_postprocess(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            model = self._loader(model_path).load()
        self.assertTrue(torch.equal(model.layers[0].weight, _weights()["layers.0.weight"]))
        self.assertTrue(torch.equal(model.final, _weights()["final"]))
        self.assertEqual(model.post_device, "cpu")
        self.assertEqual(model.post_count, 1)

    def test_wrapped_pytorch_state_dict(self):
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(
                {"state_dict": _weights()}, os.path.join(model_path, "pytorch_model.bin")
            )
            model = self._loader(model_path).load()
        self.assertTrue(torch.equal(model.layers[0].bias, _weights()["layers.0.bias"]))

    def test_missing_required_parameter_fails(self):
        checkpoint = _weights()
        checkpoint.pop("final")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "missing required.*final"):
                self._loader(model_path).load()

    def test_unknown_tensor_fails(self):
        checkpoint = _weights()
        checkpoint["unexpected.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
                self._loader(model_path).load()

    def test_shape_mismatch_fails(self):
        checkpoint = _weights()
        checkpoint["final"] = torch.ones(3)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                self._loader(model_path).load()

    def test_force_cpu_runs_postprocess_before_device_move(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            model = self._loader(model_path, force_cpu_load_weights=True).load()
        self.assertEqual(model.post_device, "cpu")

    def test_explicit_fastsafetensors_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "not part of.*foundation"):
                self._loader(model_path, load_method=LoadMethod.FASTSAFETENSORS).load()

    def test_invalid_load_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Unsupported load_method"):
                self._loader(model_path, load_method="typo").load()

    def test_non_string_load_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(TypeError, "load_method must be a string"):
                self._loader(model_path, load_method=object()).load()

    def test_optimizer_bin_does_not_hide_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as model_path:
            torch.save({}, os.path.join(model_path, "optimizer.bin"))
            torch.save(_weights(), os.path.join(model_path, "model.pt"))
            self.assertEqual(discover_ckpt_files(model_path), [os.path.join(model_path, "model.pt")])

    def test_duplicate_tensor_across_shards_fails(self):
        with tempfile.TemporaryDirectory() as model_path:
            first = os.path.join(model_path, "model-1.safetensors")
            second = os.path.join(model_path, "model-2.safetensors")
            save_file({"duplicate": torch.ones(1)}, first)
            save_file({"duplicate": torch.zeros(1)}, second)
            with self.assertRaisesRegex(RuntimeError, "more than one shard"):
                list(get_all_weights([first, second]))

    def test_lazy_registry_loads_declared_class(self):
        module_name = "_newloader_foundation_lazy_test"
        module = types.ModuleType(module_name)

        class LazyModel(nn.Module):
            pass

        LazyModel.__module__ = module_name
        module.LazyModel = LazyModel
        sys.modules[module_name] = module
        try:
            register_lazy_model("foundation_lazy_model", module_name, "LazyModel")
            self.assertIs(get_model_class("foundation_lazy_model"), LazyModel)
        finally:
            sys.modules.pop(module_name, None)

    def test_partition_config_validation(self):
        with self.assertRaisesRegex(ValueError, "Invalid TP"):
            LoadConfig(tp_size=0)
        with self.assertRaisesRegex(ValueError, "Invalid EP"):
            LoadConfig(ep_size=2, ep_rank=2)


if __name__ == "__main__":
    unittest.main()

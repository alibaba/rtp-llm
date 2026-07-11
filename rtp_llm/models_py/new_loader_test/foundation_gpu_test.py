import os
import tempfile
import types
import unittest

import torch
import torch.nn as nn
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


class _GpuModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2))
        self.register_buffer("scale", torch.empty(1), persistent=True)
        self.validation_devices = []

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.validation_devices.append(self.scale.device.type)
        super().validate_weights_loaded(loaded_tensor_ids)


register_model("foundation_gpu_model")(_GpuModel)


class _GpuAliasModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        same = torch.empty(1)
        self.register_buffer("same", same, persistent=True)
        self.register_buffer("same_alias", same, persistent=True)
        self.child = RtpModule()
        cross = torch.empty(1)
        self.register_buffer("cross", cross, persistent=True)
        self.child.register_buffer("cross_alias", cross, persistent=True)
        mixed = nn.Parameter(torch.empty(1))
        self.mixed_parameter = mixed
        self.register_buffer("mixed_buffer", mixed, persistent=True)
        parent_first = nn.Parameter(torch.empty(1))
        self.register_buffer("parent_buffer", parent_first, persistent=True)
        self.child.parent_parameter = parent_first
        self.left = RtpModule()
        self.right = RtpModule()
        sibling = nn.Parameter(torch.empty(1))
        self.left.register_buffer("sibling_buffer", sibling, persistent=True)
        self.right.sibling_parameter = sibling


register_model("foundation_gpu_alias_model")(_GpuAliasModel)


class FoundationGpuTest(unittest.TestCase):
    def test_persistent_buffer_marker_survives_device_migration(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"weight": torch.ones(2), "scale": torch.ones(1)},
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                types.SimpleNamespace(model_type="foundation_gpu_model"),
                NewLoaderConfig(device="cuda:0", compute_dtype=torch.float32),
                model_path=model_path,
            ).load()
        self.assertEqual(model.scale.device.type, "cuda")
        self.assertEqual(model.validation_devices, ["cpu", "cuda"])

    def test_buffer_aliases_survive_device_migration(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    "same_alias": torch.ones(1),
                    "child.cross_alias": torch.ones(1),
                    "mixed_buffer": torch.ones(1),
                    "parent_buffer": torch.ones(1),
                    "left.sibling_buffer": torch.ones(1),
                },
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                types.SimpleNamespace(model_type="foundation_gpu_alias_model"),
                NewLoaderConfig(device="cuda:0"),
                model_path=model_path,
            ).load()
        self.assertIs(model.same, model.same_alias)
        self.assertIs(model.cross, model.child.cross_alias)
        self.assertIs(model.mixed_parameter, model.mixed_buffer)
        self.assertIs(model.parent_buffer, model.child.parent_parameter)
        self.assertIs(model.left.sibling_buffer, model.right.sibling_parameter)
        self.assertIsInstance(model.child.parent_parameter, nn.Parameter)
        self.assertIsInstance(model.right.sibling_parameter, nn.Parameter)
        with torch.inference_mode():
            model.same.add_(1)
            model.child.cross_alias.add_(2)
            model.mixed_parameter.add_(3)
        self.assertEqual(model.same_alias.item(), 2)
        self.assertEqual(model.cross.item(), 3)
        self.assertEqual(model.mixed_buffer.item(), 4)
        with torch.inference_mode():
            model.to(dtype=torch.float16)
        self.assertIs(model.parent_buffer, model.child.parent_parameter)
        self.assertIs(model.left.sibling_buffer, model.right.sibling_parameter)
        self.assertIsInstance(model.child.parent_parameter, nn.Parameter)
        self.assertIsInstance(model.right.sibling_parameter, nn.Parameter)

    def test_cross_device_copy_does_not_allocate_full_target_temporary(self):
        class CudaTarget(RtpModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.float32)
                )

        model = CudaTarget()
        source = torch.ones(model.weight.shape, dtype=torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.load_weights({"weight": source})
        torch.cuda.synchronize()
        extra_peak = torch.cuda.max_memory_allocated() - baseline
        self.assertLess(extra_peak, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()

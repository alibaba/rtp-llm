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
        self.validation_device = None

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.validation_device = self.scale.device.type
        super().validate_weights_loaded(loaded_tensor_ids)


register_model("foundation_gpu_model")(_GpuModel)


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
        self.assertEqual(model.validation_device, "cpu")

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

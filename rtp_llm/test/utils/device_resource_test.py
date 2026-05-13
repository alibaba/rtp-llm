import importlib.util
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_device_resource_module():
    path = PROJECT_ROOT / "rtp_llm" / "test" / "utils" / "device_resource.py"
    spec = importlib.util.spec_from_file_location("_device_resource_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


device_resource = _load_device_resource_module()


class DeviceResourceMainContractTest(TestCase):
    def test_default_requires_one_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 1)

    def test_gpu_count_zero_is_explicit_no_lock_opt_out(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 0)

    def test_gpu_count_takes_precedence_over_world_size(self):
        with patch.dict(os.environ, {"GPU_COUNT": "2", "WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 2)

    def test_world_size_used_when_gpu_count_absent(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 4)

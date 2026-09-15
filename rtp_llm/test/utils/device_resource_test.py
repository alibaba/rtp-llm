import importlib.util
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_device_resource_module():
    path = PROJECT_ROOT / "rtp_llm" / "test" / "utils" / "device_resource.py"
    spec = importlib.util.spec_from_file_location("_device_resource_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_jit_sys_path_setup_module():
    path = PROJECT_ROOT / "rtp_llm" / "test" / "utils" / "jit_sys_path_setup.py"
    spec = importlib.util.spec_from_file_location(
        "_jit_sys_path_setup_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


device_resource = _load_device_resource_module()


class DeviceResourceMainContractTest(TestCase):
    def test_default_requires_one_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 1)

        resource = object.__new__(device_resource.DeviceResource)
        with (
            patch.dict(
                os.environ,
                {"RTP_REMOTE_SESSION_ID": "remote-test"},
                clear=True,
            ),
            patch.object(device_resource.subprocess, "run") as run,
        ):
            run.return_value.returncode = 0
            run.return_value.stdout = "151859\n"
            self.assertTrue(resource._has_excess_preexisting_memory("0"))
            run.return_value.stdout = "512\n"
            self.assertFalse(resource._has_excess_preexisting_memory("0"))

        resource._candidate_gpu_groups = lambda: [[0]]
        resource._gpu_bad_until = {}
        resource.gpu_status_root_path = "/tmp/test-gpu-status"
        resource.required_gpu_count = 1
        resource._has_zombie_gpu_contexts = lambda gpu_id: False
        resource._has_non_session_live_cuda_pids = lambda gpu_id: False
        resource._has_excess_preexisting_memory = lambda gpu_id: True
        with (
            patch.object(device_resource, "FileLock"),
            self.assertRaisesRegex(
                device_resource.GpuLockTimeoutError, "pre-existing memory"
            ),
        ):
            resource._lock_gpus()

        jit_setup = _load_jit_sys_path_setup_module()
        cached_paths = ["/cache/torch/site-packages", "/cache/deep_gemm/site-packages"]
        with (
            TemporaryDirectory() as venv_dir,
            patch.dict(os.environ, {"PYTHONPATH": "/workspace"}, clear=True),
            patch.object(sys, "argv", ["device_resource.py", "pytest"]),
            patch.object(jit_setup.site, "getsitepackages", return_value=[venv_dir]),
            patch.object(
                jit_setup,
                "copy_package_with_lock",
                Mock(side_effect=[*cached_paths, None, None]),
            ),
        ):
            cupti_dir = Path(venv_dir) / "nvidia" / "cuda_cupti" / "lib"
            cupti_dir.mkdir(parents=True)
            (Path(venv_dir) / "nvidia" / "not_a_library").mkdir()
            os.environ["LD_LIBRARY_PATH"] = "/runtime/lib"
            self.assertEqual(jit_setup.setup_jit_cache(), cached_paths)
            self.assertEqual(
                os.environ["PYTHONPATH"], os.pathsep.join([*cached_paths, "/workspace"])
            )
            self.assertEqual(
                os.environ["LD_LIBRARY_PATH"],
                os.pathsep.join(["/runtime/lib", str(cupti_dir)]),
            )

    def test_default_uses_cpu_when_no_device_is_available(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                device_resource._get_required_gpu_count(device_available=False), 0
            )

    def test_explicit_gpu_request_is_preserved_without_a_device(self):
        with patch.dict(os.environ, {"GPU_COUNT": "1"}, clear=True):
            self.assertEqual(
                device_resource._get_required_gpu_count(device_available=False), 1
            )

    def test_gpu_count_zero_is_explicit_no_lock_opt_out(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 0)

    def test_gpu_count_takes_precedence_over_world_size(self):
        with patch.dict(os.environ, {"GPU_COUNT": "2", "WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 2)

    def test_world_size_used_when_gpu_count_absent(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 4)

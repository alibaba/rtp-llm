"""Exercise real PPU registration files without loading accelerator libraries."""

import os
import subprocess
import sys
import unittest


class PpuEntrypointTest(unittest.TestCase):
    def test_device_isolation_and_repeat_import_in_fresh_process(self):
        for device in ("Ppu", "Cuda", "ROCm", "Cpu", "PpuFailure"):
            with self.subTest(device=device):
                script = """
import importlib
import sys
import types
from enum import IntEnum

class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    ROCm = 4
    Ppu = 5

device = types.ModuleType("rtp_llm.device.device_type")
device.DeviceType = DeviceType
device.get_device_type = lambda: DeviceType[sys.argv[1].replace("Failure", "")]
sys.modules[device.__name__] = device
import rtp_llm.utils.backend_registry as registry
if sys.argv[1] == "PpuFailure":
    import importlib.util
    def fail_spec(*args, **kwargs):
        raise ImportError("injected missing registration file")
    importlib.util.spec_from_file_location = fail_spec
    try:
        importlib.import_module("rtp_llm.models_py")
    except ImportError as error:
        assert "injected missing" in str(error)
        sys.exit(0)
    raise AssertionError("PPU registration failure was swallowed")
backend = importlib.import_module("rtp_llm.models_py")
assert backend.__file__, "Missing real initializer in runfiles"
importlib.reload(backend)
slots = {"attention", "linear", "fused_moe", "moe_strategy_choices"}
if sys.argv[1] == "Ppu":
    assert set(registry._hooks) == slots, registry._hooks
    assert all(len(hooks) == 1 for hooks in registry._hooks.values())
    assert "/ppu-entrypoint-test-cache" in sys.path
else:
    assert not registry._hooks, registry._hooks
    assert "/ppu-entrypoint-test-cache" not in sys.path
assert "rtp_llm.models_py.modules.factory" not in sys.modules
"""
                env = dict(os.environ, _JIT_CACHE_PATHS="/ppu-entrypoint-test-cache")
                subprocess.run([sys.executable, "-c", script, device], env=env, check=True)


if __name__ == "__main__":
    unittest.main()

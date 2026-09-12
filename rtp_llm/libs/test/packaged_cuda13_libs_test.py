import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PackagedCuda13LibsTest(unittest.TestCase):
    def test_relocated_package_loads_without_bazel_library_paths(self):
        source = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/libs"
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for library in source.glob("*.so*"):
                shutil.copyfile(library, destination / library.name)

            # Inspect the actual linked dependencies, not a second hardcoded
            # list that could drift from the CUDA13 build.
            required = set()
            tool_env = {
                key: value
                for key, value in os.environ.items()
                if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")
            }
            for name in ("librtp_compute_ops.so", "libth_transformer.so"):
                dynamic = subprocess.check_output(
                    ["readelf", "-d", str(destination / name)],
                    text=True,
                    env=tool_env,
                )
                required.update(
                    re.findall(r"NEEDED.*\[(libflashinfer[^\]]+)\]", dynamic)
                )
            self.assertTrue(required, "CUDA13 runtime has no FlashInfer dependencies")
            self.assertEqual(
                [],
                sorted(name for name in required if not (destination / name).is_file()),
            )

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ":".join(
                path
                for path in env.get("LD_LIBRARY_PATH", "").split(":")
                if path
                and not any(part in path for part in ("bazel", "_solib", "runfiles"))
            )
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes, pathlib, sys, sysconfig, torch; "
                    "ctypes.CDLL(sysconfig.get_config_var('LIBDIR') + '/libpython3.10.so', "
                    "mode=ctypes.RTLD_GLOBAL); "
                    "sys.path.insert(0, sys.argv[1]); "
                    "import libth_transformer_config; import librtp_compute_ops; "
                    "torch.ops.load_library(str(pathlib.Path(sys.argv[1]) / 'libth_transformer.so')); "
                    "assert hasattr(librtp_compute_ops, 'PyModelInputs'); "
                    "print('RELOCATED_PACKAGE_LOADED')",
                    directory,
                ],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(0, probe.returncode, probe.stdout + probe.stderr)
            self.assertIn("RELOCATED_PACKAGE_LOADED", probe.stdout)


if __name__ == "__main__":
    unittest.main()

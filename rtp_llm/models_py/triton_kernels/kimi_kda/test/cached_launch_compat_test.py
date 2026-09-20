"""CPU import/dispatch compatibility for older Triton configurations."""

import builtins
import importlib.util
import runpy
import unittest
from unittest.mock import patch

import torch
import triton


class CachedLaunchCompatibilityTest(unittest.TestCase):
    def test_no_knobs_module_keeps_import_and_reference_launch_working(self):
        name = "rtp_llm.models_py.triton_kernels.kimi_kda.cached_launch"
        source = importlib.util.find_spec(name).origin
        original_import = builtins.__import__

        def older_triton(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton" and "knobs" in (fromlist or ()):
                raise ImportError("older Triton does not expose knobs")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=older_triton):
            namespace = runpy.run_path(source)

        class CpuKernel:
            # Exercise the caller's normal dispatch without requiring CUDA or
            # pretending CPU tensors can enter a compiled CUDA launcher.
            def __getitem__(self, grid):
                def run(x, y, factor):
                    y.copy_(x * factor)

                return run

        launch = namespace["CachedLaunch"](CpuKernel())
        x = torch.tensor([1.0, -3.0])
        y = torch.empty_like(x)
        launch((1, 1, 1), (x, y), (2,))
        self.assertEqual(y.tolist(), [2.0, -6.0])

    def test_unverified_triton_version_does_not_enable_compiled_cache(self):
        from rtp_llm.models_py.triton_kernels.kimi_kda.cached_launch import CachedLaunch

        @triton.jit
        def kernel():
            pass

        x = torch.tensor([2.0, -4.0])
        y = torch.empty_like(x)

        def ordinary_launch(x, y, factor, **kwargs):
            y.copy_(x * factor)

        for version in ("3.2.0", "3.4.0", "3.5.0", "3.7.0"):
            with self.subTest(version=version), patch.object(
                triton, "__version__", version
            ), patch.object(
                kernel, "run", side_effect=ordinary_launch
            ) as jit, patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("unsupported fast path accessed CUDA"),
            ):
                launch = CachedLaunch(kernel)
                self.assertFalse(launch.enabled)
                for factor in (3, -2, 7):
                    launch((1, 1, 1), (x, y), (factor,))
                    torch.testing.assert_close(y, x * factor, rtol=0, atol=0)
                self.assertEqual(jit.call_count, 3)
                self.assertEqual(launch.cache, {})


if __name__ == "__main__":
    unittest.main()

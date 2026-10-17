"""Regression tests for CPU-suite runtime isolation and failure cleanup."""

import io
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

_HELPER = pathlib.Path(__file__).with_name("cpu_test_utils.py")
isolate_cpu_test_module = run_path(str(_HELPER))["isolate_cpu_test_module"]


class CpuTestModuleIsolation(unittest.TestCase):
    def test_runtime_state_is_restored(self):
        original_init = torch.cuda._lazy_init
        runtime = types.ModuleType("rtp_llm.runtime")
        test_module = types.ModuleType("rtp_llm.models_py.modules.dsv4.test.example")
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "0"}
        ):
            with patch.dict(
                sys.modules,
                {runtime.__name__: runtime, test_module.__name__: test_module},
            ):
                original_modules = {
                    name: mod
                    for name, mod in sys.modules.copy().items()
                    if name == "rtp_llm" or name.startswith("rtp_llm.")
                }
                try:
                    cleanup = isolate_cpu_test_module()
                    self.assertNotIn(runtime.__name__, sys.modules)
                    self.assertIs(sys.modules[test_module.__name__], test_module)
                    self.assertEqual(os.environ["TRITON_INTERPRET"], "1")
                    with self.assertRaisesRegex(AssertionError, "CUDA initialization"):
                        torch.cuda._lazy_init()
                    sys.modules["rtp_llm.fixture_only"] = types.ModuleType(
                        "fixture_only"
                    )
                finally:
                    cleanup.close()
                self.assertIs(torch.cuda._lazy_init, original_init)
                self.assertEqual(os.environ["TRITON_INTERPRET"], "0")
                restored = {
                    name: mod
                    for name, mod in sys.modules.copy().items()
                    if name == "rtp_llm" or name.startswith("rtp_llm.")
                }
                self.assertEqual(restored, original_modules)

    def test_failed_setup_restores_runtime(self):
        original_init = torch.cuda._lazy_init
        module = types.ModuleType("_cpu_test_failed_setup")

        def set_up():
            isolate_cpu_test_module()
            raise RuntimeError("intentional fixture failure")

        class Body(unittest.TestCase):
            def test_body(self):
                self.fail("test body must not run after setup failure")

        Body.__module__ = module.__name__
        module.Body = Body
        module.setUpModule = set_up
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "0"}
        ):
            with patch.dict(sys.modules, {module.__name__: module}):
                result = unittest.TextTestRunner(stream=io.StringIO()).run(
                    unittest.defaultTestLoader.loadTestsFromModule(module)
                )
            self.assertEqual(result.testsRun, 0)
            self.assertEqual(len(result.errors), 1)
            self.assertIn("intentional fixture failure", result.errors[0][1])
            self.assertIs(torch.cuda._lazy_init, original_init)
            self.assertEqual(os.environ["TRITON_INTERPRET"], "0")

    def test_requires_hidden_cuda_before_changing_state(self):
        original_init = torch.cuda._lazy_init
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
                isolate_cpu_test_module()
        self.assertIs(torch.cuda._lazy_init, original_init)


if __name__ == "__main__":
    unittest.main()

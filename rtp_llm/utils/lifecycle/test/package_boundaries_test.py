"""Protect lifecycle packaging and the control/resource dependency boundary."""

import ast
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = "rtp_llm.utils.lifecycle"
ROOT = Path(__file__).resolve().parents[1]
CONTROL = ("controller", "quiesce", "rpc")
CORE = ("lease", "shutdown", "status", "timing", "validation")
RESOURCES = (
    "gpu_mem_probe",
    "nccl_memory",
    "nccl_memory_backend",
    "nccl_memory_policy",
    "sleep_gpu_reclaim",
)


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


class PackageBoundariesTest(unittest.TestCase):
    def test_all_modules_are_in_the_runtime_package(self):
        modules = list(CORE + CONTROL) + [f"resources.{name}" for name in RESOURCES]
        for module in modules:
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(f"{PACKAGE}.{module}"))

    def test_control_never_imports_gpu_resources(self):
        forbidden = (
            f"{PACKAGE}.resources",
            "torch",
            "rtp_llm.model_loader",
            "rtp_llm.models_py",
        )
        for name in CONTROL:
            for module in imports(ROOT / f"{name}.py"):
                with self.subTest(source=name, dependency=module):
                    self.assertFalse(module.startswith(forbidden))

    def test_resources_never_import_control_or_http(self):
        forbidden = tuple(f"{PACKAGE}.{name}" for name in CONTROL) + (
            "rtp_llm.frontend",
            "rtp_llm.utils.grpc_client_wrapper",
            "grpc",
            "fastapi",
        )
        for name in RESOURCES:
            for module in imports(ROOT / "resources" / f"{name}.py"):
                with self.subTest(source=name, dependency=module):
                    self.assertFalse(module.startswith(forbidden))

    def test_shared_contracts_do_not_depend_on_http(self):
        for name in CORE + CONTROL:
            for module in imports(ROOT / f"{name}.py"):
                with self.subTest(source=name, dependency=module):
                    self.assertFalse(module.startswith(("rtp_llm.frontend", "fastapi")))

    def test_importing_core_does_not_initialize_grpc_or_gpu_adapters(self):
        script = f"""
import importlib
import sys
importlib.import_module({PACKAGE!r})
importlib.import_module({(PACKAGE + '.resources')!r})
for name in {CORE!r}:
    importlib.import_module({PACKAGE!r} + '.' + name)
for name in sys.modules:
    assert not name.startswith(('torch', 'grpc', 'fastapi')), name
    assert not name.startswith({(PACKAGE + '.resources.')!r}), name
    assert name not in {tuple(PACKAGE + '.' + name for name in CONTROL)!r}, name
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

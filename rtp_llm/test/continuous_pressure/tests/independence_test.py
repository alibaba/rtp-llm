"""The independent core is importable without an inference runtime."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

CORE_MODULES = frozenset(
    {
        "config",
        "diagnostics",
        "phases",
        "ports",
        "results",
        "seed_plan",
        "sender",
        "sequence",
        "session",
    }
)
PACKAGE = "rtp_llm.test.continuous_pressure"


class IndependenceTest(unittest.TestCase):
    def test_public_api_imports_with_framework_and_native_packages_blocked(self):
        program = """
import importlib
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if (
            set(parts) & {"perf_test", "cache_grid", "torch", "proto", "protobuf"}
            or fullname.startswith(("rtp_llm.cpp", "rtp_llm.ops", "rtp_llm.server"))
        ):
            raise AssertionError(f"Unexpected runtime dependency: {fullname}")

sys.meta_path.insert(0, BlockRuntime())
from rtp_llm.test.continuous_pressure import (
    EventSink, ExecutionLimits, JsonlSink, PressureConfig, PressurePoll,
    PressureSession, RequestHandle, SeedPlan,
)
for module in CORE_MODULES:
    importlib.import_module(PACKAGE + "." + module)
"""
        process = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                f"CORE_MODULES = {sorted(CORE_MODULES)!r}\nPACKAGE = {PACKAGE!r}\n"
                + program,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).absolute().parents[4],
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_core_imports_only_stdlib_or_core_even_inside_functions(self):
        root = Path(__file__).parent.parent
        for name in sorted(CORE_MODULES | {"__init__"}):
            path = root / f"{name}.py"
            with self.subTest(module=path.name):
                self.assertTrue(path.is_file(), f"Missing core module {path}")
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        self._check_absolute(node, [part.name for part in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.level == 0:
                            self._check_absolute(node, [node.module])
                        elif node.module is None:
                            self.assertEqual(node.level, 1)
                            self.assertTrue(
                                {part.name for part in node.names} <= CORE_MODULES
                            )
                        else:
                            self.assertEqual(node.level, 1)
                            self.assertIn(node.module, CORE_MODULES)

    def _check_absolute(self, node, names):
        for name in names:
            if name.startswith(PACKAGE + "."):
                self.assertIn(name.removeprefix(PACKAGE + "."), CORE_MODULES)
            else:
                self.assertIn(
                    name.split(".")[0],
                    sys.stdlib_module_names,
                    f"Forbidden import {name!r} on line {node.lineno}",
                )


if __name__ == "__main__":
    unittest.main()

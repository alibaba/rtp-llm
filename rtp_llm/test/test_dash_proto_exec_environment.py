"""Dash proto action environment contracts; no Native library substitutes."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "rtp_llm/dash_sc/proto"


class DashProtoExecutionEnvironmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dash-proto-contract-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        tree = ast.parse((PACKAGE / "BUILD").read_text())
        self.rule = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "genrule"
        )
        self.attrs = {kw.arg: ast.literal_eval(kw.value) for kw in self.rule.keywords}

    def command(self, tool, output):
        return (
            self.attrs["cmd"]
            .replace("$(execpath :create_grpc_proto)", str(tool))
            .replace("$(@D)", str(output))
            .replace("$$", "$")
        )

    def run_tool(self, tool, output=None, extra_env=None):
        output = output or self.directory / "output with spaces"
        env = {"PATH": "/usr/bin:/bin", "PYTHON_BIN_PATH": sys.executable}
        env.update(extra_env or {})
        return subprocess.run(
            ["/bin/bash", "-c", self.command(tool, output)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

    def recorder(self, code=0):
        script = self.directory / "ordinary Python recorder.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, subprocess, sys\n"
            "child = subprocess.check_output(['python3', '-c', 'import sys; print(sys.executable)'], text=True).strip()\n"
            "print(json.dumps({'args': sys.argv[1:], 'python': sys.executable, 'child': child}))\n"
            f"raise SystemExit({code})\n"
        )
        return script

    def test_original_rule_inputs_outputs_and_exec_tool_are_preserved(self):
        self.assertEqual("predict_v2_py", self.attrs["name"])
        self.assertEqual([], self.attrs["srcs"])
        self.assertEqual([":create_grpc_proto"], self.attrs["tools"])
        self.assertEqual(
            [
                "model_config_pb2.py",
                "predict_v2_pb2.py",
                "predict_v2_pb2_grpc.py",
                "__init__.py",
            ],
            self.attrs["outs"],
        )
        source = (PACKAGE / "BUILD").read_text()
        self.assertIn(
            'deps = ["//rtp_llm:grpcio-tools", "//rtp_llm:setuptools"]', source
        )
        self.assertIn('data = ["model_config.proto", "predict_v2.proto"]', source)
        self.assertIn(
            'build --action_env PYTHON_BIN_PATH="/opt/conda310/bin/python3"',
            (ROOT / ".bazelrc").read_text(),
        )

    def test_configured_python_and_child_lookup_preserve_argument_with_spaces(self):
        result = self.run_tool(self.recorder())
        self.assertEqual(0, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual([str(self.directory / "output with spaces")], report["args"])
        self.assertEqual(
            Path(sys.executable).resolve(), Path(report["python"]).resolve()
        )
        self.assertEqual(
            Path(sys.executable).resolve(), Path(report["child"]).resolve()
        )

    def test_nonzero_original_tool_exit_propagates_without_retry(self):
        result = self.run_tool(self.recorder(23))
        self.assertEqual(23, result.returncode)
        self.assertEqual(1, len(result.stdout.splitlines()))

    def test_missing_configured_python_is_a_failure(self):
        result = self.run_tool(self.recorder(), extra_env={"PYTHON_BIN_PATH": ""})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("PYTHON_BIN_PATH must identify", result.stderr)
        self.assertEqual("", result.stdout)

    def test_unavailable_explicit_python_does_not_fallback(self):
        result = self.run_tool(
            self.recorder(), extra_env={"PYTHON_BIN_PATH": "/no/such/python3"}
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("/no/such/python3", result.stderr)
        self.assertEqual("", result.stdout)

    def test_actual_generator_preserves_original_failure_stderr(self):
        output = self.directory / "existing regular file"
        output.write_text("must not overwrite")
        result = self.run_tool(PACKAGE / "create_grpc_proto.py", output=output)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("FileExistsError", result.stderr)
        self.assertIn(str(output), result.stderr)
        self.assertEqual("must not overwrite", output.read_text())

    def test_actual_original_producer_generates_both_protos_and_package_imports(self):
        output = self.directory / "actual generated output"
        result = self.run_tool(PACKAGE / "create_grpc_proto.py", output=output)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        for filename in self.attrs["outs"]:
            generated = output / filename
            self.assertTrue(generated.is_file(), filename)
            self.assertGreater(generated.stat().st_size, 0)
            ast.parse(generated.read_text(), filename=str(generated))
        for filename in ("predict_v2_pb2.py", "predict_v2_pb2_grpc.py"):
            self.assertIn("from . import", (output / filename).read_text())
        # Original producer may also emit model_config_pb2_grpc.py; outputs/producer
        # are kept verbatim rather than suppressing that existing side file.
        self.assertIn("Generated Python protos in", result.stdout)


if __name__ == "__main__":
    unittest.main()

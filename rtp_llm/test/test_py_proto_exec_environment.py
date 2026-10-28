import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MACRO = ROOT / "bazel/py_proto.bzl"
PROTO_ROOT = Path("rtp_llm/cpp/model_rpc/proto")


class PyProtoExecutionEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pyproto-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.tool = SimpleNamespace(
            executable=SimpleNamespace(path="launcher with spaces")
        )
        self.action = None

        def declare_file(name):
            return SimpleNamespace(path=name)

        def record(**kwargs):
            self.action = kwargs

        self.proto = SimpleNamespace(
            basename="input.proto", path="input with spaces.proto"
        )
        self.ctx = SimpleNamespace(
            file=SimpleNamespace(proto=self.proto),
            bin_dir=SimpleNamespace(path="output with spaces"),
            executable=SimpleNamespace(create_grpc_proto=self.tool.executable),
            attr=SimpleNamespace(),
            actions=SimpleNamespace(declare_file=declare_file, run_shell=record),
        )
        function = next(
            node
            for node in ast.parse(MACRO.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_generate_grpc_proto_impl"
        )
        namespace = {
            "PyInfo": lambda **kwargs: kwargs,
            "depset": lambda values: values,
        }

        class DefaultInfoFactory:
            def __call__(self, **kwargs):
                return kwargs

        default_factory = DefaultInfoFactory()
        self.ctx.attr.create_grpc_proto = {
            default_factory: SimpleNamespace(files_to_run=self.tool)
        }
        namespace["DefaultInfo"] = default_factory
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(MACRO), "exec"),
            namespace,
        )
        self.providers = namespace["_generate_grpc_proto_impl"](self.ctx)

    def run_action(
        self, tool, proto="proto argument", output="output argument", env=None
    ):
        environment = {"PATH": "/usr/bin:/bin", "PYTHON_BIN_PATH": sys.executable}
        if env is not None:
            environment.update(env)
        return subprocess.run(
            [
                "/bin/bash",
                "-c",
                self.action["command"],
                "--",
                str(tool),
                str(proto),
                str(output),
            ],
            cwd=self.directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )

    def recorder(self, code=0):
        script = self.directory / "ordinary Python argument recorder.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, subprocess, sys\n"
            "child = subprocess.check_output(['python3', '-c', 'import sys; print(sys.executable)'], text=True).strip()\n"
            "print(json.dumps({'argv': sys.argv[1:], 'interpreter': sys.executable, 'child': child}))\n"
            f"raise SystemExit({code})\n"
        )
        return script

    def test_real_files_to_run_and_providers_preserved(self):
        self.assertEqual([self.tool], self.action["tools"])
        self.assertIs(self.tool, self.action["tools"][0])
        self.assertTrue(self.action["use_default_shell_env"])
        self.assertEqual([self.proto], self.action["inputs"])
        self.assertEqual(
            ["input_pb2.py", "input_pb2_grpc.py"],
            [value.path for value in self.action["outputs"]],
        )
        self.assertEqual(
            [self.tool.executable.path, self.proto.path, self.ctx.bin_dir.path],
            self.action["arguments"],
        )
        self.assertEqual(
            self.providers[0]["files"], self.providers[1]["transitive_sources"]
        )

    def test_exec_configuration_and_existing_python_contract(self):
        tree = ast.parse(MACRO.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign))
        attrs = next(
            value.value for value in assignment.value.keywords if value.arg == "attrs"
        )
        tool = next(
            value
            for key, value in zip(attrs.keys, attrs.values)
            if key.value == "create_grpc_proto"
        )
        self.assertEqual(
            "exec",
            next(item.value.value for item in tool.keywords if item.arg == "cfg"),
        )
        self.assertTrue(
            next(item.value.value for item in tool.keywords if item.arg == "executable")
        )
        self.assertIn(
            'build --action_env PYTHON_BIN_PATH="/opt/conda310/bin/python3"',
            (ROOT / ".bazelrc").read_text(),
        )

    def test_explicit_python_and_child_python_use_configured_bin(self):
        result = self.run_action(
            self.recorder(), "source with spaces", "out with spaces"
        )
        self.assertEqual(0, result.returncode, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(["source with spaces", "out with spaces"], value["argv"])
        self.assertEqual(
            Path(sys.executable).resolve(), Path(value["interpreter"]).resolve()
        )
        self.assertEqual(Path(sys.executable).resolve(), Path(value["child"]).resolve())

    def test_arguments_are_not_shell_code(self):
        malicious = "$(touch owned-marker); 'quoted value'"
        result = self.run_action(self.recorder(), malicious, malicious)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([malicious, malicious], json.loads(result.stdout)["argv"])
        self.assertFalse((self.directory / "owned-marker").exists())

    def test_nonzero_tool_exit_is_preserved(self):
        result = self.run_action(self.recorder(23))
        self.assertEqual(23, result.returncode)
        self.assertEqual(1, len(result.stdout.splitlines()))

    def test_missing_configured_python_fails_without_running_tool(self):
        result = self.run_action(self.recorder(), env={"PYTHON_BIN_PATH": ""})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("PYTHON_BIN_PATH must identify", result.stderr)
        self.assertEqual("", result.stdout)

    def test_unavailable_explicit_python_does_not_fallback(self):
        result = self.run_action(
            self.recorder(), env={"PYTHON_BIN_PATH": "/no/such/python3"}
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("/no/such/python3", result.stderr)
        self.assertEqual("", result.stdout)

    def generate_actual_proto(self, filename):
        source = ROOT / PROTO_ROOT / filename
        tool = ROOT / PROTO_ROOT / "create_grpc_proto.py"
        output = self.directory / "actual generated output"
        output.mkdir()
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                self.action["command"],
                "--",
                str(tool),
                str(PROTO_ROOT / filename),
                str(output),
            ],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "PYTHON_BIN_PATH": sys.executable},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        self.assertTrue(source.is_file())
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        generated = output / PROTO_ROOT
        expected = {source.stem + "_pb2.py", source.stem + "_pb2_grpc.py"}
        self.assertEqual(expected, {item.name for item in generated.iterdir()})
        for item in generated.iterdir():
            self.assertGreater(item.stat().st_size, 0)
            ast.parse(item.read_text(), filename=str(item))

    def test_actual_model_rpc_proto_generation(self):
        self.generate_actual_proto("model_rpc_service.proto")

    def test_actual_flexlb_schedule_proto_generation(self):
        self.generate_actual_proto("flexlb_schedule_service.proto")


if __name__ == "__main__":
    unittest.main()

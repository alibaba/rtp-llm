import ast
import os
import re
import unittest
from pathlib import Path


def _runfile(path: str) -> Path:
    return Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / path


class ExecOpsStubConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._source = _runfile(
            "rtp_llm/models_py/bindings/core/ExecOps.cc"
        ).read_text()
        self._stub = ast.parse(
            _runfile("rtp_llm/ops/librtp_compute_ops/__init__.pyi").read_text()
        )

    def test_all_module_functions_are_declared_in_stub(self) -> None:
        bound_functions = set(re.findall(r'm\.def\(\s*"([^"]+)"', self._source))
        stub_functions = {
            node.name for node in self._stub.body if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(
            bound_functions - stub_functions,
            set(),
            "librtp_compute_ops module functions missing from __init__.pyi",
        )

    def test_comm_and_cpu_tp_broadcaster_signatures(self) -> None:
        functions = {
            node.name: node
            for node in self._stub.body
            if isinstance(node, ast.FunctionDef)
        }
        expected = {
            "init_cpu_tp_broadcaster": (
                ["tp_rank", "tp_size", "base_path"],
                ["int", "int", "str"],
            ),
            "destroy_cpu_tp_broadcaster": ([], []),
            "register_comm_ops": (
                ["broadcast_fn", "allreduce_fn", "allgather_fn"],
                ["typing.Callable", "typing.Callable", "typing.Callable"],
            ),
            "clear_comm_ops": ([], []),
        }
        all_node = next(
            node
            for node in self._stub.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        )
        public_names = {
            element.value
            for element in all_node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        self.assertLessEqual(set(expected), public_names)
        for name, (arg_names, arg_types) in expected.items():
            function = functions[name]
            self.assertEqual([arg.arg for arg in function.args.args], arg_names)
            self.assertEqual(
                [ast.unparse(arg.annotation) for arg in function.args.args], arg_types
            )
            self.assertEqual(ast.unparse(function.returns), "None")


if __name__ == "__main__":
    unittest.main()

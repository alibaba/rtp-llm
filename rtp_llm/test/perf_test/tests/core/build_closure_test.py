"""Performance framework behavior and failure regressions."""

import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT
from rtp_llm.test.perf_test.tests.support.fixtures import (
    PERF_TEST_ROOT as public_boundary_ROOT,
)


class RuntimeBuildClosureTest(unittest.TestCase):

    def test_cuda13_reports_declare_platform_plotly_dependencies(self):
        tree = ast.parse((public_boundary_ROOT / "BUILD").read_text())
        rules = {"perf_test_runtime_lib", "perf_test_lib"}
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and (node.value.func.id == "load" or node.value.func.id == "py_library")
        ]
        for platform, repository in (
            ("using_cuda13_x86", "@pip_gpu_cuda13_torch"),
            ("using_cuda13_arm", "@pip_cuda13_arm_torch"),
        ):
            with self.subTest(platform=platform):
                declarations = {}
                namespace = dict(
                    py_library=lambda **kw: declarations.update({kw["name"]: kw}),
                    glob=lambda _: [],
                    select=lambda options: options.get(
                        "@rtp_llm//:" + platform,
                        options.get("//conditions:default", []),
                    ),
                )

                def load(label, *symbols, **aliases):
                    for name in aliases:
                        namespace[name] = (
                            lambda package, repo=label.split("//")[0]: repo
                            + "//"
                            + package
                        )

                namespace["load"] = load
                exec(
                    compile(ast.Module(body=nodes, type_ignores=[]), "BUILD", "exec"),
                    namespace,
                )

                def dependencies(name, visited=frozenset()):
                    if name in visited:
                        return set()
                    direct = set(declarations[name].get("deps", ()))
                    result = set(direct)
                    for label in direct:
                        if label.startswith(":"):
                            result.update(dependencies(label[1:], visited | {name}))
                    return result

                for name in rules:
                    self.assertIn(repository + "//plotly", dependencies(name))

    def test_declared_runtime_sources_import_in_isolated_tree(self):
        root = PERF_TEST_ROOT
        build = ast.parse((root / "BUILD").read_text())
        libraries = {}
        for node in build.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "py_library":
                    attrs = {item.arg: item.value for item in call.keywords}
                    libraries[ast.literal_eval(attrs["name"])] = ast.literal_eval(
                        attrs["srcs"]
                    )
        self.assertIn("aone_lib", libraries)
        sources = set(
            libraries["perf_test_runtime_lib"]
            + libraries["perf_test_entry_lib"]
            + libraries["aone_lib"]
            + libraries["perf_test_results_lib"]
            + libraries["perf_test_ssh_lib"]
        )
        prefix = "rtp_llm.test.perf_test."
        missing = set()
        for source in sources:
            for node in ast.walk(ast.parse((root / source).read_text())):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                    modules += [node.module + "." + alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    if module.startswith(prefix):
                        candidate = module[len(prefix) :].replace(".", "/") + ".py"
                        if (root / candidate).is_file() and candidate not in sources:
                            missing.add(candidate)
        self.assertEqual(
            set(), missing, "runtime imports missing from declared closure"
        )
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)
            package = isolated / "rtp_llm/test/perf_test"
            for source in sources:
                target = package / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / source, target)
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys, types; sys.path.insert(0, sys.argv[1]); sys.modules['yaml'] = types.ModuleType('yaml'); tokenizers = types.ModuleType('transformers'); tokenizers.AutoTokenizer = object; sys.modules['transformers'] = tokenizers; fuser = types.ModuleType('rtp_llm.utils.fuser'); fuser.fetch_remote_file_to_local = lambda path: path; sys.modules['rtp_llm.utils.fuser'] = fuser; from rtp_llm.test.perf_test.entry import plan_from_source; from rtp_llm.test.perf_test.ssh.source import resolve_source; assert callable(plan_from_source) and callable(resolve_source)",
                    str(isolated),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()

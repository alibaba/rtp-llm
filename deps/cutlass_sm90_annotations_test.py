"""Checkout tests for isolated CUTLASS 4.8 split-wheel annotations."""
import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Bazel execution reads declared runfiles only; checkout mode is explicit.
RUNFILES = os.environ.get("TEST_SRCDIR")
ROOT = None if RUNFILES else Path(os.environ.get("DSV41_DEPS_WORKSPACE", Path(__file__).absolute().parents[1]))


def data_path(repository, relative):
    if RUNFILES:
        result = Path(RUNFILES) / repository / relative
    elif repository == "rtp_deps" and (ROOT / "internal_source").is_dir():
        result = ROOT / "internal_source/deps" / relative
    elif repository in ("rtp_public_deps", "rtp_deps"):
        result = ROOT / "deps" / relative
    else:
        result = ROOT / relative
    if not result.is_file():
        raise FileNotFoundError(str(result))
    return result


def annotations(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Call) or not isinstance(n.value.func, ast.Name) or n.value.func.id != "load"]
    namespace = {"package_annotation": lambda **kwargs: kwargs,
                 "json": types.SimpleNamespace(encode=json.dumps)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class CutlassSm90AnnotationsTest(unittest.TestCase):
    def test_patch_hunks_retain_standard_context_prefixes(self):
        def validate(text):
            lines = text.splitlines()
            index = 0
            while index < len(lines):
                if lines[index].startswith("diff --git "):
                    self.assertTrue(lines[index + 1].startswith("--- "))
                    index += 1
                    continue
                if lines[index].startswith("--- "):
                    self.assertTrue(index > 0 and lines[index - 1].startswith("diff --git "))
                    self.assertTrue(lines[index + 1].startswith("+++ "))
                    index += 2
                    continue
                match = re.fullmatch(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", lines[index])
                self.assertIsNotNone(match)
                old_count, new_count = int(match[2]), int(match[4])
                index += 1
                old_lines = new_lines = 0
                while index < len(lines) and not lines[index].startswith(("@@ ", "--- ", "diff --git ")):
                    line = lines[index]
                    self.assertTrue(line and line[0] in " +-", repr(line))
                    old_lines += line[0] in " -"
                    new_lines += line[0] in " +"
                    index += 1
                self.assertEqual((old_lines, new_lines), (old_count, new_count))

        for name in ("0003-sm90-481-split-package-paths.patch", "0004-sm90-481-toolkit-library-paths.patch"):
            patch = data_path("rtp_llm", "patches/cutlass/" + name).read_text()
            validate(patch)
            # GNU patch accepts these malformed hunks; Bazel ctx.patch does not.
            without_boundaries = "\n".join(line for line in patch.splitlines() if not line.startswith("diff --git "))
            with self.assertRaises(AssertionError):
                validate(without_boundaries)
            malformed = patch.replace("\n \n", "\n\n")
            self.assertNotEqual(malformed, patch)
            with self.assertRaises(AssertionError):
                validate(malformed)

    def sources(self):
        yield data_path("rtp_public_deps", "pip.bzl")
        yield data_path("rtp_deps", "pip.bzl")

    def test_annotations_are_isolated_and_exact_version(self):
        for source in self.sources():
            ns = annotations(source)
            default, sm90 = ns["CUDA13_ANNOTATIONS"], ns["CUDA13_SM90_ANNOTATIONS"]
            for package in ("nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-base"):
                self.assertIsNot(default[package], sm90[package])
                for annotation, version in ((default, "4.7.1"), (sm90, "4.8.1")):
                    patches = annotation[package]["whl_patches"]
                    self.assertEqual(len(patches), 1)
                    label, spec = next(iter(patches.items()))
                    spec = json.loads(spec)
                    self.assertTrue(all("-" + version + "-" in name for name in spec["whls"]))
                    self.assertEqual(spec["patch_strip"], 1)
                    self.assertTrue(data_path("rtp_llm", label.split("//", 1)[1].replace(":", "/")).is_file())
                self.assertEqual(len(json.loads(next(iter(sm90[package]["whl_patches"].values())))["whls"]), 1)
            for package in default.keys() - {"nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-base"}:
                self.assertEqual(default[package], sm90[package])

    def test_only_sm90_repository_selects_new_annotations(self):
        for source in self.sources():
            ns = annotations(source)
            calls = []
            ns["pip_parse"] = lambda **kwargs: calls.append(kwargs)
            ns["pip_deps"]()
            calls = {c["name"]: c for c in calls}
            self.assertIs(calls["pip_gpu_cuda13_sm90_torch"]["annotations"], ns["CUDA13_SM90_ANNOTATIONS"])
            for repo in ("pip_gpu_cuda13_torch", "pip_cuda13_arm_torch"):
                self.assertIs(calls[repo]["annotations"], ns["CUDA13_ANNOTATIONS"])

    def test_namespace_patch_resolves_split_mlir_and_preserves_precedence(self):
        patch = data_path("rtp_llm", "patches/cutlass/0003-sm90-481-split-package-paths.patch").read_text()
        added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
        added = added.split("        if not dsl_libs:", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name / "nvidia_cutlass_dsl" for name in ("core", "base", "toolkit")]
            for root in roots:
                (root / "dsl_packages/cutlass").mkdir(parents=True)
            (roots[1] / "dsl_packages/cutlass/_mlir").mkdir()
            namespace = {"__path__": [str(roots[0] / "dsl_packages/cutlass")]}
            module = types.ModuleType("nvidia_cutlass_dsl")
            module.__path__ = [str(p) for p in roots]
            previous = sys.modules.get("nvidia_cutlass_dsl")
            sys.modules["nvidia_cutlass_dsl"] = module
            try:
                self.assertIsNone(importlib.machinery.PathFinder.find_spec("_mlir", namespace["__path__"]))
                exec(added, namespace)
                self.assertEqual(namespace["__path__"], [str(p / "dsl_packages/cutlass") for p in roots])
                self.assertIsNotNone(importlib.machinery.PathFinder.find_spec("_mlir", namespace["__path__"]))
                exec(added, namespace)
                self.assertEqual(len(namespace["__path__"]), 3)
            finally:
                if previous is None:
                    del sys.modules["nvidia_cutlass_dsl"]
                else:
                    sys.modules["nvidia_cutlass_dsl"] = previous


if __name__ == "__main__":
    unittest.main()

"""
Verify all transformers imports across source code and model custom code resolve correctly.

Scans two sources:
1. rtp_llm/ source directories — always runs
2. Model checkpoint directories from smoke test JSON configs — skips if not founded
"""

import ast
import importlib
import json
import os
import unittest

import rtp_llm.frontend.tokenizer_factory.tokenizers  # triggers compat shims

SCAN_SUBDIRS = [
    "rtp_llm/models",
    "rtp_llm/models_py",
    "rtp_llm/frontend",
    "rtp_llm/config",
    "rtp_llm/openai",
    "rtp_llm/utils",
    "rtp_llm/tools",
    "rtp_llm/test",
    "internal_source/rtp_llm/models",
    "internal_source/rtp_llm/tokenizers",
    "internal_source/rtp_llm/tools",
    "internal_source/rtp_llm/openai_renderers",
]

SMOKE_JSON_DIRS = [
    "rtp_llm/test/smoke/data/model",
    "internal_source/rtp_llm/test/smoke/data/model",
]

EXCLUDE_DIRS = {"__pycache__", "3rdparty"}


def _find_workspace_root():
    srcdir = os.environ["TEST_SRCDIR"]
    workspace = os.environ.get("TEST_WORKSPACE", "")
    return os.path.join(srcdir, workspace)


def _extract_model_paths(workspace_root):
    """Extract model_path values from smoke test JSON configs."""
    model_paths = set()
    for subdir in SMOKE_JSON_DIRS:
        search_dir = os.path.join(workspace_root, subdir)
        if not os.path.isdir(search_dir):
            continue
        for dirpath, _, files in os.walk(search_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                json_path = os.path.join(dirpath, f)
                try:
                    with open(json_path, "r") as fh:
                        content = fh.read()
                    lines = [
                        l
                        for l in content.split("\n")
                        if not l.lstrip().startswith("//")
                    ]
                    data = json.loads("\n".join(lines))
                    if "model_path" in data:
                        model_paths.add(data["model_path"])
                except (json.JSONDecodeError, OSError, KeyError, ValueError):
                    continue
    return model_paths


def _scan_transformers_imports(directory, exclude_prefixes=None):
    imports = set()
    if not os.path.isdir(directory):
        return imports
    for dirpath, dirs, files in os.walk(directory):
        dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
        for f in files:
            if not f.endswith(".py"):
                continue
            if exclude_prefixes and any(f.startswith(p) for p in exclude_prefixes):
                continue
            filepath = os.path.join(dirpath, f)
            try:
                with open(filepath, "r", errors="ignore") as fh:
                    tree = ast.parse(fh.read(), filename=filepath)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if module.startswith("transformers"):
                    for alias in node.names:
                        imports.add((module, alias.name, filepath))
    return imports


def _check_imports(imports):
    failures = []
    checked = set()
    for module_path, attr_name, filepath in sorted(imports):
        key = (module_path, attr_name)
        if key in checked:
            continue
        checked.add(key)
        try:
            mod = importlib.import_module(module_path)
            if getattr(mod, attr_name, None) is None:
                failures.append(
                    f"{module_path}.{attr_name} — attribute not found"
                    f"\n    in: {filepath}"
                )
        except ImportError as e:
            failures.append(f"{module_path}.{attr_name} — {e}\n    in: {filepath}")
    return failures


class TestTransformersImports(unittest.TestCase):

    def test_source_imports(self):
        root = _find_workspace_root()
        all_imports = set()
        for subdir in SCAN_SUBDIRS:
            all_imports.update(_scan_transformers_imports(os.path.join(root, subdir)))

        self.assertTrue(all_imports, "No transformers imports found in source")

        failures = _check_imports(all_imports)
        if failures:
            import transformers

            self.fail(
                f"\ntransformers=={transformers.__version__}: "
                f"{len(failures)} broken import(s) in source:\n"
                + "\n".join(f"  {f}" for f in failures)
            )

    def test_model_custom_code_imports(self):
        root = _find_workspace_root()
        model_paths = _extract_model_paths(root)
        if not model_paths:
            self.skipTest("No smoke JSON configs found in runfiles")

        scanned = 0
        all_imports = set()
        for model_path in sorted(model_paths):
            if not os.path.isdir(model_path):
                continue
            scanned += 1
            all_imports.update(
                _scan_transformers_imports(model_path, exclude_prefixes=["modeling_"])
            )

        if scanned == 0:
            self.skipTest("No model directories accessible")

        failures = _check_imports(all_imports)
        if failures:
            import transformers

            self.fail(
                f"\ntransformers=={transformers.__version__}: "
                f"{len(failures)} broken import(s) in model custom code "
                f"({scanned} dirs scanned):\n" + "\n".join(f"  {f}" for f in failures)
            )


if __name__ == "__main__":
    unittest.main()

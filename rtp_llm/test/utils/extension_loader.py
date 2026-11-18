"""Locate test extensions in Native installs and declared Bazel runfiles."""

import importlib.util
import os
from pathlib import Path


def load_test_extension(name, native_path, bazel_path):
    import rtp_llm

    roots = [Path(p).absolute().parent for p in rtp_llm.__path__]
    for variable in ("RUNFILES_DIR", "TEST_SRCDIR"):
        if os.environ.get(variable):
            root = Path(os.environ[variable])
            roots.extend(
                [root / os.environ.get("TEST_WORKSPACE", "rtp_llm"), root / "rtp_llm"]
            )
    candidates = [
        root / relative for relative in (native_path, bazel_path) for root in roots
    ]
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest:
        with open(manifest) as stream:
            for line in stream:
                logical, _, actual = line.rstrip("\n").partition(" ")
                if any(
                    logical.endswith("/" + path) for path in (native_path, bazel_path)
                ):
                    candidates.append(Path(actual))
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load extension spec: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"{name} is missing; searched {candidates}")

"""Test-only loader for a package assembled from standalone source files."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Iterable


def load_source_module(
    package_name: str,
    module_name: str,
    source_path: str,
    package_paths: Iterable[str],
):
    """Load one source file with sibling imports resolved inside a tiny package."""
    normalized_paths = [os.path.abspath(path) for path in package_paths]
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = normalized_paths
        package.__package__ = package_name
        sys.modules[package_name] = package
    elif list(package.__path__) != normalized_paths:
        raise ValueError(f"package {package_name!r} already has different paths")

    qualified_name = f"{package_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {qualified_name} from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module

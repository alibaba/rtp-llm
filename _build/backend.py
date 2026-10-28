"""Setuptools backend with the project's platform-selected native build inputs."""
from pathlib import Path
import importlib.util
import re

from setuptools import build_meta as _setuptools
from setuptools.build_meta import (
    build_editable, build_sdist, build_wheel,
    get_requires_for_build_sdist,
    prepare_metadata_for_build_editable, prepare_metadata_for_build_wheel,
)


def _native_build_requirements():
    # Load the same setup helpers used for runtime metadata, without invoking
    # setup() or importing the runtime package/Torch in the bootstrap process.
    path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("_rtp_build_requirements", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.should_skip_bazel_build():
        # Preserve the existing explicit deps-only editable installation path.
        return []
    requirements = []
    names = []
    for requirement in module.get_all_dependencies():
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        name = match.group(1).lower().replace("_", "-") if match else ""
        # Stub generation imports the real package after compilation. Its tools
        # and runtime preload inputs must also exist inside PEP 517 isolation.
        if name in {"torch", "triton", "numpy", "pybind11-stubgen"}:
            if requirement not in requirements:
                requirements.append(requirement)
                names.append(name)
    if names.count("torch") != 1:
        raise RuntimeError("Native build requires exactly one platform Torch dependency")
    if names.count("pybind11-stubgen") != 1:
        raise RuntimeError("Native stub generation requires pybind11_stubgen")
    return requirements


def get_requires_for_build_wheel(config_settings=None):
    return list(dict.fromkeys(
        _setuptools.get_requires_for_build_wheel(config_settings)
        + _native_build_requirements()
    ))


def get_requires_for_build_editable(config_settings=None):
    return list(dict.fromkeys(
        _setuptools.get_requires_for_build_editable(config_settings)
        + _native_build_requirements()
    ))

import functools
import importlib
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Optional

import torch

_LOGGER = logging.getLogger(__name__)
_ORIGINAL_SYSTEM_ATTR = "_rtp_llm_original_system"
_ORIGINAL_IMPORT_MODULE_ATTR = "_rtp_llm_original_import_module"
_PATCHED_ATTR = "_rtp_llm_aiter_jit_patch_installed"
_AITER_CODEGEN_SCRIPTS = {
    "codegen.py",
    "gen_instances.py",
    "gen_instances_cktile.py",
    "generate.py",
    "generate_binaryop.py",
}


def _is_rocm_runtime() -> bool:
    return torch.version.hip is not None


@functools.lru_cache(maxsize=1)
def _aiter_package_roots() -> tuple[str, ...]:
    roots = []
    for module_name in ("aiter", "aiter_meta"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            continue
        if spec is None or spec.submodule_search_locations is None:
            continue
        roots.extend(os.path.abspath(path) for path in spec.submodule_search_locations)
    return tuple(roots)


def _is_within_root(path: str, root: str) -> bool:
    try:
        logical_path = os.path.abspath(path)
        logical_root = os.path.abspath(root)
        if os.path.commonpath((logical_path, logical_root)) == logical_root:
            return True
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        return os.path.commonpath((real_path, real_root)) == real_root
    except ValueError:
        return False


def _is_aiter_codegen_command(command: object) -> bool:
    if not isinstance(command, str):
        return False
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    package_roots = _aiter_package_roots()
    return any(
        os.path.basename(arg) in _AITER_CODEGEN_SCRIPTS
        and any(_is_within_root(arg, root) for root in package_roots)
        for arg in args
    )


def _run_aiter_codegen_command(command: str) -> int:
    try:
        args = shlex.split(command)
    except ValueError as e:
        _LOGGER.error("failed to parse aiter JIT codegen command: %s", e)
        return 1
    if not args:
        return 0
    try:
        return subprocess.run(args, check=False).returncode
    except OSError as e:
        _LOGGER.error("failed to run aiter JIT codegen command: %s", e)
        return 1


def _extract_shared_object_path(error_message: str) -> Optional[str]:
    match = re.search(r"(/[^\s:]+\.so):\s+undefined symbol:", error_message)
    if match is None:
        return None
    return match.group(1)


def _maybe_remove_stale_aiter_jit_module(module_name: str, error: ImportError) -> bool:
    if not module_name.startswith("aiter.jit."):
        return False
    so_path = _extract_shared_object_path(str(error))
    if so_path is None or "/aiter/jit/" not in so_path:
        return False
    if not os.path.exists(so_path):
        return False

    try:
        os.remove(so_path)
        module_base = os.path.splitext(os.path.basename(so_path))[0]
        build_dir = os.path.join(os.path.dirname(so_path), "build", module_base)
        shutil.rmtree(build_dir, ignore_errors=True)
        _LOGGER.warning(
            "removed stale aiter JIT module after import failure: %s", so_path
        )
        return True
    except OSError as remove_error:
        _LOGGER.warning(
            "failed to remove stale aiter JIT module %s: %s", so_path, remove_error
        )
        return False


def install_aiter_jit_patch() -> None:
    if not _is_rocm_runtime() or getattr(os, _PATCHED_ATTR, False):
        return

    original_system = getattr(os, _ORIGINAL_SYSTEM_ATTR, os.system)
    original_import_module = getattr(
        importlib, _ORIGINAL_IMPORT_MODULE_ATTR, importlib.import_module
    )

    def patched_system(command):
        if _is_aiter_codegen_command(command):
            ret = _run_aiter_codegen_command(command)
            if ret != 0:
                raise RuntimeError(f"aiter JIT codegen failed with exit code {ret}")
            return ret
        return original_system(command)

    def patched_import_module(name, package=None):
        try:
            return original_import_module(name, package)
        except ImportError as e:
            if _maybe_remove_stale_aiter_jit_module(name, e):
                raise ModuleNotFoundError(
                    f"removed stale aiter JIT module {name}; rebuild required"
                ) from e
            raise

    setattr(os, _ORIGINAL_SYSTEM_ATTR, original_system)
    os.system = patched_system
    setattr(importlib, _ORIGINAL_IMPORT_MODULE_ATTR, original_import_module)
    importlib.import_module = patched_import_module
    setattr(os, _PATCHED_ATTR, True)

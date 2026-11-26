"""
FlashInfer Path Setup Utility

This module ensures flashinfer is imported from the specified custom path
by inserting it at the beginning of sys.path before any imports.

This is particularly useful in testing environments where you want to use
a specific version of flashinfer different from the system-installed one.
"""

import hashlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from filelock import FileLock, Timeout

from rtp_llm.utils import jit_cache_env as cache_env


def get_package_info(package_name):
    """Get the version, installation path, and RECORD fingerprint."""
    try:
        import importlib

        runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
        meta_package_name = package_name
        if package_name == "tvm_ffi":
            meta_package_name = "apache-tvm-ffi"
        elif package_name == "flashinfer":
            # the wheel that provides the flashinfer module is flashinfer-python
            meta_package_name = "flashinfer-python"

        saved_path = list(sys.path)
        try:
            if runfiles_dir and os.path.exists(runfiles_dir):
                if runfiles_dir not in sys.path:
                    sys.path.insert(0, runfiles_dir)

                for item in os.listdir(runfiles_dir):
                    if item.startswith("pip_"):
                        pip_path = os.path.join(runfiles_dir, item, "site-packages")
                        if os.path.exists(pip_path) and pip_path not in sys.path:
                            sys.path.insert(0, pip_path)

            # Probe via metadata only: importing the module here would pin the
            # runfiles copy in sys.modules for the rest of the process, and any
            # JIT rules that process generates would then anchor include paths
            # to the per-action runfiles tree, which dangles for every other
            # consumer of the shared JIT workspace.
            dist = importlib.metadata.distribution(meta_package_name)
            version = dist.version
            package_path = dist.locate_file(package_name)
            record = dist.read_text("RECORD")
        finally:
            sys.path[:] = saved_path
        if Path(str(package_path)).exists():
            record_hash = (
                hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None
            )
            return version, str(package_path), record_hash
        return None, None, None
    except Exception as e:
        logging.info(f"[Package Copy] Failed to get info for {package_name}: {e}")
        return None, None, None


def _build_variant(source_path):
    """The bazel external repo a package came from, e.g. pip_gpu_rocm_torch_torch.

    A version string is not a build identity. The same version is built per
    platform into separate external repos -- pip_gpu_cuda12_9_torch_torch,
    pip_gpu_rocm_torch_torch, the ppu and arm variants -- while the cache below
    lives in ~/.cache, shared by every job that lands on the host. Keyed on
    version alone, whichever job populates the cache first decides which build
    every later job on that host imports, and it does so silently, because the
    staged directory is prepended to sys.path.

    Returns None when the layout is unrecognised, in which case the caller keeps
    the old version-only key rather than guessing.
    """
    for part in Path(source_path).parts:
        if part.startswith("pip_"):
            return part
    return None


def _cache_key(package_name, version, source_path, record_hash):
    # The interpreter ABI is part of the build identity too: a host can serve
    # jobs running different Python versions from the same pip variant (the
    # 2026-10-23 ppu312 pool swap ran cp312 jobs alongside the cp310 bazel
    # lane), and a cached copy materialized under one interpreter breaks the
    # other (cp312-abi3 tvm_ffi core.abi3.so references Py_Version, undefined
    # on 3.10). Key on it so each interpreter stages its own copy.
    py_tag = "cp%d%d" % sys.version_info[:2]
    variant = _build_variant(source_path)
    key = f"{package_name}_python-{version}-{py_tag}"
    if variant is not None:
        key += f"__{variant}"
    # Wheel patches change RECORD while keeping the version and repo name.
    if record_hash is not None:
        key += f"__{record_hash}"
    return key


def _required_companion_dirs(package_name: str, source_parent: Path) -> list[str]:
    """Return companion sibling directories (e.g. .libs, deep_gemm_sm120) beside a wheel package."""
    candidates = [f"{package_name}.libs", f"{package_name}_libs"]
    if package_name == "deep_gemm":
        # DeepGEMM 2.8.2+ SM120 wheels place JIT headers in a sibling top-level
        # deep_gemm_sm120/ directory resolved relative to deep_gemm/__init__.py.
        candidates.append("deep_gemm_sm120")
    return [name for name in candidates if (source_parent / name).is_dir()]


def _is_real_dir(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir()


def _cache_ancestors_are_real_dirs(
    target_base: Path, target_site_packages: Path
) -> bool:
    return _is_real_dir(target_base) and _is_real_dir(target_site_packages)


def _is_internal_relative_symlink(entry: Path, companion_root: Path) -> bool:
    """Return True if ``entry`` is a relative symlink whose resolved target stays inside ``companion_root``."""
    try:
        raw_target = os.readlink(entry)
    except OSError:
        return False
    if os.path.isabs(raw_target):
        return False
    root_abs = os.path.abspath(companion_root)
    resolved = os.path.abspath(os.path.normpath(os.path.join(entry.parent, raw_target)))
    try:
        return os.path.commonpath([root_abs, resolved]) == root_abs
    except ValueError:
        return False


def _entry_is_intact(
    entry: Path, source: "Path | None" = None, companion_root: "Path | None" = None
) -> bool:
    if source is None:
        return (entry.is_symlink() and entry.exists()) or (
            not entry.is_symlink() and entry.is_file()
        )
    if source.is_symlink():
        preserve_symlink = (
            True
            if companion_root is None
            else _is_internal_relative_symlink(source, companion_root)
        )
        if preserve_symlink:
            try:
                return (
                    entry.is_symlink()
                    and entry.exists()
                    and os.readlink(entry) == os.readlink(source)
                )
            except OSError:
                return False
        # External runfiles symlinks are materialized as real files/dirs.
        try:
            resolved_src = source.resolve(strict=True)
        except OSError:
            return False
        if resolved_src.is_dir():
            return _companion_dir_is_complete(resolved_src, entry)
        return (
            not entry.is_symlink()
            and entry.is_file()
            and entry.stat().st_size == resolved_src.stat().st_size
        )
    return (
        not entry.is_symlink()
        and entry.is_file()
        and entry.stat().st_size == source.stat().st_size
    )


def _companion_dir_is_complete(source: Path, target: Path) -> bool:
    """Return True when every directory/file in ``source`` exists and is intact in ``target``."""
    if not source.is_dir() or not _is_real_dir(target):
        return False
    found_file = False
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        src_dir = Path(dirpath)
        rel_dir = src_dir.relative_to(source)
        dst_dir = target / rel_dir
        if not _is_real_dir(dst_dir):
            return False
        for name in dirnames:
            src_sub = src_dir / name
            dst_sub = dst_dir / name
            if src_sub.is_symlink():
                if not _entry_is_intact(dst_sub, src_sub, companion_root=source):
                    return False
                if not _is_internal_relative_symlink(src_sub, source):
                    found_file = True
            elif not _is_real_dir(dst_sub):
                return False
        for name in filenames:
            dst_entry = dst_dir / name
            if not _entry_is_intact(dst_entry, src_dir / name, companion_root=source):
                return False
            found_file = True
    return found_file


def _install_leaf_atomically(
    src_entry: Path,
    dst_entry: Path,
    pid: int,
    *,
    validate_target: bool = True,
    companion_root: "Path | None" = None,
) -> None:
    tmp_leaf = dst_entry.parent / f".{dst_entry.name}.tmp.{pid}"
    if tmp_leaf.is_symlink():
        tmp_leaf.unlink()
    elif tmp_leaf.is_dir():
        shutil.rmtree(tmp_leaf, ignore_errors=True)
    elif tmp_leaf.exists():
        tmp_leaf.unlink()
    try:
        if src_entry.is_symlink() and (
            companion_root is None
            or _is_internal_relative_symlink(src_entry, companion_root)
        ):
            os.symlink(os.readlink(src_entry), tmp_leaf)
            if not tmp_leaf.is_symlink() or (validate_target and not tmp_leaf.exists()):
                raise RuntimeError(f"Staged companion entry is incomplete: {tmp_leaf}")
        elif src_entry.is_symlink() and src_entry.resolve().is_dir():
            _copy_companion_tree(src_entry.resolve(), tmp_leaf, root=src_entry.resolve())
            if not _companion_dir_is_complete(src_entry.resolve(), tmp_leaf):
                raise RuntimeError(f"Staged companion entry is incomplete: {tmp_leaf}")
        else:
            shutil.copy2(src_entry, tmp_leaf)
            if not _entry_is_intact(tmp_leaf, src_entry, companion_root=companion_root):
                raise RuntimeError(f"Staged companion entry is incomplete: {tmp_leaf}")
        if dst_entry.is_dir() and not dst_entry.is_symlink() and tmp_leaf.is_dir():
            shutil.rmtree(dst_entry)
        os.replace(tmp_leaf, dst_entry)
    finally:
        if tmp_leaf.is_symlink():
            with suppress(OSError):
                tmp_leaf.unlink()
        elif tmp_leaf.is_dir():
            shutil.rmtree(tmp_leaf, ignore_errors=True)
        elif tmp_leaf.exists():
            with suppress(OSError):
                tmp_leaf.unlink()


def _copy_companion_tree(
    companion_source: Path, destination: Path, *, root: "Path | None" = None
) -> None:
    """Copy a companion tree preserving internal relative symlinks and materializing external ones."""
    source_root = companion_source if root is None else root
    destination.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(companion_source, followlinks=False):
        src_dir = Path(dirpath)
        dst_dir = destination / src_dir.relative_to(companion_source)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in list(dirnames):
            src_sub = src_dir / name
            dst_sub = dst_dir / name
            if src_sub.is_symlink():
                dirnames.remove(name)
                if _is_internal_relative_symlink(src_sub, source_root):
                    os.symlink(os.readlink(src_sub), dst_sub)
                else:
                    resolved_sub = src_sub.resolve()
                    if resolved_sub.is_dir():
                        _copy_companion_tree(resolved_sub, dst_sub, root=resolved_sub)
                    else:
                        shutil.copy2(resolved_sub, dst_sub)
        for name in filenames:
            src_file = src_dir / name
            dst_file = dst_dir / name
            if src_file.is_symlink() and _is_internal_relative_symlink(
                src_file, source_root
            ):
                os.symlink(os.readlink(src_file), dst_file)
            else:
                shutil.copy2(src_file, dst_file)


def _stage_and_install_companion_dir(
    companion_source: Path, companion_target: Path
) -> None:
    """Install a companion dir atomically or repair missing leaves in place."""
    pid = os.getpid()
    if _is_real_dir(companion_target):
        # Never move or remove a live companion directory while lockless JIT
        # readers may be resolving headers from it; repair missing/broken
        # regular files before symlinks so relative links resolve when validated.
        deferred_symlinks: list[tuple[Path, Path]] = []
        for dirpath, dirnames, filenames in os.walk(
            companion_source, followlinks=False
        ):
            src_dir = Path(dirpath)
            dst_dir = companion_target / src_dir.relative_to(companion_source)
            if dst_dir.is_symlink() or (dst_dir.exists() and not dst_dir.is_dir()):
                dst_dir.unlink()
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in list(dirnames):
                src_sub = src_dir / name
                dst_sub = dst_dir / name
                if src_sub.is_symlink():
                    dirnames.remove(name)
                    if _is_internal_relative_symlink(src_sub, companion_source):
                        deferred_symlinks.append((src_sub, dst_sub))
                    elif not _entry_is_intact(
                        dst_sub, src_sub, companion_root=companion_source
                    ):
                        _install_leaf_atomically(
                            src_sub,
                            dst_sub,
                            pid,
                            companion_root=companion_source,
                        )
                elif dst_sub.is_symlink() or (
                    dst_sub.exists() and not dst_sub.is_dir()
                ):
                    dst_sub.unlink()
            for name in filenames:
                src_entry = src_dir / name
                dst_entry = dst_dir / name
                if src_entry.is_symlink() and _is_internal_relative_symlink(
                    src_entry, companion_source
                ):
                    deferred_symlinks.append((src_entry, dst_entry))
                elif not _entry_is_intact(
                    dst_entry, src_entry, companion_root=companion_source
                ):
                    _install_leaf_atomically(
                        src_entry,
                        dst_entry,
                        pid,
                        companion_root=companion_source,
                    )
        for src_link, dst_link in deferred_symlinks:
            if not _entry_is_intact(dst_link, src_link, companion_root=companion_source):
                _install_leaf_atomically(
                    src_link,
                    dst_link,
                    pid,
                    validate_target=False,
                    companion_root=companion_source,
                )
        if not _companion_dir_is_complete(companion_source, companion_target):
            raise RuntimeError(
                f"Companion directory repair failed verification: {companion_target}"
            )
        return

    for stale in companion_target.parent.glob(f".{companion_target.name}.tmp.*"):
        if stale.is_symlink():
            with suppress(OSError):
                stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        elif stale.exists():
            with suppress(OSError):
                stale.unlink()

    staging = companion_target.parent / f".{companion_target.name}.tmp.{pid}"
    if staging.is_symlink():
        staging.unlink()
    elif staging.exists():
        shutil.rmtree(staging)
    try:
        _copy_companion_tree(companion_source, staging)
        if not _companion_dir_is_complete(companion_source, staging):
            raise RuntimeError(f"Staged companion directory is incomplete: {staging}")
        if companion_target.is_symlink() or (
            companion_target.exists() and not companion_target.is_dir()
        ):
            companion_target.unlink()
        os.replace(staging, companion_target)
    finally:
        if staging.is_symlink():
            with suppress(OSError):
                staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _cached_package_is_complete(
    completion_marker: Path,
    target_package_path: Path,
    source_parent: Path,
    target_site_packages: Path,
    required_companions: list[str],
) -> bool:
    target_base = target_site_packages.parent
    return (
        _cache_ancestors_are_real_dirs(target_base, target_site_packages)
        and not completion_marker.is_symlink()
        and completion_marker.is_file()
        and not target_package_path.is_symlink()
        and target_package_path.exists()
        and all(
            _companion_dir_is_complete(
                source_parent / name, target_site_packages / name
            )
            for name in required_companions
        )
    )


def copy_package_with_lock(package_name, cache_dir):
    """
    Copy a Python package to cache directory with version in name.
    Uses file lock to prevent concurrent copies.
    Returns the path to the copied package's site-packages directory.
    """
    logging.info(f"[Package Copy] Processing {package_name}...")

    # Get package version and source path
    version, source_path, record_hash = get_package_info(package_name)
    if not version or not source_path:
        logging.info(f"[Package Copy] Package {package_name} not found, skipping")
        return None

    logging.info(f"[Package Copy] Found {package_name} v{version} at {source_path}")

    source_parent = Path(source_path).parent
    required_companions = _required_companion_dirs(package_name, source_parent)
    cache_key = _cache_key(package_name, version, source_path, record_hash)
    target_base = Path(cache_dir) / cache_key
    target_site_packages = target_base / "site-packages"
    target_package_path = target_site_packages / Path(source_path).name

    # Lock file for this specific package and version
    lock_file = Path(cache_dir) / f".{cache_key}.lock"
    completion_marker = target_base / ".copy_complete"

    # Check if already copied and complete. The marker is written last, so an
    # unmarked tree was never handed out and its copy is (re)done under the
    # lock. Also verify required companion dirs so older incomplete caches heal.
    if _cached_package_is_complete(
        completion_marker,
        target_package_path,
        source_parent,
        target_site_packages,
        required_companions,
    ):
        logging.info(
            f"[Package Copy] {package_name} v{version} already cached at {target_base}"
        )
        return str(target_site_packages)

    # Use file lock to prevent concurrent copies
    with FileLock(str(lock_file), timeout=300):
        # Double check after acquiring lock
        if _cached_package_is_complete(
            completion_marker,
            target_package_path,
            source_parent,
            target_site_packages,
            required_companions,
        ):
            logging.info(
                f"[Package Copy] {package_name} v{version} already cached (confirmed after lock)"
            )
            return str(target_site_packages)

        logging.info(
            f"[Package Copy] Copying {package_name} to {target_package_path}..."
        )

        # Create target directories after stripping any linked/non-dir cache ancestors.
        for ancestor in (target_base, target_site_packages):
            if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):
                ancestor.unlink()
        if completion_marker.is_symlink() or (
            completion_marker.exists() and not completion_marker.is_file()
        ):
            if completion_marker.is_symlink() or not completion_marker.is_dir():
                completion_marker.unlink()
            else:
                shutil.rmtree(completion_marker)
        if target_package_path.is_symlink():
            target_package_path.unlink()
        target_site_packages.mkdir(parents=True, exist_ok=True)

        # If the main package and marker were already published, only repair
        # missing/broken companion dirs atomically so active readers never lose
        # the live package or marker.
        if completion_marker.exists() and target_package_path.exists():
            try:
                for companion_name in required_companions:
                    companion_source = source_parent / companion_name
                    companion_target = target_site_packages / companion_name
                    if not _companion_dir_is_complete(
                        companion_source, companion_target
                    ):
                        logging.info(
                            f"[Package Copy] Repairing {companion_name} directory..."
                        )
                        _stage_and_install_companion_dir(
                            companion_source, companion_target
                        )
                return str(target_site_packages)
            except Exception as e:
                logging.info(
                    f"[Package Copy] Failed to repair companion dirs for {package_name}: {e}"
                )
                return None

        # Invalidate an incomplete generation before recreating its package.
        completion_marker.unlink(missing_ok=True)
        if target_package_path.exists():
            shutil.rmtree(target_package_path)

        # Copy the package
        try:
            if Path(source_path).is_dir():
                shutil.copytree(source_path, target_package_path, symlinks=True)
            else:
                shutil.copy2(source_path, target_package_path)

            # Copy companion directories (e.g. *.libs, *_libs, deep_gemm_sm120).
            for companion_name in required_companions:
                companion_source = source_parent / companion_name
                companion_target = target_site_packages / companion_name
                logging.info(f"[Package Copy] Copying {companion_name} directory...")
                _stage_and_install_companion_dir(companion_source, companion_target)

            # Verify copy
            if not target_package_path.exists() or any(
                not _companion_dir_is_complete(
                    source_parent / name, target_site_packages / name
                )
                for name in required_companions
            ):
                raise RuntimeError(f"Copy verification failed for {package_name}")

            # Create completion marker
            completion_marker.write_text(f"{package_name}=={version}\n")

            logging.info(
                f"[Package Copy] Successfully copied {package_name} v{version}"
            )
            return str(target_site_packages)

        except Exception as e:
            logging.info(f"[Package Copy] Failed to copy {package_name}: {e}")
            # Clean up incomplete copy; the missing marker means this
            # generation was never published, so deleting is safe.
            completion_marker.unlink(missing_ok=True)
            if target_package_path.exists():
                shutil.rmtree(target_package_path, ignore_errors=True)
            for companion_name in required_companions:
                companion_target = target_site_packages / companion_name
                if companion_target.exists():
                    shutil.rmtree(companion_target, ignore_errors=True)
            return None


def modify_bazel_wrapper_pythonpath(wrapper_path):
    """
    Modify Bazel-generated wrapper to inject _JIT_CACHE_PATHS at the beginning of PYTHONPATH.

    Args:
        wrapper_path: Path to the Bazel-generated wrapper file
    """
    try:
        with open(wrapper_path, "r") as f:
            lines = f.readlines()
        # Find the line index where new_env['PYTHONPATH'] = python_path (line 479)
        target_line_idx = None
        for i, line in enumerate(lines):
            if "new_env['PYTHONPATH'] = python_path" in line:
                target_line_idx = i
                break
        if target_line_idx is None:
            logging.warning(
                f"[Package Setup] Could not find target line in wrapper: {wrapper_path}"
            )
            return False

        # Create injection code to insert before line 479
        injection_lines = [
            "  # Inject _JIT_CACHE_PATHS at the beginning of PYTHONPATH\n",
            "  jit_cache_paths = os.environ.get('_JIT_CACHE_PATHS', '')\n",
            "  if jit_cache_paths:\n",
            "    jit_cache_entries = jit_cache_paths.split(os.pathsep)\n",
            "    # Prepend cache paths to the beginning of python_path\n",
            "    python_path = os.pathsep.join(jit_cache_entries) + os.pathsep + python_path\n",
        ]

        # Insert the code before the target line
        lines[target_line_idx:target_line_idx] = injection_lines

        import stat

        if os.path.exists(wrapper_path):
            # Make file writable
            current_permissions = os.stat(wrapper_path).st_mode
            os.chmod(wrapper_path, current_permissions | stat.S_IWRITE)

        # Write back to file
        with open(wrapper_path, "w") as f:
            f.writelines(lines)

        logging.info(f"[Package Setup] Modified Bazel wrapper: {wrapper_path}")
        logging.info(
            f"[Package Setup] Injected _JIT_CACHE_PATHS at the beginning of PYTHONPATH"
        )
        return True

    except Exception as e:
        logging.warning(
            f"[Package Setup] Failed to modify Bazel wrapper {wrapper_path}: {e}"
        )
        return False


def _flashinfer_op_is_stale(op_dir: Path) -> bool:
    rules_path = op_dir / "build.ninja"
    try:
        rules = rules_path.read_text(errors="ignore")
    except OSError:
        return False
    # FlashInfer embeds the absolute include paths of the package instance that
    # generated the rules (both -I and -isystem forms). A bazel action's runfiles
    # tree only lives for that action, so rules anchored there dangle for every
    # later consumer of the shared workspace -- whether or not the tree still
    # exists at eviction time -- and only rules anchored to stable locations
    # (package-copy cache, toolkit dirs) are safe to keep.
    include_paths = re.findall(r"(?:-I|-isystem\s+)(/\S+)", rules)
    if any(".runfiles" in path for path in include_paths):
        return True
    if any(not Path(path).exists() for path in include_paths):
        return True
    # Ninja files survive changes to CC/CXX. FlashInfer reuses existing rules,
    # so a removed toolchain must invalidate them just like a missing header.
    compilers = re.findall(r"^\s*(?:cxx|nvcc)\s*=\s*(/\S+)", rules, re.MULTILINE)
    compilers += re.findall(r"-ccbin(?:=|\s+)(/\S+)", rules)
    if any(not Path(compiler).is_file() for compiler in compilers):
        return True
    src_link = op_dir / "src"
    return src_link.is_symlink() and not src_link.exists()


def _evict_dangling_flashinfer_ops(workspace: Path) -> None:
    """Drop cached FlashInfer ops whose build rules anchor to include paths that
    are not stable for later consumers of the shared workspace.

    FlashInfer's generated FMHA ninja rules embed the absolute include paths of the
    flashinfer package instance that generated them. A bazel test action's runfiles
    sandbox is per-action, so rules generated from a runfiles import dangle for every
    later action that reuses the shared workspace and fail with missing-header
    errors -- including when the generating action is still alive at eviction time,
    because its sandbox disappears when the action completes. Evicting the affected
    ops makes FlashInfer regenerate them against the current stable package copy.

    FlashInfer serializes each op's build and load with a FileLock at
    ``cached_ops/tmp/<op_name>.lock``; eviction takes that same lock and re-judges
    staleness only inside it, so it cannot delete an op directory that a concurrent
    compile or load is using or has just refreshed.

    The lock wait is deliberately near-non-blocking: a stale op left behind only
    costs its consumer one rebuild, while blocking behind a long compile here
    silently stalls test startup (observed as a 664s pre-test gap).
    """
    started = time.monotonic()
    stale_candidates = []
    # FlashInfer layout only: cached_ops/<op_name>/build.ninja. Scan that one
    # directory directly instead of walking the workspace tree: the workspace
    # root must never be treated as enumerable (a HOME-sized root once dragged
    # the whole FUSE-mounted model library into this scan).
    # flashinfer >= 0.6.9 nests the ops dir as
    # <workspace>/.cache/flashinfer/<version>/<arch>/cached_ops (jit/env.py),
    # so scan both layouts: missing the nested one is how poisoned
    # runfiles-anchored ops survived eviction in CI run 85083250.
    cached_ops_roots = [workspace / "cached_ops"]
    try:
        cached_ops_roots += sorted(
            (workspace / ".cache" / "flashinfer").glob("*/*/cached_ops")
        )
    except OSError:
        pass  # neighbor scope vanished or is not listable
    for cached_ops_root in cached_ops_roots:
        try:
            if not cached_ops_root.is_dir():
                continue
            op_dirs = sorted(cached_ops_root.iterdir())
        except OSError:
            continue  # cross-UID shared tree: neighbor scope not listable, or deleted mid-scan
        for op_dir in op_dirs:
            if (op_dir / "build.ninja").is_file() and _flashinfer_op_is_stale(op_dir):
                stale_candidates.append(op_dir)
    logging.info(
        f"[Package Setup] FlashInfer eviction scan found "
        f"{len(stale_candidates)} stale candidate(s) under {workspace}"
    )
    evicted = 0
    skipped = 0
    for op_dir in sorted(stale_candidates):
        lock_dir = op_dir.parent / "tmp"
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(lock_dir / f"{op_dir.name}.lock"), timeout=3)
            with lock:
                # Re-read the rules under the op's own lock: a compile that ran
                # since the first pass may have replaced the stale rules already.
                if _flashinfer_op_is_stale(op_dir):
                    shutil.rmtree(op_dir, ignore_errors=True)
                    evicted += 1
                    logging.info(
                        f"[Package Setup] Evicted stale FlashInfer cache op: {op_dir}"
                    )
        except Timeout:
            # Somebody else owns the op (likely recompiling it); leave it alone.
            skipped += 1
            logging.info(f"[Package Setup] Skipped eviction of {op_dir}: lock busy")
        except OSError as error:
            # Cross-UID shared tree: a neighbor scope with restrictive
            # permissions, or the tree was deleted by a concurrent
            # commit_restore. Skipping one candidate only costs a rebuild.
            skipped += 1
            logging.info(f"[Package Setup] Skipped eviction of {op_dir}: {error}")
    logging.info(
        f"[Package Setup] FlashInfer eviction done in "
        f"{time.monotonic() - started:.1f}s: evicted={evicted} skipped={skipped} "
        f"of {len(stale_candidates)} candidate(s)"
    )


def setup_jit_cache(cache_dir=None, packages=None):
    logging.info("[Package Setup] setup_jit_cache starting")
    # 缓存基根前置唯一确定：传入 cache_dir 直接用，未传回落 HOME。路径解析
    # （auto 等）只在 server 参数解析（jit_group_args 的 --remote_jit_dir）
    # 一处；本函数不做任何派生，组件目录一律简单 join。
    base = Path(cache_dir) if cache_dir else Path.home()
    base.mkdir(parents=True, exist_ok=True)

    # workspace 必须是真实 FlashInfer 缓存目录，驱逐只扫其 cached_ops 子目录，
    # 不会爬基根本体。
    cache_env.configure_writable_cache_env(
        "FLASHINFER_WORKSPACE_BASE",
        base / "flashinfer",
        "flashinfer",
    )
    _evict_dangling_flashinfer_ops(Path(os.environ["FLASHINFER_WORKSPACE_BASE"]))
    # server 侧 jit_cache_manager 会把 flashinfer op 重定向到受管 v2 scope 工
    # 作区，上面的默认根扫描看不到它们：build.ninja 内嵌 per-action runfiles
    # include 路径的陈旧 op 会存活并毒化下一个冷启动消费者（run 85063732：
    # fused_moe_trtllm_sm100 从已销毁的 runfiles 树带出 include）。也扫每个
    # scope 的 flashinfer 工作区。树是跨 UID 共享的：邻居 scope 可能不可读或
    # 整树在扫描中被删，glob 有守卫。
    try:
        v2_scopes = sorted(
            Path("/tmp/rtp-llm/.jit_cache/v2").glob("*/flashinfer")
        )
    except OSError:
        v2_scopes = []
    for scope_workspace in v2_scopes:
        _evict_dangling_flashinfer_ops(scope_workspace)

    # triton 自带内容寻址 key（kernel 源/constexpr/目标 arch），只需解释器 ABI
    # tag 目录（launcher 嵌入 CPython ABI，同机 cp310/cp312 lane 并存）。
    cache_env.configure_writable_cache_env(
        "TRITON_CACHE_DIR",
        base / "triton_disk" / ("cp%d%d" % sys.version_info[:2]),
        "triton",
    )

    package_dir = cache_env.ensure_writable_directory(base / "python_packages")
    if package_dir is None:
        package_dir = cache_env.ensure_writable_directory(
            cache_env.local_jit_fallback("python_packages")
        )
    if package_dir is None:
        raise OSError("no writable Python package cache directory")

    # DeepGEMM 的 NVCC 编译前会 chdir 进 JIT tmp 目录：相对缓存路径会被二次
    # 解析导致 kernel.cu 不可达。默认与传入路径都先归一化再启动测试进程。
    cache_env.configure_writable_cache_env(
        "DG_JIT_CACHE_DIR",
        base / "deep_gemm",
        "deep_gemm",
    )
    logging.info(
        f"[Package Setup] Set DG_JIT_CACHE_DIR: {os.environ['DG_JIT_CACHE_DIR']}"
    )

    if packages is None:
        # flashinfer_cubin carries the flashInferMetaInfo.h / csrc headers that
        # fmha_gen ops embed as -I paths: without a stable package copy those
        # anchors point into per-action runfiles and poison the shared cache.
        packages = ["flashinfer", "flashinfer_cubin", "torch", "deep_gemm", "tvm_ffi"]

    # Copy packages to cache with file locking
    copied_paths = []
    for package_name in packages:
        site_packages_path = copy_package_with_lock(package_name, package_dir)
        if site_packages_path:
            copied_paths.append(site_packages_path)

    if not copied_paths:
        logging.info("[Package Setup] Warning: No packages were successfully copied")
        return None

    # Store cached package paths in environment variable for bootstrap script
    os.environ["_JIT_CACHE_PATHS"] = ":".join(copied_paths)
    logging.info(
        f"[Package Setup] Set _JIT_CACHE_PATHS: {os.environ['_JIT_CACHE_PATHS']}"
    )
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    test_binary = sys.argv[1] if len(sys.argv) > 1 else ""
    if not runfiles_dir or not test_binary:
        logging.warning(
            "[Package Setup] Bazel wrapper unavailable; skip cache path injection"
        )
        return None
    bazel_wrapper_path = Path(runfiles_dir) / "rtp_llm" / test_binary
    suffix = f"_new_{os.getpid()}"
    bazel_wrapper_path_new = bazel_wrapper_path.with_name(
        bazel_wrapper_path.name + suffix
    )
    try:
        bazel_wrapper_path_new.unlink(missing_ok=True)
        shutil.copy2(bazel_wrapper_path, bazel_wrapper_path_new)
        if not modify_bazel_wrapper_pythonpath(bazel_wrapper_path_new):
            raise OSError("wrapper injection failed")
    except Exception as error:
        with suppress(OSError):
            bazel_wrapper_path_new.unlink()
        logging.warning(f"[Package Setup] wrapper setup failed ({error})")
        return None
    logging.info(f"[Package Setup] Copied Bazel wrapper to: {bazel_wrapper_path_new}")
    sys.argv[1] = test_binary + suffix

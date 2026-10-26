"""
FlashInfer Path Setup Utility

This module ensures flashinfer is imported from the specified custom path
by inserting it at the beginning of sys.path before any imports.

This is particularly useful in testing environments where you want to use
a specific version of flashinfer different from the system-installed one.
"""

import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock, Timeout

from rtp_llm.utils import jit_cache_env as cache_env


def get_package_info(package_name):
    """Get package version and installation path"""
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
        finally:
            sys.path[:] = saved_path
        if Path(str(package_path)).exists():
            return version, str(package_path)
        return None, None
    except Exception as e:
        logging.info(f"[Package Copy] Failed to get info for {package_name}: {e}")
        return None, None


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


def _cache_key(package_name, version, source_path):
    # The interpreter ABI is part of the build identity too: a host can serve
    # jobs running different Python versions from the same pip variant (the
    # 2026-10-23 ppu312 pool swap ran cp312 jobs alongside the cp310 bazel
    # lane), and a cached copy materialized under one interpreter breaks the
    # other (cp312-abi3 tvm_ffi core.abi3.so references Py_Version, undefined
    # on 3.10). Key on it so each interpreter stages its own copy.
    py_tag = "cp%d%d" % sys.version_info[:2]
    variant = _build_variant(source_path)
    key = "%s_python-%s-%s" % (package_name, version, py_tag)
    if variant is not None:
        key += "__" + variant
    if package_name in ("flashinfer", "flashinfer_cubin"):
        # Bazel wheel patches keep upstream metadata unchanged. Include Python
        # payloads so a same-version cached copy cannot hide a dependency fix.
        source = Path(source_path)
        digest = sha256()
        for path in sorted(source.rglob("*.py")):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
        key += "__src_" + digest.hexdigest()[:16]
    return key


def copy_package_with_lock(package_name, cache_dir):
    """
    Copy a Python package to cache directory with version in name.
    Uses file lock to prevent concurrent copies.
    Returns the path to the copied package's site-packages directory.
    """
    logging.info(f"[Package Copy] Processing {package_name}...")

    # Get package version and source path
    version, source_path = get_package_info(package_name)
    if not version or not source_path:
        logging.info(f"[Package Copy] Package {package_name} not found, skipping")
        return None

    logging.info(f"[Package Copy] Found {package_name} v{version} at {source_path}")

    # Keyed on version AND build variant; see _build_variant.
    cache_key = _cache_key(package_name, version, source_path)
    target_base = Path(cache_dir) / cache_key
    target_site_packages = target_base / "site-packages"
    target_package_path = target_site_packages / Path(source_path).name

    # Lock file for this specific package and version
    lock_file = Path(cache_dir) / f".{cache_key}.lock"
    completion_marker = target_base / ".copy_complete"

    # Check if already copied and complete
    if completion_marker.exists() and target_package_path.exists():
        logging.info(
            f"[Package Copy] {package_name} v{version} already cached at {target_base}"
        )
        return str(target_site_packages)

    # Use file lock to prevent concurrent copies
    with FileLock(str(lock_file), timeout=300):
        # Double check after acquiring lock
        if completion_marker.exists() and target_package_path.exists():
            logging.info(
                f"[Package Copy] {package_name} v{version} already cached (confirmed after lock)"
            )
            return str(target_site_packages)

        logging.info(
            f"[Package Copy] Copying {package_name} to {target_package_path}..."
        )

        # Create target directories
        target_site_packages.mkdir(parents=True, exist_ok=True)

        # Remove incomplete copy if exists
        if target_package_path.exists():
            shutil.rmtree(target_package_path)

        # Copy the package
        try:
            if Path(source_path).is_dir():
                shutil.copytree(
                    source_path,
                    target_package_path,
                    symlinks=package_name != "flashinfer_cubin",
                )
            else:
                shutil.copy2(source_path, target_package_path)

            # Also copy .libs directory if it exists (for torch, flashinfer, etc.)
            source_parent = Path(source_path).parent
            libs_dirs = [f"{package_name}.libs", f"{package_name}_libs"]
            for libs_dir_name in libs_dirs:
                libs_source = source_parent / libs_dir_name
                if libs_source.exists() and libs_source.is_dir():
                    libs_target = target_site_packages / libs_dir_name
                    if not libs_target.exists():
                        logging.info(
                            f"[Package Copy] Copying {libs_dir_name} directory..."
                        )
                        shutil.copytree(libs_source, libs_target, symlinks=True)

            # Verify copy
            if not target_package_path.exists():
                raise RuntimeError(f"Copy verification failed for {package_name}")

            # Create completion marker
            completion_marker.write_text(f"{package_name}=={version}\n")

            logging.info(
                f"[Package Copy] Successfully copied {package_name} v{version}"
            )
            return str(target_site_packages)

        except Exception as e:
            logging.info(f"[Package Copy] Failed to copy {package_name}: {e}")
            # Clean up incomplete copy
            if target_package_path.exists():
                shutil.rmtree(target_package_path, ignore_errors=True)
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


def bootstrap_remote_jit_dir():
    remote = os.environ.get("REMOTE_JIT_DIR", "").strip()
    if not remote or urlparse(remote).scheme:
        return
    try:
        root = Path(remote).expanduser()
        if not root.is_absolute():
            raise OSError(f"not an absolute path: {remote}")
        if root.is_symlink() or root.parent.is_symlink():
            raise OSError(f"symlinked path: {remote}")
        try:
            root.mkdir(parents=True)
        except FileExistsError:
            if not root.is_dir() or not os.access(root, os.R_OK | os.W_OK | os.X_OK):
                raise OSError(f"unusable directory: {remote}")
        else:
            root.chmod(0o1777)  # protect direct children of a shared root
        os.environ["REMOTE_JIT_DIR"] = str(root)
    except (OSError, RuntimeError) as e:
        os.environ.pop("REMOTE_JIT_DIR", None)  # child servers inherit the env
        logging.warning(f"[JIT] REMOTE_JIT_DIR refused ({e}); cold start later")


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
    for root, _dirs, files in os.walk(workspace, followlinks=False):
        op_dir = Path(root)
        # FlashInfer layout only: cached_ops/<op_name>/build.ninja (never trust an
        # unrelated build.ninja deeper in the tree).
        if op_dir.parent.name != "cached_ops" or "build.ninja" not in files:
            continue
        if _flashinfer_op_is_stale(op_dir):
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
    logging.info(
        f"[Package Setup] FlashInfer eviction done in "
        f"{time.monotonic() - started:.1f}s: evicted={evicted} skipped={skipped} "
        f"of {len(stale_candidates)} candidate(s)"
    )


def _flashinfer_cache_scope():
    # FlashInfer keys its ops by its own version and GPU, but their shared
    # libraries also link the selected Torch/TVM FFI runtime. A runfiles path
    # may change per action; the wheel build variant remains stable.
    identities = []
    for package in ("torch", "tvm_ffi"):
        version, source = get_package_info(package)
        identities.append(_cache_key(package, version or "absent", source or ""))
    return sha256("\0".join(identities).encode()).hexdigest()[:16]


def setup_jit_cache(cache_dir=None, packages=None):
    logging.info("[Package Setup] setup_jit_cache starting")
    bootstrap_remote_jit_dir()
    flashinfer_scope = _flashinfer_cache_scope()
    cache_env.configure_writable_cache_env(
        "FLASHINFER_WORKSPACE_BASE",
        Path.home() / ".cache" / "rtp-llm" / "flashinfer" / flashinfer_scope,
        f"flashinfer-{flashinfer_scope}",
        cache_subpath=(".cache", "flashinfer"),
    )
    _evict_dangling_flashinfer_ops(Path(os.environ["FLASHINFER_WORKSPACE_BASE"]))
    cache_env.configure_writable_cache_env(
        "TRITON_CACHE_DIR", Path.home() / ".triton" / "cache", "triton"
    )

    requested_cache_dir = cache_dir or Path.home() / ".cache"
    cache_dir = cache_env.ensure_writable_directory(requested_cache_dir)
    if cache_dir is None:
        cache_dir = cache_env.ensure_writable_directory(
            cache_env.local_jit_fallback("python_packages")
        )
    if cache_dir is None:
        raise OSError("no writable Python package cache directory")

    # DeepGEMM's NVCC compiler changes into the JIT tmp directory before
    # compiling. A relative cache path would then be resolved a second time
    # and make the generated kernel.cu unreachable. Normalize both the default
    # and caller-provided path before launching the actual test process.
    cache_env.configure_writable_cache_env(
        "DG_JIT_CACHE_DIR", Path.home() / ".deep_gemm", "deep_gemm"
    )
    logging.info(
        f"[Package Setup] Set DG_JIT_CACHE_DIR: {os.environ['DG_JIT_CACHE_DIR']}"
    )

    if packages is None:
        packages = ["flashinfer", "flashinfer_cubin", "torch", "deep_gemm", "tvm_ffi"]

    # Copy packages to cache with file locking
    copied_paths = []
    for package_name in packages:
        site_packages_path = copy_package_with_lock(package_name, cache_dir)
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

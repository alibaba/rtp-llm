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
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock


def _is_pip_repo(name):
    """Whether a directory in runfiles is a pip spoke.

    WORKSPACE era = ``pip_<hub>_<pkg>``; under Bzlmod they land on disk under
    canonical repo names, e.g. ``rules_python~~pip~pip_<hub>_<pkg>`` (Bazel 7,
    ``~`` separator), ``rules_python++pip++pip_<...>`` (Bazel 8, ``+`` separator),
    and PPU with its own extension gets
    ``_main~rtp_non_module_deps~pip_ppu_torch_<pkg>``. The check only looks at
    whether the last segment starts with ``pip_``: if any of these naming schemes
    is missed, the setup process cannot import flashinfer or copy it into .cache,
    the JIT's shared ninja cache gets sandbox-temporary paths baked in, and
    recompilation is guaranteed to break once the sandbox is destroyed.
    """
    tail = name.split("~")[-1].split("+")[-1]
    return tail.startswith("pip_")


def get_package_info(package_name):
    """Get package version and installation path"""
    try:
        # Try to import the package first to get its location
        import importlib

        runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
        meta_package_name = package_name
        if package_name == "tvm_ffi":
            meta_package_name = "apache-tvm-ffi"
        elif package_name == "flashinfer_cubin":
            meta_package_name = "flashinfer-cubin"

        if runfiles_dir and os.path.exists(runfiles_dir):
            if runfiles_dir not in sys.path:
                sys.path.insert(0, runfiles_dir)

            for item in os.listdir(runfiles_dir):
                if _is_pip_repo(item):
                    pip_path = os.path.join(runfiles_dir, item, "site-packages")
                    if os.path.exists(pip_path) and pip_path not in sys.path:
                        sys.path.insert(0, pip_path)

        module = importlib.import_module(package_name)
        if hasattr(module, "__file__") and module.__file__:
            package_path = Path(module.__file__).parent
            # Get version from metadata
            try:
                dist = importlib.metadata.distribution(meta_package_name)
                version = dist.version
                return version, str(package_path)
            except:
                # If no metadata, use __version__ attribute
                if hasattr(module, "__version__"):
                    return module.__version__, str(package_path)
        return None, None
    except Exception as e:
        logging.info(f"[Package Copy] Failed to get info for {package_name}: {e}")
        return None, None


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

    # Create target directory with version
    target_base = Path(cache_dir) / f"{package_name}_python-{version}"
    target_site_packages = target_base / "site-packages"
    target_package_path = target_site_packages / Path(source_path).name

    # Lock file for this specific package and version
    lock_file = Path(cache_dir) / f".{package_name}-{version}.lock"
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
                shutil.copytree(source_path, target_package_path, symlinks=True)
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


def add_pip_bin_to_path(runfiles_dir):
    """Merge the bin/ dirs of pip spokes in runfiles into PATH.

    flashinfer's JIT backend invokes `ninja` via subprocess. ninja has no
    entry_points; the executable lives in the wheel's .data/scripts, and the spoke
    BUILD generated by rules_python only globs site-packages -- so MODULE.bazel
    uses whl_mods to attach bin/ to that spoke's pkg data (see the comment there),
    and here we merge it into PATH so the subprocess can find it. x86 has
    flashinfer-jit-cache and skips JIT, so this was never exposed; cuda12_9_arm
    has no aarch64 jit-cache and must JIT.
    """
    if not runfiles_dir or not os.path.exists(runfiles_dir):
        return
    bins = []
    for item in sorted(os.listdir(runfiles_dir)):
        if not _is_pip_repo(item):
            continue
        bin_dir = os.path.join(runfiles_dir, item, "bin")
        if os.path.isdir(bin_dir) and os.listdir(bin_dir):
            bins.append(bin_dir)
    if not bins:
        return
    current = os.environ.get("PATH", "")
    known = current.split(os.pathsep)
    missing = [b for b in bins if b not in known]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + ([current] if current else []))
        logging.info(
            "[Package Setup] Prepended %d pip bin dir(s) to PATH: %s"
            % (len(missing), missing)
        )


def setup_jit_cache(cache_dir=None, packages=None):
    bootstrap_remote_jit_dir()

    # Use defaults if not provided
    if cache_dir is None:
        cache_dir = Path.home().as_posix() + "/.cache"
    if packages is None:
        # flashinfer_cubin must be copied along: the trtllm-gen fmha JIT passes the
        # include directory *inside the cubin package* as -I via extra_include_paths,
        # and the shared ninja cache hard-codes that path. If it stays in runfiles it
        # is a sandbox path that changes per run, and the next recompilation is
        # guaranteed to fail with flashInferMetaInfo.h not found (ut-gb200 in run
        # 57973152); once copied into .cache the path is stable and writable.
        # The x86 line has flashinfer-jit-cache and skips JIT, so this was never
        # exposed; cuda12_9_arm always takes this path.
        packages = [
            "flashinfer",
            "flashinfer_cubin",
            "torch",
            "deep_gemm",
            "tvm_ffi",
        ]

    runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")

    # Copy packages to cache with file locking
    copied_paths = []
    for package_name in packages:
        site_packages_path = copy_package_with_lock(package_name, cache_dir)
        if site_packages_path:
            copied_paths.append(site_packages_path)

    add_pip_bin_to_path(runfiles_dir)

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
    # The main repo's runfiles directory name cannot be hard-coded: WORKSPACE era =
    # the workspace name (rtp_llm), under Bzlmod = _main. Bazel exports the current
    # convention via TEST_WORKSPACE (case_runner.py already uses the same variable).
    workspace = os.environ.get("TEST_WORKSPACE")
    if not workspace:
        logging.warning(
            "[Package Setup] TEST_WORKSPACE unset; skip cache path injection"
        )
        return None
    bazel_wrapper_path = Path(runfiles_dir) / workspace / test_binary
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

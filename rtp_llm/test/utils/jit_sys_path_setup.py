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
import sys
import tempfile
import zipfile
from pathlib import Path

from filelock import FileLock

# importlib.metadata / dist name for each importable package name
_META_PACKAGE_NAMES = {
    "tvm_ffi": "apache-tvm-ffi",
    "flashinfer": "flashinfer-python",
    "flashinfer_jit_cache": "flashinfer-jit-cache",
    "flashinfer_cubin": "flashinfer-cubin",
}

# substrings used to locate matching wheels under runfiles / pip cache
_WHEEL_NAME_HINTS = {
    "flashinfer": "flashinfer_python-",
    "flashinfer_jit_cache": "flashinfer_jit_cache-",
    "flashinfer_cubin": "flashinfer_cubin-",
    "tvm_ffi": "apache_tvm_ffi-",
    "deep_gemm": "deep_gemm-",
    "torch": "torch-",
}


def _meta_package_name(package_name: str) -> str:
    return _META_PACKAGE_NAMES.get(package_name, package_name)


def _add_runfiles_site_packages_to_sys_path(runfiles_dir: str) -> list[str]:
    """Add all pip_*/site-packages under runfiles to sys.path. Return those paths."""
    site_paths = []
    if not runfiles_dir or not os.path.isdir(runfiles_dir):
        return site_paths
    if runfiles_dir not in sys.path:
        sys.path.insert(0, runfiles_dir)
    try:
        entries = os.listdir(runfiles_dir)
    except OSError:
        return site_paths
    for item in entries:
        if not item.startswith("pip_"):
            continue
        pip_path = os.path.join(runfiles_dir, item, "site-packages")
        if os.path.isdir(pip_path):
            site_paths.append(pip_path)
            if pip_path not in sys.path:
                sys.path.insert(0, pip_path)
    return site_paths


def _version_from_package_path(package_name: str, package_path: Path) -> str | None:
    meta_name = _meta_package_name(package_name)
    # Prefer adjacent dist-info / METADATA near site-packages
    site_packages = package_path.parent
    try:
        dist = importlib.metadata.distribution(meta_name)
        return dist.version
    except Exception:
        pass
    for child in site_packages.iterdir() if site_packages.is_dir() else []:
        if child.name.startswith(
            meta_name.replace("-", "_") + "-"
        ) and child.name.endswith(".dist-info"):
            meta = child / "METADATA"
            if meta.exists():
                for line in meta.read_text(errors="ignore").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        if child.name.startswith(meta_name + "-") and child.name.endswith(".dist-info"):
            meta = child / "METADATA"
            if meta.exists():
                for line in meta.read_text(errors="ignore").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
    build_meta = package_path / "_build_meta.py"
    if build_meta.exists():
        for line in build_meta.read_text(errors="ignore").splitlines():
            if line.startswith("__version__"):
                # __version__ = "0.6.9"
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip("\"'")
    return None


def _find_package_dir_in_site_packages(
    site_packages: str, package_name: str
) -> str | None:
    """Find importable package dir, including broken .data/purelib wheel extracts."""
    sp = Path(site_packages)
    direct = sp / package_name
    if (direct / "__init__.py").exists():
        return str(direct)
    # rules_python / raw unzip may leave Root-Is-Purelib:false layout intact
    try:
        for child in sp.iterdir():
            if child.name.endswith(".data") and child.is_dir():
                purelib = child / "purelib" / package_name
                if (purelib / "__init__.py").exists():
                    return str(purelib)
    except OSError:
        pass
    return None


def _find_package_via_filesystem(package_name: str, runfiles_dir: str | None) -> tuple:
    """Locate package without importing (handles purelib / incomplete extracts)."""
    if not runfiles_dir or not os.path.isdir(runfiles_dir):
        return None, None
    try:
        entries = os.listdir(runfiles_dir)
    except OSError:
        return None, None

    pip_dirs = [e for e in entries if e.startswith("pip_")]
    logging.info(
        f"[Package Copy] Scanning {len(pip_dirs)} pip_* runfiles dirs for {package_name}"
    )
    for item in pip_dirs:
        site_packages = os.path.join(runfiles_dir, item, "site-packages")
        if not os.path.isdir(site_packages):
            continue
        pkg_dir = _find_package_dir_in_site_packages(site_packages, package_name)
        if pkg_dir:
            version = (
                _version_from_package_path(package_name, Path(pkg_dir)) or "unknown"
            )
            logging.info(
                f"[Package Copy] Filesystem found {package_name} under {item}: {pkg_dir}"
            )
            return version, pkg_dir
    return None, None


def _iter_candidate_wheel_paths(package_name: str, runfiles_dir: str | None):
    """Yield likely wheel paths without walking huge trees."""
    hint = _WHEEL_NAME_HINTS.get(package_name)
    if not hint:
        return

    def _list_whls(directory: str):
        try:
            for fn in os.listdir(directory):
                if fn.endswith(".whl") and hint in fn:
                    yield os.path.join(directory, fn)
        except OSError:
            return

    # CI fetch logs show wheels saved into the action cwd
    yield from _list_whls(os.getcwd())

    if runfiles_dir and os.path.isdir(runfiles_dir):
        yield from _list_whls(runfiles_dir)
        try:
            for item in os.listdir(runfiles_dir):
                if not item.startswith("pip_"):
                    continue
                base = os.path.join(runfiles_dir, item)
                yield from _list_whls(base)
                # rules_python sometimes keeps the original wheel next to site-packages
                yield from _list_whls(os.path.join(base, "site-packages"))
        except OSError:
            pass

    # pip http cache: ~/.cache/pip/http*/*/flashinfer_jit_cache-*.whl (shallow)
    pip_cache = Path.home() / ".cache" / "pip"
    if pip_cache.is_dir():
        try:
            for dirpath, dirnames, filenames in os.walk(pip_cache):
                for fn in filenames:
                    if fn.endswith(".whl") and hint in fn:
                        yield os.path.join(dirpath, fn)
                rel = os.path.relpath(dirpath, pip_cache)
                if rel.count(os.sep) >= 5:
                    dirnames.clear()
        except OSError:
            pass


def _extract_wheel_package(wheel_path: str, package_name: str) -> tuple:
    """Extract package dir (+ version) from a wheel, handling .data/purelib."""
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            version = "unknown"
            for n in names:
                if n.endswith(".dist-info/METADATA"):
                    for line in zf.read(n).decode(errors="ignore").splitlines():
                        if line.startswith("Version:"):
                            version = line.split(":", 1)[1].strip()
                            break
                    break

            # Preferred: purelib layout
            purelib_prefix = None
            for n in names:
                marker = f".data/purelib/{package_name}/"
                if marker in n:
                    purelib_prefix = n[: n.index(marker) + len(marker)]
                    break

            tmp = tempfile.mkdtemp(prefix=f"{package_name}_whl_")
            if purelib_prefix:
                for n in names:
                    if n.startswith(purelib_prefix) and not n.endswith("/"):
                        dest = Path(tmp) / package_name / n[len(purelib_prefix) :]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(n))
                pkg_dir = Path(tmp) / package_name
            else:
                prefix = package_name + "/"
                for n in names:
                    if n.startswith(prefix) and not n.endswith("/"):
                        dest = Path(tmp) / n
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(n))
                pkg_dir = Path(tmp) / package_name

            if not (pkg_dir / "__init__.py").exists():
                shutil.rmtree(tmp, ignore_errors=True)
                return None, None
            logging.info(
                f"[Package Copy] Extracted {package_name} v{version} from wheel {wheel_path}"
            )
            return version, str(pkg_dir)
    except Exception as e:
        logging.info(f"[Package Copy] Wheel extract failed for {wheel_path}: {e}")
        return None, None


def _find_package_via_wheel(package_name: str, runfiles_dir: str | None) -> tuple:
    for wheel_path in _iter_candidate_wheel_paths(package_name, runfiles_dir):
        version, pkg_dir = _extract_wheel_package(wheel_path, package_name)
        if version and pkg_dir:
            return version, pkg_dir
    return None, None


def get_package_info(package_name):
    """Get package version and installation path"""
    import importlib

    runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    meta_package_name = _meta_package_name(package_name)
    _add_runfiles_site_packages_to_sys_path(runfiles_dir)

    # 1) Normal import from runfiles / sys.path
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, "__file__") and module.__file__:
            package_path = Path(module.__file__).parent
            try:
                dist = importlib.metadata.distribution(meta_package_name)
                version = dist.version
                return version, str(package_path)
            except Exception:
                if hasattr(module, "__version__"):
                    return module.__version__, str(package_path)
                version = _version_from_package_path(package_name, package_path)
                if version:
                    return version, str(package_path)
    except Exception as e:
        logging.info(f"[Package Copy] Import failed for {package_name}: {e}")

    # 2) Filesystem scan (incl. .data/purelib leftovers)
    version, path = _find_package_via_filesystem(package_name, runfiles_dir)
    if version and path:
        return version, path

    # 3) Extract from a wheel already fetched by bazel/pip
    version, path = _find_package_via_wheel(package_name, runfiles_dir)
    if version and path:
        return version, path

    logging.info(f"[Package Copy] Failed to get info for {package_name}")
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
    with FileLock(str(lock_file), timeout=600):
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


def setup_jit_cache(cache_dir=None, packages=None):
    # Use defaults if not provided
    if cache_dir is None:
        cache_dir = Path.home().as_posix() + "/.cache"
    if packages is None:
        # flashinfer_jit_cache must be copied too: AOT .so live there. If only
        # flashinfer is prepended via Package Copy and jit_cache is missing from
        # the test PYTHONPATH, flashinfer falls back to JIT and needs ninja.
        # flashinfer_cubin provides cubins + headers (e.g. flashInferMetaInfo.h)
        # required if any JIT fallback still happens for fmha_gen.
        packages = [
            "flashinfer",
            "flashinfer_jit_cache",
            "flashinfer_cubin",
            "torch",
            "deep_gemm",
            "tvm_ffi",
        ]

    # Avoid flashinfer vs jit-cache local-version mismatches (0.6.9 vs 0.6.9+git)
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")

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
    runfiles_dir = os.environ.get("RUNFILES_DIR", None)
    test_binary = sys.argv[1]
    bazel_wrapper_path = os.path.join(runfiles_dir, "rtp_llm/" + test_binary)
    suffix = f"_new_{os.getpid()}"
    bazel_wrapper_path_new = bazel_wrapper_path + suffix
    try:
        os.remove(bazel_wrapper_path_new)
    except FileNotFoundError:
        pass
    shutil.copy2(bazel_wrapper_path, bazel_wrapper_path_new)
    logging.info(f"[Package Setup] Copied Bazel wrapper to: {bazel_wrapper_path_new}")
    modify_bazel_wrapper_pythonpath(bazel_wrapper_path_new)
    sys.argv[1] = test_binary + suffix

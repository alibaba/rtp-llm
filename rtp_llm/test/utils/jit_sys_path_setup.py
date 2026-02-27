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
from pathlib import Path

from filelock import FileLock


def get_package_info(package_name):
    """Get package version and installation path"""
    try:
        # Try to import the package first to get its location
        import importlib

        runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
        meta_package_name = package_name
        if package_name == "tvm_ffi":
            meta_package_name = "apache-tvm-ffi"

        if runfiles_dir and os.path.exists(runfiles_dir):
            if runfiles_dir not in sys.path:
                sys.path.insert(0, runfiles_dir)

            for item in os.listdir(runfiles_dir):
                if item.startswith("pip_"):
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


def setup_jit_cache(cache_dir=None, packages=None):
    # Use defaults if not provided
    if cache_dir is None:
        cache_dir = Path.home().as_posix() + "/.cache"
    if packages is None:
        packages = ["flashinfer", "torch", "deep_gemm", "tvm_ffi"]

    logging.info("=" * 80)
    logging.info("[Package Setup] Starting JIT cache setup")
    logging.info("=" * 80)

    runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if runfiles_dir and os.path.exists(runfiles_dir):
        logging.info(f"[Package Setup] RUNFILES_DIR: {runfiles_dir}")
        logging.info("[Package Setup] Listing runfiles directory contents:")
        try:
            items = sorted(os.listdir(runfiles_dir))
            for item in items:
                item_path = os.path.join(runfiles_dir, item)
                if os.path.isdir(item_path):
                    logging.info(f"  [DIR]  {item}")
                else:
                    logging.info(f"  [FILE] {item}")
            logging.info(f"[Package Setup] Total items in runfiles: {len(items)}")
        except Exception as e:
            logging.info(f"[Package Setup] Failed to list runfiles: {e}")
    else:
        logging.info("[Package Setup] RUNFILES_DIR not found or doesn't exist")
    logging.info("=" * 80)

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

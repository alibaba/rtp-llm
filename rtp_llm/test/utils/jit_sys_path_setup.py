"""
FlashInfer Path Setup Utility

This module ensures flashinfer is imported from the specified custom path
by inserting it at the beginning of sys.path before any imports.

This is particularly useful in testing environments where you want to use
a specific version of flashinfer different from the system-installed one.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from filelock import FileLock


def get_package_info(package_name):
    """Get package version and installation path"""
    try:
        # Try to import the package first to get its location
        import importlib

        runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")

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
                dist = importlib.metadata.distribution(package_name)
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


def setup_jit_cache(cache_dir=None, packages=None):
    # Use defaults if not provided
    if cache_dir is None:
        cache_dir = Path.home().as_posix() + "/.cache"
    if packages is None:
        packages = ["flashinfer", "torch", "deep_gemm"]

    logging.info("=" * 80)
    logging.info("[Package Setup] Starting JIT cache setup")
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

    os.environ["PYTHONPATH"] = ":".join(copied_paths) + ':' + os.environ.get("PYTHONPATH", "")

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

        try:
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
        except ImportError:
            pass

        # Fallback: try to get from metadata
        dist = importlib.metadata.distribution(package_name)
        version = dist.version

        # Get package location from distribution
        if dist.files:
            # Look for the main package directory
            package_normalized = package_name.lower().replace("-", "_")
            for file in dist.files:
                file_path = Path(str(file))
                # Check if this is a top-level package file
                if len(file_path.parts) > 0:
                    top_dir = file_path.parts[0]
                    if top_dir == package_normalized or top_dir.startswith(
                        package_normalized
                    ):
                        full_path = Path(dist.locate_file(file))
                        package_dir = full_path.parent
                        while package_dir.name != top_dir:
                            package_dir = package_dir.parent
                        return version, str(package_dir)

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


def setup_jit_cache_and_create_bootstrap(cache_dir=None, packages=None):
    """
    Setup JIT package cache and create Bootstrap script.

    Args:
        cache_dir: Cache directory path. Defaults to /home/yangchengjun.ycj/.cache
        packages: List of package names to copy. Defaults to ["flashinfer", "torch", "deep_gemm"]
        logger: Logger instance for output. If None, uses logging.info()

    Returns:
        Bootstrap script path (str) or None if setup fails
    """
    import tempfile

    # Use defaults if not provided
    if cache_dir is None:
        cache_dir = "/home/yangchengjun.ycj/.cache"
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

    # Store cached package paths in environment variable for bootstrap script
    os.environ["_JIT_CACHE_PATHS"] = ":".join(copied_paths)
    logging.info(
        f"[Package Setup] Set _JIT_CACHE_PATHS: {os.environ['_JIT_CACHE_PATHS']}"
    )

    # Store current working directory for Bootstrap script to use
    os.environ["_JIT_ORIGINAL_CWD"] = os.getcwd()
    logging.info(
        f"[Package Setup] Set _JIT_ORIGINAL_CWD: {os.environ['_JIT_ORIGINAL_CWD']}"
    )

    # Read Bootstrap runner template
    bootstrap_template_path = Path(__file__).parent / "bootstrap_runner.py"
    try:
        with open(bootstrap_template_path, "r") as f:
            bootstrap_code = f.read()
    except FileNotFoundError:
        logging.error(
            f"[Package Setup] ERROR: Bootstrap template not found: {bootstrap_template_path}"
        )
        return None

    # Write bootstrap script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        bootstrap_script = f.name
        f.write(bootstrap_code)

    logging.info("=" * 80)
    logging.info(f"[Package Setup] Created bootstrap script: {bootstrap_script}")
    logging.info(f"[Package Setup] Bootstrap script ready for execution")
    logging.info("=" * 80)

    return bootstrap_script

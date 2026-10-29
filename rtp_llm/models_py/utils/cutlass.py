import sys
from importlib import metadata
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    # Bazel keeps the base and CUDA 13 compiler wheels in separate runfiles.
    # Their identical namespace paths must not make import order select CUDA 12.
    try:
        cu13 = metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    except metadata.PackageNotFoundError:
        pass
    else:
        python_packages = Path(cu13.locate_file("nvidia_cutlass_dsl/python_packages"))
        if python_packages.is_dir():
            python_packages_path = str(python_packages)
            if python_packages_path in sys.path:
                sys.path.remove(python_packages_path)
            sys.path.insert(0, python_packages_path)
            return

    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    for package_root in nvidia_cutlass_dsl.__path__:
        python_packages = Path(package_root) / "python_packages"
        if not python_packages.is_dir():
            continue

        python_packages_path = str(python_packages)
        if python_packages_path not in sys.path:
            sys.path.insert(0, python_packages_path)
        return

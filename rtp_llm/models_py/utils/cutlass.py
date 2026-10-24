import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    # Bazel keeps wheel roots separate. Both base and cu13 provide cutlass;
    # prefer the CUDA 13 extra just as pip's merged installation does.
    try:
        cu13 = distribution("nvidia-cutlass-dsl-libs-cu13")
    except PackageNotFoundError:
        package_roots = nvidia_cutlass_dsl.__path__
    else:
        package_roots = [cu13.locate_file("nvidia_cutlass_dsl")]

    for package_root in package_roots:
        python_packages = Path(package_root) / "python_packages"
        if not python_packages.is_dir():
            continue

        python_packages_path = str(python_packages)
        if python_packages_path not in sys.path:
            sys.path.insert(0, python_packages_path)
        return

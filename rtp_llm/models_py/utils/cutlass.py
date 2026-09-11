import sys
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
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

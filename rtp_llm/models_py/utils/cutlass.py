import sys
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    # Bazel does not execute wheel .pth files. CUTLASS 4.7 renamed the
    # directory and split it across namespace packages; keep every root.
    paths = []
    for package_root in nvidia_cutlass_dsl.__path__:
        for directory in ("dsl_packages", "python_packages"):
            python_packages = Path(package_root) / directory
            if not python_packages.is_dir():
                continue
            paths.append(python_packages)

    # rules_python creates namespace __init__.py files in the base/toolkit
    # wheels. The real core package must precede those namespace stubs.
    paths.sort(key=lambda path: not (path / "cutlass/impl_utils.py").is_file())
    for path in reversed(paths):
        # Reapply the complete order: skipping an existing pinned root can
        # put a newly discovered system installation ahead of it.
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))

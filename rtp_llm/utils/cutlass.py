import sys
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``.

    Only call from CuTe-DSL consumers right before importing CuTe-DSL code,
    never eagerly at startup: ``flashinfer.norm`` picks its RMSNorm provider
    (CuTe-DSL vs CUDA JIT) on first import based on whether ``cutlass`` is
    importable, so an eager call changes the norm kernel for every model.

    The directory is prepended so the bundled provider keeps precedence over
    any already-importable ``cutlass`` package; CuTe-DSL consumers written
    against the bundled API must not silently bind to a different install.
    """
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

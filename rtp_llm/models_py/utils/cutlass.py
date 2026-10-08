import importlib
import os
import sys
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    explicit_root = os.environ.get("KIMI_K3_CUTLASS_DSL_ROOT")
    if explicit_root:
        if os.environ.get("MODEL_TYPE") not in ("kimi_k3", "kimi_k3_mtp"):
            raise RuntimeError("KIMI_K3_CUTLASS_DSL_ROOT is only supported for K3")
        root = Path(explicit_root).resolve(strict=True)
        packages = root / "nvidia_cutlass_dsl" / "dsl_packages"
        expected = (packages / "cutlass").resolve(strict=True)
        loaded = sys.modules.get("cutlass")
        if loaded is not None:
            actual = Path(loaded.__file__).parent.resolve(strict=True)
            if actual != expected:
                raise RuntimeError(
                    f"K3 CUTLASS was already imported from {actual}; "
                    f"restart with KIMI_K3_CUTLASS_DSL_ROOT={root}"
                )
        for path in (root, packages):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        # FlashInfer may prepend its vendored CuTeDSL path later. Import the
        # selected package now so that later path changes cannot replace it.
        selected = importlib.import_module("cutlass")
        # The import view links individual files to FlashInfer's bundled
        # package. Resolve the package directory, not __init__.py's target.
        actual = Path(selected.__file__).parent.resolve(strict=True)
        if actual != expected:
            raise RuntimeError(
                f"K3 CUTLASS resolved to {actual}; expected {expected}"
            )
        return

    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    for package_root in nvidia_cutlass_dsl.__path__:
        # CUTLASS DSL 4.7 uses dsl_packages; older wheels use python_packages.
        for directory in ("dsl_packages", "python_packages"):
            python_packages = Path(package_root) / directory
            if not python_packages.is_dir():
                continue
            python_packages_path = str(python_packages)
            if python_packages_path not in sys.path:
                sys.path.insert(0, python_packages_path)
            return

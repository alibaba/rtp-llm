import importlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


def _import_from(packages: Path) -> None:
    expected = (packages / "cutlass").resolve(strict=True)
    value = str(packages)
    if value not in sys.path:
        sys.path.insert(0, value)
    selected = importlib.import_module("cutlass")
    actual = Path(selected.__file__).parent.resolve(strict=True)
    if actual != expected:
        raise RuntimeError(
            f"CUTLASS was already imported from {actual}; expected {expected}. "
            "Initialize CUTLASS before importing dependent kernels."
        )


def setup_cutlass_import_path() -> None:
    """Select the complete Bazel or pip CUTLASS installation before kernel imports."""
    explicit_root = os.environ.get("KIMI_K3_CUTLASS_DSL_ROOT")
    if explicit_root:
        if os.environ.get("MODEL_TYPE") not in ("kimi_k3", "kimi_k3_mtp"):
            raise RuntimeError("KIMI_K3_CUTLASS_DSL_ROOT is only supported for K3")
        root = Path(explicit_root).resolve(strict=True)
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)
        _import_from(root / "nvidia_cutlass_dsl" / "dsl_packages")
        return

    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    # The Bazel assembly target exposes cutlass directly through PyInfo.imports.
    # Prefer it to the incomplete namespaces of the individual pip repositories.
    spec = importlib.util.find_spec("cutlass")
    if spec is not None and spec.origin:
        packages = Path(spec.origin).parent.parent
        if (
            packages.name in ("dsl_packages", "python_packages")
            and packages.parent.name == "nvidia_cutlass_dsl"
        ):
            _import_from(packages)
            selected = sys.modules["cutlass"]
            try:
                version = importlib.metadata.version("nvidia-cutlass-dsl")
            except importlib.metadata.PackageNotFoundError:
                return
            actual_version = getattr(selected, "__version__", version)
            if actual_version != version:
                raise RuntimeError(
                    f"CUTLASS version {actual_version} differs from installed {version}"
                )
            return

    # Installed wheels and older DSL releases can require exposing their nested
    # Python package directory. Their files already share one installation root.
    for root in nvidia_cutlass_dsl.__path__:
        for directory in ("dsl_packages", "python_packages"):
            packages = Path(root) / directory
            if (packages / "cutlass" / "__init__.py").is_file():
                _import_from(packages)
                return

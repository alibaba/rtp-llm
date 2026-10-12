import importlib
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

_install_view = None


def _package_roots() -> list[Path]:
    # Preserve runfiles paths: native loaders find siblings relative to __file__.
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return [
            Path(entry) / "nvidia_cutlass_dsl"
            for entry in sys.path
            if (Path(entry) / "nvidia_cutlass_dsl").is_dir()
        ]
    roots = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if (
            name != "nvidia-cutlass-dsl"
            and not name.startswith("nvidia-cutlass-dsl-libs-")
        ) or distribution.version != version:
            continue
        root = Path(distribution.locate_file("nvidia_cutlass_dsl"))
        # Deduplicate physical directories while preserving runfiles paths.
        if root.is_dir() and not any(root.samefile(existing) for existing in roots):
            roots.append(root)
    return roots


def _merged_install(roots: list[Path]) -> Path:
    global _install_view
    if _install_view is not None:
        return Path(_install_view.name) / "nvidia_cutlass_dsl"

    view = tempfile.TemporaryDirectory(prefix="rtp_cutlass_dsl_")
    destination = Path(view.name)
    installed = set()
    try:
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata["Name"].lower().replace("_", "-")
            if name != "nvidia-cutlass-dsl" and not name.startswith(
                "nvidia-cutlass-dsl-libs-"
            ):
                continue
            candidate = Path(distribution.locate_file("nvidia_cutlass_dsl"))
            if not candidate.is_dir():
                continue
            package_root = next(
                (root for root in roots if candidate.samefile(root)), None
            )
            if package_root is None:
                continue
            files = distribution.files
            if not files:
                # rules_python omits RECORD from runfiles. Exclude only its
                # namespace shims, never the wheel's real initializers.
                files = [
                    source.relative_to(package_root.parent)
                    for source in package_root.rglob("*")
                    if source.is_file()
                    and "__pycache__" not in source.parts
                    and not (
                        source.name == "__init__.py"
                        and source.read_text().startswith(
                            "# __path__ manipulation added by bazelbuild/rules_python"
                        )
                    )
                ]
            for relative in files:
                if relative.parts[0] != "nvidia_cutlass_dsl":
                    continue
                source = Path(distribution.locate_file(relative)).absolute()
                if not source.is_file():
                    raise RuntimeError(f"CUTLASS wheel payload is missing: {source}")
                installed.add(package_root)
                target = destination / relative
                if target.is_symlink():
                    if target.read_bytes() != source.read_bytes():
                        raise RuntimeError(
                            f"Conflicting CUTLASS wheel payload: {relative}"
                        )
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
        if installed != set(roots):
            raise RuntimeError(
                "Split CUTLASS installation has incomplete wheel manifests"
            )
    except BaseException:
        view.cleanup()
        raise
    # Keep the view alive for lazy imports and relative native-library lookup.
    _install_view = view
    return destination / "nvidia_cutlass_dsl"


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
            raise RuntimeError(f"K3 CUTLASS resolved to {actual}; expected {expected}")
        return

    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    roots = _package_roots()
    if len(roots) > 1:
        # The upstream loader scans paths relative to __file__, so extending
        # package __path__ alone cannot join the core/base/CUDA wheel payloads.
        roots = [_merged_install(roots)]
    if not roots:
        roots = [Path(root) for root in nvidia_cutlass_dsl.__path__]
    for package_root in roots:
        # CUTLASS DSL 4.7 uses dsl_packages; older wheels use python_packages.
        for directory in ("dsl_packages", "python_packages"):
            python_packages = Path(package_root) / directory
            if not python_packages.is_dir():
                continue
            python_packages_path = str(python_packages)
            if python_packages_path not in sys.path:
                sys.path.insert(0, python_packages_path)
            expected = (python_packages / "cutlass").resolve(strict=True)
            selected = importlib.import_module("cutlass")
            actual = Path(selected.__file__).parent.resolve(strict=True)
            if actual != expected:
                raise RuntimeError(
                    f"CUTLASS was already imported from {actual}; expected {expected}. "
                    "Initialize CUTLASS before importing dependent kernels."
                )
            return

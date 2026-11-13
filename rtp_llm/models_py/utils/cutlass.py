import base64
import hashlib
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


@lru_cache(maxsize=None)
def _cutlass_dsl_package_paths(cuda_major: str):
    # CuTe 4.8 splits Python, MLIR and runtime files between wheels. Its
    # compiler loader locates the CUDA library next to the Python package.
    package = "nvidia_cutlass_dsl"
    try:
        roots = [
            Path(
                metadata.distribution(f"nvidia-cutlass-dsl-libs-{part}").locate_file(
                    package
                )
            )
            for part in (f"cu{cuda_major}", "base", "core")
        ]
    except metadata.PackageNotFoundError:
        return (), None
    if not all((root / "dsl_packages").is_dir() for root in roots):
        return (), None

    root = roots[0]
    if any(other != root for other in roots[1:]):
        # pip overlays these wheel paths on install. Recreate that layout for
        # Bazel's separate site-packages, with core's package initializer last.
        overlay = TemporaryDirectory(prefix="rtp-cutlass-dsl-")
        root = Path(overlay.name) / package
        for source in roots:
            for entry in source.rglob("*"):
                if "__pycache__" in entry.parts:
                    continue
                target = root / entry.relative_to(source)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.unlink(missing_ok=True)
                    target.symlink_to(entry.absolute())
        # Keep the namespace and its compiler files available for later JITs.
        return (str(root / "dsl_packages"), str(root.parent)), overlay
    return (str(root / "dsl_packages"),), None


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    cuda_major = (torch.version.cuda or "").split(".", 1)[0]
    if cuda_major in ("12", "13"):
        paths, _overlay = _cutlass_dsl_package_paths(cuda_major)
        if paths:
            for path in paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
            return

    # Bazel keeps the base and CUDA 13 compiler wheels in separate runfiles.
    # Their identical namespace paths must not make import order select CUDA 12.
    try:
        if not torch.version.cuda or torch.version.cuda.split(".", 1)[0] != "13":
            raise metadata.PackageNotFoundError("CUDA 13 runtime is not active")
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


def cuda13_cutlass_is_intact() -> bool:
    """Reject merged installations in which the base wheel overwrote CUDA 13.

    Bazel uses separate wheel roots. pip may merge their overlapping namespace;
    dist-info alone then cannot identify which compiler files actually remain.
    """
    try:
        distribution = metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    except metadata.PackageNotFoundError:
        return False
    if distribution.version != "4.4.2" or not distribution.files:
        return False
    loaded = sys.modules.get("cutlass")
    if loaded is not None and getattr(loaded, "__file__", None):
        expected = Path(
            distribution.locate_file(
                "nvidia_cutlass_dsl/python_packages/cutlass/__init__.py"
            )
        ).resolve()
        if Path(loaded.__file__).resolve() != expected:
            return False
    for entry in distribution.files:
        if not str(entry).startswith("nvidia_cutlass_dsl/") or entry.hash is None:
            continue
        path = Path(distribution.locate_file(entry))
        try:
            digest = hashlib.new(entry.hash.mode)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except (OSError, ValueError):
            return False
        actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if actual != entry.hash.value:
            return False
    return True

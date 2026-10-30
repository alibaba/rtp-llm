import base64
import hashlib
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

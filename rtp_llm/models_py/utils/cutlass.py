import base64
import hashlib
import sys
from importlib import metadata
from pathlib import Path

import torch


def setup_cutlass_import_path() -> None:
    """Expose the ``cutlass`` package bundled by ``nvidia-cutlass-dsl``."""
    try:
        core = metadata.distribution("nvidia-cutlass-dsl-libs-core")
        base = metadata.distribution("nvidia-cutlass-dsl-libs-base")
    except metadata.PackageNotFoundError:
        core = base = None
    if core is not None and base is not None and core.version == base.version == "4.7.1":
        distributions = [core, base]
        try:
            distributions.append(metadata.distribution("nvidia-cutlass-dsl-libs-cu12"))
        except metadata.PackageNotFoundError:
            pass
        import nvidia_cutlass_dsl

        namespace_roots = [
            str(Path(distribution.locate_file("nvidia_cutlass_dsl")))
            for distribution in distributions
            if distribution.version == "4.7.1"
        ]
        # Do not expose a stale system 4.4 CUDA13 binary to the 4.7 loader.
        nvidia_cutlass_dsl.__path__ = namespace_roots
        roots = []
        for distribution in distributions:
            for name in ("dsl_packages", "python_packages"):
                path = Path(distribution.locate_file("nvidia_cutlass_dsl/" + name))
                if path.is_dir():
                    roots.append(path)
        for path in reversed(roots):
            if path.is_dir():
                if str(path) in sys.path:
                    sys.path.remove(str(path))
                sys.path.insert(0, str(path))
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


def cuda13_cutlass_is_intact() -> bool:
    """Reject merged installations in which the base wheel overwrote CUDA 13.

    Bazel uses separate wheel roots. pip may merge their overlapping namespace;
    dist-info alone then cannot identify which compiler files actually remain.
    """
    try:
        core = metadata.distribution("nvidia-cutlass-dsl-libs-core")
        base = metadata.distribution("nvidia-cutlass-dsl-libs-base")
        if core.version == base.version == "4.7.1":
            loaded = sys.modules.get("cutlass")
            if loaded is not None and getattr(loaded, "__file__", None):
                expected = Path(core.locate_file("nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py")).resolve()
                return Path(loaded.__file__).resolve() == expected
            return True
    except metadata.PackageNotFoundError:
        pass
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

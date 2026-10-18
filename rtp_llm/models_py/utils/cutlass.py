import sys
from pathlib import Path


def setup_cutlass_import_path() -> None:
    """Expose CUTLASS's wheel layout without executing arbitrary .pth files."""
    # A Bazel overlay restores split wheels' original adjacent-file layout.
    # Find it before importing a globally installed or raw split namespace.
    package_roots = [
        Path(entry) / "nvidia_cutlass_dsl"
        for entry in sys.path
        if isinstance(entry, str)
        and (Path(entry) / "nvidia_cutlass_dsl" / ".rtp_bazel_overlay").is_file()
    ]
    using_overlay = bool(package_roots)
    if not package_roots:
        try:
            import nvidia_cutlass_dsl
        except ImportError:
            return
        package_roots = [Path(root) for root in nvidia_cutlass_dsl.__path__]

    for package_root in package_roots:
        for layout in ("dsl_packages", "python_packages"):
            packages = package_root / layout
            if not (packages / "cutlass" / "__init__.py").is_file():
                continue
            packages_path = str(packages)
            preferred = [packages_path]
            if using_overlay:
                preferred.append(str(package_root.parent))
                # Do not mix an already-imported global version with this closure.
                for name, root in (
                    ("cutlass", packages / "cutlass"),
                    ("nvidia_cutlass_dsl", package_root),
                ):
                    loaded = sys.modules.get(name)
                    paths = getattr(loaded, "__path__", ())
                    origin = getattr(loaded, "__file__", None)
                    if loaded is not None and not (
                        origin
                        and Path(origin).resolve() == (root / "__init__.py").resolve()
                        and paths
                        and Path(next(iter(paths))).resolve() == root.resolve()
                    ):
                        raise ImportError(
                            f"{name} was imported before the selected CUTLASS "
                            "wheel overlay; start a fresh process"
                        )
            sys.path[:] = preferred + [
                entry for entry in sys.path if entry not in preferred
            ]
            return

def _ensure_cutlass_importable():
    """Ensure cutlass package is importable.

    nvidia-cutlass-dsl installs its Python packages under
    nvidia_cutlass_dsl/python_packages/ and relies on a .pth file to add that
    directory to sys.path. In sandboxed environments (like Bazel) .pth files
    are not processed, so we need to add the path manually.
    """
    try:
        import cutlass  # noqa: F401

        return  # already importable
    except ImportError:
        pass

    import importlib
    import sys

    try:
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        if spec is None or not spec.submodule_search_locations:
            return
        for base in spec.submodule_search_locations:
            import os

            pp = os.path.join(base, "python_packages")
            if os.path.isdir(pp) and pp not in sys.path:
                sys.path.insert(0, pp)
    except Exception:
        pass


_ensure_cutlass_importable()

try:
    import cutlass

    _ver = getattr(cutlass, "__version__", "0.0.0")
    if tuple(int(x) for x in _ver.split(".")[:3]) < (4, 4, 2):
        import logging as _log

        _log.getLogger(__name__).info(
            "nvidia-cutlass-dsl %s < 4.4.2, Blackwell GDN kernel disabled", _ver
        )
        chunk_gated_delta_rule = None  # type: ignore
    else:
        from .gdn import chunk_gated_delta_rule
except Exception:
    chunk_gated_delta_rule = None  # type: ignore

__all__ = ["chunk_gated_delta_rule"]

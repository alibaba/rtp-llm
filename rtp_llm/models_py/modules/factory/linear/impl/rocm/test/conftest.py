_aiter_available = False
try:
    import aiter  # noqa: F401
    _aiter_available = True
except (ImportError, RuntimeError):
    pass

if not _aiter_available:
    collect_ignore_glob = ["*.py"]

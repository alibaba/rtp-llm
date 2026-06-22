_aiter_available = False
try:
    import aiter  # noqa: F401

    _aiter_available = True
except (ImportError, RuntimeError):
    pass

if not _aiter_available:
    # Only ignore tests that require aiter/ROCm hardware. CPU-only regression
    # tests (e.g. torch_moe_ref_test.py, test_pure_tp_router.py) must remain
    # collectable so they run in generic CI.
    collect_ignore_glob = [
        "rocm_*_test.py",
        "deepep_*_test.py",
        "moriep_*_test.py",
    ]

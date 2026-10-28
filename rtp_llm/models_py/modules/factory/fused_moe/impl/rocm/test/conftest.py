_aiter_available = False
try:
    import aiter  # noqa: F401

    _aiter_available = True
except (ImportError, RuntimeError):
    pass

if not _aiter_available:
    # Only ignore tests that require aiter/ROCm hardware. CPU-only regression
    # tests such as test_pure_tp_router.py must remain collectable so they run
    # in generic CI. Tests that use CPU tensors but import the ROCm executor
    # stack, such as torch_moe_ref_test.py, are routed with an MI308X marker.
    #
    # NOTE: these globs key off the filename prefix, so a CPU-only test in this
    # directory must NOT be named rocm_*/deepep_*/moriep_* or it will be silently
    # ignored when aiter is unavailable.
    collect_ignore_glob = [
        "test_generic_moe_allreduce.py",
        "rocm_*_test.py",
        "deepep_*_test.py",
        "moriep_*_test.py",
    ]

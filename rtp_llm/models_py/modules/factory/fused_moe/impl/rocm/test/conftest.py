import pytest

# Skip all tests in this directory if aiter (ROCm-only package) is not available.
# pytest.importorskip only catches ImportError, but aiter raises RuntimeError
# when ROCm is not present (e.g. on CUDA hosts where aiter is installed but
# the ROCm runtime is missing).
try:
    import aiter  # noqa: F401
except (ImportError, RuntimeError) as e:
    pytest.skip(f"ROCm-only: aiter unavailable ({e})", allow_module_level=True)

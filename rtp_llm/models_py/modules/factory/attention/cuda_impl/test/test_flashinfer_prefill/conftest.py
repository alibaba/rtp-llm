import pytest

try:
    import flashinfer  # noqa: F401
except ImportError as e:
    pytest.skip(f"flashinfer unavailable: {e}", allow_module_level=True)

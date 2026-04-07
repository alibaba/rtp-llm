import pytest

try:
    import deep_ep  # noqa: F401
except ImportError as e:
    pytest.skip(f"deep_ep unavailable: {e}", allow_module_level=True)

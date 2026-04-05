import pytest

try:
    from rtp_llm.ops.compute_ops import FusedMoEOp  # noqa: F401
except ImportError as e:
    pytest.skip(f"FusedMoEOp unavailable: {e}", allow_module_level=True)

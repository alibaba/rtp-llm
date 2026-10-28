import pytest

try:
    from rtp_llm.ops.compute_ops import trt_fp8_quantize_128  # noqa: F401
except ImportError as e:
    pytest.skip(
        f"compute_ops missing trt_fp8_quantize_128: {e}", allow_module_level=True
    )

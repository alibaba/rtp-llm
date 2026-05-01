import pytest

from rtp_llm.device.device_type import is_cuda

# Layer 1: hard platform gate. The ROCm build of rtp_llm_ops loads fine, but
# does NOT export FlashInferMlaAttnParams (CUDA-only pybind binding), so test
# files would import their wrappers successfully and only fail later at
# attribute access — short-circuit on non-CUDA platforms before that.
if not is_cuda():
    pytest.skip(
        "cuda_mla_impl tests require CUDA (FlashInferMlaAttnParams is not "
        "built into the ROCm rtp_llm_ops binding)",
        allow_module_level=True,
    )

# Layer 2: even on CUDA, skip if the .so itself isn't built. Only the
# `flashmla_benchmark.py` script in this directory uses the Python-side
# `flashinfer` package, and it has no `test_*` functions so pytest doesn't
# collect it anyway. Actual tests here only need `rtp_llm_ops` (C++ binding) —
# gating on Python-side `flashinfer` would be over-broad.
try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops  # noqa: F401
except ImportError as e:
    pytest.skip(
        f"rtp_llm compute_ops unavailable (likely missing libcuda or unbuilt .so): {e}",
        allow_module_level=True,
    )

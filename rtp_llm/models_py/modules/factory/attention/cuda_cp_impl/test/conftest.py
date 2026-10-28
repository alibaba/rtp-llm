import pytest

from rtp_llm.device.device_type import is_cuda

if not is_cuda():
    pytest.skip(
        "cuda_cp_impl tests require CUDA (flashinfer + fill_mla_params are "
        "not available in the ROCm build of rtp_llm_ops)",
        allow_module_level=True,
    )

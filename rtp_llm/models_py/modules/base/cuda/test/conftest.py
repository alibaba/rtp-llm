import pytest

from rtp_llm.device.device_type import is_cuda

if not is_cuda():
    pytest.skip(
        "base/cuda tests require CUDA-side rtp_llm_ops kernels "
        "(layernorm/topk/symm_mem variants not in the ROCm build)",
        allow_module_level=True,
    )

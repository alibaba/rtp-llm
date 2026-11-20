"""Device-stable selection for DSv4 attention and projection implementations."""

from functools import cache
from typing import Optional

import torch

from rtp_llm.device.device_type import is_cuda
from rtp_llm.models_py.utils.arch import is_sm120


@cache
def _is_cuda_runtime() -> bool:
    return is_cuda()


def cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not _is_cuda_runtime():
        return None
    # Never cache the alias: current_device can change between model warmups.
    return device.index if device.index is not None else torch.cuda.current_device()


@cache
def _is_sm120_index(index: int) -> bool:
    return is_sm120(torch.device("cuda", index))


def is_sm120_device(device: torch.device) -> bool:
    index = cuda_device_index(device)
    return index is not None and _is_sm120_index(index)


def require_sm120_providers(device: torch.device) -> None:
    """Require the registered SM120 operations on the selected device."""
    index = cuda_device_index(device)
    if index is not None and _is_sm120_index(index):
        _require_sm120_providers(index)


@cache
def _require_sm120_providers(index: int) -> None:
    # Imported here: provider modules load only on SM120 workers.
    from rtp_llm.models_py.kernels.cuda.sm120_fp8_blockwise import (
        sm120_blockwise_fp8_available,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import sparse_mla_available
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
        _has_sm120_grouped_fp8_fp4_kernel,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_grouped_fp4 import (
        sm120_eager_fp4_available,
        sm120_graph_fp4_available,
    )

    device = torch.device("cuda", index)
    missing = [
        name
        for name, available in (
            (
                "the SM120 FP8 blockwise GEMM binding",
                sm120_blockwise_fp8_available(index),
            ),
            ("the SM120 sparse MLA operation", sparse_mla_available()),
            (
                "the FlashInfer eager grouped FP4 operations",
                sm120_eager_fp4_available(device),
            ),
            (
                "the SM120 graph MoE operation and workspace API",
                sm120_graph_fp4_available(device),
            ),
            (
                "the SM120 grouped FP8×FP4 decode operation",
                _has_sm120_grouped_fp8_fp4_kernel(device),
            ),
        )
        if not available
    ]
    if missing:
        raise RuntimeError(
            "DeepSeek-V4 on SM120 requires the x86 CUDA 13 RTP-LLM build; "
            "this environment lacks " + ", ".join(missing)
        )

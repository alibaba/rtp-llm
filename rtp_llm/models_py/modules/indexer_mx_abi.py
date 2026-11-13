"""DeepGEMM indexer cache ABI selected for the active device and wheel."""

from functools import lru_cache

import torch

try:
    import deep_gemm
except Exception:
    deep_gemm = None


@lru_cache(maxsize=None)
def use_mx_indexer_cache(device: torch.device) -> bool:
    """SM100+ DeepGEMM 2.8.2 requires packed MX scales for FP8 MQA."""
    device = torch.device(device)
    return (
        device.type == "cuda"
        and deep_gemm is not None
        and hasattr(deep_gemm, "fp8_fp4_mqa_logits")
        and hasattr(deep_gemm, "fp8_fp4_paged_mqa_logits")
        and hasattr(deep_gemm, "get_paged_mqa_logits_metadata")
        and not hasattr(deep_gemm, "fp8_mqa_logits")
        and torch.cuda.get_device_capability(device)[0] == 10
    )
